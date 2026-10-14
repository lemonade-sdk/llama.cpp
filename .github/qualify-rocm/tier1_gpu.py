#!/usr/bin/env python3
"""
Tier 1 -- GPU functional inference. Requires a real AMD GPU (gfx1151 runners).

Drives llama-server over HTTP rather than llama-cli, because llama-server is
the binary the downstream consumer actually launches, with the flag set
reproduced in SERVER_ARGS below.

T1.2 is the check that carries the weight here. A GPU backend that fails to
load is not treated as an error by llama.cpp: it logs the failure, skips the
backend and continues on the CPU. So a test that only asks "did the server
answer?" passes happily on a build whose GPU support is entirely broken -- the
answers are just much slower. T1.2 confirms the GPU really did the work.

Checks:
  T1.1  server boot            -- llama-server starts and serves /health.
  T1.2  GPU offload confirmed  -- the device/offload lines prove work landed on
                                  the GPU (gating; the anti-CPU-fallback gate).
  T1.3  response well-formed   -- non-empty, valid UTF-8, no replacement chars,
                                  sane length, proper finish_reason.
  T1.4  greedy determinism     -- two identical temperature=0 requests return
                                  byte-identical text (gating; model-agnostic
                                  evidence the sampling loop saw sane logits).
  T1.5  closed </think> tag    -- NON-gating. A Qwen3-thinking artifact kept as
                                  a data point; T1.3 and T1.4 are the real
                                  correctness signals.

Usage:
    python3 tier1_gpu.py --archive-root ./extracted --model ./Qwen3-0.6B-Q4_0.gguf \
        --rocm-version 10.1 --gfx-target gfx1151 --output tier1.json
"""

import argparse
import glob
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

READY_TIMEOUT = 600
REQUEST_TIMEOUT = 180

# The flags the downstream consumer launches llama-server with. Using the same
# ones here means this tier exercises the real integration surface rather than
# a simplified version of it.
SERVER_ARGS = ["--ctx-size", "4096", "--jinja", "--metrics"]

# Device/offload lines that prove llama-server placed work on the GPU.
GPU_OFFLOAD_PATTERNS = [
    re.compile(r"using device ROCm"),
    re.compile(r"offloaded\s+[1-9]\d*/\d+\s+layers to GPU"),
]

CPU_FALLBACK_HINTS = [
    re.compile(r"failed to load .*ggml-hip", re.I),
    re.compile(r"no usable GPU", re.I),
    re.compile(r"offloaded\s+0/\d+\s+layers to GPU"),
]


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

def http_get(url, timeout=10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read()
    except (urllib.error.URLError, OSError):
        return None, b""


def http_post_json(url, payload, timeout=REQUEST_TIMEOUT):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    # Decode strictly: a malformed byte sequence here is itself a finding, and
    # T1.3 wants to know about it rather than have it silently replaced.
    return json.loads(raw.decode("utf-8"))


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def stage_hip_runtime(rocm_dirs, exe_dir):
    """Windows only: copy the HIP runtime next to llama-server.exe.

    Windows resolves DLLs from System32 BEFORE PATH, so a stale System32
    amdhip64_*.dll shadows the ROCm we just installed and the binary silently
    binds to the wrong runtime.
    """
    if os.name != "nt":
        return []
    staged = []
    for directory in rocm_dirs:
        for pattern in ("amdhip64*.dll", "amd_comgr*.dll", "rocm_kpack*.dll"):
            for src in glob.glob(os.path.join(directory, pattern)):
                dst = os.path.join(exe_dir, os.path.basename(src))
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    staged.append(os.path.basename(src))
    return staged


def build_env(rocm_dirs, exe_dir, gfx_target):
    env = dict(os.environ)
    joined = os.pathsep.join(rocm_dirs + [exe_dir])
    if os.name == "nt":
        env["PATH"] = joined + os.pathsep + env.get("PATH", "")
        # Raises the shared-memory limit on these targets so larger models fit.
        if gfx_target in ("gfx1151", "gfx1152"):
            env["OCL_SET_SVM_SIZE"] = "262144"
    else:
        env["LD_LIBRARY_PATH"] = joined + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    return env


class Server:
    def __init__(self, exe, model, port, log_path, env, cwd):
        self.exe = exe
        self.model = model
        self.port = port
        self.log_path = log_path
        self.env = env
        self.cwd = cwd
        self.proc = None
        self._log = None

    def start(self):
        cmd = [self.exe, "--model", self.model, "--host", "127.0.0.1",
               "--port", str(self.port), "-ngl", "99", *SERVER_ARGS]
        print("+ " + " ".join(cmd), flush=True)
        self._log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        kwargs = {}
        if os.name != "nt":
            kwargs["start_new_session"] = True
        self.proc = subprocess.Popen(cmd, stdout=self._log,
                                     stderr=subprocess.STDOUT, env=self.env,
                                     cwd=self.cwd, **kwargs)

    def wait_ready(self, timeout):
        proc = self.proc
        if proc is None:
            return False, "llama-server was never started"
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False, f"llama-server exited early (rc={proc.returncode})"
            status, _ = http_get(f"http://127.0.0.1:{self.port}/health")
            if status == 200:
                return True, None
            time.sleep(3)
        return False, f"llama-server not ready within {timeout}s"

    def stop(self):
        if self.proc and self.proc.poll() is None:
            try:
                if os.name == "nt":
                    self.proc.terminate()
                else:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=30)
            except (ProcessLookupError, OSError, subprocess.TimeoutExpired):
                try:
                    self.proc.kill()
                except OSError:
                    pass
        if self._log:
            self._log.close()

    def log_text(self):
        try:
            with open(self.log_path, encoding="utf-8", errors="replace") as handle:
                return handle.read()
        except OSError:
            return ""


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_gpu_offload(server, base):
    log = server.log_text()
    matched = [p.pattern for p in GPU_OFFLOAD_PATTERNS if p.search(log)]

    props_status, props_raw = http_get(f"{base}/props")
    props = {}
    if props_status == 200:
        try:
            props = json.loads(props_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            props = {}

    details = {
        "patterns_matched": matched,
        "offload_line": next(
            (m.group(0) for m in
             [re.search(r"offloaded\s+\d+/\d+\s+layers to GPU", log)] if m), None),
        "device_line": next(
            (m.group(0) for m in [re.search(r"using device ROCm\S*.*", log)] if m), None),
        # /props is supplementary: its schema is not stable across upstream
        # releases, so it is recorded but never gates.
        "props_status": props_status,
        "props_n_gpu_layers": (props.get("default_generation_settings") or {})
        .get("n_gpu_layers"),
    }

    if matched:
        return (report.STATUS_PASS, None, details)

    hints = [p.pattern for p in CPU_FALLBACK_HINTS if p.search(log)]
    details["cpu_fallback_hints"] = hints
    details["log_tail"] = log[-2000:]
    reason = ("the GPU backend did not load and llama.cpp fell back to the CPU"
              if hints else "no device/offload line was logged at all")
    return (report.STATUS_FAIL,
            f"GPU offload NOT confirmed -- {reason}. With GGML_BACKEND_DL=ON this "
            "is silent: the server still answers requests, just on the CPU.",
            details)


def _content(body):
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    text = (message.get("content") or "") + (message.get("reasoning_content") or "")
    return text, choice.get("finish_reason")


def chat(base, prompt, max_tokens=192, seed=0):
    return http_post_json(f"{base}/v1/chat/completions", {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
    })


def check_well_formed(base, state):
    body = chat(base, "In one short paragraph, what is the capital of France?")
    text, finish = _content(body)
    state["sample"] = text

    bad_chars = text.count("�")
    details = {
        "finish_reason": finish,
        "chars": len(text),
        "sample": text[:200],
        "replacement_chars": bad_chars,
    }
    problems = []
    if not text.strip():
        problems.append("response is empty")
    if bad_chars:
        problems.append(f"{bad_chars} U+FFFD replacement character(s) "
                        "-- the decoder produced invalid UTF-8")
    if 0 < len(text.strip()) < 8:
        problems.append(f"response is implausibly short ({len(text.strip())} chars)")
    if finish not in ("stop", "length"):
        problems.append(f"finish_reason is {finish!r}, expected 'stop' or 'length'")

    if problems:
        return (report.STATUS_FAIL, "; ".join(problems), details)
    return (report.STATUS_PASS, None, details)


def check_determinism(base):
    prompt = "Count from one to five, separated by commas."
    outputs = []
    for _ in range(2):
        body = chat(base, prompt, max_tokens=64)
        outputs.append(_content(body)[0])

    details = {"first": outputs[0][:200], "second": outputs[1][:200],
               "identical": outputs[0] == outputs[1]}
    if not outputs[0].strip():
        return (report.STATUS_FAIL, "greedy request returned nothing", details)
    if outputs[0] != outputs[1]:
        return (report.STATUS_FAIL,
                "two identical temperature=0 requests produced different text -- "
                "the sampling path is non-deterministic, which usually means "
                "uninitialised memory or NaN logits in the GPU kernels", details)
    return (report.STATUS_PASS, None, details)


def check_think_tag(state):
    text = state.get("sample") or ""
    closed = "</think>" in text
    details = {"found": closed, "sample": text[:200]}
    if closed:
        return (report.STATUS_PASS, None, details)
    return (report.STATUS_WARN,
            "no closed </think> tag. This is a Qwen3-thinking chat-template "
            "artifact, not a correctness signal -- T1.3/T1.4 are authoritative.",
            details)


def main():
    parser = argparse.ArgumentParser(description="Tier 1 GPU functional inference")
    parser.add_argument("--archive-root", required=True,
                        help="directory holding the extracted llama.cpp binaries")
    parser.add_argument("--model", required=True, help="path to the .gguf model")
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--gfx-target", default="gfx1151")
    parser.add_argument("--artifact", default=None, help="release asset filename")
    parser.add_argument("--tag", default=None)
    parser.add_argument("--rocm-dirs", default=os.environ.get("ROCM_RUNTIME_DIRS", ""),
                        help="os.pathsep-separated ROCm runtime directories")
    parser.add_argument("--port", type=int, default=8231)
    parser.add_argument("--logs-dir", default=".")
    parser.add_argument("--ready-timeout", type=int, default=READY_TIMEOUT)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output", default="tier1-report.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.archive_root)
    exe = os.path.join(root, "llama-server.exe" if os.name == "nt" else "llama-server")
    if not os.path.exists(exe):
        raise SystemExit(f"::error::llama-server not found at {exe}")
    if os.name != "nt":
        for entry in os.listdir(root):
            path = os.path.join(root, entry)
            if os.path.isfile(path):
                os.chmod(path, 0o755)

    rocm_dirs = [d for d in args.rocm_dirs.split(os.pathsep) if d]
    if not rocm_dirs:
        raise SystemExit(
            "::error::No ROCm runtime directories supplied. This repo's archives "
            "do not bundle ROCm -- run install_rocm.py first and pass its "
            "ROCM_RUNTIME_DIRS through --rocm-dirs.")

    staged = stage_hip_runtime(rocm_dirs, root)
    if staged:
        print(f"Staged HIP runtime next to llama-server: {', '.join(staged)}")

    meta = report.build_meta(
        artifact=args.artifact,
        os_name="win" if os.name == "nt" else "ubuntu",
        rocm_version=args.rocm_version,
        llamacpp_tag=args.tag,
        gfx_target=args.gfx_target,
        hardware_validated=True,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    tier = report.TierReport("tier1", meta)

    os.makedirs(args.logs_dir, exist_ok=True)
    log_path = os.path.join(
        args.logs_dir, f"llama-server-{args.gfx_target}-rocm{args.rocm_version}.log")
    base = f"http://127.0.0.1:{args.port}"
    env = build_env(rocm_dirs, root, args.gfx_target)
    server = Server(exe, os.path.abspath(args.model), args.port, log_path, env, root)
    state = {}

    try:
        server.start()
        ready, err = server.wait_ready(args.ready_timeout)
        if ready:
            tier.add("T1.1", "server boot", report.STATUS_PASS,
                     details={"port": args.port, "args": SERVER_ARGS})
        else:
            tier.add("T1.1", "server boot", report.STATUS_FAIL, error=err,
                     details={"log_tail": server.log_text()[-3000:]})

        if ready:
            tier.run("T1.2", "GPU offload confirmed",
                     lambda: check_gpu_offload(server, base))
            tier.run("T1.3", "response well-formed",
                     lambda: check_well_formed(base, state))
            tier.run("T1.4", "greedy determinism", lambda: check_determinism(base))
            tier.run("T1.5", "closed </think> tag",
                     lambda: check_think_tag(state), gating=False)
        else:
            for test_id, name, gating in [
                ("T1.2", "GPU offload confirmed", True),
                ("T1.3", "response well-formed", True),
                ("T1.4", "greedy determinism", True),
                ("T1.5", "closed </think> tag", False),
            ]:
                tier.add(test_id, name, report.STATUS_SKIP,
                         gating=gating, error="server not ready")
    finally:
        server.stop()

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
