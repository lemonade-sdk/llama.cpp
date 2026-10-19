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
  T1.2  GPU really did the work -- GATING. A ROCm device was selected AND ROCm
                                  memory was allocated. The familiar
                                  "offloaded N/N layers to GPU" line is NOT
                                  accepted as evidence: llama.cpp prints it
                                  from what -ngl requested, so it appears
                                  unchanged on a machine with no GPU at all.
  T1.3  response well-formed   -- GATING. Non-empty, valid UTF-8, no
                                  replacement characters, sensible length,
                                  proper finish_reason.
  T1.4  greedy determinism     -- NON-gating. The same question asked twice
                                  with randomness off should give the same
                                  answer. A difference is a hint, not a
                                  verdict: see check_determinism().
  T1.5  reasoning pass         -- NON-gating. The model thought, answered, and
                                  stopped cleanly.

Usage:
    python3 tier1_gpu.py --meta qualify-meta-ubuntu-rocm-10.1.json \
        --archive-dir ./artifact --output tier1.json
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
import tarfile
import tempfile
import zipfile
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

READY_TIMEOUT = 600
REQUEST_TIMEOUT = 180

# The flags the downstream consumer launches llama-server with, so this tier
# exercises the real integration surface rather than a simplified version.
# `-lv 4` raises the log verbosity, and it is not optional.
# At the default setting llama-server prints about nine lines -- "model loaded", "listening
# on ..." -- and none of the device or memory information T1.2 relies on.
SERVER_ARGS = ["--ctx-size", "4096", "--jinja", "--metrics", "-lv", "4"]

# Evidence that the GPU genuinely did the work. Both name a ROCm device, so
# neither can appear unless a ROCm device was found and used:
#   "llama_prepare_model_devices: using device ROCm0 (...) - 12345 MiB free"
#   "load_tensors:        ROCm0 model buffer size =  357.09 MiB"
GPU_PROOF_PATTERNS = [
    re.compile(r"using device (ROCm\d+)"),
    re.compile(r"(ROCm\d+)[^\n]*buffer size"),
]

# Deliberately NOT treated as proof, despite how it reads.
# llama.cpp prints "offloaded 29/29 layers to GPU" from the number of layers
# REQUESTED through -ngl, before any GPU is involved. It appears word for word
# on a machine with no GPU at all, where every buffer is then reported as CPU.
# A check that accepted this line would pass on exactly the silent CPU fallback
# it exists to catch, so it is recorded only for context and nothing more.
REQUESTED_OFFLOAD_RE = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")

# Lines that explain WHY the GPU went unused, so the failure message can say
# something more useful than "no evidence found".
CPU_FALLBACK_HINTS = [
    re.compile(r"failed to load .*ggml-hip", re.I),
    re.compile(r"no usable GPU", re.I),
    re.compile(r"offloaded\s+0/\d+\s+layers to GPU"),
]


# ---------------------------------------------------------------------------
# Inputs this script fetches for itself
#
# Extraction and downloading steps live here so the job works on any runner regardless of which shell it has.
# A self-hosted machine is not guaranteed to provide the same tools a hosted one does.
# ---------------------------------------------------------------------------

MODEL_URL = ("https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/"
             "Qwen3-0.6B-Q4_0.gguf")


def extract_archive(archive, dest):
    """Unpack a published archive and return the folder holding the binaries.

    Linux archives nest everything inside a llama-<tag>/ folder; Windows
    archives put the files at the top level. Rather than encode that
    difference, find llama-server and use whatever folder it landed in.
    """
    os.makedirs(dest, exist_ok=True)
    if archive.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tar:
            try:
                tar.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest)
    else:
        raise SystemExit(f"::error::unsupported archive type: {archive}")

    # Accept either name rather than the one matching the host, so an archive
    # can also be unpacked and inspected from the other operating system.
    for dirpath, _dirnames, filenames in os.walk(dest):
        if {"llama-server", "llama-server.exe"} & set(filenames):
            return dirpath
    raise SystemExit(f"::error::llama-server not found anywhere inside {archive}")


def download_model(url, path, attempts=3):
    """Fetch the test model, reusing an existing copy when there is one."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        print(f"Model already present: {path} ({os.path.getsize(path):,} bytes)")
        return path
    last = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"Downloading model (attempt {attempt}/{attempts}): {url}")
            with urllib.request.urlopen(url, timeout=600) as resp, \
                    open(path, "wb") as handle:
                shutil.copyfileobj(resp, handle)
            print(f"Downloaded {os.path.getsize(path):,} bytes to {path}")
            return path
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            if os.path.exists(path):
                os.remove(path)
            time.sleep(5 * attempt)
    raise SystemExit(f"::error::could not download the test model: {last}")


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
    """Confirm the GPU actually did the work, not merely that it was asked to.

    This is the check the tier exists for. A GPU backend that fails to load is
    not an error to llama.cpp: it logs the failure, skips the backend and keeps
    serving from the CPU. Every other check here passes happily in that state.
    """
    log = server.log_text()

    proof = []
    for pattern in GPU_PROOF_PATTERNS:
        for match in pattern.finditer(log):
            line = match.group(0).strip()
            if line not in proof:
                proof.append(line)

    requested = REQUESTED_OFFLOAD_RE.search(log)
    devices = re.findall(r"^\s*-\s+(\S+)\s*:.*MiB", log, re.MULTILINE)

    props_status, props_raw = http_get(f"{base}/props")
    props = {}
    if props_status == 200:
        try:
            props = json.loads(props_raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            props = {}

    details = {
        "gpu_proof": proof[:10],
        # Recorded, never decisive -- see REQUESTED_OFFLOAD_RE above.
        "requested_offload_line": requested.group(0) if requested else None,
        "devices_reported": sorted(set(devices)),
        # /props is supplementary: its shape changes between upstream releases,
        # so it is recorded but never gates.
        "props_status": props_status,
        "props_n_gpu_layers": (props.get("default_generation_settings") or {})
        .get("n_gpu_layers"),
    }

    if proof:
        print("GPU confirmed by:")
        for line in proof[:10]:
            print(f"    {line}")
        return (report.STATUS_PASS, None, details)

    hints = [p.pattern for p in CPU_FALLBACK_HINTS if p.search(log)]
    details["cpu_fallback_hints"] = hints
    details["log_tail"] = log[-3000:]

    if hints:
        reason = "the GPU backend did not load, so llama.cpp used the CPU"
    elif requested:
        reason = (f"the only offload line present, '{requested.group(0)}', reports "
                  "what -ngl asked for rather than what happened; no ROCm device "
                  "and no ROCm memory allocation were logged")
    else:
        reason = "no ROCm device and no ROCm memory allocation were logged"

    return (report.STATUS_FAIL,
            f"GPU was NOT used -- {reason}. This is silent at run time: the "
            "server still starts and still answers, just on the CPU.", details)


def _parts(body):
    """Split a reply into (reasoning, answer, stop reason).

    A model that thinks before answering has that thinking separated out by the
    server into its own field, so the two arrive apart rather than as one piece
    of text with markers in it.
    """
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    return ((message.get("reasoning_content") or ""),
            (message.get("content") or ""),
            choice.get("finish_reason"))


def _content(body):
    reasoning, answer, finish = _parts(body)
    return reasoning + answer, finish


def chat(base, prompt, max_tokens=192, seed=0, cache_prompt=True):
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
    }
    if not cache_prompt:
        # Stops the second of two identical requests from reusing work cached
        # by the first, so comparing them is like for like.
        payload["cache_prompt"] = False
    return http_post_json(f"{base}/v1/chat/completions", payload)


def check_well_formed(base, state):
    # A model that thinks before answering spends most of its budget on the
    # thinking, so this needs enough room to reach an actual answer.
    body = chat(base, "In one short paragraph, what is the capital of France?",
                max_tokens=512)
    reasoning, answer, finish = _parts(body)
    text = reasoning + answer
    state.update(reasoning=reasoning, answer=answer, finish=finish)

    bad_chars = text.count("�")
    details = {
        "finish_reason": finish,
        "chars": len(text),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(answer),
        "sample": text[:400],
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
    """Ask the same question twice with randomness off and compare the answers.

    Non-gating, and worth being honest about why. On a GPU thousands of threads
    add numbers in whatever order they happen to finish, so results can differ
    in the last decimal place from one run to the next. That is normal and
    harmless. It only changes the reply when two candidate words are almost
    exactly tied -- then the pick flips, and because each word feeds the next,
    everything after it diverges.

    So a difference here is a hint worth a look, not proof of a defect. The
    prompt is one the model is confident about and the reply is kept short,
    because both make a near-tie less likely. Gross corruption, which this
    would also catch, is already caught by T1.3.
    """
    prompt = "What is 2 + 2?"
    outputs = []
    for _ in range(2):
        body = chat(base, prompt, max_tokens=64, cache_prompt=False)
        outputs.append(_content(body)[0])

    identical = outputs[0] == outputs[1]

    # Printed in full, both of them, so anyone reading the log can see for
    # themselves what matched or where the two runs parted company.
    print(f"Determinism check -- prompt: {prompt!r}")
    print(f"  run 1 ({len(outputs[0])} chars): {outputs[0]!r}")
    print(f"  run 2 ({len(outputs[1])} chars): {outputs[1]!r}")
    print(f"  verdict: {'IDENTICAL' if identical else 'DIFFERENT'}")

    # Stored in full as well -- a truncated record cannot show where they differ.
    details = {"prompt": prompt, "identical": identical,
               "first": outputs[0], "second": outputs[1]}

    if not outputs[0].strip():
        return (report.STATUS_WARN, "the greedy request returned nothing", details)
    if not identical:
        prefix = os.path.commonprefix(outputs)
        details["diverged_at_char"] = len(prefix)
        return (report.STATUS_WARN,
                f"the same question asked twice gave different answers, diverging "
                f"after {len(prefix)} characters. Usually harmless -- two words "
                "were near-tied and rounding decided between them -- but worth "
                "investigating if it happens on every run.", details)
    return (report.STATUS_PASS, None, details)


def check_reasoning_pass(state):
    """Did the model think first, then answer, and stop cleanly?

    Looks at the structure of the reply rather than searching it for the
    markers a thinking model writes around its reasoning. Those markers never
    reach us: the server strips them and returns the reasoning in its own
    field, so a check that searched the text for them could never pass.

    Non-gating. It describes how one model behaves rather than whether the
    build is correct; T1.2 and T1.3 are the signals that matter.
    """
    reasoning = state.get("reasoning") or ""
    answer = state.get("answer") or ""
    finish = state.get("finish")
    details = {"reasoning_chars": len(reasoning), "answer_chars": len(answer),
               "finish_reason": finish,
               "reasoning_sample": reasoning[:200],
               "answer_sample": answer[:200]}

    if finish == "length":
        return (report.STATUS_WARN,
                "the reply ran out of tokens before the model finished, so a "
                "complete reasoning pass could not be observed", details)
    if reasoning.strip() and answer.strip():
        return (report.STATUS_PASS, None, details)
    if answer.strip():
        return (report.STATUS_WARN,
                "the model answered without a separate reasoning pass, which is "
                "normal for a model that does not think out loud", details)
    return (report.STATUS_WARN, "no answer was produced", details)


def main():
    parser = argparse.ArgumentParser(description="Tier 1 GPU functional inference")
    parser.add_argument("--meta", default=None,
                        help="qualify-meta-*.json written by the build job. Supplies "
                             "--rocm-version/--tag and names the archive to find "
                             "under --archive-dir, so the command line needs no "
                             "other arguments in CI.")
    parser.add_argument("--archive-dir", default=".",
                        help="where to look for the archive named by --meta")
    parser.add_argument("--archive", default=None,
                        help="published archive to test; extracted automatically")
    parser.add_argument("--archive-root", default=None,
                        help="an already-extracted archive, instead of --archive")
    parser.add_argument("--model", default=None,
                        help="path to the .gguf model; downloaded if absent")
    parser.add_argument("--model-url", default=MODEL_URL)
    parser.add_argument("--rocm-version", default=None)
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

    if args.meta:
        with open(args.meta, encoding="utf-8-sig") as handle:
            info = json.load(handle)
        args.rocm_version = args.rocm_version or info.get("rocm_version")
        args.tag = args.tag or info.get("tag")
        args.artifact = args.artifact or info.get("artifact")
        if not args.archive and not args.archive_root and info.get("artifact"):
            args.archive = os.path.join(args.archive_dir, info["artifact"])

    if not args.rocm_version:
        parser.error("--rocm-version is required when --meta is not given")

    workdir = tempfile.mkdtemp(prefix="tier1-")
    if args.archive_root:
        root = os.path.abspath(args.archive_root)
    elif args.archive:
        if not os.path.exists(args.archive):
            raise SystemExit(f"::error::archive not found: {args.archive}")
        root = os.path.abspath(extract_archive(args.archive, os.path.join(workdir, "x")))
        print(f"Extracted {args.archive} -> {root}")
    else:
        parser.error("one of --meta, --archive or --archive-root is required")

    model = args.model or os.path.join(workdir, "Qwen3-0.6B-Q4_0.gguf")
    download_model(args.model_url, model)

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
            "::error::No ROCm runtime directories supplied. These archives do "
            "not bundle ROCm, so install_rocm.py must run first -- it exports "
            "ROCM_RUNTIME_DIRS, which this script reads automatically.")

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
    server = Server(exe, os.path.abspath(model), args.port, log_path, env, root)
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
            tier.run("T1.2", "GPU really did the work",
                     lambda: check_gpu_offload(server, base))
            tier.run("T1.3", "response well-formed",
                     lambda: check_well_formed(base, state))
            tier.run("T1.4", "greedy determinism",
                     lambda: check_determinism(base), gating=False)
            tier.run("T1.5", "reasoning pass completed",
                     lambda: check_reasoning_pass(state), gating=False)
        else:
            for test_id, name, gating in [
                ("T1.2", "GPU really did the work", True),
                ("T1.3", "response well-formed", True),
                ("T1.4", "greedy determinism", False),
                ("T1.5", "reasoning pass completed", False),
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
