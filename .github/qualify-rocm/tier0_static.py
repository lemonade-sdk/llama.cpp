#!/usr/bin/env python3
"""
Tier 0 -- static verification of a published ROCm archive. No GPU required.

Runs on a free hosted runner against the ARCHIVE WE ARE ABOUT TO PUBLISH (or
have just published), not against the build tree. That distinction matters because
every bug this tier exists to catch -- a dropped gfx target, a missing CPU
backend after the Windows merge, a leaked build path -- is invisible in the
build tree and only appears in the packaged bytes.

Checks:
  T0.1  archive contents           -- every binary a user needs is present.
  T0.2  HIP backend present        -- ggml-hip exists on BOTH OSes and is
                                      large enough to contain GPU code.
  T0.3  gfx target audit           -- the GPU backend really contains code for
                                      every gfx target the build asked for.
                                      See binfmt.py for how and why.
  T0.4  CPU backend variants       -- ggml-cpu-* present. On Windows after the
                                      release job's CPU merge this is also the
                                      proof that the merge landed.
  T0.5  runtime dependency survey  -- every non-ROCm dependency resolves from
                                      the archive or the base system; the ROCm
                                      ones are recorded as expected-external.
  T0.6  Linux version support      -- Linux only. The archive must not require
                                      a newer C library than the oldest
                                      distribution we support.
  T0.7  library search path        -- Linux only. No absolute build-machine
                                      paths baked into the binaries.
  T0.8  llama-server flag surface  -- the flags the downstream consumer
                                      launches with, and the one Tier 1 relies
                                      on, are all still accepted (needs
                                      --run-binaries).
  T0.9  archive tag consistency    -- the directory inside the tarball matches
                                      the tag in the asset filename.
  T0.12 ROCm dependency resolution -- the ROCm libraries this archive needs
                                      exist in the ROCm a user will install
                                      (needs --rocm-dirs).

(T0.10 and T0.11 are emitted separately by check_post_merge.py, which runs on
the final archives after the release job assembles them.)

Usage:
    python3 tier0_static.py --archive llama-bin-ubuntu-rocm-10.1-x64.tar.gz \
        --os ubuntu --rocm-version 10.1 --gpu-targets "gfx900;gfx906;..." \
        --output tier0-ubuntu-10.1.json
"""

import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import binfmt  # noqa: E402
import report  # noqa: E402

# Tools a consumer actually invokes.
REQUIRED_TOOLS = ["llama-server", "llama-cli", "llama-bench", "llama-perplexity"]

REQUIRED_LIBS = {
    "ubuntu": ["libggml-base.so", "libggml.so", "libllama.so", "libggml-hip.so"],
    "win": ["ggml-base.dll", "ggml.dll", "llama.dll", "ggml-hip.dll"],
}

CPU_VARIANT_GLOB = {"ubuntu": "libggml-cpu-*.so", "win": "ggml-cpu-*.dll"}
HIP_BACKEND = {"ubuntu": "libggml-hip.so", "win": "ggml-hip.dll"}

MIN_HIP_BACKEND_BYTES = 64 << 20

# Libraries that come with any Linux installation, so the archive does not and
# should not carry them.
BASE_LIBS_LINUX = {
    "libc.so.6", "libm.so.6", "libdl.so.2", "librt.so.1", "libpthread.so.0",
    "libstdc++.so.6", "libgcc_s.so.1", "ld-linux-x86-64.so.2", "libutil.so.1",
    "libresolv.so.2", "libgomp.so.1",
    # llama-server uses the distribution's OpenSSL 3 rather than a copy built
    # into the binary. Every supported distribution ships it.
    "libssl.so.3", "libcrypto.so.3",
}

# ROCm libraries. These are deliberately NOT in the archive: a user installs
# ROCm separately, and llama.cpp loads it from there at run time. T0.5 records
# which ones an archive needs; T0.12 confirms they exist in the ROCm that will
# actually be installed.
ROCM_LIB_PREFIXES = (
    "libamdhip64", "libhipblas", "librocblas", "libhipblaslt", "librocsolver",
    "libamd_comgr", "libhsa-runtime64", "librocm", "libroctx", "librocprofiler",
)
ROCM_DLL_PREFIXES = (
    "amdhip64", "hipblas", "rocblas", "hipblaslt", "rocsolver", "amd_comgr",
    "amdhsa", "rocm",
)

# Windows DLLs supplied by the OS or the MSVC redistributable.
BASE_DLLS_WIN = ("kernel32", "user32", "advapi32", "ws2_32", "shell32",
                 "ole32", "oleaut32", "bcrypt", "crypt32", "dbghelp",
                 "msvcp140", "vcruntime140", "vcruntime140_1", "msvcrt",
                 "api-ms-win-", "ucrtbase", "concrt140", "powrprof",
                 "psapi", "setupapi", "cfgmgr32", "version", "winmm")

# --- oldest Linux we support ------------------------------------------------
# The Linux ROCm artifact is built on Ubuntu 24.04, which is also the oldest
# release the downstream consumer supports. Ubuntu 24.04 ships glibc 2.39, so
# an archive that needs anything newer than that would fail to start even on a
# supported machine.
#
# Raise this only together with the runner image in the build job, and only
# after deciding that dropping the older distributions is acceptable. See
# binfmt.elf_glibc_floor() for what this number means.
SUPPORTED_GLIBC = (2, 39)
SUPPORTED_DISTRO = "Ubuntu 24.04"

# Flags Lemonade passes to llama-server.
# Upstream renames flags routinely and this repo rebuilds upstream every 12 hours,
# so a rename should fail here within hours rather than surfacing as a Lemonade bug.
LEMONADE_SERVER_FLAGS = [
    "--model", "--ctx-size", "--device", "--port", "--jinja", "--metrics",
    "--mmproj", "--no-mmproj-offload", "--model-draft", "--embeddings",
    "--reranking",
    # Raises log verbosity. Listed because Tier 1 depends on it: without it the
    # server never prints which device it used, and the GPU check has nothing
    # to inspect. If upstream renames it, failing here names the cause instead
    # of leaving Tier 1 to report a confusing absence of evidence.
    "--verbosity",
]


# ---------------------------------------------------------------------------
# Archive handling
# ---------------------------------------------------------------------------

def extract(archive, dest):
    """Extract `archive` into `dest`; return (content_root, top_level_names).

    Linux tarballs nest everything under llama-<tag>/; Windows zips put the
    files at the archive root. Both layouts are normal -- return the directory
    that actually holds the binaries.
    """
    if archive.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tar:
            names = tar.getnames()
            # data filter: never let a published archive write outside dest.
            try:
                tar.extractall(dest, filter="data")
            except TypeError:  # Python < 3.12
                tar.extractall(dest)
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = zf.namelist()
            zf.extractall(dest)
    else:
        raise ValueError(f"unsupported archive type: {archive}")

    tops = sorted({n.split("/")[0] for n in names if n.split("/")[0] not in (".", "")})
    root = dest
    if len(tops) == 1 and os.path.isdir(os.path.join(dest, tops[0])):
        root = os.path.join(dest, tops[0])
    return root, tops


def listing(root):
    return sorted(os.listdir(root))


def exe(name, os_name):
    return f"{name}.exe" if os_name == "win" else name


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_contents(root, os_name):
    files = set(listing(root))
    missing = [t for t in (exe(n, os_name) for n in REQUIRED_TOOLS) if t not in files]
    missing += [lib for lib in REQUIRED_LIBS[os_name] if lib not in files]
    if "LICENSE" not in files:
        missing.append("LICENSE")

    # The windows-rocm build job is the only one that does not pass the shared
    # CMAKE_ARGS (which carry -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF),
    # so its zip ships the upstream unit-test binaries to end users.
    tests = sorted(f for f in files if f.startswith("test-"))
    details = {"entries": len(files), "missing": missing,
               "test_binaries_shipped": len(tests), "test_binaries": tests[:10]}

    if missing:
        return (report.STATUS_FAIL, f"missing from archive: {', '.join(missing)}", details)
    if tests:
        return (report.STATUS_WARN,
                f"{len(tests)} upstream test binary/binaries shipped to users "
                "(this job does not pass -DLLAMA_BUILD_TESTS=OFF)", details)
    return (report.STATUS_PASS, None, details)


def check_hip_backend(root, os_name):
    name = HIP_BACKEND[os_name]
    path = os.path.join(root, name)
    details = {"backend": name, "exists": os.path.exists(path)}
    if not details["exists"]:
        return (report.STATUS_FAIL,
                f"{name} is missing -- the HIP backend silently failed to build "
                "or was dropped at packaging", details)
    size = os.path.getsize(path)
    details["bytes"] = size
    if size < MIN_HIP_BACKEND_BYTES:
        return (report.STATUS_FAIL,
                f"{name} is only {size / 1e6:.1f} MB; a multi-target HIP fat "
                "binary is ~1 GB, so this contains little or no device code",
                details)
    return (report.STATUS_PASS, None, details)


def check_gfx_targets(root, os_name, gpu_targets):
    name = HIP_BACKEND[os_name]
    path = os.path.join(root, name)
    if not os.path.exists(path):
        return (report.STATUS_FAIL, f"{name} missing (see T0.2)", {})

    expected = [t.strip() for t in re.split(r"[;,\s]+", gpu_targets) if t.strip()]
    if not expected:
        return (report.STATUS_FAIL, "no expected gfx target list was supplied", {})

    result = binfmt.audit_gfx_targets(path, expected)

    # An audit that can silently find nothing is worse than no audit: if the
    # scan returns zero targets the METHOD is broken (toolchain changed the
    # bundle-entry format), and that must fail loudly rather than pass.
    if result["present_count"] == 0:
        return (report.STATUS_FAIL,
                "found NO gfx code objects at all -- the audit method is broken, "
                "not necessarily the build. Check the offload bundle entry format "
                "against binfmt.BUNDLE_ENTRY_RE.", result)

    if result["missing"]:
        return (report.STATUS_FAIL,
                f"{len(result['missing'])} gfx target(s) requested but NOT in the "
                f"fat binary: {', '.join(result['missing'])}. Users with those GPUs "
                "fall back to CPU with no error message.", result)

    if not result["entry_count_uniform"]:
        counts = result["entries_per_target"]
        low = min(counts, key=counts.get)
        return (report.STATUS_WARN,
                f"all targets present but entry counts differ (e.g. {low} has "
                f"{counts[low]}, max is {max(counts.values())}) -- a target may be "
                "compiled into only some translation units", result)

    return (report.STATUS_PASS, None, result)


def check_cpu_variants(root, os_name, post_merge):
    files = listing(root)
    variants = fnmatch.filter(files, CPU_VARIANT_GLOB[os_name])
    details = {"count": len(variants), "variants": sorted(variants),
               "post_merge": post_merge}
    if not variants:
        extra = ""
        if os_name == "win" and post_merge:
            extra = (" -- the release job's Windows CPU-backend merge did not land "
                     "in this zip")
        return (report.STATUS_FAIL,
                f"no CPU backend matching {CPU_VARIANT_GLOB[os_name]}{extra}", details)
    # GGML_CPU_ALL_VARIANTS=ON produces well over a dozen; a single one means
    # the variant build collapsed or only part of the merge arrived.
    if len(variants) < 5:
        return (report.STATUS_WARN,
                f"only {len(variants)} CPU backend variant(s); "
                "GGML_CPU_ALL_VARIANTS=ON normally yields 14+", details)
    return (report.STATUS_PASS, None, details)


def _classify_linux(needed, present):
    rocm, missing = [], []
    for soname in needed:
        if soname.startswith(ROCM_LIB_PREFIXES):
            rocm.append(soname)
        elif soname in present or soname in BASE_LIBS_LINUX:
            continue
        else:
            missing.append(soname)
    return rocm, missing


def _classify_win(imports, present_lower):
    rocm, missing = [], []
    for dll in imports:
        low = dll.lower()
        if low.startswith(ROCM_DLL_PREFIXES):
            rocm.append(dll)
        elif low in present_lower or low.startswith(BASE_DLLS_WIN):
            continue
        else:
            missing.append(dll)
    return rocm, missing


def check_dependencies(root, os_name):
    """Survey every binary's dynamic dependencies.

    ROCm libraries are EXPECTED to be absent -- this repo ships no ROCm.
    They are recorded so the set a consumer must provide is written down per release.
    Anything else that does not resolve is a packaging defect and fails.
    """
    files = listing(root)
    present = set(files)
    present_lower = {f.lower() for f in files}
    rocm_needed, unresolved, scanned = set(), {}, 0

    for name in files:
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        try:
            if os_name == "ubuntu":
                if not binfmt.is_elf(path):
                    continue
                needed, _ = binfmt.elf_dynamic(path)
                rocm, missing = _classify_linux(needed, present)
            else:
                if not binfmt.is_pe(path):
                    continue
                imports = binfmt.pe_imports(path)
                if not imports:
                    continue
                rocm, missing = _classify_win(imports, present_lower)
        except (ValueError, OSError, struct.error) as exc:
            unresolved[name] = [f"<unreadable: {exc}>"]
            continue
        scanned += 1
        rocm_needed.update(rocm)
        if missing:
            unresolved[name] = sorted(missing)

    details = {
        "binaries_scanned": scanned,
        "rocm_runtime_required": sorted(rocm_needed),
        "unresolved": unresolved,
    }
    if unresolved:
        flat = sorted({d for deps in unresolved.values() for d in deps})
        return (report.STATUS_FAIL,
                f"{len(unresolved)} binary/binaries have dependencies that are "
                f"neither in the archive nor part of the base system: "
                f"{', '.join(flat[:6])}", details)
    if not rocm_needed:
        return (report.STATUS_WARN,
                "no ROCm runtime dependency found at all -- expected the HIP "
                "backend to link one", details)
    return (report.STATUS_PASS, None, details)


def check_linux_version_support(root, os_name):
    """Confirm the archive still runs on the oldest Linux we support.

    Each binary records the newest version of the system C library it needs.
    Whichever binary needs the newest one decides the oldest Linux release the
    whole archive can run on. If that goes past what SUPPORTED_GLIBC allows,
    users on a supported distribution get a startup error instead of a program.

    This normally only changes when the build job's runner image changes, which
    is exactly the moment it should be noticed.
    """
    if os_name != "ubuntu":
        return (report.STATUS_SKIP, "Linux-only check", {})

    needs, needs_file = (0, 0), None
    per_file = {}
    for name in listing(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path) or not binfmt.is_elf(path):
            continue
        floor = binfmt.elf_glibc_floor(path)
        if not floor:
            continue
        per_file[name] = ".".join(map(str, floor))
        if floor > needs:
            needs, needs_file = floor, name

    allowed = ".".join(map(str, SUPPORTED_GLIBC))
    actual = ".".join(map(str, needs)) if needs_file else None
    details: dict = {
        "glibc_required": actual,
        "glibc_allowed": allowed,
        "oldest_supported_distro": SUPPORTED_DISTRO,
        "required_by": needs_file,
        "per_file": per_file,
    }
    if not needs_file:
        return (report.STATUS_WARN,
                "could not determine which C library version this archive needs",
                details)
    if needs > SUPPORTED_GLIBC:
        return (report.STATUS_FAIL,
                f"this archive needs glibc {actual}, but the oldest Linux we "
                f"support ({SUPPORTED_DISTRO}) only provides {allowed}. "
                f"{needs_file} is what requires it. Users on a supported "
                "distribution will see \"GLIBC_... not found\" and the program "
                "will not start. This usually means the build job's runner image "
                "moved to a newer Linux.", details)
    return (report.STATUS_PASS, None, details)


def check_rpath(root, os_name):
    """No absolute build-machine paths in RUNPATH.

    The ROCm Linux job never runs the `patchelf --set-rpath '$ORIGIN'` normalisation.
    """
    if os_name != "ubuntu":
        return (report.STATUS_SKIP, "Linux-only check", {})

    leaks = {}
    for name in listing(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path) or not binfmt.is_elf(path):
            continue
        _, runpath = binfmt.elf_dynamic(path)
        bad = [part for entry in runpath for part in entry.split(":")
               if part and not part.startswith("$ORIGIN")]
        if bad:
            leaks[name] = bad

    details = {"leaked": leaks}
    if leaks:
        flat = sorted({p for parts in leaks.values() for p in parts})
        return (report.STATUS_FAIL,
                f"{len(leaks)} binary/binaries carry a build-machine path in "
                f"their library search path: {flat[0]}. The build job should "
                "reset it to '$ORIGIN' before packaging.", details)
    return (report.STATUS_PASS, None, details)


def check_rocm_dependencies(root, os_name, rocm_dirs):
    """Confirm the ROCm libraries this archive needs will actually be there.

    This archive does not contain ROCm. The user installs ROCm separately and
    llama.cpp loads it at run time. So the archive says "I need a library called
    libamdhip64.so.7", and whether that is true depends entirely on what the
    user installed -- something the build machine can tell us nothing about.

    Checking against the ROCm the archive was COMPILED with would be circular:
    of course the libraries it linked against are present there. The useful
    question is whether they are present in the ROCm a user ends up with, which
    is installed from the same release but with a different set of components:
    the build needs compilers, the user needs per-GPU kernels. So `rocm_dirs`
    must point at a user-style ROCm install, not at the build toolchain.

    When nothing is supplied, this is skipped rather than failed -- the rest of
    Tier 0 does not depend on having a ROCm install available.
    """
    if not rocm_dirs:
        return (report.STATUS_SKIP,
                "no ROCm install supplied via --rocm-dirs; nothing to resolve "
                "the archive's ROCm dependencies against", {})

    available = {}
    for directory in rocm_dirs:
        if not os.path.isdir(directory):
            continue
        for entry in os.listdir(directory):
            available.setdefault(entry.lower(), os.path.join(directory, entry))

    required, unresolved = set(), {}
    for name in listing(root):
        path = os.path.join(root, name)
        if not os.path.isfile(path) or os.path.islink(path):
            continue
        try:
            if os_name == "ubuntu":
                if not binfmt.is_elf(path):
                    continue
                deps = [d for d in binfmt.elf_dynamic(path)[0]
                        if d.startswith(ROCM_LIB_PREFIXES)]
            else:
                if not binfmt.is_pe(path):
                    continue
                deps = [d for d in binfmt.pe_imports(path)
                        if d.lower().startswith(ROCM_DLL_PREFIXES)]
        except (ValueError, OSError, struct.error):
            continue
        required.update(deps)
        missing = [d for d in deps if d.lower() not in available]
        if missing:
            unresolved[name] = sorted(missing)

    details: dict = {
        "rocm_dirs_searched": len(rocm_dirs),
        "files_available": len(available),
        "rocm_libraries_required": sorted(required),
        "unresolved": unresolved,
    }
    if not required:
        return (report.STATUS_WARN,
                "this archive does not reference any ROCm library, which is "
                "unexpected for a ROCm build", details)
    if unresolved:
        flat = sorted({d for deps in unresolved.values() for d in deps})
        return (report.STATUS_FAIL,
                f"the installed ROCm does not provide {', '.join(flat)}, which "
                f"{len(unresolved)} binary/binaries in this archive need. On "
                "Linux llama-server will refuse to start; on Windows the GPU "
                "backend fails to load and llama.cpp quietly runs on the CPU "
                "instead.", details)
    return (report.STATUS_PASS, None, details)


def check_server_flags(root, os_name, run_binaries):
    """The llama-server flags Lemonade launches with must still parse."""
    if not run_binaries:
        return (report.STATUS_SKIP,
                "not running binaries on this runner (OS mismatch)", {})

    server = os.path.join(root, exe("llama-server", os_name))
    if not os.path.exists(server):
        return (report.STATUS_FAIL, "llama-server missing (see T0.1)", {})
    if os_name == "ubuntu":
        os.chmod(server, 0o755)

    proc = subprocess.run([server, "--help"], capture_output=True, text=True,
                          check=False, timeout=180, cwd=root)
    text = (proc.stdout or "") + (proc.stderr or "")
    details: dict = {"returncode": proc.returncode, "help_bytes": len(text)}
    if proc.returncode != 0 or not text.strip():
        details["output_tail"] = text[-600:]
        return (report.STATUS_FAIL,
                f"llama-server --help exited {proc.returncode} with "
                f"{len(text)} bytes of output", details)

    missing = [f for f in LEMONADE_SERVER_FLAGS
               if not re.search(rf"(?<![\w-]){re.escape(f)}(?![\w-])", text)]
    details["flags_checked"] = len(LEMONADE_SERVER_FLAGS)
    details["flags_missing"] = missing
    if missing:
        return (report.STATUS_FAIL,
                f"llama-server no longer accepts {', '.join(missing)} -- upstream "
                "renamed or removed them and Lemonade's launch command will break",
                details)
    return (report.STATUS_PASS, None, details)


def check_tag_consistency(archive, tops):
    """The folder inside the archive should match the tag in its name.

    Every job in a release resolves its upstream revision from a single shared
    value, so these two should always agree. If they ever stop agreeing, jobs
    have gone back to resolving it independently -- which means they may also
    have compiled different source, and the archive's name no longer tells you
    what is inside it.

    Non-gating: a mismatch does not make the binaries wrong, it makes their
    provenance unclear.
    """
    asset = re.search(r"llama-(b\d+)-", os.path.basename(archive))
    inner = None
    for top in tops:
        match = re.fullmatch(r"llama-(b\d+)", top)
        if match:
            inner = match.group(1)
            break
    details = {"asset_tag": asset.group(1) if asset else None,
               "inner_dir_tag": inner, "top_level": tops}
    if not asset or inner is None:
        return (report.STATUS_SKIP, "no tag to compare (flat archive layout)", details)
    if asset.group(1) != inner:
        return (report.STATUS_WARN,
                f"asset is named {asset.group(1)} but contains {inner}/ -- the jobs "
                "in this run did not all build the same upstream revision", details)
    return (report.STATUS_PASS, None, details)


def main():
    parser = argparse.ArgumentParser(description="Tier 0 static archive verification")
    parser.add_argument("--meta", default=None,
                        help="qualify-meta-*.json written by the build job; supplies "
                             "--os/--rocm-version/--gpu-targets/--tag and names the "
                             "archive to find under --archive-dir")
    parser.add_argument("--archive-dir", default=".",
                        help="where to look for the archive named by --meta")
    parser.add_argument("--archive", help="published .tar.gz or .zip")
    parser.add_argument("--os", dest="os_name", choices=["ubuntu", "win"])
    parser.add_argument("--rocm-version")
    parser.add_argument("--gpu-targets",
                        help="the build job's gpu_targets matrix entry")
    parser.add_argument("--tag", default=None, help="llama.cpp build tag")
    parser.add_argument("--upstream-sha", default=None,
                        help="upstream llama.cpp commit this archive was built from")
    parser.add_argument("--run-binaries", action="store_true",
                        help="this runner's OS matches the archive, so executables "
                             "in it can be launched (enables T0.8)")
    parser.add_argument("--post-merge", action="store_true",
                        help="the archive has been through the release job's "
                             "Windows CPU-backend merge")
    parser.add_argument("--rocm-dirs", default=os.environ.get("ROCM_RUNTIME_DIRS", ""),
                        help="directories of a user-style ROCm install, separated "
                             "by the platform path separator. Supplied by "
                             "install_rocm.py; enables T0.12 when present.")
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output", default="tier0-report.json")
    parser.add_argument("--fail-on-error", action="store_true")
    args = parser.parse_args()

    if args.meta:
        with open(args.meta, encoding="utf-8-sig") as handle:
            info = json.load(handle)
        args.os_name = args.os_name or info["os"]
        args.rocm_version = args.rocm_version or info["rocm_version"]
        args.gpu_targets = args.gpu_targets or info["gpu_targets"]
        args.tag = args.tag or info.get("tag")
        args.upstream_sha = args.upstream_sha or info.get("upstream_sha")
        if not args.archive:
            args.archive = os.path.join(args.archive_dir, info["artifact"])

    for required in ("archive", "os_name", "rocm_version", "gpu_targets"):
        if not getattr(args, required):
            parser.error(f"--{required.replace('_name', '').replace('_', '-')} is "
                         "required when --meta is not given")
    if not os.path.exists(args.archive):
        raise SystemExit(f"::error::archive not found: {args.archive}")

    rocm_dirs = [d for d in args.rocm_dirs.split(os.pathsep) if d]

    meta = report.build_meta(
        artifact=os.path.basename(args.archive),
        os_name=args.os_name,
        rocm_version=args.rocm_version,
        llamacpp_tag=args.tag,
        upstream_sha=args.upstream_sha,
        hardware_validated=False,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    tier = report.TierReport("tier0", meta)

    workdir = tempfile.mkdtemp(prefix="tier0-")
    try:
        root, tops = extract(args.archive, workdir)
        print(f"Extracted {args.archive} -> {root} ({len(listing(root))} entries)")

        tier.run("T0.1", "archive contents",
                 lambda: check_contents(root, args.os_name))
        tier.run("T0.2", "HIP backend present",
                 lambda: check_hip_backend(root, args.os_name))
        tier.run("T0.3", "gfx target audit",
                 lambda: check_gfx_targets(root, args.os_name, args.gpu_targets))
        tier.run("T0.4", "CPU backend variants",
                 lambda: check_cpu_variants(root, args.os_name, args.post_merge))
        tier.run("T0.5", "runtime dependency survey",
                 lambda: check_dependencies(root, args.os_name))
        tier.run("T0.6", "Linux version support",
                 lambda: check_linux_version_support(root, args.os_name))
        tier.run("T0.7", "library search path",
                 lambda: check_rpath(root, args.os_name))
        tier.run("T0.8", "llama-server flag surface",
                 lambda: check_server_flags(root, args.os_name, args.run_binaries))
        tier.run("T0.9", "archive tag consistency",
                 lambda: check_tag_consistency(args.archive, tops),
                 gating=False)
        tier.run("T0.12", "ROCm dependency resolution",
                 lambda: check_rocm_dependencies(root, args.os_name, rocm_dirs))
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")

    if args.fail_on_error and tier.rollup_status() == report.STATUS_FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
