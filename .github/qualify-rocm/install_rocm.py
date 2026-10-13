#!/usr/bin/env python3
"""
Install a ROCm runtime into a throwaway virtual environment, matching the way
an end user's machine gets one.

The archives this repo publishes do not contain ROCm -- a user installs it
separately and llama.cpp loads it at run time. So a qualification job that only
unpacks the archive and runs it is testing nothing: on Linux llama-server never
starts, and on Windows it is worse, because a GPU backend that fails to load is
not treated as an error. llama.cpp logs it, skips the backend, and quietly runs
on the CPU, so the test passes while the thing it was meant to verify is
completely broken.

What gets installed:

  pip install --index-url <index> "rocm[libraries,device-<arch>]==<version>"

The part in brackets matters. A BUILD needs ROCm's compilers, so the build jobs
ask for the "devel" component. A USER needs the kernels for their own GPU, so
they get "device-<arch>" instead. Verifying against the build's own components
would be circular -- those are the exact files the binary was compiled against,
so of course they are present. Installing the user's set instead means the
checks answer the question that actually matters: will this archive find what
it needs on a real machine?

Everything lands under --prefix and nothing is installed system-wide, so
repeated runs on a persistent machine stay isolated from each other.

Usage:
    python3 install_rocm.py --version 10.1.0 --arch gfx1151 --prefix /tmp/rocm
"""

import argparse
import glob
import json
import os
import subprocess
import sys

# AMD publishes ROCm wheels on different package indexes depending on the
# major version. Keep this in step with the index each build job installs from.
INDEX_BY_MAJOR = {
    "7": "https://repo.amd.com/rocm/whl-multi-arch/",
    "10": "https://stable.repo.amd.com/rocm/whl-next/",
}


def venv_python(venv_dir):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", "python.exe")
    return os.path.join(venv_dir, "bin", "python")


def venv_script(venv_dir, name):
    if os.name == "nt":
        return os.path.join(venv_dir, "Scripts", f"{name}.exe")
    return os.path.join(venv_dir, "bin", name)


def run(cmd, **kwargs):
    print("+ " + " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=False, **kwargs)


def site_packages(venv_dir):
    if os.name == "nt":
        matches = [os.path.join(venv_dir, "Lib", "site-packages")]
    else:
        matches = sorted(glob.glob(os.path.join(venv_dir, "lib", "python3*", "site-packages")))
    for path in matches:
        if os.path.isdir(path):
            return path
    raise SystemExit(f"no site-packages under {venv_dir}")


def runtime_dirs(venv_dir):
    """Every directory that has to be on the library search path.

    This returns a list, not a single root directory, because ROCm spreads its
    runtime across several packages: the core HIP libraries live in one, and
    the maths libraries (rocBLAS, hipBLAS and friends) live in another. Adding
    only the first means the core loads and then the maths libraries cannot be
    found, which surfaces as a confusing "DLL not found" on Windows or a
    missing-library error on Linux.
    """
    site = site_packages(venv_dir)
    dirs = []
    for pkg in sorted(glob.glob(os.path.join(site, "_rocm_sdk_*"))):
        for sub in ("bin", "lib", os.path.join("lib", "llvm", "bin"),
                    os.path.join("lib", "llvm", "lib")):
            candidate = os.path.join(pkg, sub)
            if os.path.isdir(candidate) and candidate not in dirs:
                dirs.append(candidate)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description="Install a user-style ROCm runtime into a throwaway venv")
    parser.add_argument("--version", required=True, help="e.g. 10.1.0")
    parser.add_argument("--arch", required=True, help="e.g. gfx1151")
    parser.add_argument("--prefix", required=True, help="scratch dir for the venv")
    parser.add_argument("--index-url", default=None,
                        help="override the index chosen from the ROCm major version")
    parser.add_argument("--output-env", default=None,
                        help="write KEY=VALUE loader-path lines here (e.g. $GITHUB_ENV)")
    parser.add_argument("--output-json", default=None,
                        help="write the resolved runtime dirs here")
    args = parser.parse_args()

    major = args.version.split(".")[0]
    index = args.index_url or INDEX_BY_MAJOR.get(major)
    if not index:
        raise SystemExit(
            f"No package index known for ROCm {args.version} (major {major}). "
            f"Known: {sorted(INDEX_BY_MAJOR)}. Add it to INDEX_BY_MAJOR.")

    venv_dir = os.path.join(args.prefix, f"rocm-{args.version}-{args.arch}")
    os.makedirs(args.prefix, exist_ok=True)

    if not os.path.exists(venv_python(venv_dir)):
        if run([sys.executable, "-m", "venv", venv_dir]).returncode != 0:
            raise SystemExit(f"could not create venv at {venv_dir}")

    python = venv_python(venv_dir)
    run([python, "-m", "pip", "install", "--upgrade", "pip", "--quiet"])

    spec = f"rocm[libraries,device-{args.arch}]=={args.version}"
    proc = run([python, "-m", "pip", "install", "--only-binary", ":all:",
                "--index-url", index, spec])
    if proc.returncode != 0:
        raise SystemExit(
            f"\n::error::Could not install '{spec}' from {index}.\n"
            f"::error::ROCm {args.version} device wheels for {args.arch} were not "
            f"resolvable at that index. If this is the 7.x leg, the multi-arch "
            f"index may not publish per-device wheels -- check what "
            f"`pip index versions rocm-sdk-device-{args.arch}` offers there, and "
            f"either point --index-url elsewhere or drop this leg from the matrix.")

    dirs = runtime_dirs(venv_dir)
    if not dirs:
        raise SystemExit(
            "::error::ROCm installed but no _rocm_sdk_*/{bin,lib} directories were "
            f"found under {site_packages(venv_dir)} -- the wheel layout changed.")

    # Informational only -- the search path comes from runtime_dirs() above.
    sdk = venv_script(venv_dir, "rocm-sdk")
    if os.path.exists(sdk):
        run([sdk, "path", "--root"])

    print(f"\nROCm {args.version} ({args.arch}) runtime directories:")
    for directory in dirs:
        print(f"  {directory}")

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump({"version": args.version, "arch": args.arch,
                       "index": index, "venv": venv_dir,
                       "runtime_dirs": dirs}, handle, indent=2)

    if args.output_env:
        joined = os.pathsep.join(dirs)
        with open(args.output_env, "a", encoding="utf-8") as handle:
            handle.write(f"ROCM_RUNTIME_DIRS={joined}\n")
            handle.write(f"ROCM_VENV={venv_dir}\n")
    print("\nROCm runtime ready.")


if __name__ == "__main__":
    main()
