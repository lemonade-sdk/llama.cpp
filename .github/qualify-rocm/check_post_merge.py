#!/usr/bin/env python3
"""
Post-merge checks on the exact bytes about to be uploaded. No GPU required.

Runs inside the release job AFTER "Move artifacts", which is the only place
these two questions can be answered:

  T0.10  Windows CPU-backend merge landed -- the release job unzips
         llama-bin-win-cpu-x64.zip into every other Windows zip, including the
         ROCm ones. Nothing downstream of that merge has ever looked inside the
         result, so a merge that silently did nothing would ship a ROCm zip
         with no CPU backend in it.
  T0.11  final archive integrity -- the renamed, moved archive still opens and
         still contains llama-server and the HIP backend.

Deliberately cheap: it reads the archive index rather than extracting ~1 GB fat binaries.
The expensive static work (gfx audit, dependency survey, glibc) already ran in the qualify-tier0 job on identical pre-merge bytes.
The merge only ADDS CPU files to Windows zips and never rewrites ggml-hip.

Usage:
    python3 check_post_merge.py --archive release/llama-bNNNNN-bin-win-rocm-10.1-x64.zip \
        --os win --rocm-version 10.1 --output frag.json
"""

import argparse
import fnmatch
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

CPU_VARIANT_GLOB = {"ubuntu": "libggml-cpu-*.so", "win": "ggml-cpu-*.dll"}
HIP_BACKEND = {"ubuntu": "libggml-hip.so", "win": "ggml-hip.dll"}
SERVER = {"ubuntu": "llama-server", "win": "llama-server.exe"}

MIN_CPU_VARIANTS = 5


def archive_names(archive):
    """Base filenames inside the archive, without extracting it."""
    if archive.endswith((".tar.gz", ".tgz")):
        with tarfile.open(archive) as tar:
            names = [m.name for m in tar.getmembers() if m.isfile()]
    elif archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
    else:
        raise ValueError(f"unsupported archive type: {archive}")
    return [os.path.basename(n) for n in names]


def check_cpu_merge(names, os_name):
    variants = fnmatch.filter(names, CPU_VARIANT_GLOB[os_name])
    details = {"cpu_variants": len(variants), "names": sorted(variants)}

    if not variants:
        hint = ""
        if os_name == "win":
            hint = (" The release job's \"Move artifacts\" step merges "
                    "llama-bin-win-cpu-x64.zip into this zip; that merge did not "
                    "land, so the published artifact has no CPU backend at all.")
        return (report.STATUS_FAIL,
                f"no CPU backend matching {CPU_VARIANT_GLOB[os_name]} in the final "
                f"archive.{hint}", details)

    if len(variants) < MIN_CPU_VARIANTS:
        return (report.STATUS_WARN,
                f"only {len(variants)} CPU backend variant(s) in the final archive; "
                "a complete GGML_CPU_ALL_VARIANTS build yields 14+", details)
    return (report.STATUS_PASS, None, details)


def check_integrity(names, os_name, archive):
    required = [SERVER[os_name], HIP_BACKEND[os_name]]
    missing = [r for r in required if r not in names]
    details = {"entries": len(names), "missing": missing,
               "bytes": os.path.getsize(archive)}
    if missing:
        return (report.STATUS_FAIL,
                f"final archive is missing {', '.join(missing)} -- it was damaged "
                "between the build job and upload", details)
    return (report.STATUS_PASS, None, details)


def main():
    parser = argparse.ArgumentParser(description="Post-merge release-archive checks")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--os", dest="os_name", required=True, choices=["ubuntu", "win"])
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    parser.add_argument("--output", default="tier0-post-merge.json")
    args = parser.parse_args()

    meta = report.build_meta(
        artifact=os.path.basename(args.archive),
        os_name=args.os_name,
        rocm_version=args.rocm_version,
        llamacpp_tag=args.tag,
        hardware_validated=False,
        run_id=args.run_id,
        run_attempt=args.run_attempt,
    )
    tier = report.TierReport("tier0", meta)
    names = archive_names(args.archive)

    tier.run("T0.10", "Windows CPU-backend merge landed",
             lambda: check_cpu_merge(names, args.os_name))
    tier.run("T0.11", "final archive integrity",
             lambda: check_integrity(names, args.os_name, args.archive))

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
