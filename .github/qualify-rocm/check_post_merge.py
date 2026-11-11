#!/usr/bin/env python3
"""
Post-merge checks on the exact bytes about to be uploaded. No GPU required.

Runs inside the release job AFTER "Move artifacts", which is the only place
these two questions can be answered:

  T0.10  Windows CPU-backend merge landed -- the release job unzips
         llama-bin-win-cpu-x64.zip over every other Windows zip, including the
         ROCm ones. The ROCm build ships its own ggml-cpu-*.dll, so counting CPU
         backends cannot tell whether the merge happened. Instead every file in
         the final zip is traced back to the zip it came from: each file of the
         CPU zip must be there byte for byte, and every other file must be the
         ROCm build's, unchanged. Linux archives are not merged, so it is a skip
         there.
  T0.11  final archive integrity -- the final archive opens, reads back intact,
         still contains llama-server and the HIP backend, and comes from the
         bytes the qualify jobs tested: the SHA-256 the build job recorded must
         match the tarball (Linux) or the zip as it was before the merge
         (Windows).

The merge REPLACES every file the CPU zip shares with the ROCm zip
(llama-server.exe, llama.dll, ggml-base.dll, the ggml-cpu-*.dll, ...). The
qualify jobs tested the ROCm build's copies, so on Windows they did not test
the shipped versions of those files. T0.10 lists which files were replaced.

Deliberately cheap apart from reading each archive back once: it compares zip
directories (name, CRC-32, size) rather than extracting ~1 GB fat binaries.

Every file is opened inside a check, so an archive that cannot be read is
recorded as a FAIL instead of crashing the script and leaving no result.

Usage:
    python3 check_post_merge.py --archive release/llama-bNNNNN-bin-win-rocm-10.1-x64.zip \
        --os win --rocm-version 10.1 --meta meta/qualify-meta-win-rocm-10.1.json \
        --cpu-zip release/llama-bNNNNN-bin-win-cpu-x64.zip \
        --pre-merge premerge/llama-bin-win-rocm-10.1-x64.zip --output frag.json
"""

import argparse
import fnmatch
import hashlib
import json
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

CPU_VARIANT_GLOB = "ggml-cpu-*.dll"
HIP_BACKEND = {"ubuntu": "libggml-hip.so", "win": "ggml-hip.dll"}
SERVER = {"ubuntu": "llama-server", "win": "llama-server.exe"}

MIN_CPU_VARIANTS = 5


def zip_index(path):
    """{name: (CRC-32, size)} for every file in a zip, read from its directory."""
    with zipfile.ZipFile(path) as zf:
        return {i.filename: (i.CRC, i.file_size)
                for i in zf.infolist() if not i.is_dir()}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_back(archive):
    """Base filenames inside the archive, after reading every byte of it.

    A zip is decompressed entry by entry and each CRC-32 checked; a tarball is
    read to the end, which checks the gzip CRC. Either raises if the archive is
    damaged.
    """
    if archive.endswith((".tar.gz", ".tgz")):
        names = []
        with tarfile.open(archive, "r|gz") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                names.append(os.path.basename(member.name))
                stream = tar.extractfile(member)
                while stream.read(1 << 20):
                    pass
        return names
    if archive.endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"{bad} fails its CRC check")
            return [os.path.basename(i.filename) for i in zf.infolist() if not i.is_dir()]
    raise ValueError(f"unsupported archive type: {archive}")


def check_cpu_merge(archive, os_name, cpu_zip, pre_merge):
    if os_name != "win":
        return (report.STATUS_SKIP, "Windows-only check: Linux archives are not merged", {})
    if not cpu_zip or not pre_merge:
        return (report.STATUS_FAIL,
                "the release job did not pass --cpu-zip and --pre-merge, so the merge "
                "cannot be traced", {})

    final = zip_index(archive)
    cpu = zip_index(cpu_zip)
    rocm = zip_index(pre_merge)
    # What the merge should produce: the ROCm zip with every CPU file laid over it.
    expected = {**rocm, **cpu}
    missing = sorted(set(expected) - set(final))
    unexpected = sorted(set(final) - set(expected))
    changed = sorted(n for n in set(expected) & set(final) if final[n] != expected[n])
    variants = fnmatch.filter(final, CPU_VARIANT_GLOB)
    details = {"entries": len(final),
               "added": sorted(set(cpu) - set(rocm)),
               "replaced": sorted(set(cpu) & set(rocm)),
               "missing": missing, "unexpected": unexpected, "changed": changed,
               "cpu_variants": len(variants)}

    problems = []
    if missing:
        problems.append(f"missing {', '.join(missing)}")
    if unexpected:
        problems.append(f"not from either zip: {', '.join(unexpected)}")
    if changed:
        problems.append(f"bytes differ from their source zip: {', '.join(changed)}")
    if problems:
        return (report.STATUS_FAIL,
                "the final zip is not the ROCm zip with the CPU zip merged over it -- "
                + "; ".join(problems), details)
    if len(variants) < MIN_CPU_VARIANTS:
        return (report.STATUS_WARN,
                f"only {len(variants)} CPU backend variant(s) in the final archive; "
                "a complete GGML_CPU_ALL_VARIANTS build yields 14+", details)
    return (report.STATUS_PASS, None, details)


def check_integrity(archive, os_name, meta_path, pre_merge):
    names = read_back(archive)
    required = [SERVER[os_name], HIP_BACKEND[os_name]]
    missing = [r for r in required if r not in names]
    # Windows zips are rewritten by the merge, so it is the zip as it was before
    # the merge that has to match what the build job produced.
    tested = pre_merge if os_name == "win" else archive
    details = {"entries": len(names), "missing": missing,
               "bytes": os.path.getsize(archive),
               "hashed": os.path.basename(tested) if tested else None}

    if missing:
        return (report.STATUS_FAIL,
                f"final archive is missing {', '.join(missing)} -- it was damaged "
                "between the build job and upload", details)
    if not tested:
        return (report.STATUS_FAIL,
                "the release job did not pass --pre-merge, so the zip cannot be traced "
                "back to the tested bytes", details)

    with open(meta_path, encoding="utf-8-sig") as handle:
        expected = json.load(handle).get("sha256")
    actual = sha256(tested)
    details.update(sha256=actual, expected_sha256=expected)
    if not expected:
        return (report.STATUS_FAIL,
                f"{os.path.basename(meta_path)} records no sha256, so the archive cannot "
                "be traced back to the bytes the build job produced", details)
    if actual != expected.lower():
        return (report.STATUS_FAIL,
                f"{os.path.basename(tested)} is not the archive the build job produced "
                "and the qualify jobs tested (SHA-256 mismatch)", details)
    return (report.STATUS_PASS, None, details)


def main():
    parser = argparse.ArgumentParser(description="Post-merge release-archive checks")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--os", dest="os_name", required=True, choices=["ubuntu", "win"])
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--meta", required=True,
                        help="qualify-meta-*.json written by the build job; supplies the "
                             "SHA-256 of the archive it produced")
    parser.add_argument("--cpu-zip", default=None,
                        help="Windows: the CPU zip the release job merged in")
    parser.add_argument("--pre-merge", default=None,
                        help="Windows: a copy of this zip taken before the merge")
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

    tier.run("T0.10", "Windows CPU-backend merge landed",
             lambda: check_cpu_merge(args.archive, args.os_name,
                                     args.cpu_zip, args.pre_merge))
    tier.run("T0.11", "final archive integrity",
             lambda: check_integrity(args.archive, args.os_name,
                                     args.meta, args.pre_merge))

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
