#!/usr/bin/env python3
"""
Final-archive check on the exact bytes about to be uploaded. No GPU required.

Runs inside the release job AFTER "Move artifacts", which is the only place
this question can be answered:

  T0.11  final archive integrity -- the final archive opens, reads back intact,
         still contains llama-server and the HIP backend, and is the archive the
         build job produced and the qualify jobs tested: its SHA-256 must match
         the one the build job recorded.

(T0.10, which checked the merge of the Windows CPU zip into the ROCm zips, is
retired: the release job ships the ROCm zips exactly as built. Its id is not
reused.)

Every file is opened inside the check, so an archive that cannot be read is
recorded as a FAIL instead of crashing the script and leaving no result.

Usage:
    python3 check_post_merge.py --archive release/llama-bNNNNN-bin-win-rocm-10.1-x64.zip \
        --os win --rocm-version 10.1 --meta meta/qualify-meta-win-rocm-10.1.json \
        --output frag.json
"""

import argparse
import hashlib
import json
import os
import sys
import tarfile
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402

HIP_BACKEND = {"ubuntu": "libggml-hip.so", "win": "ggml-hip.dll"}
SERVER = {"ubuntu": "llama-server", "win": "llama-server.exe"}


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


def check_integrity(archive, os_name, meta_path):
    names = read_back(archive)
    required = [SERVER[os_name], HIP_BACKEND[os_name]]
    missing = [r for r in required if r not in names]
    details = {"entries": len(names), "missing": missing,
               "bytes": os.path.getsize(archive)}

    if missing:
        return (report.STATUS_FAIL,
                f"final archive is missing {', '.join(missing)} -- it was damaged "
                "between the build job and upload", details)

    with open(meta_path, encoding="utf-8-sig") as handle:
        expected = json.load(handle).get("sha256")
    actual = sha256(archive)
    details.update(sha256=actual, expected_sha256=expected)
    if not expected:
        return (report.STATUS_FAIL,
                f"{os.path.basename(meta_path)} records no sha256, so the archive cannot "
                "be traced back to the bytes the build job produced", details)
    if actual != expected.lower():
        return (report.STATUS_FAIL,
                f"{os.path.basename(archive)} is not the archive the build job produced "
                "and the qualify jobs tested (SHA-256 mismatch)", details)
    return (report.STATUS_PASS, None, details)


def main():
    parser = argparse.ArgumentParser(description="Final release-archive check")
    parser.add_argument("--archive", required=True)
    parser.add_argument("--os", dest="os_name", required=True, choices=["ubuntu", "win"])
    parser.add_argument("--rocm-version", required=True)
    parser.add_argument("--meta", required=True,
                        help="qualify-meta-*.json written by the build job; supplies the "
                             "SHA-256 of the archive it produced")
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

    tier.run("T0.11", "final archive integrity",
             lambda: check_integrity(args.archive, args.os_name, args.meta))

    tier.write(args.output)
    tier.print_summary()
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
