#!/usr/bin/env python3
"""
Merge tier reports into one qualification record per published archive, decide
which ROCm versions may be published, and write the job summary.

Reports are grouped by (OS, ROCm version), because that pair identifies exactly
one release asset -- the unit a user downloads and the unit a bug report names.

A ROCm version is publishable when BOTH its archives (Windows and Linux) pass.
Versions are judged independently of each other: they are separate products and
a consumer installs exactly one of them, so a problem in one is no reason to
withhold another.

Within one archive:
  - the static checks must be present and must pass.
  - the GPU checks must pass IF they ran. An archive with no GPU report is
    recorded as not hardware-validated rather than being failed -- each build
    covers 20 or more gfx targets and only one of them has test hardware, so
    the record states which is which instead of implying they are all equal.

With --fail-on-no-promote the exit code is non-zero if any version failed. The
release job uses that to drop only the affected ROCm versions, leaving every
other backend untouched.

Usage:
    python3 aggregate.py --fragments-dir ./fragments --tag bNNNNN \
        --records-dir ./qualification --output-summary summary.json \
        --fail-on-no-promote
"""

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import report  # noqa: E402


def load_fragments(fragments_dir):
    fragments = []
    for path in sorted(glob.glob(os.path.join(fragments_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"::warning::skipping unreadable fragment {path}: {exc}")
            continue
        if data.get("kind") == "tier_fragment":
            fragments.append(data)
    return fragments


def group_key(fragment):
    build = fragment.get("build", {})
    return (build.get("os") or "?", build.get("rocm_version") or "?")


def published_name(tag, os_name, rocm_version):
    """The release asset filename for this (os, rocm) pair.
    """
    if not tag:
        return None
    if os_name == "win":
        return f"llama-{tag}-bin-win-rocm-{rocm_version}-x64.zip"
    if os_name == "ubuntu":
        return f"llama-{tag}-bin-ubuntu-rocm-{rocm_version}-x64.tar.gz"
    return None


def decide(record):
    """Return (publishable, reasons) for one archive's merged record."""
    reasons = []
    tiers = record.get("tiers", {})

    if "tier0" not in tiers:
        reasons.append("tier0 missing")
    elif tiers["tier0"]["status"] == report.STATUS_FAIL:
        reasons.append("tier0=fail")

    if "tier1" in tiers and tiers["tier1"]["status"] == report.STATUS_FAIL:
        reasons.append("tier1=fail")

    return not reasons, reasons


def gfx_summary(record):
    """Pull the gfx audit result out of T0.3 for the record's top level.

    This is the field that answers "was my GPU ever verified?" -- present in
    the binary is not the same claim as validated on hardware, and the record
    should not blur them.
    """
    tests = record.get("tiers", {}).get("tier0", {}).get("tests", [])
    audit = next((t for t in tests if t.get("id") == "T0.3"), None)
    details = (audit or {}).get("details", {})
    validated = []
    if record.get("build", {}).get("hardware_validated"):
        target = record.get("build", {}).get("gfx_target")
        if target:
            validated = [target]
    return {
        "requested": details.get("expected", []),
        "present_in_binary": details.get("present", []),
        "missing_from_binary": details.get("missing", []),
        "hardware_validated": validated,
    }


def write_step_summary(records, promote_all):
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    lines = ["## ROCm build qualification", "",
             f"**ROCm family publishable: {'YES' if promote_all else 'NO'}**", "",
             "| Artifact | Tier 0 | Tier 1 | gfx in binary | HW validated | Promote |",
             "|---|---|---|---|---|---|"]
    for record in records:
        build = record["build"]
        tiers = record.get("tiers", {})
        gfx = record["gfx"]
        t0 = tiers.get("tier0", {}).get("status", "missing").upper()
        t1 = tiers.get("tier1", {}).get("status", "not run").upper()
        hw = ", ".join(gfx["hardware_validated"]) or "none"
        lines.append(
            f"| `{build.get('artifact')}` | {t0} | {t1} | "
            f"{len(gfx['present_in_binary'])}/{len(gfx['requested'])} | {hw} | "
            f"{'yes' if record['promoted'] else 'NO'} |")

    problems = []
    for record in records:
        for tier in sorted(record.get("tiers", {})):
            for test in record["tiers"][tier].get("tests", []):
                if test["status"] in (report.STATUS_FAIL, report.STATUS_WARN):
                    gate = "" if test.get("gating") else " _(non-gating)_"
                    problems.append(
                        f"- `{record['build'].get('artifact')}` `{test['id']}` "
                        f"**{test['status']}**{gate} — {test['name']}: "
                        f"{test.get('error') or ''}")
    if problems:
        lines += ["", "### Findings", *problems]

    with open(path, "a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Aggregate ROCm tier fragments")
    parser.add_argument("--fragments-dir", required=True)
    parser.add_argument("--tag", default=None, help="llama.cpp release tag")
    parser.add_argument("--records-dir", default=None,
                        help="write one qualification-<artifact>.json per archive here")
    parser.add_argument("--output-summary", default=None,
                        help="write the combined decision here")
    parser.add_argument("--fail-on-no-promote", action="store_true")
    args = parser.parse_args()

    fragments = load_fragments(args.fragments_dir)
    if not fragments:
        print(f"::error::No tier fragments found in {args.fragments_dir}. The "
              "qualification jobs produced nothing, so nothing can be vouched for.")
        if args.fail_on_no_promote:
            sys.exit(1)
        return

    grouped = {}
    for fragment in fragments:
        grouped.setdefault(group_key(fragment), []).append(fragment)

    records = []
    for (os_name, rocm_version), group in sorted(grouped.items()):
        record = report.merge_fragments(group, extra_meta={
            "llamacpp_tag": args.tag,
            "artifact": published_name(args.tag, os_name, rocm_version),
        })
        promote, reasons = decide(record)
        record["promoted"] = promote
        record["promotion"] = {"blocked_by": reasons}
        record["gfx"] = gfx_summary(record)
        records.append(record)

        artifact = record["build"].get("artifact") or f"{os_name}-rocm-{rocm_version}"
        if args.records_dir:
            os.makedirs(args.records_dir, exist_ok=True)
            stem = artifact
            for suffix in (".tar.gz", ".zip", ".tgz"):
                if stem.endswith(suffix):
                    stem = stem[: -len(suffix)]
                    break
            out = os.path.join(args.records_dir, f"qualification-{stem}.json")
            with open(out, "w", encoding="utf-8") as handle:
                json.dump(record, handle, indent=2)
                handle.write("\n")
            print(f"Wrote {out}")

        print(f"{artifact}: overall={record['overall']} promote={promote} "
              f"blocked_by={reasons} "
              f"gfx={len(record['gfx']['present_in_binary'])}/"
              f"{len(record['gfx']['requested'])} "
              f"hw_validated={record['gfx']['hardware_validated'] or 'none'}")

    # A ROCm version is publishable only when every archive carrying it passed.
    # Versions are independent, so one bad version never withholds the others.
    by_version = {}
    for record in records:
        version = record["build"].get("rocm_version") or "?"
        by_version.setdefault(version, []).append(record)
    blocked = sorted(v for v, group in by_version.items()
                     if not all(r["promoted"] for r in group))
    promote_all = not blocked

    write_step_summary(records, promote_all)

    if args.output_summary:
        with open(args.output_summary, "w", encoding="utf-8") as handle:
            json.dump({
                "tag": args.tag,
                "all_versions_publishable": promote_all,
                "blocked_rocm_versions": blocked,
                "artifacts": [{"artifact": r["build"].get("artifact"),
                               "rocm_version": r["build"].get("rocm_version"),
                               "publishable": r["promoted"],
                               "overall": r["overall"],
                               "blocked_by": r["promotion"]["blocked_by"]}
                              for r in records],
            }, handle, indent=2)
            handle.write("\n")

    github_output = os.environ.get("GITHUB_OUTPUT")
    if github_output:
        with open(github_output, "a", encoding="utf-8") as handle:
            handle.write(f"promote_rocm={'true' if promote_all else 'false'}\n")
            handle.write(f"blocked_rocm_versions={' '.join(blocked)}\n")

    if blocked:
        print(f"::error::ROCm {', '.join(blocked)} did not qualify and will be "
              "dropped from this release. Other ROCm versions and every other "
              "backend are unaffected.")
    if args.fail_on_no_promote and blocked:
        sys.exit(1)


if __name__ == "__main__":
    main()
