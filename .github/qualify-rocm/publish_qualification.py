#!/usr/bin/env python3
"""
Publish the qualification records from one release run to the dashboard feed
on the `qualification-data` branch.

Reads the records `aggregate.py` wrote (one qualification-<archive>.json per
archive, plus qualification-summary.json) and upserts them into --data-dir,
which is the `qualification/` folder of a checkout of that branch:

  index.json                    one compact row per archive per run, newest first
  reports/<build_id>.json       the full record, for drill-down
  latest/<os>-rocm-<ver>.json   the newest full record for that archive

build_id is "<run_id>-<os>-rocm-<rocm_version>".
Publishing the same run again (a re-run of the release job) replaces its rows and files
rather than duplicating them, so this script is safe to run any number of times.

A test the record does not contain is reported as "missing" and counted separately -- NOT folded into "failed".
A missing Tier 1 is a legitimate state: the archive was not tested on hardware, and aggregate.py lets it ship.
Counting it as a failure would show a published archive as red.

The contract this writes is FEED.md, copied to the branch as SCHEMA.md.

Usage:
    python3 publish_qualification.py --records-dir ./qualification \
        --data-dir ./data/qualification --event schedule
"""

import argparse
import glob
import json
import os
import re
import sys

SCHEMA_VERSION = 1

# Every test the suite runs, in the order the README lists them.
# A test absent from a record shows up as "missing", so the dashboard always has the same columns to draw.
# A test id not listed here is still published, never dropped.
TIER_TESTS = {
    "tier0": ["T0.1", "T0.2", "T0.3", "T0.4", "T0.5", "T0.6",
              "T0.7", "T0.8", "T0.9", "T0.10", "T0.11", "T0.12"],
    "tier1": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"],
}
KNOWN_TESTS = [test_id for ids in TIER_TESTS.values() for test_id in ids]

STATUS_MISSING = "missing"
SUMMARY_FILE = "qualification-summary.json"

# build_id becomes a filename on the data branch, so it may only contain
# characters that cannot escape the folder it is written to.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _test_order(test_id):
    """Sort "T0.10" after "T0.9", and unknown ids after known ones."""
    match = re.match(r"^T(\d+)\.(\d+)$", test_id or "")
    if match:
        return (int(match.group(1)), int(match.group(2)), "")
    return (sys.maxsize, sys.maxsize, test_id or "")


def load_records(records_dir):
    """Return (records, summary) from the folder aggregate.py wrote."""
    records = []
    summary = None
    for path in sorted(glob.glob(os.path.join(records_dir, "*.json"))):
        try:
            with open(path, encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError) as exc:
            print(f"::warning::skipping unreadable record {path}: {exc}")
            continue
        if os.path.basename(path) == SUMMARY_FILE:
            summary = data
        elif data.get("kind") == "qualification_record":
            records.append(data)
    return records, summary


def build_id_for(record):
    build = record.get("build", {})
    run_id = build.get("run_id")
    os_name = build.get("os")
    rocm_version = build.get("rocm_version")
    if not (run_id and os_name and rocm_version):
        raise SystemExit(
            f"::error::record for {build.get('artifact')!r} has no run_id, os or "
            "rocm_version, so it cannot be given a build_id")
    build_id = f"{run_id}-{os_name}-rocm-{rocm_version}"
    if not _SAFE_ID.match(build_id):
        raise SystemExit(f"::error::refusing unsafe build_id {build_id!r}")
    return build_id


def collect_results(record):
    """Per-test status across the known suite, plus any test it does not list."""
    by_id = {}
    for tier in record.get("tiers", {}).values():
        for test in tier.get("tests", []):
            by_id.setdefault(test.get("id"), test)

    results = {test_id: by_id[test_id]["status"] if test_id in by_id else STATUS_MISSING
               for test_id in KNOWN_TESTS}
    for test_id in sorted((t for t in by_id if t not in results), key=_test_order):
        results[test_id] = by_id[test_id]["status"]

    blocking = sorted((test_id for test_id, test in by_id.items()
                       if test.get("gating") and test.get("status") == "fail"),
                      key=_test_order)
    return results, blocking


def build_entry(record, build_id, summary, run_url, event):
    """Derive the compact, dashboard-facing index row from a full record."""
    build = record.get("build", {})
    results, blocking = collect_results(record)
    statuses = list(results.values())

    version_qualified = None
    if summary is not None:
        blocked = summary.get("blocked_rocm_versions") or []
        version_qualified = build.get("rocm_version") not in blocked

    gfx = record.get("gfx", {})
    return {
        "build_id": build_id,
        "tag": build.get("llamacpp_tag"),
        "upstream_sha": build.get("upstream_sha"),
        "artifact": build.get("artifact"),
        "os": build.get("os"),
        "rocm_version": build.get("rocm_version"),
        "gfx_target": build.get("gfx_target"),
        "hardware_validated": bool(build.get("hardware_validated")),
        "gfx": {
            "requested": len(gfx.get("requested", [])),
            "present": len(gfx.get("present_in_binary", [])),
            "missing": gfx.get("missing_from_binary", []),
        },
        "generated_at": record.get("generated_at"),
        "run_id": build.get("run_id"),
        "run_attempt": _int(build.get("run_attempt")),
        "run_url": run_url,
        "event": event,
        "tests": {
            "passed": statuses.count("pass"),
            "failed": statuses.count("fail"),
            "warned": statuses.count("warn"),
            "skipped": statuses.count("skip"),
            "missing": statuses.count(STATUS_MISSING),
            "total": len(statuses),
        },
        "qualified": bool(record.get("promoted")),
        "version_qualified": version_qualified,
        "overall": record.get("overall"),
        "blocked_by": record.get("promotion", {}).get("blocked_by", []),
        "blocking_tests": blocking,
        "results": results,
        "report_url": f"reports/{build_id}.json",
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def update_latest(path, record):
    """Write the record unless the file already holds a newer one.

    Re-publishing an old run (a late re-run) must not roll the status tile
    back to an older result.
    """
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                current = json.load(handle).get("generated_at") or ""
        except (OSError, ValueError):
            current = ""
        if current > (record.get("generated_at") or ""):
            return False
    write_json(path, record)
    return True


def upsert_index(index_path, repo, entries):
    """Replace-or-append each entry by build_id, newest first."""
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as handle:
            index = json.load(handle)
    else:
        index = {}

    replaced = {entry["build_id"] for entry in entries}
    builds = [b for b in index.get("builds", []) if b.get("build_id") not in replaced]
    builds.extend(entries)
    # Newest run first; the archives within one run in a fixed, readable order.
    builds.sort(key=lambda b: b.get("build_id") or "")
    builds.sort(key=lambda b: b.get("generated_at") or "", reverse=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo or index.get("repo"),
        "updated_at": builds[0].get("generated_at") if builds else None,
        "builds": builds,
    }


def main():
    parser = argparse.ArgumentParser(description="Publish qualification records to the dashboard feed")
    parser.add_argument("--records-dir", required=True,
                        help="folder holding the qualification-*.json records aggregate.py wrote")
    parser.add_argument("--data-dir", required=True,
                        help="the qualification/ folder of the qualification-data branch")
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME"),
                        help="what triggered the run, e.g. schedule or workflow_dispatch")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                        help="owner/name, used to build run_url")
    parser.add_argument("--server-url", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    args = parser.parse_args()

    records, summary = load_records(args.records_dir)
    if not records:
        raise SystemExit(f"::error::no qualification records in {args.records_dir}; nothing to publish")
    if summary is None:
        print(f"::warning::no {SUMMARY_FILE} in {args.records_dir}; version_qualified will be null")

    os.makedirs(os.path.join(args.data_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "latest"), exist_ok=True)

    entries = []
    for record in records:
        build = record["build"]
        build_id = build_id_for(record)
        run_url = (f"{args.server_url}/{args.repo}/actions/runs/{build['run_id']}"
                   if args.repo else None)
        entry = build_entry(record, build_id, summary, run_url, args.event)
        entries.append(entry)

        write_json(os.path.join(args.data_dir, "reports", f"{build_id}.json"), record)
        latest = os.path.join(args.data_dir, "latest",
                              f"{build['os']}-rocm-{build['rocm_version']}.json")
        moved = update_latest(latest, record)

        tests = entry["tests"]
        print(f"{build_id}: {tests['passed']}/{tests['total']} passed, "
              f"{tests['failed']} failed, {tests['missing']} missing, "
              f"qualified={entry['qualified']} version_qualified={entry['version_qualified']}"
              f"{'' if moved else ' (latest/ kept: it already holds a newer run)'}")

    index_path = os.path.join(args.data_dir, "index.json")
    index = upsert_index(index_path, args.repo, entries)
    write_json(index_path, index)
    print(f"Published {len(entries)} record(s); {len(index['builds'])} build(s) in the index")


if __name__ == "__main__":
    main()
