#!/usr/bin/env python3
"""
Publish the qualification records from one release run to the dashboard feed on the `qualification-data` branch.

Reads the records `aggregate.py` wrote (one qualification-<archive>.json per archive, plus qualification-summary.json) and upserts them into --data-dir, which is the `qualification/` folder of a checkout of that branch:

  index.json                    one compact row per archive per run, newest run first
  reports/<build_id>.json       the full record, for drill-down
  latest/<os>-rocm-<ver>.json   the newest full record for that archive

build_id is "<run_id>-<os>-rocm-<rocm_version>".
Publishing the same run again (a re-run) replaces its rows and files rather than duplicating them, so this script is safe to run any number of times.

Every archive the run was meant to build gets a row.
An archive with no record (its build failed / no check ever reported on it) gets a placeholder row and a placeholder report saying so. Therefore, a failing build shows up in the feed instead of silently leaving a gap.

A test the record does not contain is reported as either:
  - "not_applicable": it is never run for this archive -- the post-merge checks T0.10 and T0.11 on an archive that never reached them (it was dropped from the release before they ran); or
  - "missing": it should have run and did not.
Neither is folded into "failed"; `qualified` is the verdict.

A feed written by an older schema_version is not mixed with this one: the first publish starts a fresh index.json, reports/ and latest/.
The old files stay in the branch's git history.

The contract this writes is FEED.md, copied to the branch as SCHEMA.md.

Usage:
    python3 publish_qualification.py --records-dir ./records \
        --data-dir ./data/qualification --operating-systems ubuntu,win \
        --rocm-versions 7.14,10.0,10.1 --tag b12062 --upstream-sha <sha> \
        --jobs-file jobs.jsonl --published-artifacts "<names>" --post-merge-checked "<names>"
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import aggregate  # noqa: E402
import report  # noqa: E402

SCHEMA_VERSION = 2

# Every test the suite runs, in the order the README lists them.
# A test absent from a record shows up as "missing" or "not_applicable", so the dashboard always has the same columns to draw.
# A test id not listed here is still published, never dropped.
TIER_TESTS = {
    "tier0": ["T0.1", "T0.2", "T0.3", "T0.4", "T0.5", "T0.6",
              "T0.7", "T0.8", "T0.9", "T0.10", "T0.11", "T0.12"],
    "tier1": ["T1.1", "T1.2", "T1.3", "T1.4", "T1.5"],
}
# Run by the release job on the archives still in the release after its completeness check.
POST_MERGE_TESTS = {"T0.10", "T0.11"}

STATUS_MISSING = "missing"
STATUS_NOT_APPLICABLE = "not_applicable"
SUMMARY_FILE = "qualification-summary.json"

# build_result is GitHub's conclusion for the archive's build job, in the feed's own vocabulary.
# Any other conclusion (failure, timed_out, startup_failure, ...) is "fail".
BUILD_PASS = "pass"
BUILD_FAIL = "fail"
BUILD_UNKNOWN = "unknown"
BUILD_RESULTS = {"success": BUILD_PASS, "cancelled": "cancelled", "skipped": "skipped"}

# Why a placeholder row has no record, by its build_result.
PLACEHOLDER_REASONS = {
    BUILD_FAIL: "build failed",
    "cancelled": "build cancelled",
    "skipped": "build skipped",
}
PLACEHOLDER_NO_REPORT = "no report"

# build_id becomes a filename on the data branch, so it may only contain characters that cannot escape the folder it is written to.
_SAFE_ID = re.compile(r"^[A-Za-z0-9._-]+$")

# The ROCm build jobs in release.yml. GitHub names a matrix job after its matrix values, the ROCm version first:
# "windows-rocm (10.1.0, gfx1010;..., x64, https://...)". Long names are cut short with "...", so only the first value is read.
_BUILD_JOBS = {"ubuntu-24-rocm": "ubuntu", "windows-rocm": "win"}
_BUILD_JOB = re.compile(r"^(?P<job>[a-z0-9-]+) \((?P<first>[^,)]+)")


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


def _run_order(run_id):
    """GitHub run ids only grow, so they order runs by when they started.

    generated_at cannot: a re-run writes its records later than a newer run did.
    """
    number = _int(run_id)
    return number if number is not None else -1


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


def load_build_results(jobs_file):
    """Map (os, rocm_version) -> result of its build job.

    jobs_file holds one {"name": ..., "conclusion": ...} object per line, as
    `gh api .../jobs --jq '.jobs[] | {name, conclusion}'` prints them.
    """
    results = {}
    if not jobs_file:
        return results
    try:
        with open(jobs_file, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        print(f"::warning::could not read {jobs_file}: {exc}; build_result will be {BUILD_UNKNOWN!r}")
        return results
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            job = json.loads(line)
        except ValueError:
            print(f"::warning::skipping unreadable job line {line!r}")
            continue
        match = _BUILD_JOB.match(job.get("name") or "")
        if not match or match.group("job") not in _BUILD_JOBS:
            continue
        # "10.1.0" -> "10.1", the form the archive names and records use.
        version = ".".join(match.group("first").strip().split(".")[:2])
        conclusion = job.get("conclusion")
        results[(_BUILD_JOBS[match.group("job")], version)] = (
            BUILD_RESULTS.get(conclusion, BUILD_FAIL) if conclusion else BUILD_UNKNOWN)
    return results


def make_build_id(run_id, os_name, rocm_version):
    if not (run_id and os_name and rocm_version):
        raise SystemExit(
            f"::error::cannot form a build_id from run_id={run_id!r} os={os_name!r} "
            f"rocm_version={rocm_version!r}")
    build_id = f"{run_id}-{os_name}-rocm-{rocm_version}"
    if not _SAFE_ID.match(build_id):
        raise SystemExit(f"::error::refusing unsafe build_id {build_id!r}")
    return build_id


def absent_status(record, test_id, post_merge_checked):
    """What a test the record does not contain is reported as."""
    if test_id in POST_MERGE_TESTS and record.get("build", {}).get("artifact") not in post_merge_checked:
        return STATUS_NOT_APPLICABLE
    return STATUS_MISSING


def tier_status(record, tier):
    """pass | warn | fail for a tier that reported, otherwise missing."""
    reported = record.get("tiers", {}).get(tier)
    return reported.get("status") if reported else STATUS_MISSING


def collect_results(record, post_merge_checked):
    """Per-test status across the known suite, plus any test it does not list."""
    by_id = {}
    for tier in record.get("tiers", {}).values():
        for test in tier.get("tests", []):
            by_id.setdefault(test.get("id"), test)

    results = {test_id: by_id[test_id]["status"] if test_id in by_id
               else absent_status(record, test_id, post_merge_checked)
               for ids in TIER_TESTS.values() for test_id in ids}
    for test_id in sorted((t for t in by_id if t not in results), key=_test_order):
        results[test_id] = by_id[test_id]["status"]

    blocking = sorted((test_id for test_id, test in by_id.items()
                       if test.get("gating") and test.get("status") == "fail"),
                      key=_test_order)
    return results, blocking


def make_placeholder(os_name, rocm_version, build_result, tag, upstream_sha,
                     run_id, run_attempt, generated_at):
    """A record-shaped stand-in for an archive that has no record."""
    build = report.build_meta(
        artifact=aggregate.published_name(tag, os_name, rocm_version),
        os_name=os_name,
        rocm_version=rocm_version,
        llamacpp_tag=tag,
        upstream_sha=upstream_sha,
        run_id=run_id,
        run_attempt=str(run_attempt) if run_attempt is not None else None,
    )
    return {
        "schema_version": report.SCHEMA_VERSION,
        "kind": "qualification_placeholder",
        "generated_at": generated_at,
        "build": build,
        "overall": report.STATUS_FAIL,
        "promoted": False,
        "counts": {report.STATUS_PASS: 0, report.STATUS_FAIL: 0,
                   report.STATUS_WARN: 0, report.STATUS_SKIP: 0, "total": 0},
        "tiers": {},
        "promotion": {"blocked_by": [PLACEHOLDER_REASONS.get(build_result, PLACEHOLDER_NO_REPORT)]},
        "build_result": build_result,
        # No archive, so nothing was audited. The gfx target list lives in the build job.
        "gfx": {
            "requested": [],
            "present_in_binary": [],
            "missing_from_binary": [],
            "hardware_validated": [],
        },
    }


def build_entry(record, build_id, build_result, version_qualified, published,
                post_merge_checked, run_url, event, run_attempt):
    """Derive the compact, dashboard-facing index row from a full record."""
    build = record.get("build", {})
    results, blocking = collect_results(record, post_merge_checked)
    statuses = list(results.values())

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
        # The attempt that published this row, the same for every row of the run.
        "run_attempt": run_attempt if run_attempt is not None else _int(build.get("run_attempt")),
        "run_url": run_url,
        "event": event,
        "build_result": build_result,
        "tests": {
            "passed": statuses.count("pass"),
            "failed": statuses.count("fail"),
            "warned": statuses.count("warn"),
            "skipped": statuses.count("skip"),
            "missing": statuses.count(STATUS_MISSING),
            "not_applicable": statuses.count(STATUS_NOT_APPLICABLE),
            "total": len(statuses),
        },
        "qualified": bool(record.get("promoted")),
        "version_qualified": version_qualified,
        "published": published,
        "overall": record.get("overall"),
        "tier0": tier_status(record, "tier0"),
        "tier1": tier_status(record, "tier1"),
        "blocked_by": record.get("promotion", {}).get("blocked_by", []),
        "blocking_tests": blocking,
        "results": results,
        "report_url": f"reports/{build_id}.json",
    }


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
        handle.write("\n")


def start_fresh_if_older(data_dir):
    """Drop a feed written by an older schema_version instead of mixing rows of two shapes."""
    index_path = os.path.join(data_dir, "index.json")
    if not os.path.exists(index_path):
        return
    with open(index_path, encoding="utf-8") as handle:
        version = json.load(handle).get("schema_version")
    if (_int(version) or 0) >= SCHEMA_VERSION:
        return
    print(f"The feed holds schema_version {version}; starting a fresh schema_version "
          f"{SCHEMA_VERSION} feed. The old files stay in the branch's git history.")
    os.remove(index_path)
    for folder in ("reports", "latest"):
        shutil.rmtree(os.path.join(data_dir, folder), ignore_errors=True)


def update_latest(path, record):
    """Write the record unless the file already holds one from a newer run.

    Re-running an old run must not roll the status tile back to an older
    result; a re-run of the same run replaces it.
    """
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as handle:
                current = json.load(handle).get("build", {}).get("run_id")
        except (OSError, ValueError, AttributeError):
            current = None
        if _run_order(current) > _run_order(record.get("build", {}).get("run_id")):
            return False
    write_json(path, record)
    return True


def upsert_index(index_path, repo, entries):
    """Replace-or-append each entry by build_id, newest run first."""
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
    builds.sort(key=lambda b: _run_order(b.get("run_id")), reverse=True)

    return {
        "schema_version": SCHEMA_VERSION,
        "repo": repo or index.get("repo"),
        "updated_at": max((b.get("generated_at") or "" for b in builds), default="") or None,
        "builds": builds,
    }


def main():
    parser = argparse.ArgumentParser(description="Publish qualification records to the dashboard feed")
    parser.add_argument("--records-dir", required=True,
                        help="folder holding the qualification-*.json records aggregate.py wrote")
    parser.add_argument("--data-dir", required=True,
                        help="the qualification/ folder of the qualification-data branch")
    parser.add_argument("--operating-systems", required=True,
                        help="the operating systems every ROCm version is built for, e.g. ubuntu,win")
    parser.add_argument("--rocm-versions", required=True,
                        help="the ROCm versions this run builds, e.g. 7.14,10.0,10.1")
    parser.add_argument("--tag", required=True, help="the llama.cpp build number this run pinned")
    parser.add_argument("--upstream-sha", help="the llama.cpp commit this run pinned")
    parser.add_argument("--jobs-file",
                        help="this run's jobs, one {name, conclusion} JSON object per line")
    parser.add_argument("--published-artifacts", default="",
                        help="the ROCm archives the release job uploaded, space separated")
    parser.add_argument("--post-merge-checked", default="",
                        help="the ROCm archives the post-merge checks ran on, space separated")
    parser.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME"),
                        help="what triggered the run, e.g. schedule or workflow_dispatch")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"),
                        help="owner/name, used to build run_url")
    parser.add_argument("--server-url", default=os.environ.get("GITHUB_SERVER_URL", "https://github.com"))
    parser.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID"))
    parser.add_argument("--run-attempt", default=os.environ.get("GITHUB_RUN_ATTEMPT"))
    args = parser.parse_args()

    expected = [(o, v) for v in aggregate.split_list(args.rocm_versions)
                for o in aggregate.split_list(args.operating_systems)]
    published_artifacts = set(args.published_artifacts.split())
    post_merge_checked = set(args.post_merge_checked.split())
    run_attempt = _int(args.run_attempt)

    records, summary = load_records(args.records_dir)
    if summary is None:
        print(f"::warning::no {SUMMARY_FILE} in {args.records_dir}; version_qualified will be null")
    blocked_versions = set((summary or {}).get("blocked_rocm_versions") or [])
    build_results = load_build_results(args.jobs_file)
    if args.jobs_file and not build_results:
        print(f"::warning::no ROCm build jobs found in {args.jobs_file}; build_result will be {BUILD_UNKNOWN!r}")

    # Placeholders share the run's timestamp, so every row of a run shows the same time.
    run_time = max((r.get("generated_at") or "" for r in records), default="") or report.now_iso()

    covered = set()
    items = []  # (record, build_id, is_placeholder)
    for record in records:
        build = record.get("build", {})
        key = (build.get("os"), build.get("rocm_version"))
        covered.add(key)
        items.append((record, make_build_id(build.get("run_id") or args.run_id, *key), False))
    for os_name, rocm_version in expected:
        if (os_name, rocm_version) in covered:
            continue
        build_result = build_results.get((os_name, rocm_version), BUILD_UNKNOWN)
        record = make_placeholder(os_name, rocm_version, build_result, args.tag, args.upstream_sha,
                                  args.run_id, run_attempt, run_time)
        items.append((record, make_build_id(args.run_id, os_name, rocm_version), True))

    if not items:
        raise SystemExit(f"::error::no qualification records in {args.records_dir} and no "
                         "expected archives; nothing to publish")

    start_fresh_if_older(args.data_dir)
    os.makedirs(os.path.join(args.data_dir, "reports"), exist_ok=True)
    os.makedirs(os.path.join(args.data_dir, "latest"), exist_ok=True)

    entries = []
    for record, build_id, is_placeholder in items:
        build = record["build"]
        run_url = (f"{args.server_url}/{args.repo}/actions/runs/{build.get('run_id') or args.run_id}"
                   if args.repo else None)
        build_result = build_results.get((build.get("os"), build.get("rocm_version")), BUILD_UNKNOWN)
        # An archive without a record can never let its version qualify.
        if is_placeholder:
            version_qualified = False
        elif summary is None:
            version_qualified = None
        else:
            version_qualified = build.get("rocm_version") not in blocked_versions
        published = not is_placeholder and build.get("artifact") in published_artifacts
        entry = build_entry(record, build_id, build_result, version_qualified, published,
                            post_merge_checked, run_url, args.event, run_attempt)
        entries.append(entry)

        write_json(os.path.join(args.data_dir, "reports", f"{build_id}.json"), record)
        latest = os.path.join(args.data_dir, "latest",
                              f"{build['os']}-rocm-{build['rocm_version']}.json")
        moved = update_latest(latest, record)

        tests = entry["tests"]
        print(f"{build_id}: {tests['passed']}/{tests['total']} passed, "
              f"{tests['failed']} failed, {tests['missing']} missing, "
              f"build={build_result} qualified={entry['qualified']} "
              f"version_qualified={entry['version_qualified']} published={entry['published']}"
              f"{' (placeholder: ' + ', '.join(entry['blocked_by']) + ')' if is_placeholder else ''}"
              f"{'' if moved else ' (latest/ kept: it already holds a newer run)'}")

    index_path = os.path.join(args.data_dir, "index.json")
    index = upsert_index(index_path, args.repo, entries)
    write_json(index_path, index)
    print(f"Published {len(entries)} row(s); {len(index['builds'])} build(s) in the index")


if __name__ == "__main__":
    main()
