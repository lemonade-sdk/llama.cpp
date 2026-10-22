# Qualification feed — published contract

Every release run publishes its ROCm qualification records to the
**`qualification-data`** branch of this repository, in a fixed layout an
external dashboard can poll. This file is that contract. A copy is kept on the
branch as `qualification/SCHEMA.md`, refreshed on every publish, so the data and
its description never drift apart. Treat it as versioned (`schema_version`).

## Why a branch and not the releases

The records are already attached to each release, but that copy cannot serve a
dashboard:

- A release only exists when something shipped. The dashboard needs every run,
  including the ones whose ROCm archives were withheld.
- Reading history out of releases means one API call per release, and the
  Releases API allows only 60 unauthenticated requests an hour.

A file on a branch is one plain HTTP GET with no authentication and no rate
limit. The branch is only the transport; the contract below would hold
unchanged if the files moved to GitHub Pages, a bucket or a database.

## Layout

```
qualification/
  index.json                    # the feed: one compact row per archive per run
  reports/<build_id>.json       # the full record for that row (drill-down)
  latest/<os>-rocm-<ver>.json   # the newest full record for each archive
  SCHEMA.md                     # a copy of this file
```

Public raw base URL (no authentication, about 5 minutes of CDN caching):

```
https://raw.githubusercontent.com/lemonade-sdk/llama.cpp/qualification-data/qualification/
```

The dashboard's one regular fetch is `.../qualification/index.json`.

## One row per archive

A release run builds six ROCm archives — Windows and Linux for each of ROCm
7.14, 10.0 and 10.1 — and each is qualified on its own. Each one gets its own
row, because an archive is what a user downloads and what a bug report names.
Rows from the same run share `run_id` and `tag`; group on those to show a run as
a whole.

## `build_id` and re-runs

```
build_id = "<run_id>-<os>-rocm-<rocm_version>"     e.g. 39775718636-win-rocm-10.1
```

Unique per archive per run. Publishing is an upsert keyed on `build_id`:
re-running the release job replaces that run's rows and files instead of adding
new ones. `run_attempt` records which attempt the row came from.

## Which runs appear

Runs of `.github/workflows/release.yml` that reach its `release` job: the
scheduled runs, and manual runs with `create_release` ticked. Pass or fail,
every record that job writes is published.

Not in the feed:

- **Pull requests**, and manual runs without `create_release`. They run the
  checks, but never reach the step that turns them into records.
- **An archive whose build failed.** There was nothing to check, so there is no
  record and no row. A run normally has six rows; fewer means a build leg
  failed (see `run_url`).
- **A run that failed before qualifying anything**, for example because no
  artifact at all was built.

## `index.json`

```jsonc
{
  "schema_version": 1,
  "repo": "lemonade-sdk/llama.cpp",
  "updated_at": "2026-10-22T14:51:14Z",       // generated_at of the newest row
  "builds": [                                   // newest run first
    {
      "build_id": "39775718636-win-rocm-10.1",
      "tag": "b11819",                          // llama.cpp build number = the release tag
      "upstream_sha": "0693dbfb915f33233818373b5f6969396658d02c",
      "artifact": "llama-b11819-bin-win-rocm-10.1-x64.zip",
      "os": "win",                              // win | ubuntu
      "rocm_version": "10.1",                   // currently 7.14 | 10.0 | 10.1
      "gfx_target": "gfx1151",                  // the target tested on real hardware, or null
      "hardware_validated": true,               // did Tier 1 run on a GPU
      "gfx": { "requested": 20, "present": 20, "missing": [] },
      "generated_at": "2026-10-22T14:51:14Z",   // ISO-8601 UTC, the time axis
      "run_id": "39775718636",
      "run_attempt": 1,
      "run_url": "https://github.com/lemonade-sdk/llama.cpp/actions/runs/39775718636",
      "event": "schedule",                      // schedule | workflow_dispatch
      "tests": { "passed": 13, "failed": 0, "warned": 1, "skipped": 3, "missing": 0, "total": 17 },
      "qualified": true,                        // this archive passed its checks
      "version_qualified": true,                // both archives of this ROCm version qualified
      "overall": "warn",                        // pass | warn | fail
      "blocked_by": [],                         // e.g. ["tier0=fail"]
      "blocking_tests": [],                     // e.g. ["T0.3"]
      "results": {
        "T0.1": "warn", "T0.2": "pass", "T0.3": "pass", "T0.4": "pass",
        "T0.5": "pass", "T0.6": "skip", "T0.7": "skip", "T0.8": "pass",
        "T0.9": "skip", "T0.10": "pass", "T0.11": "pass", "T0.12": "pass",
        "T1.1": "pass", "T1.2": "pass", "T1.3": "pass", "T1.4": "pass", "T1.5": "pass"
      },
      "report_url": "reports/39775718636-win-rocm-10.1.json"
    }
  ]
}
```

### Field notes

- **`results`** has one entry per test: T0.1–T0.12 and T1.1–T1.5, as listed in
  [README.md](README.md). Values are `pass | fail | warn | skip | missing`. A
  test that did not run at all is `missing`. A test id added to the suite later
  appears here as soon as it runs, so the failure-frequency and heatmap views
  can be built from `index.json` alone.
- **`tests`** counts the values in `results`, so
  `passed + failed + warned + skipped + missing == total` always holds.
- **`missing` is not `failed`.** This is the one deliberate difference from the
  vllm-rocm feed, which counts missing tests as failed. Here, Tier 1 needs a
  machine with an AMD GPU; when none ran it, all five T1 tests are `missing`,
  `hardware_validated` is `false`, and the archive can still pass
  (`qualified: true`). Showing that as a failure would paint a shipped archive
  red. Use `qualified`, not the counts, to decide pass or fail.
- **`qualified`** is the verdict for this one archive: Tier 0 ran and did not
  fail, and Tier 1 did not fail if it ran. A `warn` does not block, and tests
  marked non-gating in README.md never affect it.
- **`version_qualified`** is the verdict the release acts on. A ROCm version is
  withheld from the release unless its Windows **and** Linux archives both
  qualify, so an archive can have `qualified: true` and still not ship because
  its counterpart failed. `null` means that run did not record the decision.
  (A version can also be withheld because one of its archives failed to build;
  that archive then has no row at all.)
- **`blocked_by`** gives the reason per tier (`tier0=fail`, `tier0 missing`,
  `tier1=fail`); **`blocking_tests`** lists the release-blocking tests that
  failed. Both are empty when the archive qualified.
- **`gfx`** summarises the check that every gfx target was compiled into the
  GPU backend (T0.3): how many were requested, how many are present, and the
  names of any that are missing.
- **`gfx_target` / `hardware_validated`** say which single target was actually
  tested on a GPU. Every other target in `gfx.requested` was only checked for
  presence, never run. Don't present those as tested.
- **`report_url`** is relative to the base URL; it holds every check's
  `status`, `gating`, `error` and `details`.

## `reports/<build_id>.json`

The `qualification-<archive>.json` record exactly as `aggregate.py` wrote it
(`kind: "qualification_record"`): the `build` metadata, `overall`, `promoted`,
`promotion.blocked_by`, raw `counts`, the `gfx` audit, and `tiers` with every
test. The same content is attached to the release when the run published one.
Its own `build.build_id` (`<artifact>@<run_id>`) differs from the feed's
`build_id`; use the feed's.

## `latest/<os>-rocm-<ver>.json`

The newest full record for one archive (for example `latest/win-rocm-10.1.json`),
for status tiles that want the current state without scanning the index. It is
replaced only by a newer record, so re-running an old release cannot move it
backwards.

## Versioning and growth

Bump `schema_version` on any breaking change to `index.json` or the row shape.
Adding a field is not breaking. The dashboard should read `schema_version` and
show a notice, rather than break, on a version it does not know.

`index.json` grows by about six rows per run, roughly a dozen a day. That is
comfortable to poll for a long time, especially compressed. If it ever is not,
older rows will move into per-month files under a new `schema_version`.
