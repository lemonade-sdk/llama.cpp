# Qualification feed — published contract

Every release run publishes its ROCm qualification records to the
**`qualification-data`** branch of this repository, in a fixed layout an
external dashboard can poll. This file is that contract. A copy is kept on the
branch as `qualification/SCHEMA.md`, refreshed on every publish, so the data and
its description never drift apart. Treat it as versioned (`schema_version`).

The layout and field names match the
[lemonade-sdk/llamacpp-rocm feed](https://github.com/lemonade-sdk/llamacpp-rocm/blob/qualification-data/qualification/SCHEMA.md),
so one dashboard can read both. Where the two repositories' archives really
differ, the values differ too;
[Differences from lemonade-sdk/llamacpp-rocm](#differences-from-lemonade-sdkllamacpp-rocm)
lists every one.

## Why a branch and not the releases

The records are already attached to each release, but that copy cannot serve a
dashboard:

- A release only holds what shipped. The dashboard needs every run, including
  the ones whose ROCm archives were withheld.
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
7.14, 10.0 and 10.1 (`ROCM_VERSIONS` in `release.yml`) — and each is qualified
on its own. Each one gets its own row, because an archive is what a user
downloads and what a bug report names. Rows from the same run share `run_id`;
group on it to show a run as a whole.

Every archive the run was meant to build gets a row, including one whose build
failed (see [Placeholder rows](#placeholder-rows)). A run therefore always has
six rows.

## `build_id` and re-runs

```
build_id = "<run_id>-<os>-rocm-<rocm_version>"     e.g. 41550569320-win-rocm-10.1
```

Unique per archive per run. Publishing is an upsert keyed on `build_id`. A
re-run of a run keeps its `run_id`, so it replaces that run's rows and its
`reports/` files instead of adding new ones; the latest attempt's values win.
`run_attempt` records which attempt published the rows. A new run has a new
`run_id` and adds new rows.

## Which runs appear

Runs of `.github/workflows/release.yml` whose `release` job runs: the scheduled
runs, and manual runs with `create_release` ticked. Pass or fail, every one of
them is published. The records normally come from the `release` job; if it
stopped before writing them, `publish-qualification` builds them from the
Tier 0 and Tier 1 reports instead.

Not in the feed:

- **Pull requests**, and manual runs without `create_release`.
- **A run that was cancelled.**
- **A run whose `resolve-upstream` job failed.** No llama.cpp commit was
  pinned, so nothing was built.

## Example `index.json`

```jsonc
{
  "schema_version": 2,
  "repo": "lemonade-sdk/llama.cpp",
  "updated_at": "2026-11-03T03:40:49Z",       // the most recent generated_at of any row
  "builds": [                                   // newest run (highest run_id) first
    // ... the other rows of run 41550569320 ...

    // A placeholder: this archive's build failed, so there was nothing to check.
    {
      "build_id": "41550569320-win-rocm-10.1",
      "tag": "b12062",                          // llama.cpp build number; set even though nothing shipped
      "upstream_sha": "29c720eaf72fcf8e4e61a56a88b994d92668261d",
      "artifact": "llama-b12062-bin-win-rocm-10.1-x64.zip",   // the name it would have had
      "os": "win",
      "rocm_version": "10.1",
      "gfx_target": null,
      "hardware_validated": false,
      "gfx": { "requested": 0, "present": 0, "missing": [] },  // no archive, so nothing was audited
      "generated_at": "2026-11-03T03:40:49Z",
      "run_id": "41550569320",
      "run_attempt": 1,
      "run_url": "https://github.com/lemonade-sdk/llama.cpp/actions/runs/41550569320",
      "event": "schedule",
      "build_result": "fail",                   // the windows-rocm (10.1.0) job failed
      "tests": { "passed": 0, "failed": 0, "warned": 0, "skipped": 0, "missing": 14, "not_applicable": 2, "total": 16 },
      "qualified": false,
      "version_qualified": false,
      "published": false,
      "overall": "fail",
      "tier0": "missing",                       // should have run, but there was no archive
      "tier1": "missing",
      "blocked_by": ["build failed"],
      "blocking_tests": [],
      "results": {
        "T0.1": "missing", "T0.2": "missing", "T0.3": "missing", "T0.4": "missing",
        "T0.5": "missing", "T0.6": "missing", "T0.7": "missing", "T0.8": "missing",
        "T0.10": "not_applicable", "T0.11": "not_applicable", "T0.12": "missing",
        "T1.1": "missing", "T1.2": "missing", "T1.3": "missing", "T1.4": "missing", "T1.5": "missing"
      },
      "report_url": "reports/41550569320-win-rocm-10.1.json"
    },

    // The Linux archive of the same version, from the same run: it passed every
    // check, but did not ship, because its Windows counterpart above never built.
    {
      "build_id": "41550569320-ubuntu-rocm-10.1",
      "tag": "b12062",
      "upstream_sha": "29c720eaf72fcf8e4e61a56a88b994d92668261d",
      "artifact": "llama-b12062-bin-ubuntu-rocm-10.1-x64.tar.gz",
      "os": "ubuntu",
      "rocm_version": "10.1",
      "gfx_target": "gfx1151",
      "hardware_validated": true,
      "gfx": { "requested": 27, "present": 27, "missing": [] },
      "generated_at": "2026-11-03T03:40:49Z",
      "run_id": "41550569320",
      "run_attempt": 1,
      "run_url": "https://github.com/lemonade-sdk/llama.cpp/actions/runs/41550569320",
      "event": "schedule",
      "build_result": "pass",
      "tests": { "passed": 14, "failed": 0, "warned": 0, "skipped": 0, "missing": 0, "not_applicable": 2, "total": 16 },
      "qualified": true,                        // this archive passed its own checks
      "version_qualified": false,               // its Windows counterpart has no record
      "published": false,                       // so ROCm 10.1 was withheld from b12062
      "overall": "pass",
      "tier0": "pass",
      "tier1": "pass",
      "blocked_by": [],
      "blocking_tests": [],
      "results": {
        "T0.1": "pass", "T0.2": "pass", "T0.3": "pass", "T0.4": "pass",
        "T0.5": "pass", "T0.6": "pass", "T0.7": "pass", "T0.8": "pass",
        "T0.10": "not_applicable", "T0.11": "not_applicable",   // dropped before the post-merge checks
        "T0.12": "pass",
        "T1.1": "pass", "T1.2": "pass", "T1.3": "pass", "T1.4": "pass", "T1.5": "pass"
      },
      "report_url": "reports/41550569320-ubuntu-rocm-10.1.json"
    },

    // ... the rows of earlier runs ...

    // An archive that passed and shipped in release b12003.
    {
      "build_id": "40977810710-win-rocm-10.1",
      "tag": "b12003",                          // llama.cpp build number = the release tag
      "upstream_sha": "36a0e2ffc0b352b774f8d71c43668b75b3061ef5",
      "artifact": "llama-b12003-bin-win-rocm-10.1-x64.zip",
      "os": "win",                              // win | ubuntu
      "rocm_version": "10.1",                   // currently 7.14 | 10.0 | 10.1
      "gfx_target": "gfx1151",                  // the one gfx target Tier 1 ran on; null when Tier 1 did not run
      "hardware_validated": true,               // Tier 1 ran on a real GPU (see tier1 for how it went)
      "gfx": { "requested": 20, "present": 20, "missing": [] },
      "generated_at": "2026-10-30T03:47:31Z",   // ISO-8601 UTC, when this row's record was written
      "run_id": "40977810710",
      "run_attempt": 1,                         // the attempt that published this row
      "run_url": "https://github.com/lemonade-sdk/llama.cpp/actions/runs/40977810710",
      "event": "schedule",                      // schedule | workflow_dispatch
      "build_result": "pass",                   // pass | fail | cancelled | skipped | unknown
      "tests": { "passed": 13, "failed": 0, "warned": 1, "skipped": 2, "missing": 0, "not_applicable": 0, "total": 16 },
      "qualified": true,                        // passed every gating check required of it: Tier 0, plus
                                                // Tier 1 if it ran. Not whether it shipped.
      "version_qualified": true,                // both archives of ROCm 10.1 exist and qualified
      "published": true,                        // this run attached it to release b12003
      "overall": "warn",                        // pass | warn | fail; fail exactly when qualified is false
      "tier0": "warn",                          // pass | warn | fail | missing
      "tier1": "pass",                          // pass | warn | fail | missing
      "blocked_by": [],                         // why it did not qualify, e.g. ["tier0=fail"]
      "blocking_tests": [],                     // the gating tests that failed, e.g. ["T0.3"]
      "results": {
        "T0.1": "warn", "T0.2": "pass", "T0.3": "pass", "T0.4": "pass",
        "T0.5": "pass", "T0.6": "skip", "T0.7": "skip", "T0.8": "pass",
        "T0.10": "pass", "T0.11": "pass", "T0.12": "pass",
        "T1.1": "pass", "T1.2": "pass", "T1.3": "pass", "T1.4": "pass", "T1.5": "pass"
      },
      "report_url": "reports/40977810710-win-rocm-10.1.json"
    }
  ]
}
```

### Field notes

- **`results`** has one entry per test: T0.1–T0.8, T0.10–T0.12 and T1.1–T1.5,
  as listed in [README.md](README.md). A test id added to the suite later
  appears here as soon as it runs, so the failure-frequency and heatmap views
  can be built from `index.json` alone. A retired test stops appearing in new
  rows and keeps its value in older ones: rows published before T0.9 was
  retired also carry T0.9. The values:
  - `pass`: the check ran and passed.
  - `fail`: the check ran and found a problem. It blocks the archive only if
    the test is gating; T1.4 and T1.5 never block.
  - `warn`: the check ran and found something worth recording that does not
    block, e.g. upstream test programs shipped in the archive (T0.1).
  - `skip`: the check ran but could not apply, e.g. T0.6 and T0.7 on Windows
    (Linux-only checks).
  - `missing`: the check should have run for this archive and did not. For
    Tier 1 this does not block the archive (see `qualified`), but it is still a
    gap: the archive was not tested on a GPU.
  - `not_applicable`: the check was never going to run for this archive.
    T0.10 and T0.11 check the final archive inside the release job, and only
    run on archives still in the release at that point. An archive dropped
    before then — because the other archive of its ROCm version, or the
    Windows CPU backend, was missing — or never built has them
    `not_applicable`.
- **`tests`** counts the values in `results`, so
  `passed + failed + warned + skipped + missing + not_applicable == total`
  always holds. Use `qualified`, not the counts, to decide pass or fail.
- **`qualified`** is the verdict for this one archive: Tier 0 ran and no gating
  Tier 0 check failed, and no gating Tier 1 check failed if Tier 1 ran. A
  `warn` does not block, and non-gating tests never affect it. It does not say
  whether the archive shipped (`published`) or ran on a GPU (`tier1`,
  `hardware_validated`).
- **`version_qualified`** is specific to this repository. It is `true` when
  both archives of this ROCm version — Windows **and** Linux — exist and
  qualified. A version ships only as a pair, so an archive can have
  `qualified: true` and `version_qualified: false` because its counterpart
  failed a check or never built. `null` means the run recorded no decision.
- **`published`** says whether this run attached this archive to the release
  `tag`. It can be `false` even when `version_qualified` is `true`: the Windows
  ROCm archive also needs the Windows CPU backend merged into it, and an upload
  can fail. Two runs built from the same upstream commit share a tag and a
  release; the later one replaces that release's files with its own, so
  `published` describes what this run shipped, not what the release holds
  today.
- **`overall`** is `fail` exactly when `qualified` is `false`. Otherwise it is
  the worst result of the gating tests that ran: `warn` if any of them warned,
  else `pass`.
- **`tier0` / `tier1`** are each tier's result: `pass`, `warn` or `fail` when
  the tier ran, and `missing` when it did not.
- **`tag`** is the llama.cpp build number this run pinned, which is also the
  release tag. Every row of a run has it, whether or not anything shipped.
- **`build_result`** is how the archive's build job (`ubuntu-24-rocm` or
  `windows-rocm`) ended: `pass`, `fail` (including a timeout), `cancelled`,
  `skipped`, or `unknown` when the run's job list could not be read. The three
  `windows-rocm` builds stop together: when one fails, GitHub cancels the
  others, so they show `cancelled`.
- **`run_attempt`** is the attempt of the run that published the row. It is the
  same on every row of a run.
- **`generated_at`** is when the row's record was written. A re-run writes it
  again, later than any run that started after it, so order runs by `run_id`
  (rows are already newest-run-first) rather than by `generated_at`.
- **`blocked_by`** gives the reasons the archive did not qualify. It is empty
  when it qualified. The values:
  - `tier0=fail`: a gating Tier 0 check failed;
  - `tier0 missing`: no Tier 0 report;
  - `tier1=fail`: a gating Tier 1 check failed;
  - on placeholder rows only: `build failed`, `build cancelled`, `build skipped`
    or `no report`.
- **`blocking_tests`** lists the gating tests that failed; empty when none did.
- **`gfx`** summarises the check that every gfx target was compiled into the
  GPU backend (T0.3): how many were requested, how many are present, and the
  names of any that are missing.
- **`gfx_target`** is the one gfx target Tier 1 actually ran on, and `null`
  when Tier 1 did not run. **`hardware_validated`** is `true` when Tier 1 ran on
  a real GPU, whether or not it then passed; `tier1` gives the outcome. Every
  other target in `gfx.requested` was only checked for presence, never run.
  Don't present those as tested.
- **`report_url`** is relative to the base URL; it holds every check's
  `status`, `gating`, `error` and `details`.

### Placeholder rows

An archive with no record at all gets a placeholder row. This happens when its
build failed, so there was nothing to check, or when no check reported on it.
In a placeholder row:

- every test is `missing` except T0.10 and T0.11, which are `not_applicable`;
  `tier0` and `tier1` are `missing`; `qualified`, `version_qualified` and
  `published` are `false`; and `overall` is `fail`;
- `blocked_by` follows `build_result`: `build failed`, `build cancelled`,
  `build skipped`, or `no report` when the build passed or its result is
  `unknown`;
- `gfx` is all zero and empty: there is no binary, and the target list lives in
  the build job;
- `gfx_target` is `null` and `hardware_validated` is `false`; `tag` and
  `artifact` are the ones the archive would have had.

The first row of the [`index.json`](#indexjson) example is a placeholder.

## `reports/<build_id>.json`

The `qualification-<archive>.json` record exactly as `aggregate.py` wrote it (`kind: "qualification_record"`):
- the `build` metadata,
- `overall`,
- `promoted`,
- `promotion.blocked_by`,
- raw `counts`,
- the `gfx` audit, and
- `tiers` with every test.

The same content is attached to the release. Its own `build.build_id` (`<artifact>@<run_id>`) differs from the feed's `build_id`; use the feed's.

For a placeholder row it is a placeholder (`kind: "qualification_placeholder"`) with the same top-level fields:
- `build` metadata,
- `overall: "fail"`,
- `promoted: false`,
- `promotion.blocked_by`,
- `build_result` (in the feed's vocabulary),
- the empty `gfx` block,
- zero `counts` and
- empty `tiers`.

## `latest/<os>-rocm-<ver>.json`

The newest full record for one archive (for example `latest/win-rocm-10.1.json`),
for status tiles that want the current state without scanning the index. A
placeholder is written here too, so a tile shows a failing build instead of the
last good one. It is replaced only by a record from the same run or a newer one
(a higher `run_id`), so re-running an old run cannot move it backwards.

## Differences from lemonade-sdk/llamacpp-rocm

That feed has one archive per GPU family, with ROCm bundled inside; this one
has one archive per ROCm version, with ROCm installed separately by the user.
Everything else in the row means the same.

| Field | this repository | lemonade-sdk/llamacpp-rocm |
|---|---|---|
| `schema_version` | `2` | `1` (same row shape) |
| `build_id` | `<run_id>-<os>-rocm-<rocm_version>` | `<run_id>-<os>-rocm-<gfx_family>` |
| rows per run | 6 | 14 |
| `tag` | llama.cpp build number; set on every row | that repository's release tag; `null` unless `published` |
| `artifact` | always the release name | the untagged CI name unless `published` |
| `os` | `win` \| `ubuntu` | `windows` \| `ubuntu` |
| `rocm_version` | `7.14` \| `10.0` \| `10.1`, one per archive | full nightly, e.g. `10.3.0a20261102`, same for the whole run |
| `gfx_family` | — | the GPU family the archive serves |
| `version_qualified` | both archives of a ROCm version exist and qualified | not present (a release there is all or nothing) |
| `tier1` | Tier 1 runs on every archive; `missing` does not block | Tier 1 runs only on gfx1151 (`not_applicable` elsewhere); `missing` blocks |
| `not_applicable` | T0.10, T0.11 on an archive dropped before the release checks | T1.x outside gfx1151; T0.11 on a run with no release |
| `blocked_by` | no `tier1 missing`, no `T0.11 missing` | also `tier1 missing`, `T0.11 missing` |
| `results` keys | T0.1–T0.8, T0.10–T0.12, T1.1–T1.5 (older rows also T0.9) | T0.1–T0.3, T0.5–T0.9, T0.11, T0.13, T1.1–T1.6 |
| placeholder `gfx` | all zero | requested targets, all in `missing` |
| `latest/` file | `<os>-rocm-<rocm_version>.json` | `<os>-rocm-<gfx_family>.json` |

## Versioning and growth

Bump `schema_version` on any breaking change to `index.json` or the row shape.
Adding a field is not breaking. The dashboard should read `schema_version` and
show a notice, rather than break, on a version it does not know.

Version 2 replaced version 1 because the meaning of existing fields changed:
`version_qualified` now also counts an archive that never built, `overall` is
`fail` whenever `qualified` is `false`, and `missing` now only means "should
have run". It also added `published`, `build_result`, `tier0`, `tier1`, the
`not_applicable` result, and placeholder rows, and orders runs by `run_id`. A
version 1 feed is not mixed with version 2 rows: the first version 2 publish
starts a fresh `index.json`, `reports/` and `latest/`. The version 1 files stay
in the branch's git history.

`index.json` grows by six rows per run, roughly a dozen a day. That is
comfortable to poll for a long time, especially compressed. If it ever is not,
older rows will move into per-month files under a new `schema_version`.
