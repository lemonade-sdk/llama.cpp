# ROCm build qualification

Verification of the ROCm archives this repository publishes. Driven from
`.github/workflows/release.yml`, but every script here can be run by hand
against a downloaded release asset, which is the easiest way to understand what
any of them does.

## Why this exists

A broken ROCm build does not look broken.

llama.cpp loads its GPU support as a separate plug-in at run time, and treats a
plug-in that fails to load as a non-event: it writes a line to the log, skips
it, and carries on using the CPU. So a build whose GPU support is completely
broken still starts, still answers every request, and still returns correct
answers. It is simply ten times slower, with no error anywhere.

The same quietness applies further up. If the compiler produces GPU code for
only some of the gfx targets it was asked to cover, nothing reports an error — the
output file is still created, with the expected name and a plausible size.

Both of those ship green unless something deliberately looks inside the
archive. That is what this suite is for.

## Tiers

| Tier | Runs on | Blocks a release? |
|---|---|---|
| Tier 0 — static checks | standard GitHub runners, including on PRs | yes, affected ROCm version only |
| Tier 1 — GPU inference | self-hosted machines with a real AMD GPU | yes, affected ROCm version only |

Each ROCm version is treated as an independent product. A consumer downloads
exactly one of them, so a failure in one version removes only that version's
Windows and Linux pair; the other versions, and every non-ROCm backend, publish
as normal. The release job then exits non-zero so the gap is noticed.

### Tier 0 — `tier0_static.py`, `check_post_merge.py`

Runs against the archive as it will be published, not against the build
directory, because most of what it looks for only exists once the files have
been packaged.

| ID | Check | Blocks release |
|---|---|---|
| T0.1 | the binaries a user needs are all present | yes |
| T0.2 | the GPU backend exists on both operating systems and is big enough to hold GPU code | yes |
| T0.3 | the GPU backend really contains code for every gfx target the build asked for | yes |
| T0.4 | the CPU backend variants are present | yes |
| T0.5 | every library the archive needs is either inside it, part of the OS, or a ROCm library | yes |
| T0.6 | the archive still runs on the oldest Linux release we support | yes |
| T0.7 | no build-machine paths left inside the binaries | yes |
| T0.8 | `llama-server` still accepts the flags it is launched with | yes |
| T0.9 | the folder inside the archive matches the archive's own name | no |
| T0.10 | the Windows CPU backend merge landed | yes |
| T0.11 | the finished archive still opens and is complete | yes |
| T0.12 | the ROCm libraries the archive needs exist in the ROCm a user installs | yes |

T0.10 and T0.11 come from `check_post_merge.py`, which runs later than the
rest: the release job merges the Windows CPU backend into the Windows ROCm
archives, and those two questions can only be answered after that has happened.

### Tier 1 — `tier1_gpu.py`

Starts `llama-server` with the flags the downstream consumer uses, loads a
small model, and talks to it over HTTP.

| ID | Check | Blocks release |
|---|---|---|
| T1.1 | the server starts and reports itself healthy | yes |
| T1.2 | the GPU actually did the work | yes |
| T1.3 | the reply is non-empty, valid text of a sensible length, with a proper stop reason | yes |
| T1.4 | two identical requests with randomness disabled give identical replies | yes |
| T1.5 | the model closed its reasoning block | no |

**T1.2 is the one that matters.** Every other check in this tier passes
perfectly well when the GPU was never used at all.

T1.4 is worth a word too: with randomness switched off, the same question must
produce the same answer every time. When it does not, something underneath is
reading uninitialised memory or producing invalid numbers — a problem that
otherwise shows up as occasional nonsense rather than a clean failure.

## How some of this works

### Counting the gfx targets inside the backend

`binfmt.py` searches the GPU backend file for the short text labels the
compiler writes next to each gfx target's code, rather than asking an AMD tool to
list them.

The reason is that the GPU code lives in a named block inside the file, and
Windows shortens those names to eight characters once a program is linked — so
a block Linux calls `.hip_fatbin` is called `.hip_fat` there. Any tool asked to
find the block by name comes back empty on Windows. Searching the file's bytes
does not care what the block is called, so one implementation covers both
operating systems and needs no AMD tooling installed, which in turn means the
check can run on an ordinary runner against the archive we actually ship.

Two deliberate behaviours:

- Finding **no** gfx targets at all is a failure, not a pass. A check that can
  silently come up empty is worse than no check.
- The number of entries per target is reported, not just which targets are
  present. The compiler writes one entry per source file, so in a healthy build
  every target has the same count; a target with a lower count was built into
  only part of the project, which a present/absent check would miss.

### Which ROCm the checks verify against

The archives do not contain ROCm. A user installs it separately, and llama.cpp
loads it at run time. So "are this archive's libraries available?" depends
entirely on what the user installed.

Checking against the ROCm used to *build* the archive would always pass — those
are the exact files it was compiled against. `install_rocm.py` therefore
installs the components an end user gets (the kernels for their GPU) rather
than the components a build needs (compilers), and T0.12 and Tier 1 both verify
against that. Everything goes into a throwaway virtual environment under the
runner's temporary directory; nothing is installed system-wide.

### Which Linux versions the archive runs on

Every Linux program records the newest version of the system C library it
relied on, and refuses to start on a machine with an older one. Newer Linux
releases ship newer versions of that library, so building on a newer Linux
silently narrows who can run the result — nothing warns about it at build time.

T0.6 reads that number out of the archive and compares it against
`SUPPORTED_GLIBC` in `tier0_static.py`, which is set to what the oldest Linux
release we support provides. The check passes today and would fail if the build
job's runner image moved to a newer Linux without that decision being made
deliberately.

## Running by hand

```bash
# Check a downloaded release asset
python3 .github/qualify-rocm/tier0_static.py \
    --archive llama-bNNNNN-bin-ubuntu-rocm-10.1-x64.tar.gz \
    --os ubuntu --rocm-version 10.1 \
    --gpu-targets "gfx900;gfx906;...;gfx1201" \
    --output tier0.json

# Combine reports into one record per archive and decide what may ship
python3 .github/qualify-rocm/aggregate.py \
    --fragments-dir ./fragments --tag bNNNNN \
    --records-dir ./qualification --fail-on-no-promote
```

Each run writes a `qualification-<archive>.json` record that is attached to the
release. It lists which gfx targets the archive contains code for, which of them
were actually tested on hardware, and the result of every check — so questions
like "was this gfx target ever verified in that build?" stay answerable long after
the CI logs have expired.

## Out of scope

- **Performance.** Tracked downstream against these same artifacts; a second
  set of numbers here would split the history.
- **Answer quality.** T1.3 and T1.4 cover "the output is well-formed and
  reproducible", which is what catches broken GPU kernels. Judging whether an
  answer is *good* is a separate problem.
