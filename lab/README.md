# River Heads-Up Lab | Opponent Modeling

A reproducible heads-up no-limit Hold'em research environment with persistent opponent models, a bounded river policy, and paired strategy evaluation.

This is the Python lab inside [River](../README.md). It runs its own heuristic policies and does not currently consume the Rust MCCFR solver's blueprint.

The main question is whether learning a particular opponent's public betting responses improves a fixed poker baseline. A second experiment extends that baseline with explicit river action values. Results include uncertainty, failed promotion gates, and the conditions under which they were measured. This is an empirical research codebase; its policies are heuristics, and a profitable human-player edge has not been established.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12.14 and package versions are pinned. Node.js 24 is used for adapter tests. Linux and macOS are supported; persistent memory uses POSIX file locks.

```sh
git clone https://github.com/Smokeybear10/103.RSCH.river.git
cd 103.RSCH.river/lab
uv sync --locked
uv run --locked python -m headsup benchmark --pairs 16 --samples 128 --output runs/first-benchmark
```

Run the automated checks:

```sh
uv run --locked python -m unittest discover -s tests -v
node --test tests/*.test.mjs
```

To open the local play-money arena:

```sh
uv run --locked python -m headsup serve --port 8788 --hands 100 --output runs/local-table
```

Open http://127.0.0.1:8788. The host creates separate private seat credential files; load only the appropriate seat's file in each browser. See [private connections](ARCHITECTURE.md#private-connections) for bot clients and recovery.

## What It Does

**Whole-hand engine and evaluation.** PokerKit owns legal betting and settlement. The wrapper applies explicit simulated rake, refunds uncalled chips, and exposes immutable own-card observations. Evaluation matches deals and policy seeds across both seats. Saved hands support action replay and, when the recorded environment is available, exact policy replay.

**Player-specific memory.** SQLite stores public fold/call/raise responses from verified completed hands. Profiles separate room, hero ID, opponent ID, and blinds; display names and changing seats cannot merge players. Context includes street, position, betting stage, and price bucket. At least 24 matching individual responses are required before targeted adaptive adjustments. Lifetime and bounded recent memory are available; a frozen population prior is optional.

**River research.** `RiverPolicy` extends `AdaptivePolicy` in the same bot. It uses exact card enumeration within a heuristic range, a finite set of bet sizes, modeled fold/call/raise responses, rake-aware terminal values, and bounded call/fold continuation. It is not a full-game equilibrium solver. The policy remains opt-in because the recorded river experiment failed its improvement gate.

**Private integration.** An authenticated local table supports independent bot clients, durable receipts, and restart recovery. An experimental PokerNow adapter reads visible state through a supplied browser-control interface and stops on unexpected identities, rules, deadlines, or unconfirmed actions. It requires a supervised private play-chip setup; the repository does not supply an unattended casino connector or ClubGG adapter.

## Research and Results

Read the [research report](research/REPORT.md) for the prespecified configurations, independent units, uncertainty intervals, historical comparisons, and limitations. Compact manifests and paired scores are versioned under `research/results/`; raw journals, credentials, and actual opponent databases stay local.

The latest 20,016-hand replication measured **+4.13 BB/100** for adaptive memory versus the same baseline without memory, with a **95% interval of +0.57 to +7.69** across three independent training seeds. This is evidence for the tested synthetic opponent pool, with limited precision from only three seeds. The separate historical river comparison failed its promotion gate; these learning results do not validate that river extension.

Run a small mechanical check of the research pipeline:

```sh
uv run --locked python scripts/run_research.py \
  --output runs/research-smoke --export runs/research-smoke-export \
  --seeds 20261107000 --warmup 8 --pairs 4 --samples 32 --jobs 2
```

The smoke run checks execution and artifacts; it is too small to estimate playing strength. The report contains the full replication command. The experiment runner freezes source and profiles, keeps training and evaluation deals disjoint, and retains failures instead of silently substituting actions or selecting favorable seeds.

Inspect saved individual behavior:

```sh
uv run --locked python -m headsup opponent-profiles
```

This reports observed rates, sample sizes, uncertainty, and adaptation readiness. An unseen player starts without individual observations. Counts for calls and raises are separate; the default adaptive policy primarily uses fold probability for bounded bluff adjustments. The optional river policy additionally models raise responses.

## Tech Stack

| Layer | Tools |
|---|---|
| Rules and settlement | Python 3.12, PokerKit |
| Card evaluation and statistics | eval7, SciPy |
| Private table and protocol | FastAPI, Pydantic, HTTPX |
| Durable memory and receipts | SQLite, POSIX file locks |
| Tests and CI | unittest, Node.js test runner, GitHub Actions |
| Environment | uv, pinned lockfile |

## Project Structure

```text
headsup/
  engine.py               # Legal hands and explicit rake settlement
  policies.py             # Fixed, adaptive, and river policies
  river.py                # Bounded river action-value model
  learning.py             # Contextual individual and population estimates
  opponent_memory.py      # Persistent player identity and behavior reports
  training.py             # Frozen-profile paired comparison
  river_evaluation.py     # River experiments and promotion gates
  table.py                # Authenticated private play-money host
  connection.py           # Recoverable local bot client
  pokernow_*.mjs          # Experimental private-table adapter
tests/                    # Deterministic regression and integration tests
scripts/                  # Reproducible research and artifact verification
research/                 # Methods, compact evidence, and findings
../.github/workflows/lab.yml  # Tests and package build
```

[Architecture](ARCHITECTURE.md) documents the information boundaries, memory lifecycle, and connection protocols. [Design notes](DESIGN.md) preserve the initial scope and subsequent changes.

Built by Thomas Ou
