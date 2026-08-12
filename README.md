# Bayesian Opponent Modeling in Poker

Heads-up Hold'em research, multi-player tournament bots, and competition preparation, alongside a Rust Monte Carlo counterfactual regret minimization (MCCFR) solver.

Bayesian Opponent Modeling in Poker includes **Citadel Poker**, the competition bot previously maintained as Shove. Both Python packages use the same pinned environment; their game rules, policies, and opponent state stay separate. The Rust solver trains an abstract strategy blueprint and is independent of both Python bots.

| Mode | Entry point | Guide |
| --- | --- | --- |
| Heads-up research and local table | `python -m headsup` | [Architecture and connections](docs/architecture.md) |
| Citadel Poker: tournament simulation and competition preparation | `python -m competition` | [Competition guide](docs/competition.md) |
| Rust abstraction and strategy training | `cargo run` with the relevant features | [Rust pipeline](#rust-pipeline) |

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12.14 and dependencies are pinned. Linux and macOS are supported.

```sh
git clone https://github.com/Smokeybear10/bayesian-opponent-modeling-in-poker.git
cd bayesian-opponent-modeling-in-poker
uv sync --locked
uv run --locked python -m headsup benchmark --pairs 16 --samples 128 --output runs/first-benchmark
```

Run the local play-money table:

```sh
uv run --locked python -m headsup serve --port 8788 --hands 100 --output runs/local-table
```

Open http://127.0.0.1:8788. Load the appropriate generated `player-N.json` credential file for each seat. [Connection details](docs/architecture.md#private-connections) cover bot clients, recovery, and replay.

Rehearse the competition adapter and checked submission packaging:

```sh
uv run --locked python -m competition.eventday rehearse --deterministic --output day/rehearsal
uv run --locked python -m competition.session start competition/rehearsal --output day/session
uv run --locked python -m competition.session status day/session
```

These use a synthetic starter kit, without an external account or submission. The [competition guide](docs/competition.md) covers tournament comparisons and adapting an official starter. Each run requires a new output directory.

## What It Does

**Complete games.** PokerKit owns Python betting rules and gross settlement. The engine handles explicit simulated rake and uncalled refunds. Policies receive their own cards and public information. Saved hands support settlement replay and exact policy replay with the recorded environment.

**Opponent memory.** SQLite stores verified completed-hand fold/call/raise responses, separated by room, hero ID, opponent ID, and blinds. Names and seat changes cannot merge players. Default targeted adjustments require 24 matching individual responses. Lifetime memory, recent memory, and optional frozen population priors are supported.

**Precision candidate.** The opt-in `precision` policy uses the range baseline's betting rules and spends more samples on equity estimates near decision thresholds. An independent refinement batch reduces sampling noise without changing the action-mixing random stream. Total sampling is bounded; river equity remains exact within the heuristic range. This does not resolve range-model error or make the policy an equilibrium strategy.

**River decisions.** The optional river policy extends the adaptive baseline with exact card enumeration within heuristic ranges, bounded bet sizes, modeled responses, and rake-aware terminal values. Its historical benchmark did not establish improvement, so it remains opt-in.

**Rust strategy research.** Card isomorphisms, hierarchical clustering, and optimal transport reduce the card space. MCCFR samples abstract game trees and accumulates strategy weights. The native trainer and PostgreSQL query API are available; WebAssembly bindings are opt-in through the `wasm` feature.

**Private integration.** The local table supports authenticated clients and durable action receipts. Experimental PokerNow modules validate visible state and confirm actions through a supplied browser-control interface. No ClubGG adapter is included.

**External AI benchmark.** The Slumbot connector plays a bounded anonymous match through its documented research API at 200BB, checks each action receipt, and verifies settlement when the necessary cards are available. It stops on ambiguous requests instead of retrying them. Session tokens stay in memory and are excluded from saved journals.

**Citadel Poker.** The competition bot includes multi-player Hold'em simulation, payout-aware policies, opponent leagues, SDK adapters, starter inspection, runtime checks, and verified candidate packaging. Its prepared strategy assumes standard no-limit Hold'em; official rules and interfaces still need to be supplied and validated for an event.

## Research and Results

The [research report](research/REPORT.md) includes methods, failure records, negative findings, and reproduction commands. The 20,016-hand Python replication measured **+4.13 BB/100** for adaptive memory versus the same baseline without memory, with a **95% interval of +0.57 to +7.69** across three independent training seeds. This supports improvement in the tested synthetic pool, with limited precision from three seeds. It does not establish human-player profitability or validate the Rust solver.

Compact evidence is versioned under `research/results/`. Raw journals, credentials, and real opponent databases remain local. Historical source hashes describe the code used for those experiments and are not rewritten during cleanup.

Shove's saved tournament reports are preserved under [`reports/`](reports/). Its original [`eventday-kit.json`](eventday-kit.json) and [`toolkit archive`](artifacts/event-day-kit.zip) describe an older preparation snapshot, not the current complete source. Use `competition/` for new work. Those historical files retain their original repository names, hashes, and test counts; they do not validate the heads-up bot or certify an official competition entry.

```sh
uv run --locked python scripts/verify_research.py research/results/replication-20261107
uv run --locked python -m headsup opponent-profiles
```

Run an optional external AI test with a new output directory:

```sh
uv run --locked python -m headsup.slumbot --hands 20 --policy precision --samples 128 --output runs/slumbot-test
```

This sends decisions to [Slumbot's public API](https://slumbot.com/sample_api.py), using simulated chips. The match uses sequential server deals, with no paired control or online learning. The `precision` candidate and expanded historical population prior remain experimental; see the research report for their separate validation results.

## Checks

Node.js 24 runs the browser-adapter tests. Rust 1.90.0 is pinned in `rust-toolchain.toml`.

```sh
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m unittest discover -s competition -t . -p 'test*.py' -v
node --test tests/*.test.mjs
cargo test --locked --lib
cargo test --locked --lib --features shortdeck
```

Python tests cover both game engines, private-card boundaries, identity isolation, persistence, research integrity, SDK conversion, and submission packaging. The dependency-free competition integration tests also run on Python 3.10–3.12. Rust tests cover card evaluation, settlement, transport, and regret matching. Exhaustive and dataset-dependent Rust fixtures marked `#[ignore]` are excluded from normal runs.

## Rust Pipeline

The native trainer uses two players with 100-chip stacks and 1/2 blinds, or 50BB. Python defaults to 100BB. The reduced Rust configuration uses 36 cards with existing Hold'em rankings; it is not a complete standard short-deck ruleset.

```sh
cargo run --release --locked --features "cluster demo shortdeck"
cargo run --release --locked --features "trainer demo shortdeck"
```

Clustering must finish before training. Even `demo` enumerates card observations and can be expensive. Keep the same feature settings across stages; use separate checkouts for different deck configurations because outputs share `pgcopy/demo/` or `pgcopy/`.

With PostgreSQL running, publish a trained dataset to a fresh database:

```sh
createdb bayesian_poker_demo
export DB_URL='postgresql://localhost/bayesian_poker_demo'
cargo run --release --locked --features "publish demo shortdeck"
cargo run --release --locked --features "analyze demo shortdeck"
```

The API defaults to `http://127.0.0.1:3002`. `/health` checks abstraction queries; `POST /blueprint` requires a trained and published strategy. See the [request types](src/analysis/request.rs).

The Rust implementation has unresolved correctness limits: [biased card drawing](src/cards/deck.rs), [river equity that excludes ties](src/cards/observation.rs), and [resume loading that resets the iteration count](src/mccfr/nlhe/profile.rs). There is no committed trained Rust blueprint or exploitability estimate. These limits must be resolved before interpreting its playing strength.

## Project Structure

```text
headsup/              # Python game engine, policies, memory, and private arena
competition/          # Tournament bot, event tooling, SDK rehearsal, and tests
src/                  # Rust cards, abstraction, MCCFR, and query API
tests/                # Python and JavaScript regression tests
benches/              # Rust Criterion benchmarks
scripts/              # Experiment runner and independent verification
research/             # Methods and compact experiment evidence
reports/              # Historical tournament evaluation reports
artifacts/            # Original competition toolkit snapshot
docs/architecture.md  # Runtime boundaries and connection protocols
docs/competition.md   # Tournament and event preparation commands
.github/workflows/    # Research, competition, package, and Rust checks
```

## References

1. [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).
2. [Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf).
3. [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040).

These describe related methods, not measured results for Bayesian Opponent Modeling in Poker.

Maintained by Thomas Ou
