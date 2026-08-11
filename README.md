# River | Poker Strategy Research

Heads-up Hold'em simulation, persistent opponent models, and reproducible evaluation, alongside a Rust card-abstraction and Monte Carlo counterfactual regret minimization (MCCFR) solver.

The Python bot plays complete hands using heuristic policies and learned public responses. The Rust solver trains an abstract strategy blueprint. They share this repository; the Python policy currently runs independently of the Rust blueprint.

## Quick Start

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/). Python 3.12.14 and dependencies are pinned. Linux and macOS are supported.

```sh
git clone https://github.com/Smokeybear10/103.RSCH.river.git
cd 103.RSCH.river
uv sync --locked
uv run --locked python -m headsup benchmark --pairs 16 --samples 128 --output runs/first-benchmark
```

Run the local play-money table:

```sh
uv run --locked python -m headsup serve --port 8788 --hands 100 --output runs/local-table
```

Open http://127.0.0.1:8788. Load the appropriate generated `player-N.json` credential file for each seat. [Connection details](docs/architecture.md#private-connections) cover bot clients, recovery, and replay.

## What It Does

**Complete games.** PokerKit owns Python betting rules and gross settlement. The engine handles explicit simulated rake and uncalled refunds. Policies receive their own cards and public information. Saved hands support settlement replay and exact policy replay with the recorded environment.

**Opponent memory.** SQLite stores verified completed-hand fold/call/raise responses, separated by room, hero ID, opponent ID, and blinds. Names and seat changes cannot merge players. Default targeted adjustments require 24 matching individual responses. Lifetime memory, recent memory, and optional frozen population priors are supported.

**River decisions.** The optional river policy extends the adaptive baseline with exact card enumeration within heuristic ranges, bounded bet sizes, modeled responses, and rake-aware terminal values. Its historical benchmark did not establish improvement, so it remains opt-in.

**Rust strategy research.** Card isomorphisms, hierarchical clustering, and optimal transport reduce the card space. MCCFR samples abstract game trees and accumulates strategy weights. The native trainer and PostgreSQL query API are available; WebAssembly bindings are opt-in through the `wasm` feature.

**Private integration.** The local table supports authenticated clients and durable action receipts. Experimental PokerNow modules validate visible state and confirm actions through a supplied browser-control interface. No ClubGG adapter is included.

## Research and Results

The [research report](research/REPORT.md) includes methods, failure records, negative findings, and reproduction commands. The 20,016-hand Python replication measured **+4.13 BB/100** for adaptive memory versus the same baseline without memory, with a **95% interval of +0.57 to +7.69** across three independent training seeds. This supports improvement in the tested synthetic pool, with limited precision from three seeds. It does not establish human-player profitability or validate the Rust solver.

Compact evidence is versioned under `research/results/`. Raw journals, credentials, and real opponent databases remain local. Historical source hashes describe the code used for those experiments and are not rewritten during cleanup.

```sh
uv run --locked python scripts/verify_research.py research/results/replication-20261107
uv run --locked python -m headsup opponent-profiles
```

## Checks

Node.js 24 runs the browser-adapter tests. Rust 1.90.0 is pinned in `rust-toolchain.toml`.

```sh
uv run --locked python -m unittest discover -s tests -v
node --test tests/*.test.mjs
cargo test --locked --lib
cargo test --locked --lib --features shortdeck
```

Python tests cover engine accounting, private-card boundaries, identity isolation, persistence, action confirmation, and research integrity. Rust tests cover card evaluation, settlement, transport, and regret matching. Exhaustive and dataset-dependent Rust fixtures marked `#[ignore]` are excluded from normal runs.

## Rust Pipeline

The native trainer uses two players with 100-chip stacks and 1/2 blinds, or 50BB. Python defaults to 100BB. The reduced Rust configuration uses 36 cards with existing Hold'em rankings; it is not a complete standard short-deck ruleset.

```sh
cargo run --release --locked --features "cluster demo shortdeck"
cargo run --release --locked --features "trainer demo shortdeck"
```

Clustering must finish before training. Even `demo` enumerates card observations and can be expensive. Keep the same feature settings across stages; use separate checkouts for different deck configurations because outputs share `pgcopy/demo/` or `pgcopy/`.

With PostgreSQL running, publish a trained dataset to a fresh database:

```sh
createdb river_demo
export DB_URL='postgresql://localhost/river_demo'
cargo run --release --locked --features "publish demo shortdeck"
cargo run --release --locked --features "analyze demo shortdeck"
```

The API defaults to `http://127.0.0.1:3002`. `/health` checks abstraction queries; `POST /blueprint` requires a trained and published strategy. See the [request types](src/analysis/request.rs).

The Rust implementation has unresolved correctness limits: [biased card drawing](src/cards/deck.rs), [river equity that excludes ties](src/cards/observation.rs), and [resume loading that resets the iteration count](src/mccfr/nlhe/profile.rs). There is no committed trained Rust blueprint or exploitability estimate. These limits must be resolved before interpreting its playing strength.

## Project Structure

```text
headsup/              # Python game engine, policies, memory, and private arena
src/                  # Rust cards, abstraction, MCCFR, and query API
tests/                # Python and JavaScript regression tests
benches/              # Rust Criterion benchmarks
scripts/              # Experiment runner and independent verification
research/             # Methods and compact experiment evidence
docs/architecture.md  # Runtime boundaries and connection protocols
.github/workflows/    # Python and Rust checks
```

## References

1. [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper_files/paper/2007/hash/08d98638c6fcd194a4b1e6992063e944-Abstract.html).
2. [Potential-Aware Imperfect-Recall Abstraction with Earth Mover's Distance](https://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf).
3. [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040).

These describe related methods, not measured results for River.

Maintained by Thomas Ou
