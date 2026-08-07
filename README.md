# River | Poker Intelligence

**Live demo: [river-002.vercel.app](https://river-002.vercel.app)**

> The landing page and write-up are live. The Hand Explorer and Strategy viewer query the Rust solver API, which isn't hosted — run it locally with the Quick Start below to use them.

A Rust solver and Next.js frontend for No-Limit Texas Hold'em. Implements the Pluribus abstraction and blueprint-training stack — hierarchical clustering over equity distributions and external-sampling MCCFR. Depth-limited subgame search, the other half of Pluribus, is not built yet.

## Quick Start

```bash
# Solver (Rust)
cargo build
cargo run

# Frontend (Next.js)
cd web
npm install
npm run dev          # localhost:2002
```

The frontend expects the Rust API at `localhost:3002` (configurable via `NEXT_PUBLIC_API_URL`).

## What It Does

Trains a game-theoretic poker solver from scratch. Collapses the ~10¹⁶⁰-node NLHE game tree by clustering 3.1T strategically-equivalent situations into 542 buckets using Earth Mover's Distance over equity distributions, then runs external-sampling MCCFR self-play to produce a blueprint Nash strategy. The result is queryable in real time through an Actix Web API and an interactive explorer.

## Architecture

![River Architecture](docs/architecture/r1ver-architecture.png)

The four streets of River: **Abstract** → **Solve** → **Publish** → **Serve**. Full interactive doc at [`docs/architecture/index.html`](docs/architecture/index.html).

## Tech Stack

| Layer    | Tools                                       |
|----------|---------------------------------------------|
| Solver   | Rust, Rayon, Petgraph, Tokio                |
| API      | Actix Web, PostgreSQL, tokio-postgres       |
| Frontend | Next.js 16, React 19, Tailwind 4, GSAP      |
| CI       | GitHub Actions                              |

## Project Structure

```
R1VER/
├── src/
│   ├── cards/          Hand evaluation, equity, isomorphisms, iterators
│   ├── clustering/     K-means, EMD, Sinkhorn, histogram abstraction
│   ├── transport/      Optimal transport, Wasserstein distance
│   ├── gameplay/       Game engine, actions, settlements, showdowns
│   ├── mccfr/          Monte Carlo CFR solver, blueprint convergence
│   ├── save/           Disk persistence, Postgres binary format
│   ├── analysis/       API server, CLI, SQL queries
│   ├── players/        Human player interface
│   └── search/         Real-time subgame solving (in progress)
├── web/
│   └── app/
│       ├── components/ Landing page sections, card picker, charts
│       ├── lib/        API client, card utilities
│       ├── explorer/   Hand explorer — equity, clusters, neighbors
│       └── strategy/   Strategy viewer — blueprint query interface
├── docs/architecture/  System architecture (SVG, JSX, PNG)
├── pgcopy/             Pre-computed Postgres binary data
├── benches/            Criterion benchmarks
└── Cargo.toml
```

## Training Pipeline

1. **Abstraction** — Exhaustively iterate 3.1T isomorphic situations per street, project equity distributions, cluster with hierarchical k-means.
2. **Metrics** — Compute Earth Mover's Distance between all cluster pairs via Sinkhorn optimal transport.
3. **Solve** — External-sampling MCCFR with linear strategy weighting and regret-based pruning.
4. **Search** — Depth-limited subgame solving with the blueprint as prior (in progress).

### Data Sizes

| Street  | Abstraction | Metric |
|---------|-------------|--------|
| Preflop | 4 KB        | 301 KB |
| Flop    | 32 MB       | 175 KB |
| Turn    | 347 MB      | 175 KB |
| River   | 3.02 GB     | —      |

## Modules

**cards** — Nanosecond 7-card evaluator via lazy bitwise operations. Faster than Cactus Kev. Exact equity enumeration, Monte Carlo simulation, full isomorphism iteration, short-deck variant.

**clustering** — Plays out every possible situation respecting suit/rank symmetries. Hierarchical k-means over distribution space with Earth Mover's Distance. Sinkhorn-regularized optimal transport for efficient distance computation.

**transport** — Wasserstein distance via greedy coupling and Greenkhorn/Sinkhorn. Supports arbitrary distributions over joint metric spaces.

**gameplay** — Complete NLHE engine: side pots, all-ins, multi-way ties, configurable payouts. Generic Node/Edge/Tree types and a pluggable Decider trait.

**mccfr** — External-sampling MCCFR with dynamic tree construction, linear strategy weighting, discount schemes. Validated on Rock-Paper-Scissors before scaling to full NLHE.

**analysis** — Actix Web API backed by PostgreSQL. Streams Postgres binary files into indexed tables for sub-millisecond lookups across abstractions, metrics, and blueprint strategies.

## References

1. Superhuman AI for Multiplayer Poker (2019) — [Science](https://science.sciencemag.org/content/early/2019/07/10/science.aay2400)
2. Potential-Aware Imperfect-Recall Abstraction with EMD (2014) — [AAAI](http://www.cs.cmu.edu/~sandholm/potential-aware_imperfect-recall.aaai14.pdf)
3. Regret Minimization in Games with Incomplete Information (2007) — [NIPS](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information)
4. A Fast and Optimal Hand Isomorphism Algorithm (2013) — [AAAI](https://www.cs.cmu.edu/~waugh/publications/isomorphism13.pdf)
5. Near-linear Time Approximation for Optimal Transport via Sinkhorn (2018) — [NIPS](https://arxiv.org/abs/1705.09634)
6. Solving Imperfect-Information Games via Discounted Regret Minimization (2019) — [AAAI](https://arxiv.org/pdf/1809.04040.pdf)
7. Action Translation in Extensive-Form Games (2013) — [IJCAI](http://www.cs.cmu.edu/~sandholm/reverse%20mapping.ijcai13.pdf)
8. Discretization of Continuous Action Spaces (2015) — [AAMAS](http://www.cs.cmu.edu/~sandholm/discretization.aamas15.fromACM.pdf)
9. Regret-Based Pruning in Extensive-Form Games (2015) — [NIPS](http://www.cs.cmu.edu/~sandholm/regret-basedPruning.nips15.withAppendix.pdf)
10. Depth-Limited Solving for Imperfect-Information Games (2018) — [NeurIPS](https://arxiv.org/pdf/1805.08195.pdf)
11. Reduced Space and Faster Convergence via Pruning (2017) — [ICML](http://www.cs.cmu.edu/~sandholm/reducedSpace.icml17.pdf)
12. Safe and Nested Subgame Solving (2017) — [NIPS](https://www.cs.cmu.edu/~noamb/papers/17-NIPS-Safe.pdf)

---

Built by Thomas Ou
