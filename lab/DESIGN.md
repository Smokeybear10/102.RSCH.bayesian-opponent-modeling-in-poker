# Heads-Up Poker Lab

Initial design: 2026-11-05. Updated for the public research release: 2026-11-07.

This project is separate from the competition poker bot. The initial terminal-first scope has expanded to persistent opponent memory, a local browser arena, and an optional river experiment.

## Problem

Build a standalone heads-up no-limit Hold'em AI and local test arena. The user chose this scope after considering online platform integration and offline hand review. Their longer-term interest is financial performance; this milestone must measure playing strength honestly and must not imply that it earns money.

## Premises

1. Independent project, source tree, dependencies, output records, and strategy configuration. No imports from or edits to the competition project.
2. First ruleset is two-player standard no-limit Texas Hold'em, 100 big blinds per player, no antes, fresh stacks per hand, and alternating button through paired evaluation. Stack depth is configurable.
3. The first version was simulation only. Version 0.4 adds a private play-money table and network client. Experimental PokerNow modules now parse visible state and confirm actions through a supplied browser-control interface. They require supervised private play-chip testing; the project does not provide a casino connection or ClubGG adapter.
4. A functioning baseline is not a solved HUNL strategy. All policy descriptions and reports must state their actual approximation.
5. Performance means net chip return after explicit simulated rake, with uncertainty and opponent scope. Passing software checks does not establish profitable play.

## First usable version

- Play complete hands against the bot from a local terminal.
- Run reproducible bot-versus-bot matches from the same command-line entry point.
- Inspect decision inputs, legal actions, action history, and final settlement.
- Replay a saved hand deterministically and verify that the recorded actions and result match.
- Compare an equity/range baseline with calling, tight, and aggressive reference opponents.
- Save a manifest and per-hand records for every benchmark, including failures.

## AI baseline

Use explicit preflop hand-strength/position rules and postflop equity against an action-conditioned opponent range. Flop/turn and preflop all-in evaluations use fixed Monte Carlo samples. River evaluations enumerate every legal opponent combination exactly within the declared heuristic range. Terminal call-versus-fold decisions use separate win and tie probabilities, capped rake, uncalled refunds and integer odd-chip awards; this does not make the opening/value-bet heuristics or nonterminal decisions fully rake-optimized. Apply card removal to every range and sample. Select legal actions with bounded value/bluff frequencies and stack-aware sizing. Give each policy its own seeded random generator. Keep an unconditioned equity baseline for comparison.

This is a heuristic baseline, not GTO, an equilibrium blueprint, or a learned professional-level strategy. Do not add opaque LLM calls to the action loop. A failed or illegal action aborts evaluation and is retained as a failure; do not silently replace it with a favorable fallback.

## Advanced component

The implemented river layer extends the existing adaptive policy. It enumerates legal opponent cards within heuristic ranges, evaluates a finite action menu, models fold/call/raise responses, and includes a bounded call/fold response to reraises. It reuses the same opponent profiles; it is not a separate bot or an equilibrium solver. An initial matched benchmark did not establish an improvement, so the layer remains opt-in. Restricted-tree exploitability and equilibrium convergence have not been measured.

## Evaluation contract

Freeze configuration, opponent versions and weights, campaign pair count, seed range, policy sample count, and rake before starting a campaign. For each independently sampled deck, play both policy seat assignments. Within each pair, keep the physical seats, button, hole-card allocation, and runout fixed; swap only policy identities. Group each complete pair as one statistical unit. Incomplete pairs are retained but excluded from estimates. Use separate deck and policy random streams, and never expose a deck seed or future cards to either policy.

Report hand count and independent pair count separately. Show gross and net big blinds per 100 hands, standard error/confidence interval over paired outcomes, total rake, action failures, latency distribution, and maximum observed latency. Describe intervals as estimates for the tested fixed opponent distribution only; no event-winning probability or human-profitability inference. Small pilot campaigns verify the harness and cannot certify an edge.

Compute each policy's gross and net outcome directly from explicit pre-rake and post-rake settlement vectors. Do not use half the difference between the two policies' payoffs as net return in a raked game. Report opponents separately before any fixed-weight aggregate. Keep tuning and evaluation seeds separate. Do not rerun failed campaigns unchanged until passing. Benchmark results must record source hashes and dependency versions. Game logs include both hole cards for offline replay only; policies receive only their own cards and public information. Human play reveals no opponent private cards or complete replay record during an active hand. Recorded-action replay checks engine transitions and settlement. Decision reproducibility is a separate check requiring identical policy RNG/configuration and dependencies.

## Rake

Default to zero rake to establish a two-player zero-sum baseline. Chips are integers, with default small blind 1, big blind 2, and stack 200. Configure rake in integer basis points and its cap in integer chips. Rake is floor(eligible contested pot × basis points / 10000), capped at the configured chip amount. No-flop-no-drop applies if no flop was actually dealt. One engine component owns uncalled refunds, rake, and settlement, using PokerKit 0.7.7 as the rules authority. Never rake returned uncalled excess. At a split, each winner receives floor(distributable pot / winner count), with any odd chip awarded to the first winner clockwise after the button. Preserve explicit pre-rake and post-rake settlement vectors. Settlement conserves starting chips when rake is added back. Describe this as a simulation model, not a particular site's schedule.

Pot-dependent rake makes the two players' combined payoff depend on actions. Therefore zero-sum equilibrium guarantees must not be asserted for raked simulations.

## Hot-path budgets

| Interaction | Target | Failure behavior |
|---|---|---|
| Legal action and observation update | Under 20 ms locally | Abort on invalid game transition |
| Baseline decision | Under 500 ms locally at default samples | Measure actual duration; no competition 1 ms rule |
| Saved-hand replay | Under 2 seconds for a normal hand | Reject corrupt or inconsistent logs |

These are initial usability targets, not measured guarantees. No deadline-driven sampling during benchmark runs: fixed sample counts preserve reproducibility.

## Degradation matrix

| Situation | Behavior |
|---|---|
| No network | Play and benchmark work after dependency installation |
| No account | No account is required |
| Invalid CLI arguments | Explain the invalid value before starting a run |
| Existing output path | Refuse overwrite |
| Interrupted campaign | Keep manifest and completed-hand journal; mark incomplete |
| Invalid policy action | Preserve context and fail the run |

## State map

| State | Owner | Lifetime | Visible to policies |
|---|---|---|---|
| Full deck and both hole-card pairs | Engine | One hand | Own cards only |
| Public board/bets/action history | Engine | One hand | Immutable observation |
| Opponent range | Policy | One hand initially | Local policy state |
| Decision RNG | Each policy instance | One hand | Policy itself |
| Paired seeds and configuration | Arena | One campaign | No future deck access |
| Manifest and hand journal | Local files | Persistent | Offline only |

## Approaches considered

A. Standalone AI and local arena: selected by the user. Produces controllable experiments and complete games without site integration.
B. Offline hand-review trainer: useful for studying completed games, but not the selected first scope.
C. Direct online automation: experimental private-table adapter only; public hosting and real-money operation are outside this build.

## Technical shape

Python 3.12 with a pinned project environment. PokerKit 0.7.7 owns legal state transitions and settlement. Canonical actions are fold, check, call, and raise_to with an integer total contribution on the current betting street. Never mix additional chips with raise-to values. A direct PokerKit 0.7.7 probe confirmed that raw blinds (1, 2) in a two-player game post [2, 1]: internal seat 0 is the big blind and acts first postflop; internal seat 1 is the button/small blind and acts first preflop. Preserve or explicitly map this convention. Policy observations are frozen values containing own hole cards, public board/history, stacks, bets, pot and legal bounds; no live engine object, opponent hole cards, hidden deck, or deck seed crosses this boundary. A native evaluator may accelerate equity sampling. Keep engine/observation, policies, evaluation, and CLI as small distinct modules. Do not import the competition bot or copy its tournament objective. No database, server, cloud training, browser extension, or frontend is needed for the first usable arena.

## Verification

Use independent known-hand settlement checks, all-in and uncalled-bet cases, split pots/odd chips, blind order, action legality, hidden-card boundaries, deterministic replay, rake conservation, seat-swap pairing, and a small complete-game pilot. The public release includes durable Python and JavaScript suites, independent research replication, compact evidence verification, and CI. Never present a smoke test as a strategy benchmark.

## Distribution

Public source repository: `Smokeybear10/103.RSCH.river`, under `lab/`, runnable with uv. The Python lab uses its own heuristic policies; it does not currently consume the Rust solver's blueprint. Versioned research artifacts contain synthetic profiles and compact outcomes. Raw journals, real opponent databases, credentials and local browser state remain excluded from source and distribution archives.

## Success criteria

1. A human can start a local hand and finish it against the baseline.
2. All reference policies complete a fixed pilot without illegal actions or settlement errors.
3. Replaying saved hands reproduces the exact result.
4. Net returns and uncertainty are calculated from paired outcomes, with source/config identity saved.
5. Competition source and evidence remain untouched.

## Open limits

No claim of positive expected profit, superiority to real players, unrestricted HUNL equilibrium, official platform compatibility, or guaranteed latency. The first policy is a baseline for subsequent solver and opponent-model research.

## Research

- Libratus architecture and historical computation: https://www.ijcai.org/proceedings/2017/0772.pdf
- DeepStack and restricted search limits: https://johanson.ca/publications/poker/2017-science-deepstack/2017-science-deepstack.pdf
- Confidence-dependent opponent exploitation: https://proceedings.mlr.press/v5/johanson09a/johanson09a.pdf
- Variance reduction in evaluation: https://poker.cs.ualberta.ca/publications/aaai18-burch-aivat.pdf
- PokerKit simulation: https://pokerkit.readthedocs.io/en/stable/simulation.html

## Next steps

The first-version engine and local arena are implemented. Version 0.2 adds the user's requested automatic opponent learning without manual play.

## Automatic opponent learning

`uv run python -m headsup train` plays fresh training hands, saves opponent memory, warms up against three separately implemented synthetic styles, then compares the adaptive bot with the unchanged range baseline. `--resume PATH --seed NEW_INTEGER` adds training to a prior checkpoint in a new run directory. Resuming requires matching rules, fees, source and dependencies; prior output remains intact.

Memory contains only public fold/call/raise response counts, split by street, position, first bet versus reraise and call-price bucket. Blind completion is excluded. A Beta prior and at least 24 matching observations limit sparse-data adjustments. The policy suppresses some bluffs against persistent callers and adds bounded bets against sufficiently supported over-folders. It does not infer hidden cards or opponents' bluff frequencies. Each hand receives a detached profile snapshot; updates occur only after complete training hands.

Evaluation freezes all profiles. Each independent deck creates four games: baseline and adaptive, each in both seats, with matched policy seeds. Paired net-return differences measure the effect of using the learned memory. Training, held-out warmup and evaluation use distinct seed domains. Held-out warmup measures adaptation to unfamiliar synthetic styles, not zero-shot generalization. Conditional intervals exclude variation across independently trained checkpoints. Save every candidate and result; no automatic default replacement or profitable-play claim.

Alternatives considered: full-game neural self-play requires a substantially larger training system and does not directly solve specific-opponent memory; repeated threshold search risks overfitting the synthetic benchmark. Public-response learning reuses the verified engine and produces inspectable, measurable changes now.

## Changing opponents

Version 0.3 adds optional recent memory. `train --memory recent` keeps at most 64 first eligible responses per context from distinct completed hands and expires evidence at 256 completed-hand age. The 24-response gate then means 24 distinct hands in that context. These are fixed engineering choices, not theoretically optimal poker parameters. Lifetime memory remains the default/comparator until independent evidence justifies changing it.

Schema 1 keeps the original lifetime meaning and exact replay identity. Schema 2 records public response chronology for recent memory; old aggregate counts cannot reconstruct chronology and are not silently converted. Profiles still never learn from hidden cards, realized payoffs or opponent strategy labels.

`uv run python -m headsup adaptation` compares the fixed baseline, lifetime learner and recent learner across independently seeded sessions. Four prospective scenarios include stationary frequent callers/folders and both directions of style change. Each arm maintains its own memory and receives matched fresh decks and policy seeds. Public evidence updates after each completed hand; the scored outcome is not a learning input. Whole-session differences, including learning variability, are the statistical units. Full-session and post-switch windows are declared before running; 20 lift endpoints use multiplicity-adjusted approximate intervals.

Private connection testing is separate from strategy evaluation. No real-player result may be inferred from local sessions or synthetic opponent campaigns.

## Private table connection

Version 0.4 connects separate bot or human terminal clients to our own authenticated play-money table protocol. The server binds to loopback, deals private unpredictable decks, alternates player positions, and resets stacks each hand. Both players acknowledge each completed hand before the next begins. A disconnection pauses play without taking a fallback action.

SQLite commits table state, accepted request receipts and completed replay records together. A lost response can be retried with the same request ID after either process restarts. The client saves its pending command before transmission, keeps the policy's random state and frozen per-hand profile, and commits each completed-hand learning update with its completion cursor. New table/opponent identities start with empty memory; synthetic opponent profiles are not relabeled as people.

This connects to our private protocol and includes a local browser interface. The experimental PokerNow adapter is a separate transport, with visible-state validation and fail-closed action confirmation. No ClubGG adapter, public hosting, matchmaking or human recruitment is included. Network checks establish transport and recovery correctness, not poker strength.

## Persistent identity and river evidence

Saved profiles use room, hero ID, opponent ID and blind structure as identity boundaries. Names and seats are display attributes, so a rename or seat swap cannot assign one player's evidence to another. Repeated completed-hand receipts cannot add the same evidence twice. Each decision uses a detached snapshot; public evidence commits after a verified completed hand.

The default adaptive policy requires 24 matching individual responses before targeted changes. The optional river model has a different guard: candidate actions must clear its modeled scenario-margin test. It can reason from prior response assumptions without 24 individual observations, so sparse-data river results must not be described as learned player-specific certainty.

The release replication trains independent profiles under three seeds and then freezes them for matched evaluation. Independent seed aggregates capture training variability; the small number of seeds still limits precision. Historical positive and negative results are retained in the research report. Software correctness, synthetic-opponent improvement and real-player profitability remain separate claims.
