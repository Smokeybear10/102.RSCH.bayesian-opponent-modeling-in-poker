# Bayesian Opponent Modeling in Poker | Competition Toolkit

This is the competition mode of Bayesian Opponent Modeling in Poker, imported from Shove. Run it from the Bayesian Opponent Modeling in Poker repository root with the same environment as the heads-up package:

```sh
uv sync --locked
uv run --locked python -m competition --help
```

The two modes retain separate game state and policies. Tournament results below do not describe the heads-up research bot or the Rust solver.

The goal is to arrive with the tooling built, then give the coding assistant the
official starter and rules. The assistant can handle the SDK integration,
variant changes, checks, and packaging. The new session workflow prepares that
handoff in one command. It does not yet automate the integration itself.

The prepared playing strategy is **standard no-limit Hold'em only**. The event
reveals its game and interface when the build period starts. A different deck,
hand-ranking rule, or betting structure can require a different evaluator and
strategy. See the [official rules](https://pokerbot.sequoiacap.com/rules.html).
Permission to reuse prewritten code still needs to be confirmed from the event's
instructions. Preparing this toolkit does not establish that permission.

## Start with the released kit

Run these commands from the repository root using Bayesian Opponent Modeling in Poker's pinned Python 3.12
environment. The dependency-free session and SDK tools also support Python
3.10–3.12 when used separately.

```sh
uv run --locked python -m competition.session start /path/to/official-starter.zip --output day/session
uv run --locked python -m competition.session status day/session
```

The input can also be a directory. Session creation snapshots the starter,
creates a separate candidate workspace, runs source inspection, and writes an
unresolved contract plus a handoff for the coding assistant. It does not execute
the starter or infer the game rules from names in its source. Use a new output
directory for a new session.

The session contains `starter/`, `candidate/`, `intake.json`, `contract.json`,
`session.json`, and `AGENT_HANDOFF.md`. Git history, virtual environments, and
caches are excluded and recorded in the manifest. Snapshot limits are 5,000
entries, 25 MB per file, and 100 MB total; these are local tool limits, not a
statement of the official starter format. Symlinks and secret `.env` files are
rejected; example environment templates are allowed.

Provide the starter and any separate rules, technical instructions, or organizer
clarifications together. The generated handoff tells the assistant to establish
the official baseline first, resolve the contract against the supplied evidence,
and adapt the bot only after the game's mechanics are understood.

## What is built

| Capability | Current scope |
| --- | --- |
| Starter intake | Lists source files, hashes, imports, interfaces, and likely rule references without importing starter code. |
| Event session | Accepts a starter directory or ZIP, preserves its contents, creates a working copy, and prepares an explicit contract and assistant handoff. |
| Hold'em integration | Bridge for observations and legal actions, explicit raise conventions, public-history tracking, table resets, and an injectable equity backend. |
| Runtime checks | Shared callback budget, synthetic SDK rehearsal, full-game wire checks, and latency reporting. Local measurements need repeating in the official environment. |
| Strategy experiments | Hold'em tournament simulator, opponent styles, paired comparisons, parameter search, and validation on unused seeds. |
| Opponent league | Board-aware value, pressure, polar, and tricky opponents plus frozen baseline self-play; paired point comparisons with uncertainty and source fingerprints. |
| Experimental river policy | Opt-in comparison of checking and several bet sizes in restricted heads-up river spots, with a heuristic fold/call response model. It has not established a tournament advantage. |
| Candidate packaging | Checks staged code and the extracted archive, records source hashes and imports, and preserves the previous candidate when checks fail. The `starter` backend packages selected starter files without adding the Hold'em runtime. |

## Test a strategy candidate

The existing `TournamentPolicy` remains the default. `RiverPolicy` is an optional
research candidate. It uses shared sampled hands to compare checking against
different bet sizes when checking closes the river action. Its response model
does not model opponent raises, and its values are chips rather than tournament
points. Neither this module nor the practice opponents are solved strategies.

With the practice dependencies installed, screen it reproducibly:

```sh
uv run --locked python -m competition.league --fixed-samples --workers 2 \
  --fields value pressure selfplay --tables 24 --seed 20261102 \
  --freeze-manifest day/river-frozen.json --output day/river-screen.json
```

Freeze the candidate before using untouched seeds and opponent families:

```sh
uv run --locked python -m competition.league --fixed-samples --workers 2 \
  --fields polar tricky --tables 128 --seed 20262102 \
  --validate-manifest day/river-frozen.json --output day/river-validation.json
```

The manifest rejects changed policy sources or parameters and overlapping seed
intervals. Opponent-family holdout is a research procedure, not an automatically
enforced rule. Each field's interval is approximate; a small positive average
does not establish an improvement. Fixed-sample mode disables sampling deadlines
and simulator timeout fallback for strategy comparisons. Omit `--fixed-samples`
for a separate real-clock check, and use the adapter and official environment to
measure the full callback. No candidate is promoted automatically.

## Preparation priorities

The session workflow removes setup and handoff work. These are the next useful
pieces to build before the event, in order:

1. **Practice the complete takeover.** Rehearse from an unfamiliar starter ZIP
   through a working baseline, adapter, official-style checks, and an extracted
   candidate. Change method signatures, action formats, and raise conventions
   between drills. Measure time to the first verified candidate.
2. **Exercise actual game changes.** Add drills with different deck rules,
   hand-selection rules, and betting structures. A successful SDK adapter alone
   does not establish that the policy understands a new game.
3. **Separate game mechanics from strategy.** Build tested evaluator and game
   modules for chosen variants, with known-answer hand comparisons and explicit
   capability checks. Broad variant support is not implemented yet.
4. **Prepare cheap decision policies.** Build and benchmark bounded policies and
   lookup tables for the variants we choose to support. Keep a legal baseline
   available while stronger policies are being checked. The current Hold'em
   policy is not a universal fallback.
5. **Automate candidate comparisons.** Connect the official simulator after
   release, use its placement points and payouts, and compare candidates on
   matching seeds plus held-out opponents. Automatic promotion based on official
   scores is not implemented yet.
6. **Make round changes recoverable.** Extend the existing candidate retention
   into a record of which artifact was tested, submitted, and accepted each
   round. Retain the accepted version until a replacement passes the required
   checks. Packaging success alone is not submission acceptance.

## What remains on the day

Thomas supplies the released kit and any instructions unavailable to the
assistant. The assistant can then inspect the source, run the supplied baseline,
implement the adapter or variant changes, run official checks, and prepare the
candidate. Unclear organizer rules may still need a clarification. Authentication,
personal attestations, and submission approval stay with Thomas when required.

The practical target is one initial handoff and a reviewable candidate. We cannot
promise a ready strategy for an undisclosed game, a fixed integration time, or
official acceptance before seeing the actual kit.

## Local rehearsal and packaging

This existing rehearsal exercises the provided mock SDK and verifies an extracted
candidate using Python's standard library:

```sh
uv run --locked python -S -m competition.eventday rehearse --deterministic --output day/rehearsal
```

This deterministic mode freezes the mock SDK's budget clocks so host scheduling
does not turn an expected raise into a legal timeout fallback during a wiring
test. Its recorded wall times include mock overhead and are not performance
results. Omit `--deterministic` to retain the real-clock budget check. Use the
full-game `competition.wirecheck` and official runtime for actual latency checks;
the playing bot's deadline behavior is unchanged.

For a prepared starter directory, the packaging interface is:

```text
uv run --locked python -m competition.eventday build <working-starter-directory> \
  --entry <entry.py> --include <required-files-or-directories> \
  --backend starter --output day/candidate.zip \
  --check <official-check-command-using-{entry}-or-{candidate}>
```

This is a template. The entry, included files, runtime, check command, and upload
format come from the actual kit. Use `--backend starter` for an independent
baseline or a custom variant implementation. The other backends add the prepared
Hold'em core: `provided` uses an injected equity backend, and `eval7` adds the
native Hold'em evaluator.

The check command must fail for illegal actions, runtime errors, and deadline
violations. The packager runs that supplied check twice; it cannot determine
whether the check covers the official requirements. A ZIP is a local candidate
artifact until the official upload format is confirmed.

Optional Hold'em simulation and native evaluation depend on the packages pinned
in `competition/requirements.txt`. Session creation and the synthetic rehearsal
do not need those packages. Saved reports describe earlier local preparation
runs, not certification of a future official submission.
