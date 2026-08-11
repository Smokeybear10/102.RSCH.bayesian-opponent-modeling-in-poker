# Persistent opponent models for heads-up Hold'em

An empirical study of public-action memory, conservative exploitation and restricted river decisions in a reproducible two-player poker environment.

The research question is whether a policy that remembers an individual opponent improves on the same policy without that memory. Software correctness and playing strength are evaluated separately. The project includes a working heuristic baseline, persistent learning, paired experiments, and negative results; it does not establish a professional-level or profitable player.

## Models under comparison

| Component | What it does | What it does not establish |
|---|---|---|
| `range` | Preflop rules and postflop equity against an action-conditioned heuristic range; seeded mixed actions | A solved or equilibrium heads-up strategy |
| `adaptive` | Keeps the `range` baseline and adjusts selected bets using contextual individual fold-response estimates | Full opponent range learning, or a separate aggressive-player counterstrategy |
| Persistent memory | Saves completed public fold/call/raise responses transactionally, with duplicate-hand protection and identity checks | A player identity inferred from a name or playing style |
| `river` | Builds on `adaptive`; enumerates legal river combinations and compares a bounded set of bet/raise sizes and call/fold continuations under response scenarios | A full-game solver, unrestricted action-tree equilibrium, or measured exploitability bound |

Both learning policies consume the same profile format. The river policy also separates fold, call and raise response probabilities. It remains opt-in: the earlier river comparison did not pass its promotion gate.

Memory contexts distinguish street, opponent position, first bet versus reraise, and the price of calling relative to the pot. Default `adaptive` adjustments require at least 24 matching individual responses, plus uncertainty, size and exposure gates. The optional river policy can evaluate prior response scenarios before that threshold, under a separate margin gate. A frozen population prior is supported but is disabled in the new replication below, matching the default individual-memory configuration.

Player identity and behavior are separate. The private adapter keys a profile by room, hero ID, opponent ID and blind settings, rather than by display name or current seat. Identical names, renames and dealer rotation cannot merge profiles. An unexpected lineup change stops the session. Cross-room identity linking is not assumed.

## Independent training-seed replication

The prespecified campaign contains three independent training seeds and six fixed synthetic styles: calling, tight, aggressive, fit/fold, pressure and loose. Each seed/style stratum starts with empty lifetime memory, learns from 600 completed hands, closes and reopens SQLite, then freezes the restored profile. No reward, private opponent cards, hidden deck state or style label enters the learning model.

Evaluation uses 128 new matched deck units per stratum. Each unit contains four games: baseline and adaptive, each in both seats. Decks and policy seeds match across strategies. Fresh 100BB stacks, blinds 10/20, zero rake and 128 equity samples are fixed before play. Learning and evaluation use disjoint seed domains. Evaluation never updates the frozen memory.

This yields **10,800 learning hands and 9,216 evaluation hands**, with **2,304 matched deck units** and **18 independently persisted profiles**. The three training seeds, not the 20,016 hands, are the independent units for the primary uncertainty estimate.

The primary endpoint is the equal-weight mean adaptive-minus-baseline net return across all six styles. A Student-t 95% interval is calculated over the three independent seed-level aggregates. Six style-specific secondary comparisons use Bonferroni-adjusted intervals over the same three seeds. With only two degrees of freedom, these intervals are fragile; their normality assumption has not been validated.

The completed campaign estimates **+4.13 bb/100** for adaptive minus range, with a **95% interval of +0.57 to +7.69 bb/100**. The three seed-level effects are +4.56, +5.29 and +2.53 bb/100. This is evidence of improvement in this fixed synthetic pool; the small number of independent replicates limits the claim.

| Synthetic opponent | Adaptive − range, bb/100 | Multiplicity-adjusted 95% interval |
|---|---:|---|
| Calling | +6.78 | −22.92 to +36.49 |
| Tight | +6.38 | −3.02 to +15.78 |
| Aggressive | +5.73 | −28.29 to +39.75 |
| Fit/fold | −5.29 | −97.35 to +86.76 |
| Pressure | +4.97 | −46.43 to +56.38 |
| Loose | +6.18 | −13.27 to +25.62 |

No individual style has a conclusive adjusted improvement. Fit/fold has a negative point estimate. The policy makes relatively few targeted changes, so a small number of changed decisions can materially affect an individual stratum.

The [portable bundle](results/replication-20261107) includes the prespecified manifest, all 2,304 paired-unit returns, 18 frozen profiles and file checksums. Independent verification recomputes the reported statistics. The raw audit replays all **20,016 hands and 97,731 actions**, reconstructs all learned profiles, verifies each arm/style/seat assignment and reproduces policy choices and profile fingerprints on 89 selected hands. Full policy decisions are not recomputed for every hand. A compact [audit receipt](results/replication-audit-20261107.json) is public; full synthetic hand journals remain local.

An infrastructure interruption exhausted local disk space after 13 of 18 strata completed. Those complete strata were retained; the five failed strata were repeated from their first learning hand using the original frozen policy source, seeds and settings. The original incomplete report, failure files and partial journals remain in the local run's `attempts/` directory; the public [failure ledger](results/infrastructure-interruption-20261107.json) records their hashes and 3,673 complete JSON records from partial attempts, excluded from the final estimates. Recovery changed journal storage to gzip and records the recovery runner's hash separately. No parameter, sample-size or strategy change was selected from outcomes. Subsequent harness guards reject environment drift during recovery and existing export destinations before play; these do not alter the frozen policy or published results.

## Earlier experiments

These are separate historical campaigns with different configurations, versions and statistical units. Their effects and hand counts must not be pooled into one performance estimate. Compact metrics, underlying aggregate values and original source/report hashes are retained in [`historical-campaigns.json`](results/historical-campaigns.json). Earlier raw journals and full run directories remain local.

| Campaign | Recorded hands | Comparison | Estimated difference, bb/100 | Reported interval |
|---|---:|---|---:|---|
| Frozen memory, 5% simulated rake capped at 3BB | 9,744 | Adaptive − range | +4.87 | Adjusted 95%: +0.91 to +8.83 |
| Recent-memory style-change sessions, same rake model | 55,296 | Recent − range | +2.41 | Adjusted 95%: +1.07 to +3.75 |
| Same recent-memory campaign | Included above | Recent − lifetime | +0.08 | Adjusted 95%: −0.77 to +0.93 |
| Earlier experimental river campaign, same rake model | 21,504 | River − adaptive | −3.08 | Primary 95%: −18.68 to +12.52 |
| Persistent memory, no rake, one training seed | 9,744 | Adaptive − range | +4.06 | Adjusted 95%: −0.06 to +8.17 |

The frozen-memory studies condition their uncertainty on one trained profile per style. The session studies use independent complete paired sessions, including learning variability. Their interval widths are not directly comparable.

The **river result is a failed promotion**, not evidence of improvement. Its negative point estimate and wide interval motivate retaining the baseline and examining response-range and raise-continuation assumptions. Later river implementation changes require their own fresh comparison; the historical estimate cannot validate them. Positive results against a synthetic calling opponent are not sufficient to promote a strategy across a broader pool.

Development runs are retained as well. Before the earlier 9,744-hand persistent-memory campaign, a sequential pilot was interrupted after 423 learning hands to parallelize execution. An initial parallel harness rejected a domain/protocol event conversion before learning; the corrected full campaign was evaluated separately. These were mechanical failures, not discarded losing strategy results.

## Reproduction

Run from the repository root with Python 3.12 and the locked dependencies:

```sh
uv sync --locked
uv run --locked python scripts/run_research.py \
  --output runs/reproduction \
  --export runs/reproduced-results \
  --seeds 20261107101 20261107102 20261107103 \
  --warmup 600 --pairs 128 --samples 128 --jobs 6
uv run --locked python scripts/verify_research.py runs/reproduced-results
uv run --locked python scripts/audit_research_run.py runs/reproduction
```

Output directories must be new and separate. Worker count changes scheduling, not seeds or the experiment design. The runner snapshots the source before play and records Python, dependency versions, source hashes and configuration. The published campaign used `runs/replication-20261107` and exported to `research/results/replication-20261107`; the commands above preserve that bundle.

For an infrastructure interruption, repeat the identical command with `--resume`. Completed strata are retained; failed strata are archived and rerun from the beginning. Source, Python, platform, dependency versions and experimental settings must match the original campaign. A completed campaign cannot be extended or resumed with different sample sizes under that identity.

The artifact verifier recomputes means and uncertainty from the public CSV, checks all planned units and profile counts, and validates file hashes. The raw-run auditor replays every action and settlement, reconstructs every learned profile from public actions, checks train/evaluation separation, and reproduces selected policy decisions. Raw journals contain full synthetic decks for offline verification and stay in the ignored `runs/` directory.

To exercise the pipeline quickly without measuring strength:

```sh
uv run --locked python scripts/run_research.py \
  --output runs/research-smoke --export runs/research-smoke-export \
  --seeds 20261107000 --warmup 8 --pairs 4 --samples 32 --jobs 2
uv run --locked python scripts/verify_research.py runs/research-smoke-export
uv run --locked python scripts/audit_research_run.py runs/research-smoke
```

A one-seed smoke run intentionally has no replicate-level confidence interval. It verifies mechanics, not playing strength.

## Interpretation limits

- Synthetic opponents expose controlled behaviors, not the full range of human strategies. They can be much easier to exploit than real players.
- New styles receive their own learning hands. This is opponent adaptation, not zero-shot transfer to an unseen population.
- A high observed call or raise rate is a conditional response frequency, not proof of bluffing, hand strength, or a stable psychological trait.
- Historical raked and current no-rake experiments answer different questions. Neither establishes a particular site's rake economics.
- Profiles start empty for new room/player identities. Synthetic training profiles are never copied into real opponent memory.
- The local/private-table interface and the PokerNow adapter have separate verification histories. Browser integration failures, unsupported formats and incomplete hands stop action; a successful simulation does not establish live-site reliability.
- Functional tests, faster actions and more stored data do not by themselves establish positive expected profit.
