"""Board-aware practice opponents and paired candidate evaluation.

Nothing runs or writes on import. These are diagnostic adversaries for standard
Hold'em, not equilibrium opponents or a model of the unrevealed event variant.
"""

import argparse
import hashlib
import importlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from math import sqrt
from pathlib import Path
from random import Random
from statistics import mean, stdev
from time import perf_counter

import eval7

from .arena import Configuration, play, summary
from .cards import starting_strength
from .equity import Equity, Estimate
from .holdem import CARDS, HoldemEquity
from .model import Action, Event, Observation, Player
from .policy import Parameters, TournamentPolicy

FAMILIES = ("value", "pressure", "polar", "tricky", "selfplay")


class FixedSampleEquity:
    """Remove wall-clock sampling drift for strategy screening only."""

    research_fixed_samples = True

    def __init__(self, backend: Equity) -> None:
        self.backend = backend

    @property
    def cards(self):
        return self.backend.cards

    @property
    def evaluate(self):
        return self.backend.evaluate

    def estimate(
        self,
        obs: Observation,
        ranges: tuple[float, ...],
        pressure: tuple[int, ...],
        samples: int = 40,
        deadline: int | None = None,
        information: tuple[float, ...] | None = None,
    ) -> Estimate:
        return self.backend.estimate(
            obs, ranges, pressure, samples, None, information
        )


class BoardOpponent:
    """Independent, small public-information policies with different weaknesses.

    Postflop strength is sampled against uniformly drawn legal hands and runouts.
    Opponent ranges and future betting are deliberately simple. No simulator state
    or opponents' private cards enter the policy.
    """

    def __init__(self, style: str, seed: int = 0, samples: int = 20) -> None:
        if style not in FAMILIES[:-1] or samples < 1:
            raise ValueError("unknown opponent style or nonpositive sample count")
        self.style = style
        self.rng = Random(seed)
        self.samples = samples
        self.last_raise: tuple[int, int, int] | None = None

    def observe(self, event: Event) -> None:
        if event.action.kind == "raise":
            self.last_raise = (event.hand, event.street, event.seat)

    def strength(self, obs: Observation) -> float:
        if not obs.board:
            return starting_strength(obs.hole)
        known = set(obs.hole + obs.board)
        available = [c for c in range(52) if c not in known]
        players = len(obs.opponents)
        if not players:
            return 1.0
        runout_count = 5 - len(obs.board)
        board = [CARDS[c] for c in obs.board]
        own = [CARDS[c] for c in obs.hole]
        total = 0.0
        for _ in range(self.samples):
            dealt = self.rng.sample(available, runout_count + players * 2)
            runout = board + [CARDS[c] for c in dealt[:runout_count]]
            ours = eval7.evaluate(own + runout)
            ties = 1
            for offset in range(runout_count, len(dealt), 2):
                theirs = eval7.evaluate(
                    [CARDS[dealt[offset]], CARDS[dealt[offset + 1]]] + runout
                )
                if theirs > ours:
                    break
                ties += theirs == ours
            else:
                total += 1.0 / ties
        return total / self.samples

    def act(self, obs: Observation) -> Action:
        strength = self.strength(obs)
        call = obs.legal.call
        passive = Action("call" if call else "check")
        if not obs.opponents:
            return passive
        odds = call / max(1, obs.pot + call)
        chance = self.rng.random()
        if obs.street == 0:
            cutoff = {"value": 0.76, "pressure": 0.42, "polar": 0.58, "tricky": 0.50}[self.style]
            if strength < cutoff and call:
                return Action("fold")
            if strength > 0.93 or (self.style == "pressure" and strength > 0.68):
                return obs.legal.raise_to(obs.player.bet + call + 2 * obs.big_blind)
            return passive
        if call:
            edge = {"value": 0.12, "pressure": -0.025, "polar": 0.04, "tricky": -0.015}[self.style]
            if strength < odds + edge:
                return Action("fold")
        raise_ok = obs.legal.min_raise_to is not None and any(s.stack > 0 for s in obs.opponents)
        if not raise_ok:
            return passive
        fraction = 0.0
        # Value family refuses speculative aggression. Pressure attacks frequent
        # folds with small bets. Polar uses large value bets and occasional air.
        # Tricky checks strong hands and calls wider to stress thin value betting.
        if self.style == "value" and strength >= 0.78:
            fraction = 0.75
        elif self.style == "pressure" and (strength >= 0.60 or (not call and chance < 0.70)):
            fraction = 0.25
        elif self.style == "polar" and (strength >= 0.84 or (strength < 0.22 and not call and chance < 0.18)):
            fraction = 1.5
        elif self.style == "tricky" and strength >= 0.85:
            faced_raise = self.last_raise is not None and self.last_raise[:2] == (obs.hand, obs.street) and self.last_raise[2] != obs.hero
            if (call and faced_raise and chance < 0.65) or (not call and chance < 0.22):
                fraction = 0.85
        if fraction:
            increment = max(obs.big_blind, round((obs.pot + call) * fraction))
            return obs.legal.raise_to(obs.player.bet + call + increment)
        return passive


def policy_factory(spec: str, seed: int, parameters: Parameters, seats: int, fixed_samples: bool) -> Player:
    """Import a trusted local policy class; no source strings are evaluated."""
    module_name, separator, class_name = spec.partition(":")
    if not separator or not class_name.isidentifier() or not all(part.isidentifier() for part in module_name.split(".")):
        raise ValueError("policy must be a module:Class name")
    policy_class = getattr(importlib.import_module(module_name), class_name)
    equity = HoldemEquity(seed ^ 0x73B9)
    if fixed_samples:
        equity = FixedSampleEquity(equity)
    agent = policy_class(seed=seed, parameters=parameters, seats=seats, equity=equity)
    if not callable(getattr(agent, "act", None)) or not callable(getattr(agent, "observe", None)):
        raise TypeError("policy must implement act and observe")
    return agent


def opponents(field: str, seed: int, hero: int, seats: int, fixed_samples: bool) -> tuple[Player | None, ...]:
    if field not in FAMILIES + ("mixed",):
        raise ValueError(f"unknown field: {field}")
    agents = []
    for seat in range(seats):
        if seat == hero:
            agents.append(None)
            continue
        style = FAMILIES[(seat - hero - 1) % len(FAMILIES)] if field == "mixed" else field
        agent_seed = seed * 997 + seat * 7919
        if style == "selfplay":
            # Frozen constructor/parameters: candidate changes do not change the
            # opponent roster. Source hashes additionally record the exact code.
            equity = HoldemEquity(agent_seed ^ 0x73B9)
            if fixed_samples:
                equity = FixedSampleEquity(equity)
            agent = TournamentPolicy(agent_seed, Parameters(), seats, equity)
        else:
            agent = BoardOpponent(style, agent_seed)
        agents.append(agent)
    return tuple(agents)


def paired_summary(baseline: list[dict], candidate: list[dict]) -> dict:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired batches must be nonempty and equal length")
    differences = []
    for old, new in zip(baseline, candidate):
        if any(old[key] != new[key] for key in ("seed", "hero_seat", "field")):
            raise ValueError("paired results must use matching seeds, seats, and fields")
        differences.append(new["points"] - old["points"])
    delta = mean(differences)
    half_width = 1.96 * stdev(differences) / sqrt(len(differences)) if len(differences) > 1 else None
    return {
        "baseline": summary(baseline),
        "candidate": summary(candidate),
        "paired_delta": {
            "mean": delta,
            "95pct_interval": [delta - half_width, delta + half_width] if half_width is not None else None,
            "tables": len(differences),
            "positive_tables": sum(value > 0 for value in differences),
            "negative_tables": sum(value < 0 for value in differences),
        },
    }


def _pair_job(arguments: tuple) -> tuple[dict, dict]:
    seed, hero, field, baseline, candidate, baseline_parameters, candidate_parameters, config, fixed_samples = arguments
    results = []
    for spec, parameters in ((baseline, baseline_parameters), (candidate, candidate_parameters)):
        agent = policy_factory(spec, seed * 997 + hero * 7919, parameters, len(config.stacks), fixed_samples)
        result = play(
            seed, hero, "mixed", parameters, config, hero_agent=agent,
            opponent_agents=opponents(field, seed, hero, len(config.stacks), fixed_samples),
        )
        result["field"] = field
        results.append(result)
    return tuple(results)


def play_harness(
    baseline: str = "competition.policy:TournamentPolicy",
    candidate: str = "competition.river_policy:RiverPolicy",
    *,
    fields: tuple[str, ...] = FAMILIES,
    tables: int = 24,
    seed: int = 20261102,
    workers: int = 1,
    config: Configuration | None = None,
    baseline_parameters: Parameters | None = None,
    candidate_parameters: Parameters | None = None,
    fixed_samples: bool = False,
) -> dict:
    config = config or Configuration()
    if fixed_samples:
        config = replace(config, enforce_deadline=False)
    config.validate()
    if tables < 2 or workers < 1 or not fields or len(set(fields)) != len(fields) or any(f not in FAMILIES + ("mixed",) for f in fields):
        raise ValueError("use at least two tables, positive workers, and unique known fields")
    baseline_parameters = baseline_parameters or Parameters()
    candidate_parameters = candidate_parameters or Parameters()
    # Fail promptly on a bad factory rather than after scheduling a long run.
    for spec, parameters in ((baseline, baseline_parameters), (candidate, candidate_parameters)):
        policy_factory(spec, seed, parameters, len(config.stacks), fixed_samples)
    jobs = [
        (seed + table, table % len(config.stacks), field, baseline, candidate, baseline_parameters, candidate_parameters, config, fixed_samples)
        for field in fields for table in range(tables)
    ]
    started = perf_counter()
    if workers == 1:
        results = list(map(_pair_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_pair_job, jobs, chunksize=1))
    report = {
        "practice_only": True,
        "variant": "No-Limit Texas Hold'em",
        "baseline": baseline,
        "candidate": candidate,
        "baseline_parameters": asdict(baseline_parameters),
        "candidate_parameters": asdict(candidate_parameters),
        "configuration": asdict(config),
        "seed": seed,
        "tables_per_field": tables,
        "workers": workers,
        "fixed_samples": fixed_samples,
        "elapsed_seconds": perf_counter() - started,
        "fields": {},
        "limitations": [
            "Practice opponent families are hand-built diagnostics, not actual competition entrants or equilibrium strategies.",
            "Shared per-hand deck seeds and rotating seats reduce noise; different actions and eliminations can change later card assignments.",
            "Approximate normal 95% intervals describe each field separately; small samples, model selection, and multiple comparisons limit their interpretation.",
            "Fixed-sample screening disables internal equity deadlines and simulator timeout fallback; it cannot establish 1 ms runtime compliance.",
            "Timing measures act() only, excludes adapter overhead, and varies with host scheduling and concurrent workers.",
            "Real-clock timeout fallback applies to every seat, but overrun counts cover only the hero; unrecorded opponent timeouts can weaken the field under host load.",
            "Freeze candidates before validation; new seeds do not substitute for holding out opponent families and variants.",
        ],
    }
    for index, field in enumerate(fields):
        pairs = results[index * tables:(index + 1) * tables]
        report["fields"][field] = paired_summary([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return report


def fingerprint(baseline: str, candidate: str, baseline_parameters: Parameters, candidate_parameters: Parameters) -> dict:
    sources = {}
    for spec in (baseline, candidate, __name__ + ":BoardOpponent"):
        module_name = spec.split(":")[0]
        module = importlib.import_module(module_name)
        path = Path(module.__file__).resolve()
        # Hash the local package, including imported strategy helpers. Third-party
        # dependencies and runtime versions remain a separate deployment concern.
        package = module_name.rpartition(".")[0]
        files = sorted(path.parent.rglob("*.py")) if package else [path]
        for source in files:
            key = (package + "/" if package else "") + str(source.relative_to(path.parent))
            sources[key] = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "baseline": baseline,
        "candidate": candidate,
        "baseline_parameters": asdict(baseline_parameters),
        "candidate_parameters": asdict(candidate_parameters),
        "source_sha256": sources,
    }


def validate_manifest(manifest: dict, identity: dict, seed: int, tables: int) -> None:
    if manifest["identity"] != identity:
        raise ValueError("frozen policy source or parameters changed; freeze a new candidate before validation")
    screening = manifest["screening_seed_interval"]
    if max(screening[0], seed) <= min(screening[1], seed + tables - 1):
        raise ValueError("validation seeds overlap the recorded screening run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", default="competition.policy:TournamentPolicy")
    parser.add_argument("--candidate", default="competition.river_policy:RiverPolicy")
    parser.add_argument("--baseline-parameters", type=Path, help="JSON object matching Parameters fields")
    parser.add_argument("--candidate-parameters", type=Path, help="JSON object matching Parameters fields")
    parser.add_argument("--fields", nargs="+", choices=FAMILIES + ("mixed",), default=list(FAMILIES))
    parser.add_argument("--tables", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument("--fixed-samples", action="store_true", help="deterministic strategy screening; no timing enforcement")
    parser.add_argument("--output", type=Path, help="write report here; default prints only")
    manifests = parser.add_mutually_exclusive_group()
    manifests.add_argument("--freeze-manifest", type=Path, help="save candidate source/parameters and screening seeds")
    manifests.add_argument("--validate-manifest", type=Path, help="require frozen candidate and disjoint validation seeds")
    args = parser.parse_args()
    old_parameters = Parameters(**json.loads(args.baseline_parameters.read_text())) if args.baseline_parameters else Parameters()
    new_parameters = Parameters(**json.loads(args.candidate_parameters.read_text())) if args.candidate_parameters else Parameters()
    identity = fingerprint(args.baseline, args.candidate, old_parameters, new_parameters)
    if args.validate_manifest:
        validate_manifest(json.loads(args.validate_manifest.read_text()), identity, args.seed, args.tables)
    report = play_harness(
        args.baseline, args.candidate, fields=tuple(args.fields), tables=args.tables,
        seed=args.seed, workers=args.workers, config=Configuration(max_hands=args.max_hands),
        baseline_parameters=old_parameters, candidate_parameters=new_parameters,
        fixed_samples=args.fixed_samples,
    )
    if fingerprint(args.baseline, args.candidate, old_parameters, new_parameters) != identity:
        raise RuntimeError("source changed during evaluation; discard the run and retry")
    report["identity"] = identity
    report["phase"] = "validation" if args.validate_manifest else "screening"
    encoded = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    if args.freeze_manifest:
        manifest = {"identity": identity, "screening_seed_interval": [args.seed, args.seed + args.tables - 1]}
        args.freeze_manifest.parent.mkdir(parents=True, exist_ok=True)
        args.freeze_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(encoded, end="")


if __name__ == "__main__":
    main()
