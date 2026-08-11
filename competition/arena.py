from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from itertools import groupby
from math import isfinite, sqrt
from random import Random
from statistics import mean, stdev
from time import perf_counter_ns

from pokerkit import Automation, Card, NoLimitTexasHoldem, State

from .cards import CARD_NAMES
from .icm import hero_value
from .model import Event, LegalActions, Observation, Player, Seat
from .policy import Baseline, Parameters, TournamentPolicy

DECK = tuple(Card.parse("".join(CARD_NAMES)))
CARD_IDS = {card: i for i, card in enumerate(DECK)}
AUTOMATIONS = tuple(
    a
    for a in Automation
    if a not in (Automation.HOLE_DEALING, Automation.BOARD_DEALING)
)


@dataclass(frozen=True, slots=True)
class Configuration:
    stacks: tuple[int, ...] = (2000,) * 8
    payouts: tuple[float, ...] = (50, 30, 20, 0, 0, 0, 0, 0)
    blinds: tuple[int, ...] = (20, 40, 80, 160, 320, 640, 1280, 2560)
    hands_per_level: int = 25
    max_hands: int = 200
    enforce_deadline: bool = True

    def validate(self) -> None:
        n = len(self.stacks)
        if (
            not 2 <= n <= 8
            or len(self.payouts) != n
            or any(s <= 0 for s in self.stacks)
        ):
            raise ValueError(
                "supply 2–8 positive starting stacks and one payout per player"
            )
        if (
            any(not isfinite(p) or p < 0 for p in self.payouts)
            or any(a < b for a, b in zip(self.payouts, self.payouts[1:]))
            or sum(self.payouts) <= 0
        ):
            raise ValueError(
                "payouts must be descending, nonnegative, and have a positive total"
            )
        if (
            not self.blinds
            or any(b < 2 for b in self.blinds)
            or self.hands_per_level < 1
            or self.max_hands < 1
        ):
            raise ValueError("positive blinds and hand limits are required")


def observation(
    state: State,
    mapping: tuple[int, ...],
    initial: tuple[int, ...],
    stacks: list[int],
    hero: int,
    button: int,
    hand: int,
    bb: int,
    payouts: tuple[float, ...],
) -> Observation:
    local = state.actor_index
    seats = [Seat(i, s, 0, 0, True) for i, s in enumerate(stacks)]
    for i, global_id in enumerate(mapping):
        seats[global_id] = Seat(
            global_id,
            state.stacks[i],
            initial[i] - state.stacks[i],
            state.bets[i],
            not state.statuses[i],
        )
    legal = LegalActions(
        state.checking_or_calling_amount,
        state.min_completion_betting_or_raising_to_amount,
        state.max_completion_betting_or_raising_to_amount,
    )
    return Observation(
        hand,
        hero,
        button,
        state.street_index,
        tuple(CARD_IDS[c] for c in state.hole_cards[local]),
        tuple(CARD_IDS[c] for c in state.get_board_cards(0)),
        tuple(seats),
        legal,
        bb,
        payouts,
        sum(s.committed for s in seats),
    )


def _agent(
    style: str, seed: int, parameters: Parameters, seats: int
) -> TournamentPolicy | Baseline:
    if style == "hero":
        return TournamentPolicy(seed, parameters, seats)
    if style == "selfplay":
        return TournamentPolicy(seed, Parameters(), seats)
    return Baseline(style, seed)


def play(
    seed: int,
    hero: int,
    field: str,
    parameters: Parameters,
    config: Configuration,
    *,
    hero_agent: Player | None = None,
    opponent_agents: tuple[Player | None, ...] | None = None,
) -> dict:
    config.validate()
    n = len(config.stacks)
    if not 0 <= hero < n:
        raise ValueError("hero must be an occupied seat")
    if opponent_agents is not None:
        if len(opponent_agents) != n:
            raise ValueError("opponent roster must have one entry per seat")
        if opponent_agents[hero] is not None:
            raise ValueError("hero roster entry must be None; use hero_agent")
    if field not in {"mixed", "calling", "tight", "maniac", "random", "selfplay"}:
        raise ValueError(f"unknown field: {field}")
    styles = (
        "calling",
        "tight",
        "maniac",
        "random",
        "tight",
        "selfplay",
        "calling",
        "selfplay",
    )
    agents: list[Player] = [
        _agent(
            "hero"
            if i == hero
            else styles[(i - hero - 1) % len(styles)]
            if field == "mixed"
            else field,
            seed * 997 + i * 7919,
            parameters,
            n,
        )
        for i in range(n)
    ]
    if hero_agent is not None:
        agents[hero] = hero_agent
    if opponent_agents is not None:
        for seat, opponent in enumerate(opponent_agents):
            if opponent is not None:
                agents[seat] = opponent
    stacks = list(config.stacks)
    points = [0.0] * n
    button = -1
    times = []
    overruns = 0
    decisions = 0
    hands_played = 0
    for hand in range(config.max_hands):
        live = [i for i, stack in enumerate(stacks) if stack > 0]
        if len(live) < 2:
            break
        button = next(
            (button + offset) % n
            for offset in range(1, n + 1)
            if stacks[(button + offset) % n] > 0
        )
        rotation = live[live.index(button) :] + live[: live.index(button)]
        mapping = tuple(rotation[1:] + rotation[:1])
        initial = tuple(stacks[i] for i in mapping)
        bb = config.blinds[min(hand // config.hands_per_level, len(config.blinds) - 1)]
        state = NoLimitTexasHoldem.create_state(
            AUTOMATIONS, True, 0, (max(1, bb // 2), bb), bb, initial, len(live)
        )
        deck = list(DECK)
        Random(seed * 104729 + hand * 13007).shuffle(deck)
        state.deck_cards = deque(deck)
        while state.can_deal_hole():
            state.deal_hole()
        steps = 0
        while state.status:
            steps += 1
            if steps > 1000:
                raise RuntimeError("hand exceeded 1000 operations")
            if state.actor_index is None:
                if state.can_deal_board():
                    state.deal_board()
                    continue
                raise RuntimeError("unhandled simulator operation")
            actor = mapping[state.actor_index]
            obs = observation(
                state, mapping, initial, stacks, actor, button, hand, bb, config.payouts
            )
            started = perf_counter_ns()
            action = agents[actor].act(obs)
            elapsed = perf_counter_ns() - started
            if not obs.legal.accepts(action):
                raise RuntimeError(f"illegal {action} at hand {hand}: {obs.legal}")
            if actor == hero:
                times.append(elapsed)
                decisions += 1
                if elapsed > 1_000_000:
                    overruns += 1
            if elapsed > 1_000_000 and config.enforce_deadline:
                action = obs.legal.passive()
            event = Event(hand, actor, obs.street, action, obs.legal.call > 0, bb)
            if action.kind == "fold":
                state.fold()
            elif action.kind == "raise":
                state.complete_bet_or_raise_to(action.amount)
            else:
                state.check_or_call()
            for agent in agents:
                agent.observe(event)
        before = stacks.copy()
        hands_played += 1
        for i, global_id in enumerate(mapping):
            stacks[global_id] = state.stacks[i]
        if sum(stacks) != sum(config.stacks) or any(s < 0 for s in stacks):
            raise RuntimeError("chip conservation failed")
        busted = sorted((i for i in live if stacks[i] == 0), key=lambda i: before[i])
        remaining = len(live)
        for _, group in groupby(busted, key=lambda i: before[i]):
            tied = list(group)
            payout = mean(config.payouts[remaining - len(tied) : remaining])
            for i in tied:
                points[i] = payout
            remaining -= len(tied)
    live = [i for i, stack in enumerate(stacks) if stack > 0]
    for index, global_id in enumerate(live):
        points[global_id] = hero_value(
            tuple(stacks[i] for i in live), config.payouts[: len(live)], index
        )
    if abs(sum(points) - sum(config.payouts)) > 1e-7:
        raise RuntimeError("tournament payout conservation failed")
    return {
        "seed": seed,
        "hero_seat": hero,
        "field": field,
        "points": points[hero],
        "all_points": points,
        "final_stacks": stacks,
        "hands": hands_played,
        "decisions": decisions,
        "overruns": overruns,
        "times_ns": times,
    }


def _job(arguments: tuple[int, int, str, Parameters, Configuration]) -> dict:
    return play(*arguments)


def summary(results: list[dict]) -> dict:
    points = [r["points"] for r in results]
    timings = sorted(t for r in results for t in r["times_ns"])
    intervals = 1.96 * stdev(points) / sqrt(len(points)) if len(points) > 1 else None

    def percentile(p: float) -> float:
        return (
            timings[min(len(timings) - 1, round(p * (len(timings) - 1)))] / 1000
            if timings
            else 0
        )

    return {
        "tables": len(results),
        "mean_points": mean(points),
        "95pct_interval": [mean(points) - intervals, mean(points) + intervals]
        if intervals is not None
        else None,
        "decisions": len(timings),
        "overruns": sum(r["overruns"] for r in results),
        "latency_us": {
            "p50": percentile(0.5),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(timings, default=0) / 1000,
        },
        "mean_hands": mean(r["hands"] for r in results),
    }


def compare(
    variants: dict[str, Parameters],
    fields: tuple[str, ...],
    tables: int,
    seed: int,
    workers: int,
    config: Configuration,
) -> dict:
    if tables < 2 or workers < 1:
        raise ValueError("at least two tables and one worker are required")
    jobs = [
        (seed + t, t % len(config.stacks), field, parameters, config)
        for field in fields
        for parameters in variants.values()
        for t in range(tables)
    ]
    if workers == 1:
        results = list(map(_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_job, jobs, chunksize=2))
    output = {
        "practice_only": True,
        "variant": "No-Limit Texas Hold'em",
        "parameters": {name: asdict(p) for name, p in variants.items()},
        "configuration": asdict(config),
        "seed": seed,
        "paired_design": "Shared per-hand deck seeds and rotating hero seats; differing actions and eliminations can change subsequent card assignments. Table-level paired intervals are descriptive and do not establish performance against actual entrants.",
        "fields": {},
    }
    offset = 0
    names = list(variants)
    for field in fields:
        batches = {}
        for name in names:
            batch = results[offset : offset + tables]
            offset += tables
            batches[name] = batch
        records = {name: summary(batch) for name, batch in batches.items()}
        for index, name in enumerate(names):
            for reference in names[:index]:
                differences = [
                    a["points"] - b["points"]
                    for a, b in zip(batches[name], batches[reference])
                ]
                difference = mean(differences)
                half_width = 1.96 * stdev(differences) / sqrt(tables)
                records[name]["paired_delta_vs_" + reference] = {
                    "mean": difference,
                    "95pct_interval": [
                        difference - half_width,
                        difference + half_width,
                    ],
                }
        output["fields"][field] = records
    return output
