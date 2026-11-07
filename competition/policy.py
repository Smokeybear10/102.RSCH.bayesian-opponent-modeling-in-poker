from dataclasses import dataclass
from random import Random
from time import perf_counter_ns

from .cards import starting_strength
from .equity import Equity
from .icm import hero_value
from .model import Action, Event, Observation, pots, settle_pots
from .opponents import Opponents


@dataclass(frozen=True, slots=True)
class Parameters:
    open_scale: float = 1.0
    call_edge: float = 0.025
    value_edge: float = 0.14
    bet_fraction: float = 0.6
    bluff_frequency: float = 0.08
    samples: int = 40
    equity_budget_us: int = 400
    use_icm: bool = True
    adapt: bool = True
    learn_continuation: bool = True


def _all_in_values(obs: Observation) -> tuple[float, float, float, float]:
    opponent = obs.opponents[0]
    live = tuple(s.id for s in obs.seats if s.stack + s.committed > 0)
    payouts = obs.payouts[: len(live)]
    hero_index = live.index(obs.hero)
    contributions = tuple(
        s.committed + (obs.legal.call if s.id == obs.hero else 0) for s in obs.seats
    )
    layers = pots(contributions, tuple(s.folded for s in obs.seats))
    base = tuple(
        s.stack - (obs.legal.call if s.id == obs.hero else 0) for s in obs.seats
    )
    values = []
    for own, other in ((1, 0), (0, 1), (1, 1)):
        ranks = tuple(
            own if s.id == obs.hero else other if s.id == opponent.id else -1
            for s in obs.seats
        )
        awards = settle_pots(layers, ranks)
        values.append(
            hero_value(tuple(base[i] + awards[i] for i in live), payouts, hero_index)
        )
    folded = tuple(s.folded or s.id == obs.hero for s in obs.seats)
    layers = pots(tuple(s.committed for s in obs.seats), folded)
    awards = settle_pots(
        layers, tuple(1 if s.id == opponent.id else -1 for s in obs.seats)
    )
    fold = hero_value(
        tuple(obs.seats[i].stack + awards[i] for i in live), payouts, hero_index
    )
    return fold, *values


class TournamentPolicy:
    def __init__(
        self,
        seed: int = 0,
        parameters: Parameters | None = None,
        seats: int = 8,
        equity: Equity | None = None,
    ) -> None:
        self.parameters = parameters or Parameters()
        self.rng = Random(seed)
        if equity is None:
            from .holdem import HoldemEquity

            equity = HoldemEquity(seed ^ 0x73B9)
        self.equity = equity
        self.opponents = Opponents(seats, self.parameters.learn_continuation)
        self.last_samples = 0

    def observe(self, event: Event) -> None:
        self.opponents.observe(event)

    def _opening(self, obs: Observation) -> Action:
        player = obs.player
        live = [s.id for s in obs.seats if s.stack + s.committed > 0]
        button = live.index(obs.button)
        rotated = live[button:] + live[:button]
        order = rotated if len(live) == 2 else rotated[3:] + rotated[:3]
        after = order[order.index(obs.hero) + 1 :]
        behind = sum(not obs.seats[i].folded and obs.seats[i].stack > 0 for i in after)
        limpers = sum(
            s.id != obs.hero and not s.folded and s.bet == obs.big_blind
            for s in obs.seats
        )
        limpers = max(0, limpers - 1)
        fraction = min(
            0.6,
            max(
                0.1,
                (0.15 + 0.055 * (7 - behind) - 0.035 * limpers)
                * self.parameters.open_scale,
            ),
        )
        strength = starting_strength(obs.hole)
        if strength < 1 - fraction:
            return obs.legal.passive()
        if obs.legal.min_raise_to is None:
            return Action("call" if obs.legal.call else "check")
        effective = min(
            player.stack + player.bet, max(s.stack + s.bet for s in obs.opponents)
        )
        if effective <= 10 * obs.big_blind:
            return obs.legal.raise_to(obs.legal.max_raise_to)
        return obs.legal.raise_to(round((2.3 + limpers) * obs.big_blind))

    def act(self, obs: Observation, *, deadline_ns: int | None = None) -> Action:
        started = perf_counter_ns()
        self.last_samples = 0
        self.opponents.begin_hand(obs.hand)
        self.opponents.begin_street(obs.street)
        if obs.street == 0 and max(s.bet for s in obs.seats) <= obs.big_blind:
            return self._opening(obs)
        config = self.parameters
        ranges = (
            tuple(self.opponents.ranges) if config.adapt else (1.0,) * len(obs.seats)
        )
        pressure = (
            tuple(self.opponents.pressure) if config.adapt else (0,) * len(obs.seats)
        )
        equity_deadline = started + config.equity_budget_us * 1000
        if deadline_ns is not None:
            equity_deadline = min(equity_deadline, deadline_ns)
        estimate = self.equity.estimate(
            obs,
            ranges,
            pressure,
            config.samples,
            equity_deadline,
            self.opponents.information() if config.adapt else None,
        )
        self.last_samples = estimate.samples
        if estimate.samples < 4:
            return obs.legal.passive()
        player = obs.player
        opponents = obs.opponents
        call = obs.legal.call
        terminal_call = (
            call
            and len(opponents) == 1
            and (opponents[0].stack == 0 or call == player.stack)
        )
        if terminal_call and not config.use_icm:
            return Action("call") if estimate.call_return >= call else Action("fold")
        if config.use_icm and terminal_call:
            fold, win, lose, tie = _all_in_values(obs)
            expected = (
                estimate.win * win
                + estimate.tie * tie
                + (1 - estimate.win - estimate.tie) * lose
            )
            return Action("call") if expected >= fold else Action("fold")
        # A showdown estimate omits future betting; discount deep, unfinished hands.
        realization = (
            1.0
            if obs.street == 3
            else 0.96
            if player.stack <= 10 * obs.big_blind
            else 0.86
        )
        if call:
            net = estimate.call_return * realization - call
            margin = config.call_edge * (obs.pot + call)
            if estimate.effective_samples < 8:
                margin += 0.04 * (obs.pot + call)
            if net < margin:
                return Action("fold")
        if obs.legal.min_raise_to is None:
            return Action("call" if call else "check")
        contenders = len(opponents) + 1
        threshold = max(0.4, 1 / contenders + config.value_edge)
        confidence = estimate.share - 0.35 * estimate.standard_error
        if confidence > threshold:
            amount = max(obs.big_blind, round((obs.pot + call) * config.bet_fraction))
            return obs.legal.raise_to(player.bet + call + amount)
        if not call and len(opponents) == 1 and opponents[0].stack > 0 and config.adapt:
            tendency = self.opponents.seats[opponents[0].id]
            if (
                tendency.postflop_faced >= 16
                and tendency.fold_rate > 0.6
                and self.rng.random() < config.bluff_frequency
            ):
                return obs.legal.raise_to(
                    player.bet + max(obs.big_blind, round(obs.pot * 0.4))
                )
        return Action("call" if call else "check")


class Baseline:
    def __init__(self, style: str, seed: int = 0) -> None:
        if style not in {"calling", "tight", "maniac", "random"}:
            raise ValueError(f"unknown baseline: {style}")
        self.style = style
        self.rng = Random(seed)

    def observe(self, event: Event) -> None:
        pass

    def act(self, obs: Observation) -> Action:
        if self.style == "calling":
            return Action("call" if obs.legal.call else "check")
        strength = starting_strength(obs.hole)
        if self.style == "tight":
            if obs.street == 0 and strength < 0.86:
                return obs.legal.passive()
            if obs.legal.min_raise_to is not None and strength > 0.96:
                return obs.legal.raise_to(max(3 * obs.big_blind, round(obs.pot * 0.8)))
            if obs.legal.call > obs.pot / 3 and strength < 0.96:
                return Action("fold")
            return Action("call" if obs.legal.call else "check")
        chance = self.rng.random()
        if obs.legal.min_raise_to is not None and chance < (
            0.65 if self.style == "maniac" else 0.2
        ):
            if self.style == "maniac" and chance < 0.08:
                return obs.legal.raise_to(obs.legal.max_raise_to)
            return obs.legal.raise_to(
                obs.player.bet + obs.legal.call + max(obs.big_blind, obs.pot // 2)
            )
        if obs.legal.call and chance > (0.9 if self.style == "maniac" else 0.6):
            return Action("fold")
        return Action("call" if obs.legal.call else "check")
