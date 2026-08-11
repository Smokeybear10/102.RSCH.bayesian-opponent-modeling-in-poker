"""Opt-in river action-value experiment, not a solved poker strategy.

The continuation model is a hand-strength/size heuristic, shrunk towards a
minimum-defense-style prior. It does not model raises. Only a heads-up river
check that closes betting is compared with bets; other spots use the original
policy. Values are chip EV, not tournament payout EV. This must beat held-out
opponents and meet real-clock deadlines before being promoted.
"""

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from math import sqrt
from time import perf_counter_ns

from .cards import RANGES
from .model import Action, Observation
from .policy import TournamentPolicy


@dataclass(frozen=True, slots=True)
class RiverDiagnostics:
    eligible: bool = False
    used: bool = False
    reason: str = "not_run"
    samples: int = 0
    check_ev: float | None = None
    expected_ev: float | None = None
    bet_to: int = 0
    # (raise-to amount, modeled expected chips from the current decision)
    action_evs: tuple[tuple[int, float], ...] = ()
    raises_modeled: bool = False


def terminal_bet_value(share: float, pot: int, bet: int, call_probability: float) -> float:
    """Net chips relative to current stack under a fold/call-only response.

    The already committed pot is sunk. A called win earns pot + bet, a loss
    loses bet, and a tie earns half the existing pot. An opponent fold earns
    the existing pot. ``bet`` must be the matched, effective additional bet.
    """
    if not 0 <= share <= 1 or not 0 <= call_probability <= 1 or pot < 0 or bet < 0:
        raise ValueError("invalid showdown share, response probability, pot, or bet")
    return (1 - call_probability) * pot + call_probability * (
        share * (pot + 2 * bet) - bet
    )


class RiverPolicy(TournamentPolicy):
    """A separately selectable experimental policy with baseline constructor."""

    MIN_SAMPLES = 12
    FINISH_RESERVE_NS = 70_000

    def __init__(self, seed=0, parameters=None, seats=8, equity=None):
        super().__init__(seed=seed, parameters=parameters, seats=seats, equity=equity)
        self.last_river = RiverDiagnostics()
        self.river_eligible_spots = 0
        self.river_used_spots = 0
        self._river_known = -1
        self._river_bucket = -1
        self._river_choices = ()

    def _eligible(self, obs: Observation) -> bool:
        if obs.street != 3 or len(obs.board) != 5 or len(obs.opponents) != 1:
            return False
        other = obs.opponents[0]
        if obs.legal.call or obs.player.stack <= 0 or other.stack <= 0:
            return False
        if obs.player.committed != other.committed or obs.player.bet != other.bet:
            return False
        # A folded player's unmatched excess would make simple pot EV invalid.
        if any(s.committed > obs.player.committed for s in obs.seats):
            return False
        if obs.pot <= 0 or obs.pot != sum(s.committed for s in obs.seats):
            return False
        # Postflop action starts after the button. Checking must close betting.
        n = len(obs.seats)
        position = lambda seat: (seat - obs.button - 1) % n
        if position(obs.hero) < position(other.id):
            return False
        return True

    def _bet_sizes(self, obs: Observation) -> tuple[int, ...]:
        legal = obs.legal
        if legal.min_raise_to is None or legal.max_raise_to is None:
            return ()
        low = legal.min_raise_to - obs.player.bet
        high = min(
            legal.max_raise_to - obs.player.bet,
            obs.player.stack,
            obs.opponents[0].stack,
        )
        if low <= 0 or high < low:
            return ()
        return tuple(sorted({
            max(low, min(high, max(obs.big_blind, round(obs.pot * fraction))))
            for fraction in (1 / 3, 2 / 3, 1.0)
        }))

    def _call_probability(self, strength: float, bet: int, pot: int, seat: int) -> float:
        """Heuristic response, using public observations only.

        Strength is the sampled opponent hand's percentile on this board,
        independent of the hero's actual showdown rank. Larger bets select a
        stronger calling range. Aggregate fold data shifts the prior cautiously;
        it is not bet-size-specific training data or a learned action model.
        """
        threshold = bet / (pot + bet)
        if self.parameters.adapt:
            tendency = self.opponents.seats[seat]
            reliability = tendency.postflop_faced / (tendency.postflop_faced + 40)
            threshold += reliability * (tendency.fold_rate - 0.4)
        threshold = max(0.08, min(0.92, threshold))
        return max(0.0, min(1.0, 0.5 + (strength - threshold) / 0.24))

    def _fallback(self, reason: str, *, eligible: bool, samples: int = 0) -> Action:
        self.last_samples = samples
        self.last_river = RiverDiagnostics(eligible=eligible, reason=reason, samples=samples)
        return Action("check")

    def act(self, obs: Observation, *, deadline_ns: int | None = None) -> Action:
        started = perf_counter_ns()
        self.last_river = RiverDiagnostics(reason="ineligible")
        if deadline_ns is not None and started >= deadline_ns:
            self.last_samples = 0
            self.last_river = RiverDiagnostics(reason="expired_deadline")
            return obs.legal.passive()
        if not self._eligible(obs):
            return super().act(obs, deadline_ns=deadline_ns)
        cards = getattr(self.equity, "cards", ())
        evaluate = getattr(self.equity, "evaluate", None)
        if len(cards) != 52 or not callable(evaluate):
            self.last_river = RiverDiagnostics(reason="incompatible_evaluator")
            return super().act(obs, deadline_ns=deadline_ns)
        self.river_eligible_spots += 1
        self.opponents.begin_hand(obs.hand)
        self.opponents.begin_street(obs.street)
        self.last_samples = 0
        config = self.parameters
        finish_by = started + max(0, config.equity_budget_us) * 1000
        if deadline_ns is not None:
            finish_by = min(finish_by, deadline_ns)
        # Explicit opt-in for reproducible offline experiments. A real caller's
        # deadline always wins, even if a research wrapper is accidentally used.
        if deadline_ns is None and getattr(self.equity, "research_fixed_samples", False) is True:
            finish_by = None
        sample_until = None if finish_by is None else finish_by - self.FINISH_RESERVE_NS
        if sample_until is not None and perf_counter_ns() >= sample_until:
            return self._fallback("insufficient_time", eligible=True)
        bets = self._bet_sizes(obs)
        if not bets:
            return self._fallback("no_effective_legal_bet", eligible=True)
        other = obs.opponents[0]
        fraction = self.opponents.ranges[other.id] if config.adapt else 1.0
        bucket = min(20, max(2, round(fraction * 20)))
        known = sum(1 << card for card in obs.hole + obs.board)
        if known != self._river_known or bucket != self._river_bucket:
            self._river_choices = tuple(c for c in RANGES[bucket] if not c[2] & known)
            self._river_known, self._river_bucket = known, bucket
        available = self._river_choices
        if not available:
            return self._fallback("empty_range", eligible=True)
        board = [cards[c] for c in obs.board]
        own_rank = evaluate([cards[c] for c in obs.hole] + board)
        ranks = []
        for _ in range(max(0, config.samples)):
            if sample_until is not None and perf_counter_ns() >= sample_until:
                break
            a, b, _ = available[self.rng.randrange(len(available))]
            ranks.append(evaluate([cards[a], cards[b]] + board))
        completed = len(ranks)
        self.last_samples = completed
        if completed < self.MIN_SAMPLES:
            return self._fallback("insufficient_samples", eligible=True, samples=completed)
        ordered = sorted(ranks)
        shares = [float(own_rank > rank) + 0.5 * (own_rank == rank) for rank in ranks]
        check_ev = sum(shares) * obs.pot / completed
        # Mid-ranks prevent arbitrary tie breaking on shared-board hands.
        strengths = [
            (bisect_left(ordered, rank) + bisect_right(ordered, rank)) / (2 * completed)
            for rank in ranks
        ]
        best_score = 0.0
        selected = 0
        expected_ev = check_ev
        action_evs = []
        for bet in bets:
            if finish_by is not None and perf_counter_ns() >= finish_by:
                return self._fallback("computation_deadline", eligible=True, samples=completed)
            values = [
                terminal_bet_value(
                    share, obs.pot, bet,
                    self._call_probability(strength, bet, obs.pot, other.id),
                )
                for share, strength in zip(shares, strengths)
            ]
            value = sum(values) / completed
            action_evs.append((obs.player.bet + bet, value))
            deltas = [v - share * obs.pot for v, share in zip(values, shares)]
            improvement = value - check_ev
            variance = sum((d - improvement) ** 2 for d in deltas) / (completed - 1)
            uncertainty = 1.5 * sqrt(variance / completed)
            # This is a deliberately conservative heuristic model-risk penalty,
            # not a statistical guarantee against opponent raises/model error.
            model_margin = 0.025 * obs.pot + 0.05 * bet
            score = improvement - uncertainty - model_margin
            if score > best_score:
                best_score, selected, expected_ev = score, bet, value
        if finish_by is not None and perf_counter_ns() >= finish_by:
            return self._fallback("computation_deadline", eligible=True, samples=completed)
        self.river_used_spots += 1
        self.last_river = RiverDiagnostics(
            eligible=True, used=True, reason="modeled_fold_call",
            samples=completed, check_ev=check_ev, expected_ev=expected_ev,
            bet_to=obs.player.bet + selected if selected else 0,
            action_evs=tuple(action_evs),
        )
        return Action("raise", obs.player.bet + selected) if selected else Action("check")
