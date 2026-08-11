from collections import deque
from dataclasses import asdict
from random import Random

from pokerkit import Automation, Card, Mode, NoLimitTexasHoldem

from .models import Action, Event, GameConfig, HandResult, Observation


CARDS = tuple(rank + suit for rank in "23456789TJQKA" for suit in "cdhs")
STREETS = ("preflop", "flop", "turn", "river")


class Hand:
    def __init__(self, config: GameConfig, seed: int = 0, *, deck: tuple[str, ...] | None = None) -> None:
        self.config = config
        if deck is None:
            shuffled = list(CARDS)
            Random(seed).shuffle(shuffled)
            deck = tuple(shuffled)
        self._deck = tuple(deck)
        if len(self._deck) != 52 or any(card not in CARDS for card in self._deck) or len(set(self._deck)) != 52:
            raise ValueError("deck must contain each of the 52 standard cards exactly once")
        self._cursor = 0
        self._holes = ((self._deck[0], self._deck[2]), (self._deck[1], self._deck[3]))
        self._history: list[Event] = []
        self._contributions = [config.big_blind, config.small_blind]
        self._result: HandResult | None = None
        self._state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            True, 0, (config.small_blind, config.big_blind),
            config.big_blind, config.starting_stack, 2, mode=Mode.CASH_GAME,
        )
        self._state.deck_cards = deque(Card.parse("".join(self._deck)))
        self._advance()

    @property
    def finished(self) -> bool:
        return not self._state.status

    @property
    def public_history(self) -> tuple[Event, ...]:
        return tuple(self._history)

    def _take(self, count: int) -> str:
        cards = self._deck[self._cursor:self._cursor + count]
        self._cursor += count
        return "".join(cards)

    def _advance(self) -> None:
        state = self._state
        while state.status and state.actor_index is None:
            if state.card_burning_status:
                state.burn_card(self._take(1))
            elif state.hole_dealee_index is not None:
                state.deal_hole(self._take(1))
            elif state.board_dealing_count is not None:
                state.deal_board(self._take(state.board_dealing_count))
            elif state.can_select_runout_count():
                state.select_runout_count(1)
            elif state.can_show_or_muck_hole_cards():
                state.show_or_muck_hole_cards(True)
            else:
                raise RuntimeError("engine stopped without a betting or dealing action")

    def _board(self) -> tuple[str, ...]:
        return tuple(repr(card) for board_slot in self._state.board_cards for card in board_slot)

    def observation(self) -> Observation:
        if self.finished:
            raise ValueError("the hand is finished")
        state = self._state
        actor = state.actor_index
        if actor is None or state.street_index is None:
            raise RuntimeError("engine has no acting player")
        can_raise = state.can_complete_bet_or_raise_to()
        to_call = state.checking_or_calling_amount
        if to_call is None:
            raise RuntimeError("engine has no check or call action")
        return Observation(
            actor=actor, hole=self._holes[actor], board=self._board(),
            street=STREETS[state.street_index], button=1,
            stacks=tuple(state.stacks), bets=tuple(state.bets),
            pot=state.total_pot_amount, to_call=to_call,
            min_raise_to=state.min_completion_betting_or_raising_to_amount if can_raise else None,
            max_raise_to=state.max_completion_betting_or_raising_to_amount if can_raise else None,
            can_fold=to_call > 0 and state.can_fold(), history=tuple(self._history),
            big_blind=self.config.big_blind,
            rake_bps=self.config.rake_bps, rake_cap=self.config.rake_cap,
            no_flop_no_drop=self.config.no_flop_no_drop,
        )

    def act(self, action: Action) -> None:
        obs = self.observation()
        state = self._state
        if action.kind == "fold":
            if not obs.can_fold:
                raise ValueError("fold is unavailable when checking is free")
            paid = 0
            state.fold()
        elif action.kind in ("check", "call"):
            if (action.kind == "check") != (obs.to_call == 0):
                raise ValueError("check requires zero call cost; call requires positive call cost")
            paid = obs.to_call
            state.check_or_call()
        elif action.kind == "raise_to":
            if not state.can_complete_bet_or_raise_to(action.amount):
                raise ValueError(f"illegal raise_to amount: {action.amount}")
            paid = action.amount - obs.bets[obs.actor]
            state.complete_bet_or_raise_to(action.amount)
        else:
            raise ValueError(f"unknown action: {action.kind}")
        self._contributions[obs.actor] += paid
        self._history.append(Event(obs.actor, obs.street, action, obs.pot, obs.to_call, paid, obs.board))
        self._advance()

    def result(self) -> HandResult:
        if not self.finished:
            raise ValueError("the hand has not finished")
        if self._result is not None:
            return self._result
        starting = self.config.starting_stack
        matched = min(self._contributions)
        contested = 2 * matched
        refunded_stacks = (starting - matched, starting - matched)
        gross = tuple(self._state.stacks)
        winners = tuple(seat for seat in range(2) if gross[seat] > refunded_stacks[seat])
        if not winners or sum(gross) != 2 * starting:
            raise RuntimeError("native settlement did not conserve chips")
        share, odd = divmod(contested, len(winners))
        expected = list(refunded_stacks)
        for index, seat in enumerate(winners):
            expected[seat] += share + (index < odd)
        if tuple(expected) != gross:
            raise RuntimeError("native settlement disagrees with matched-pot accounting")
        board = self._board()
        rake = 0 if self.config.no_flop_no_drop and len(board) < 3 else min(
            self.config.rake_cap, contested * self.config.rake_bps // 10000,
        )
        share, odd = divmod(contested - rake, len(winners))
        net = list(refunded_stacks)
        for index, seat in enumerate(winners):
            net[seat] += share + (index < odd)
        if sum(net) + rake != 2 * starting:
            raise RuntimeError("raked settlement did not conserve chips")
        self._result = HandResult(
            gross_stacks=gross, net_stacks=tuple(net),
            gross_returns=tuple(stack - starting for stack in gross),
            net_returns=tuple(stack - starting for stack in net),
            rake=rake, board=board, winners=winners, contested_pot=contested,
        )
        return self._result

    def record(self) -> dict:
        return {
            "schema_version": 1, "config": asdict(self.config), "deck": self._deck,
            "actions": [asdict(event.action) for event in self._history],
            "result": asdict(self.result()),
        }

    def snapshot(self) -> dict:
        return {
            "schema_version": 1, "config": asdict(self.config), "deck": self._deck,
            "actions": [asdict(event.action) for event in self._history],
            "status": "finished" if self.finished else "in_progress",
            "cursor": self._cursor, "board": self._board(),
            "stacks": tuple(self._state.stacks), "bets": tuple(self._state.bets),
            "gross_stacks": tuple(self._state.stacks) if self.finished else None,
            "contributions": tuple(self._contributions),
            "result": asdict(self._result) if self._result is not None else None,
        }
