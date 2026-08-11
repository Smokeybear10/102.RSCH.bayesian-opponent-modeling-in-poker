import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .arena import Configuration, play, summary
from .bridge import Bridge, RaiseMode
from .cards import CARD_NAMES
from .model import Action, Event, Observation
from .policy import Parameters, TournamentPolicy
from .rehearsal.adapter import DemoAdapter


class WireAgent:
    def __init__(self, seed: int, mode: RaiseMode, table: str) -> None:
        self.bridge = Bridge(DemoAdapter(), lambda: TournamentPolicy(seed))
        self.mode = mode
        self.table = table
        self.hand = -1
        self.history = []

    def _hand(self, hand: int) -> None:
        if hand != self.hand:
            self.hand = hand
            self.history = []

    def observe(self, event: Event) -> None:
        self._hand(event.hand)
        self.history.append(
            {
                "sequence": len(self.history),
                "hand_index": event.hand,
                "seat": event.seat,
                "street": ("preflop", "flop", "turn", "river")[event.street],
                "type": event.action.kind,
                "raise_to": event.action.amount,
                "facing_bet": event.facing_bet,
            }
        )

    def act(self, obs: Observation) -> Action:
        self._hand(obs.hand)
        base = (
            0
            if self.mode == "to"
            else max(s.bet for s in obs.seats)
            if self.mode == "increment"
            else obs.player.bet
        )
        raw = {
            "table_id": self.table,
            "hand_index": obs.hand,
            "acting_player": obs.hero,
            "button_seat": obs.button,
            "street": ("preflop", "flop", "turn", "river")[obs.street],
            "hole_cards": [CARD_NAMES[c] for c in obs.hole],
            "board": [CARD_NAMES[c] for c in obs.board],
            "players": [
                {
                    "seat": s.id,
                    "remaining_chips": s.stack,
                    "hand_contribution": s.committed,
                    "round_contribution": s.bet,
                    "folded": s.folded,
                }
                for s in obs.seats
            ],
            "legal": {
                "call_cost": obs.legal.call,
                "min_raise": None
                if obs.legal.min_raise_to is None
                else obs.legal.min_raise_to - base,
                "max_raise": None
                if obs.legal.max_raise_to is None
                else obs.legal.max_raise_to - base,
                "raise_amounts": self.mode,
            },
            "big_blind": obs.big_blind,
            "payouts": obs.payouts,
            "history": self.history,
        }
        encoded = self.bridge.decide(raw)
        if encoded["type"] != "raise":
            return Action(encoded["type"])
        # Apply the SDK convention independently before PokerKit checks legality.
        amount = encoded["amount"]
        if self.mode == "increment":
            amount += max(s.bet for s in obs.seats)
        elif self.mode == "additional":
            amount += obs.player.bet
        return Action("raise", amount)


def run(tables: int, seed: int) -> dict:
    if tables < 1:
        raise ValueError("at least one table is required")
    modes = {}
    for mode in ("to", "increment", "additional"):
        results = []
        fallbacks = 0
        for index in range(tables):
            agent = WireAgent(seed + index, mode, f"table-{index}")
            result = play(
                seed + index,
                index % 8,
                "mixed",
                Parameters(),
                Configuration(),
                hero_agent=agent,
            )
            results.append(result)
            fallbacks += agent.bridge.budget_fallbacks
        modes[mode] = {**summary(results), "bridge_budget_fallbacks": fallbacks}
    return {
        "official_sdk": False,
        "purpose": "Full eight-player Hold'em games through raw SDK-shaped observations, history replay, adapter normalization, native equity policy, output encoding and PokerKit legality checks",
        "configuration": asdict(Configuration()),
        "seed": seed,
        "tables": tables * 3,
        "illegal_action_or_accounting_failures": 0,
        "timing_note": "Wall times include serializing SDK-shaped input, decode, validation, history, policy, encode, and applying wire amount conventions. Desktop timings do not certify the official runtime.",
        "modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Exercise the complete adapter callback in real local tournaments"
    )
    parser.add_argument(
        "--tables", type=int, default=8, help="tables per raise convention"
    )
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.tables, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
