"""Example SDK mapping for integration rehearsals. This is not the event interface."""

from competition.bridge import ContractError, Frame, PublicEvent, encode_raise
from competition.cards import CARD_INDEX
from competition.model import Action, Event, LegalActions, Observation, Seat

STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


class DemoAdapter:
    variant = "nlhe-standard-52"
    has_history = True

    def decode(self, raw: object) -> Frame:
        if not isinstance(raw, dict):
            raise ContractError("the rehearsal SDK uses dictionary observations")
        seats = tuple(
            Seat(
                p["seat"],
                p["remaining_chips"],
                p["hand_contribution"],
                p["round_contribution"],
                p["folded"],
            )
            for p in raw["players"]
        )
        hero = raw["acting_player"]
        mode = raw["legal"]["raise_amounts"]
        if mode not in {"to", "increment", "additional"}:
            raise ContractError("declare the SDK's raise convention explicitly")
        base = (
            0
            if mode == "to"
            else max(s.bet for s in seats)
            if mode == "increment"
            else seats[hero].bet
        )
        minimum, maximum = raw["legal"]["min_raise"], raw["legal"]["max_raise"]
        obs = Observation(
            raw["hand_index"],
            hero,
            raw["button_seat"],
            STREETS[raw["street"]],
            tuple(CARD_INDEX[c] for c in raw["hole_cards"]),
            tuple(CARD_INDEX[c] for c in raw["board"]),
            seats,
            LegalActions(
                raw["legal"]["call_cost"],
                None if minimum is None else minimum + base,
                None if maximum is None else maximum + base,
            ),
            raw["big_blind"],
            tuple(raw["payouts"]),
            sum(s.committed for s in seats),
        )
        events = tuple(
            PublicEvent(
                e["sequence"],
                Event(
                    e["hand_index"],
                    e["seat"],
                    STREETS[e["street"]],
                    Action(e["type"], e.get("raise_to", 0)),
                    e["facing_bet"],
                    raw["big_blind"],
                ),
            )
            for e in raw["history"]
        )
        return Frame(raw["table_id"], obs, events)

    def encode(self, action: Action, raw: object, frame: Frame) -> dict:
        result = {"type": action.kind}
        if action.kind == "raise":
            result["amount"] = encode_raise(
                action.amount, frame.observation, raw["legal"]["raise_amounts"]
            )
        return result
