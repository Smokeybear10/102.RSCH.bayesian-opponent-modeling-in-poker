"""Synthetic SDK fixtures for checking integration, not poker strength."""

from competition.equity import Estimate
from competition.model import Observation


class SDKEquity:
    def estimate(
        self,
        obs: Observation,
        ranges: tuple[float, ...],
        pressure: tuple[int, ...],
        samples: int = 40,
        deadline: int | None = None,
        information: tuple[float, ...] | None = None,
    ) -> Estimate:
        return Estimate(0.8, 0.8 * (obs.pot + obs.legal.call), 0.8, 0, 0.04, 40, 40)


def fixture(mode: str = "to", table: str = "table-1") -> dict:
    base = {"to": 0, "increment": 60, "additional": 20}[mode]
    return {
        "table_id": table,
        "hand_index": 0,
        "acting_player": 0,
        "button_seat": 1,
        "street": "river",
        "hole_cards": ["As", "Qh"],
        "board": ["Ah", "Kd", "7s", "4c", "2d"],
        "players": [
            {
                "seat": 0,
                "remaining_chips": 1800,
                "hand_contribution": 200,
                "round_contribution": 20,
                "folded": False,
            },
            {
                "seat": 1,
                "remaining_chips": 1760,
                "hand_contribution": 240,
                "round_contribution": 60,
                "folded": False,
            },
        ],
        "legal": {
            "call_cost": 40,
            "min_raise": 100 - base,
            "max_raise": 1820 - base,
            "raise_amounts": mode,
        },
        "big_blind": 20,
        "payouts": [100, 0],
        "history": [
            {
                "sequence": 0,
                "hand_index": 0,
                "seat": 1,
                "street": "river",
                "type": "raise",
                "raise_to": 60,
                "facing_bet": False,
            }
        ],
    }


def legal_output(raw: dict, result: object) -> bool:
    if not isinstance(result, dict) or result.get("type") not in {
        "fold",
        "check",
        "call",
        "raise",
    }:
        return False
    kind = result["type"]
    if kind == "raise":
        amount = result.get("amount")
        return (
            type(amount) is int
            and raw["legal"]["min_raise"] <= amount <= raw["legal"]["max_raise"]
        )
    return (
        raw["legal"]["call_cost"] == 0
        if kind == "check"
        else raw["legal"]["call_cost"] > 0
    )
