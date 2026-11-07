from dataclasses import asdict

from headsup.engine import CARDS, Hand
from headsup.models import Action, Event, GameConfig
from headsup.opponent_memory import OpponentIdentity
from headsup.protocol import History, WireEvent


def stacked_deck(hole0: tuple[str, str], hole1: tuple[str, str],
                 board: tuple[str, ...]) -> tuple[str, ...]:
    reserved = set(hole0 + hole1 + board)
    unused = iter(card for card in CARDS if card not in reserved)
    prefix = [hole0[0], hole1[0], hole0[1], hole1[1], next(unused),
              *board[:3], next(unused), board[3], next(unused), board[4]]
    return tuple(prefix + [card for card in CARDS if card not in prefix])


def finish_passively(hand: Hand) -> None:
    while not hand.finished:
        hand.act(Action("call" if hand.observation().to_call else "check"))


def response_hand(response: str = "fold", opponent: int = 0) -> Hand:
    hand = Hand(GameConfig(rake_cap=0), seed=91)
    if opponent == 1:
        hand.act(Action("call"))
    hand.act(Action("raise_to", 6))
    hand.act(Action("raise_to", 18) if response == "raise" else Action(response))
    if response == "raise":
        hand.act(Action("fold"))
    finish_passively(hand)
    return hand


def wire_history(history: tuple[Event, ...]) -> History:
    return tuple(WireEvent.model_validate(asdict(event)) for event in history)


def identity(opponent: str = "opponent_a", **overrides: object) -> OpponentIdentity:
    values = {"room_url": "https://www.pokernow.com/games/synthetic_test_room",
              "hero_id": "test_hero", "opponent_id": opponent,
              "small_blind": 1, "big_blind": 2}
    return OpponentIdentity.model_validate({**values, **overrides})


class Clock:
    def __init__(self) -> None:
        self.monotonic = 100.0
        self.wall = 1_800_000_000.0

    def advance(self, seconds: float) -> None:
        self.monotonic += seconds
        self.wall += seconds
