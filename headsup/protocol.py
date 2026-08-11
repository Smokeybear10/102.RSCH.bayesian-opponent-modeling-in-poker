from dataclasses import asdict
from ipaddress import ip_address
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .models import Action, Event, GameConfig, HandResult, Observation

MAX_HANDS = 1000
MAX_EVENTS = 4096
MAX_STACK_BB = 1000
MAX_CHIPS = 1_000_000_000
Version = Annotated[int, Field(ge=1, le=1)]
Seat = Annotated[int, Field(ge=0, le=1)]
Count = Annotated[int, Field(ge=0, le=10_000_000)]
Chips = Annotated[int, Field(ge=0, le=2 * MAX_CHIPS)]
Identifier = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$")]
CardCode = Annotated[str, Field(pattern=r"^[2-9TJQKA][cdhs]$")]
Street = Literal["preflop", "flop", "turn", "river"]
STREET_LENGTHS = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


def checked_url(value: str) -> str:
    parts = urlsplit(value)
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment or parts.path not in ("", "/")):
        raise ValueError("table URL must be an HTTP(S) origin without credentials, path, query or fragment")
    if parts.port is not None and not 1 <= parts.port <= 65535:
        raise ValueError("invalid table port")
    if parts.scheme == "http":
        try:
            loopback = ip_address(parts.hostname).is_loopback
        except ValueError:
            loopback = parts.hostname == "localhost"
        if not loopback:
            raise ValueError("remote connections require HTTPS; HTTP is limited to loopback")
    return value.rstrip("/")


class Credentials(WireModel):
    schema_version: Version = 1
    table_id: Identifier
    player_id: Seat
    url: str
    token: Annotated[str, Field(min_length=32, max_length=256, pattern=r"^[A-Za-z0-9_-]+$")] = Field(repr=False)

    _url = field_validator("url")(checked_url)


class WireConfig(WireModel):
    small_blind: Annotated[int, Field(gt=0, le=MAX_CHIPS)]
    big_blind: Annotated[int, Field(gt=0, le=MAX_CHIPS)]
    stack_bb: Annotated[int, Field(ge=1, le=MAX_STACK_BB)]
    rake_bps: Annotated[int, Field(ge=0, le=10000)]
    rake_cap: Chips
    no_flop_no_drop: bool

    @model_validator(mode="after")
    def validate_config(self) -> Self:
        config = self.to_domain()
        if config.starting_stack > MAX_CHIPS:
            raise ValueError("starting stack is too large")
        return self

    def to_domain(self) -> GameConfig:
        return GameConfig(**self.model_dump())


class WireAction(WireModel):
    kind: Literal["fold", "check", "call", "raise_to"]
    amount: Chips = 0

    @model_validator(mode="after")
    def validate_action(self) -> Self:
        self.to_domain()
        return self

    def to_domain(self) -> Action:
        return Action(self.kind, self.amount)


class WireEvent(WireModel):
    actor: Seat
    street: Street
    action: WireAction
    pot_before: Annotated[int, Field(gt=0, le=2 * MAX_CHIPS)]
    to_call_before: Chips
    paid: Chips
    board: tuple[CardCode, ...]

    @model_validator(mode="after")
    def validate_event(self) -> Self:
        if len(self.board) != STREET_LENGTHS[self.street] or len(set(self.board)) != len(self.board):
            raise ValueError("event board does not match its street")
        if self.action.kind in ("check", "fold") and self.paid != 0:
            raise ValueError("check and fold cannot pay chips")
        if self.action.kind == "check" and self.to_call_before:
            raise ValueError("check cannot face a bet")
        if self.action.kind in ("call", "fold") and not self.to_call_before:
            raise ValueError("call and fold require a bet")
        if self.action.kind == "call" and self.paid != self.to_call_before:
            raise ValueError("call payment must match the call cost")
        if self.action.kind == "raise_to" and self.paid <= self.to_call_before:
            raise ValueError("raise must exceed the call cost")
        return self

    def to_domain(self) -> Event:
        return Event(self.actor, self.street, self.action.to_domain(), self.pot_before,
                     self.to_call_before, self.paid, self.board)


History = Annotated[tuple[WireEvent, ...], Field(max_length=MAX_EVENTS)]


def validate_history(history: History, board: tuple[str, ...] | None = None) -> None:
    previous: tuple[str, ...] = ()
    folded = False
    for event in history:
        if folded or event.board[:len(previous)] != previous:
            raise ValueError("public history regresses or continues after a fold")
        if board is not None and board[:len(event.board)] != event.board:
            raise ValueError("event board disagrees with the current board")
        previous = event.board
        folded = event.action.kind == "fold"


class WireObservation(WireModel):
    actor: Seat
    hole: tuple[CardCode, CardCode]
    board: tuple[CardCode, ...]
    street: Street
    button: Annotated[int, Field(ge=1, le=1)]
    stacks: tuple[Chips, Chips]
    bets: tuple[Chips, Chips]
    pot: Annotated[int, Field(gt=0, le=2 * MAX_CHIPS)]
    to_call: Chips
    min_raise_to: Chips | None
    max_raise_to: Chips | None
    can_fold: bool
    history: History
    big_blind: Annotated[int, Field(gt=0, le=MAX_CHIPS)]
    rake_bps: Annotated[int, Field(ge=0, le=10000)]
    rake_cap: Chips
    no_flop_no_drop: bool

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if len(self.board) != STREET_LENGTHS[self.street]:
            raise ValueError("board does not match the betting street")
        if len(set(self.hole + self.board)) != len(self.hole + self.board):
            raise ValueError("duplicate visible cards")
        actor = self.actor
        expected = min(self.stacks[actor], max(self.bets) - self.bets[actor])
        if self.to_call != expected or self.can_fold != bool(self.to_call):
            raise ValueError("call/fold availability disagrees with bets and stacks")
        if self.pot < sum(self.bets) or not self.stacks[actor]:
            raise ValueError("invalid pot or acting stack")
        if (self.min_raise_to is None) != (self.max_raise_to is None):
            raise ValueError("raise bounds must both be present or absent")
        if self.min_raise_to is not None:
            if not (max(self.bets) < self.min_raise_to <= self.max_raise_to
                    <= self.bets[actor] + self.stacks[actor]):
                raise ValueError("invalid raise bounds")
        validate_history(self.history, self.board)
        return self

    def to_domain(self) -> Observation:
        data = self.model_dump(exclude={"history"})
        return Observation(**data, history=tuple(event.to_domain() for event in self.history))

    def accepts(self, action: WireAction) -> bool:
        if action.kind == "fold":
            return self.can_fold
        if action.kind == "check":
            return self.to_call == 0
        if action.kind == "call":
            return self.to_call > 0
        return (self.min_raise_to is not None and self.max_raise_to is not None
                and self.min_raise_to <= action.amount <= self.max_raise_to)


class WireResult(WireModel):
    gross_stacks: tuple[Chips, Chips]
    net_stacks: tuple[Chips, Chips]
    gross_returns: tuple[int, int]
    net_returns: tuple[int, int]
    rake: Chips
    board: tuple[CardCode, ...]
    winners: Annotated[tuple[Seat, ...], Field(min_length=1, max_length=2)]
    contested_pot: Chips


class TableState(WireModel):
    schema_version: Version = 1
    play_money: bool = True
    table_id: Identifier
    player_id: Seat
    opponent_id: Seat
    hero_seat: Seat
    hand_index: Annotated[int, Field(ge=0, lt=MAX_HANDS)]
    hands_total: Annotated[int, Field(ge=1, le=MAX_HANDS)]
    revision: Count
    phase: Literal["waiting", "playing", "hand_complete", "complete"]
    ready: tuple[bool, bool]
    config: WireConfig
    observation: WireObservation | None
    history: History
    result: WireResult | None

    @model_validator(mode="after")
    def validate_state(self) -> Self:
        if not self.play_money or self.opponent_id != 1 - self.player_id:
            raise ValueError("invalid play-money table identity")
        if self.hand_index >= self.hands_total or self.hero_seat != (self.player_id ^ (self.hand_index % 2)):
            raise ValueError("invalid hand index or player/position mapping")
        validate_history(self.history)
        completed = self.phase in ("hand_complete", "complete")
        if completed != (self.result is not None):
            raise ValueError("settlement must appear exactly when a hand is complete")
        if self.phase == "waiting" and (self.hand_index or self.history):
            raise ValueError("waiting is only valid before the first hand")
        if self.phase == "complete" and self.hand_index != self.hands_total - 1:
            raise ValueError("match completed too early")
        if self.observation is not None:
            obs = self.observation
            config = self.config
            if self.phase != "playing" or obs.actor != self.hero_seat or obs.history != self.history:
                raise ValueError("observation does not belong to this turn")
            if (obs.big_blind, obs.rake_bps, obs.rake_cap, obs.no_flop_no_drop) != (
                    config.big_blind, config.rake_bps, config.rake_cap, config.no_flop_no_drop):
                raise ValueError("observation game settings changed")
            if sum(obs.stacks) + obs.pot != 2 * config.to_domain().starting_stack:
                raise ValueError("live chips do not balance")
        if self.result is not None:
            result = self.result
            starting = self.config.to_domain().starting_stack
            if (sum(result.gross_stacks) != 2 * starting
                    or sum(result.net_stacks) + result.rake != 2 * starting
                    or result.gross_returns != tuple(n - starting for n in result.gross_stacks)
                    or result.net_returns != tuple(n - starting for n in result.net_stacks)):
                raise ValueError("settled chips do not balance")
            if len(result.board) not in (0, 3, 4, 5) or len(set(result.board)) != len(result.board):
                raise ValueError("invalid settled board")
            validate_history(self.history, result.board)
            contributions = [self.config.big_blind, self.config.small_blind]
            for event in self.history:
                contributions[event.actor] += event.paid
            if max(contributions) > starting or result.contested_pot != 2 * min(contributions):
                raise ValueError("contested pot disagrees with public payments")
            if tuple(sorted(set(result.winners))) != result.winners:
                raise ValueError("winners must be unique and ordered clockwise after the button")
            if self.history and self.history[-1].action.kind == "fold":
                if result.winners != (1 - self.history[-1].actor,):
                    raise ValueError("a folded player cannot win")
            elif len(result.board) != 5:
                raise ValueError("a showdown requires a complete board")
            expected_rake = (0 if self.config.no_flop_no_drop and len(result.board) < 3 else
                             min(self.config.rake_cap, result.contested_pot * self.config.rake_bps // 10000))
            if result.rake != expected_rake:
                raise ValueError("rake disagrees with the table rules")
            base = starting - result.contested_pot // 2
            for payout, actual in ((result.contested_pot, result.gross_stacks),
                                   (result.contested_pot - result.rake, result.net_stacks)):
                expected_stacks = [base, base]
                share, odd = divmod(payout, len(result.winners))
                for index, winner in enumerate(result.winners):
                    expected_stacks[winner] += share + (index < odd)
                if tuple(expected_stacks) != actual:
                    raise ValueError("settlement disagrees with winners, refunds and odd-chip rules")
        return self


class Command(WireModel):
    schema_version: Version = 1
    table_id: Identifier
    request_id: Identifier
    hand_index: Annotated[int, Field(ge=0, lt=MAX_HANDS)]
    revision: Count
    kind: Literal["ready", "act"]
    action: WireAction | None = None

    @model_validator(mode="after")
    def validate_command(self) -> Self:
        if (self.kind == "act") != (self.action is not None):
            raise ValueError("only act commands carry an action")
        return self


class Receipt(WireModel):
    schema_version: Version = 1
    table_id: Identifier
    player_id: Seat
    request_id: Identifier
    revision: Count


def wire_observation(observation: Observation) -> WireObservation:
    return WireObservation.model_validate(asdict(observation))


def wire_result(result: HandResult) -> WireResult:
    return WireResult.model_validate(asdict(result))
