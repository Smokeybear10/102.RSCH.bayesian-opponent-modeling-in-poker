import hashlib
import json
import re
import tomllib
import unicodedata
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pokerkit import HandHistory, parse_action

from .models import Action, Event

STREETS = ("preflop", "flop", "turn", "river")
CARD = re.compile(r"(?:[2-9TJQKA][cdhs]|\?\?)")
AMOUNT = re.compile(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?")
ZONE_ALIASES = {"ET": "America/New_York", "CT": "America/Chicago",
                "MT": "America/Denver", "PT": "America/Los_Angeles"}
ZONE_OFFSETS = {"UTC": 0, "GMT": 0, "Z": 0, "EST": -5, "EDT": -4,
                "CST": -6, "CDT": -5, "MST": -7, "MDT": -6, "PST": -8, "PDT": -7}


@dataclass(frozen=True, slots=True)
class ImportedHand:
    key: str
    timestamp: str
    players: tuple[str, str]
    history: tuple[Event, ...]
    big_blind: int
    starting_stacks: tuple[int, int]
    source: str
    row: str


class _Excluded(ValueError):
    pass


def _text(value: object, reason: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise _Excluded(reason)
    return unicodedata.normalize("NFC", value.strip())


def _hash(parts: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(parts, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _identity(record: dict) -> tuple[str, str]:
    venue = _text(record.get("venue"), "missing_venue").casefold()
    hand = record.get("hand")
    if type(hand) is int:
        if hand < 0:
            raise _Excluded("missing_hand_identifier")
        hand = str(hand)
    hand = _text(hand, "missing_hand_identifier")
    return venue, _hash((venue, hand))


def _canonical(value: object) -> object:
    if isinstance(value, Decimal):
        sign, digits, exponent = value.as_tuple()
        if isinstance(exponent, int):
            while len(digits) > 1 and digits[-1] == 0:
                digits = digits[:-1]
                exponent += 1
        return {"decimal": [sign, digits, exponent]}
    if isinstance(value, (datetime, date, time)):
        return {"datetime": value.isoformat()}
    if isinstance(value, dict):
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


def _money(value: object) -> int:
    if type(value) is not int and not isinstance(value, Decimal):
        raise _Excluded("invalid_money")
    amount = Decimal(value)
    if not amount.is_finite() or amount < 0 or amount > 1_000_000_000:
        raise _Excluded("invalid_money")
    _, digits, exponent = amount.as_tuple()
    if len(digits) > 80 or abs(exponent) > 80:
        raise _Excluded("invalid_money")
    numerator, denominator = amount.as_integer_ratio()
    scaled, remainder = divmod(numerator * 100, denominator)
    if remainder:
        raise _Excluded("fractional_cent")
    return scaled


def _action_money(value: str) -> int:
    if not AMOUNT.fullmatch(value) or len(value) > 64:
        raise _Excluded("invalid_action_amount")
    try:
        return _money(Decimal(value))
    except InvalidOperation:
        raise _Excluded("invalid_action_amount") from None


def _timestamp(record: dict) -> str:
    if any(type(record.get(key)) is not int for key in ("year", "month", "day")):
        raise _Excluded("missing_timestamp")
    clock = record.get("time")
    if not isinstance(clock, time):
        raise _Excluded("missing_timestamp")
    try:
        local = datetime(record["year"], record["month"], record["day"], clock.hour,
                         clock.minute, clock.second, clock.microsecond)
    except ValueError:
        raise _Excluded("invalid_timestamp") from None
    zones = []
    for zone_name in (record.get("time_zone"), record.get("time_zone_abbreviation")):
        if zone_name is None:
            continue
        zone_name = _text(zone_name, "missing_time_zone")
        if zone_name in ZONE_OFFSETS:
            zones.append(timezone(timedelta(hours=ZONE_OFFSETS[zone_name])))
        else:
            try:
                zones.append(ZoneInfo(ZONE_ALIASES.get(zone_name, zone_name)))
            except (ZoneInfoNotFoundError, ValueError):
                raise _Excluded("unknown_time_zone") from None
    if clock.tzinfo is not None:
        zones.append(clock.tzinfo)
    if not zones:
        raise _Excluded("missing_time_zone")
    instants = set()
    for zone in zones:
        candidates = {aware.astimezone(timezone.utc) for fold in (0, 1)
                      if (aware := local.replace(tzinfo=zone, fold=fold)).astimezone(timezone.utc)
                      .astimezone(zone).replace(tzinfo=None) == local}
        if len(candidates) != 1:
            raise _Excluded("ambiguous_or_nonexistent_local_time")
        instants.update(candidates)
    if len(instants) != 1:
        raise _Excluded("conflicting_time_zone")
    instant = instants.pop()
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _cash_currency(record: dict) -> str:
    if record.get("level") is not None or record.get("event") is not None:
        raise _Excluded("tournament_or_ambiguous_event")
    for key in ("_game_type", "game_type", "_mode", "mode"):
        if key in record and str(record[key]).casefold() not in ("cash", "cash_game", "ring", "ring_game"):
            raise _Excluded("not_cash_game")
    currency = record.get("currency")
    symbol = record.get("currency_symbol")
    if currency is not None:
        currency = _text(currency, "missing_currency").upper()
        if not re.fullmatch(r"[A-Z]{3}", currency) or currency in ("XXX", "CHP"):
            raise _Excluded("unsupported_currency")
        return currency
    if symbol not in ("$", "€", "£", "¥", "₹", "₽", "₩", "¢"):
        raise _Excluded("missing_or_unsupported_cash_currency")
    return symbol


def _cards(value: str, *, public: bool) -> tuple[str, ...]:
    tokens = CARD.findall(value)
    if not tokens or "".join(tokens) != value or (public and "??" in tokens):
        raise _Excluded("unknown_or_invalid_public_cards" if public else "unsupported_hole_cards")
    known = tuple(token for token in tokens if token != "??")
    if len(set(known)) != len(known):
        raise _Excluded("duplicate_known_card")
    return known


def _public_history(record: dict, stacks: tuple[int, int], blinds: list[int], minimum: int) -> tuple[Event, ...]:
    actions = record.get("actions")
    if (not isinstance(actions, list) or not 1 <= len(actions) <= 4096
            or any(not isinstance(action, str) or len(action) > 4096 for action in actions)):
        raise _Excluded("invalid_actions")
    hand = HandHistory(variant="NT", ante_trimming_status=False, antes=[0, 0],
                       blinds_or_straddles=blinds, min_bet=minimum,
                       starting_stacks=list(stacks), actions=actions, parse_value=_action_money)
    events = []
    folded = False
    dealt = [0, 0]
    known_holes = [set(), set()]
    known_board = set()
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        try:
            state = hand.create_state()
            if tuple(state.bets) != (blinds[1], blinds[0]):
                raise _Excluded("unexpected_blind_positions")
            for raw_action in actions:
                action = raw_action.split("#", 1)[0].strip()
                if not action:
                    continue
                words = action.split()
                betting = len(words) >= 2 and words[0] in ("p1", "p2") and words[1] in ("f", "cc", "cbr")
                if words[:2] == ["d", "dh"] and len(words) == 4 and words[2] in ("p1", "p2"):
                    cards = set(_cards(words[3], public=False))
                    player = int(words[2][1]) - 1
                    if cards & (known_holes[0] | known_holes[1] | known_board):
                        raise _Excluded("duplicate_known_card")
                    known_holes[player].update(cards)
                    dealt[player] += len(words[3]) // 2
                    if dealt[player] > 2:
                        raise _Excluded("invalid_hole_dealing")
                elif words[:2] == ["d", "db"] and len(words) == 3:
                    cards = set(_cards(words[2], public=True))
                    if cards & (known_holes[0] | known_holes[1] | known_board):
                        raise _Excluded("duplicate_known_card")
                    known_board.update(cards)
                    if folded:
                        raise _Excluded("actions_after_fold")
                elif len(words) in (2, 3) and words[0] in ("p1", "p2") and words[1] == "sm":
                    if len(words) == 3 and words[2] != "-":
                        cards = set(_cards(words[2], public=False))
                        player = int(words[0][1]) - 1
                        if cards & (known_holes[1 - player] | known_board):
                            raise _Excluded("duplicate_known_card")
                        known_holes[player].update(cards)
                        if len(known_holes[player]) > 2:
                            raise _Excluded("conflicting_private_cards")
                elif not betting:
                    raise _Excluded("unsupported_action")
                if state.can_burn_card():
                    state.burn_card("??")
                event = None
                if betting:
                    if folded or dealt != [2, 2]:
                        raise _Excluded("actions_after_fold" if folded else "missing_explicit_hole_dealing")
                    actor = int(words[0][1]) - 1
                    if actor != state.actor_index or state.street_index not in range(4):
                        raise _Excluded("invalid_action_order")
                    board = tuple(map(repr, state.get_board_cards(0)))
                    to_call = state.checking_or_calling_amount
                    if to_call is None:
                        raise _Excluded("invalid_betting_state")
                    if words[1] == "cbr" and len(words) == 3:
                        amount = _action_money(words[2])
                        choice = Action("raise_to", amount)
                        paid = amount - state.bets[actor]
                    elif len(words) == 2 and words[1] == "cc":
                        choice = Action("call" if to_call else "check")
                        paid = to_call
                    elif len(words) == 2 and words[1] == "f" and to_call:
                        choice = Action("fold")
                        paid = 0
                    else:
                        raise _Excluded("invalid_betting_action")
                    event = Event(actor, STREETS[state.street_index], choice, state.total_pot_amount,
                                  to_call, paid, board)
                parse_action(state, action, hand.parse_value)
                if event is not None:
                    if event.paid < 0 or event.pot_before <= 0:
                        raise _Excluded("invalid_bet_accounting")
                    events.append(event)
                    folded = event.action.kind == "fold"
            board = tuple(map(repr, state.get_board_cards(0)))
            if not events or dealt != [2, 2] or state.actor_index is not None:
                raise _Excluded("incomplete_public_betting")
            if not folded and (len(board) != 5 or state.can_deal_board() or state.can_burn_card()):
                raise _Excluded("incomplete_public_betting")
            if len(board) not in (0, 3, 4, 5) or len(set(board)) != len(board):
                raise _Excluded("invalid_public_board")
        except _Excluded:
            raise
        except (ValueError, TypeError, IndexError, KeyError, AssertionError, UserWarning):
            raise _Excluded("illegal_or_inconsistent_history") from None
    return tuple(events)


def _import(record: dict, source: str, row: str, venue: str, key: str,
            min_stack_bb: Decimal, max_stack_bb: Decimal) -> ImportedHand:
    if record.get("variant") != "NT":
        raise _Excluded("unsupported_variant")
    raw_stacks = record.get("starting_stacks")
    if not isinstance(raw_stacks, list) or len(raw_stacks) != 2:
        raise _Excluded("not_two_active_players")
    if type(record.get("seat_count")) is not int or record["seat_count"] != 2:
        raise _Excluded("not_explicit_two_seat_table")
    seats = record.get("seats")
    if not isinstance(seats, list) or len(seats) != 2 or any(type(value) is not int for value in seats) or set(seats) != {1, 2}:
        raise _Excluded("ambiguous_seat_mapping")
    antes = record.get("antes")
    if not isinstance(antes, list) or len(antes) != 2 or any(_money(value) for value in antes):
        raise _Excluded("nonzero_or_invalid_antes")
    raw_blinds = record.get("blinds_or_straddles")
    if not isinstance(raw_blinds, list) or len(raw_blinds) != 2:
        raise _Excluded("invalid_blinds")
    blinds = [_money(value) for value in raw_blinds]
    if not 0 < blinds[0] <= blinds[1]:
        raise _Excluded("invalid_blinds_or_straddles")
    minimum = _money(record.get("min_bet"))
    if minimum != blinds[1]:
        raise _Excluded("unsupported_minimum_bet")
    stacks = tuple(_money(value) for value in raw_stacks)
    if min(stacks) <= blinds[1]:
        raise _Excluded("blind_all_in_or_inactive_player")
    if not min_stack_bb <= Decimal(min(stacks)) / blinds[1] <= max_stack_bb:
        raise _Excluded("effective_stack_outside_window")
    currency = _cash_currency(record)
    players = record.get("players")
    if not isinstance(players, list) or len(players) != 2:
        raise _Excluded("missing_player_identifiers")
    names = tuple(_text(player, "invalid_player_identifier") for player in players)
    if names[0] == names[1]:
        raise _Excluded("duplicate_player_identifier")
    timestamp = _timestamp(record)
    history = _public_history(record, stacks, blinds, minimum)
    return ImportedHand(key, timestamp, tuple(_hash((venue, currency, player)) for player in names),
                        history, blinds[1], stacks, source, row)


def import_histories(paths: list[Path], min_stack_bb: float = 80, max_stack_bb: float = 120) -> tuple[list[ImportedHand], dict]:
    minimum, maximum = Decimal(str(min_stack_bb)), Decimal(str(max_stack_bb))
    if not minimum.is_finite() or not maximum.is_finite() or not 0 < minimum <= maximum:
        raise ValueError("effective-stack bounds must be finite, positive, and ordered")
    if not paths:
        raise ValueError("at least one PHH or PHHS source is required")
    report = {"source_files": [], "total": 0, "accepted": 0, "excluded": 0,
              "reasons": {}, "duplicates": 0, "exclusion_examples": [],
              "policy": {"variant": "NT", "table_seats": 2, "active_players": 2, "ante": 0,
                         "money_unit": "one hundredth of the source currency unit",
                         "min_effective_stack_bb": str(minimum), "max_effective_stack_bb": str(maximum),
                         "timestamp": "UTC; ambiguous or nonexistent local times rejected",
                         "cash_evidence": "explicit currency and no tournament/event metadata",
                         "public_betting_replayed": True, "payouts_verified": False,
                         "private_cards_in_output": False, "implicit_betting_repairs": False}}
    reasons = Counter()
    seen = {}
    imported = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.suffix.lower() not in (".phh", ".phhs"):
            raise ValueError(f"unsupported history file extension: {path}")
        data = path.read_bytes()
        try:
            parsed = tomllib.loads(data.decode("utf-8"), parse_float=Decimal)
        except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"invalid PHH TOML source: {path}") from exc
        rows = {"1": parsed} if "variant" in parsed else parsed
        if not rows:
            raise ValueError(f"empty PHH source: {path}")
        report["source_files"].append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "records": len(rows)})
        for row, record in rows.items():
            report["total"] += 1
            try:
                if not isinstance(record, dict):
                    raise _Excluded("malformed_hand_record")
                venue, key = _identity(record)
                fingerprint = hashlib.sha256(json.dumps(_canonical(record), sort_keys=True,
                                                       separators=(",", ":"), allow_nan=False).encode()).hexdigest()
                if key in seen:
                    if seen[key] != fingerprint:
                        raise ValueError(f"conflicting records for hand key {key}: {path}, row {row}")
                    report["duplicates"] += 1
                    continue
                seen[key] = fingerprint
                imported.append(_import(record, str(path), str(row), venue, key, minimum, maximum))
            except _Excluded as exc:
                reason = str(exc)
                reasons[reason] += 1
                report["excluded"] += 1
                if reasons[reason] <= 3:
                    report["exclusion_examples"].append({"source": str(path), "row": str(row), "reason": reason})
    imported.sort(key=lambda hand: (hand.timestamp, hand.key))
    report["accepted"] = len(imported)
    report["reasons"] = dict(sorted(reasons.items()))
    return imported, report
