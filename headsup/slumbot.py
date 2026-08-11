import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from .arena import environment, seed_for, summarize
from .engine import CARDS, STREETS, Hand
from .models import MAX_SAMPLES, Action, GameConfig, Observation
from .policies import POLICY_NAMES, make_policy

BASE_URL = "https://slumbot.com/slumbot/api"
CONFIG = GameConfig(small_blind=50, big_blind=100, stack_bb=200, rake_cap=0)
MAX_RESPONSE_BYTES = 1_048_576
MAX_ACTIONS = 4096
CardCode = Annotated[str, Field(pattern=r"^[2-9TJQKA][cdhs]$")]
ActionText = Annotated[str, Field(max_length=32768, pattern=r"^[kcfb0-9/]*$")]


class SlumbotError(ValueError):
    pass


class Response(BaseModel):
    model_config = ConfigDict(strict=True, extra="allow", frozen=True)

    action: ActionText
    client_pos: Annotated[int, Field(ge=0, le=1)]
    hole_cards: Annotated[list[CardCode], Field(min_length=2, max_length=2)]
    board: Annotated[list[CardCode], Field(max_length=5)]
    old_action: ActionText | None = None
    bot_hole_cards: Annotated[list[CardCode], Field(min_length=2, max_length=2)] | None = None
    winnings: Annotated[int, Field(ge=-20000, le=20000)] | None = None
    token: Annotated[str, Field(min_length=16, max_length=256)] | None = Field(default=None, repr=False, exclude=True)

    @model_validator(mode="after")
    def valid_cards(self) -> "Response":
        cards = self.hole_cards + self.board + (self.bot_hole_cards or [])
        if len(self.board) not in (0, 3, 4, 5) or len(set(cards)) != len(cards):
            raise ValueError("invalid or repeated cards")
        if self.bot_hole_cards is not None and self.winnings is None:
            raise ValueError("opponent cards were disclosed before settlement")
        return self


@dataclass(frozen=True)
class VerifiedState:
    response: Response
    hand: Hand
    observation: Observation | None
    settlement_check: str | None


def _deck(response: Response) -> tuple[str, ...]:
    known = set(response.hole_cards + response.board + (response.bot_hole_cards or []))
    available = iter(card for card in CARDS if card not in known)
    deck = [None] * 52
    hero_slots = (0, 2) if response.client_pos == 0 else (1, 3)
    other_slots = (1, 3) if response.client_pos == 0 else (0, 2)
    for index, card in zip(hero_slots, response.hole_cards):
        deck[index] = card
    if response.bot_hole_cards is not None:
        for index, card in zip(other_slots, response.bot_hole_cards):
            deck[index] = card
    for index, card in zip((5, 6, 7, 9, 11), response.board):
        deck[index] = card
    return tuple(card if card is not None else next(available) for card in deck)


def _tokens(text: str) -> list[str]:
    tokens = re.findall(r"b[1-9][0-9]*|[kcf/]", text)
    if "".join(tokens) != text or len(tokens) > MAX_ACTIONS + 3:
        raise SlumbotError("invalid or excessive action history")
    return tokens


def _action(token: str) -> Action:
    return Action("raise_to", int(token[1:])) if token.startswith("b") else Action(
        {"k": "check", "c": "call", "f": "fold"}[token])


def verify_response(payload: dict) -> VerifiedState:
    try:
        response = Response.model_validate(payload)
    except ValidationError:
        raise SlumbotError("response fields do not match the Slumbot protocol") from None
    tokens = _tokens(response.action)
    hand = Hand(CONFIG, deck=_deck(response))
    street = 0
    terminal_at = None
    try:
        for index, token in enumerate(tokens):
            if hand.finished:
                terminal_at = index
                break
            if token == "/":
                if street >= 3 or hand.observation().street != STREETS[street + 1]:
                    raise SlumbotError("street separator disagrees with the betting history")
                street += 1
                continue
            if hand.observation().street != STREETS[street]:
                raise SlumbotError("missing street separator")
            hand.act(_action(token))
        if terminal_at is not None:
            suffix = tokens[terminal_at:]
            if (hand.public_history[-1].action.kind == "fold"
                    or suffix != ["/"] * (3 - street)):
                raise SlumbotError("unexpected actions or separators after settlement")
    except (ValueError, KeyError) as exc:
        if isinstance(exc, SlumbotError):
            raise
        raise SlumbotError("illegal betting history") from None
    if list(hand.snapshot()["board"]) != response.board:
        raise SlumbotError("board disagrees with the reconstructed street")
    if hand.finished != (response.winnings is not None):
        raise SlumbotError("terminal state and winnings disagree")
    if not hand.finished:
        observation = hand.observation()
        if observation.actor != response.client_pos:
            raise SlumbotError("response does not give the client a turn")
        return VerifiedState(response, hand, observation, None)
    if hand.public_history[-1].action.kind == "fold" or response.bot_hole_cards is not None:
        if hand.result().net_returns[response.client_pos] != response.winnings:
            raise SlumbotError("winnings disagree with verified settlement")
        check = "recomputed_fold" if hand.public_history[-1].action.kind == "fold" else "recomputed_showdown"
    else:
        matched = min(hand.snapshot()["contributions"])
        if response.winnings not in (-matched, 0, matched):
            raise SlumbotError("showdown winnings exceed the matched-pot outcomes")
        check = "server_reported_showdown; opponent cards unavailable"
    return VerifiedState(response, hand, None, check)


def encode_action(state: VerifiedState, action: Action) -> str:
    if state.observation is None:
        raise SlumbotError("cannot act on a completed hand")
    trial = verify_response(state.response.model_dump(exclude={"token"}))
    try:
        trial.hand.act(action)
    except ValueError:
        raise SlumbotError("policy proposed an illegal action") from None
    return f"b{action.amount}" if action.kind == "raise_to" else {
        "check": "k", "call": "c", "fold": "f",
    }[action.kind]


def verify_receipt(previous: VerifiedState, current: VerifiedState, action: Action) -> None:
    before, after = previous.response, current.response
    if (after.old_action != before.action or after.client_pos != before.client_pos
            or after.hole_cards != before.hole_cards or after.board[:len(before.board)] != before.board):
        raise SlumbotError("action receipt changed the prior state or player identity")
    old_history, new_history = previous.hand.public_history, current.hand.public_history
    if new_history[:len(old_history)] != old_history or len(new_history) <= len(old_history):
        raise SlumbotError("action receipt does not extend the accepted history")
    first, *rest = new_history[len(old_history):]
    if (first.actor != before.client_pos or first.action != action
            or any(event.actor == before.client_pos for event in rest)):
        raise SlumbotError("action receipt does not confirm exactly the submitted client action")


def _private_json(path: Path, value: dict) -> None:
    with path.open("w") as file:
        path.chmod(0o600)
        file.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _redact(value: object, tokens: tuple[str, ...]) -> object:
    if isinstance(value, dict):
        return {key: _redact(item, tokens) for key, item in value.items()
                if "token" not in key.casefold() and "password" not in key.casefold()}
    if isinstance(value, list):
        return [_redact(item, tokens) for item in value]
    if isinstance(value, str):
        for token in tokens:
            value = value.replace(token, "[redacted]")
    return value


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise SlumbotError("duplicate response JSON field")
        result[key] = value
    return result


def _post(client: httpx.Client, endpoint: str, data: dict) -> dict:
    with client.stream("POST", f"{BASE_URL}/{endpoint}", json=data) as response:
        if response.status_code != 200:
            raise SlumbotError(f"HTTP {response.status_code}; request was not retried")
        chunks, size = [], 0
        for chunk in response.iter_bytes():
            size += len(chunk)
            if size > MAX_RESPONSE_BYTES:
                raise SlumbotError("response exceeds the size limit")
            chunks.append(chunk)
    try:
        payload = json.loads(b"".join(chunks), object_pairs_hook=_json_pairs)
    except (ValueError, UnicodeError):
        raise SlumbotError("invalid JSON response; request was not retried") from None
    if not isinstance(payload, dict) or "error_msg" in payload:
        raise SlumbotError("server returned an error or unsupported response")
    return payload


def run_benchmark(output: Path, hands: int, policy: str = "range", samples: int = 128,
                  seed: int = 20261107, *, client: httpx.Client | None = None) -> dict:
    if type(hands) is not int or not 2 <= hands <= 1000 or hands % 2:
        raise ValueError("hands must be an even integer from 2 to 1000")
    if policy not in POLICY_NAMES or type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError("unsupported policy or sample count")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    output.chmod(0o700)
    original_environment = environment()
    _private_json(output / "manifest.json", {
        "schema_version": 1, "mode": "slumbot_anonymous_ai_benchmark",
        "created_utc": datetime.now(timezone.utc).isoformat(), "endpoint": BASE_URL,
        "requested_hands": hands, "policy": policy, "samples": samples, "seed": seed,
        "config": asdict(CONFIG), "environment": original_environment,
        "protocol_source": "https://slumbot.com/sample_api.py",
        "evaluation": "Sequential server deals; no duplicate-deal control, no online learning, no default-policy promotion.",
        "retry_policy": "Never retry a POST; stop on an ambiguous or invalid response. Tokens stay in memory.",
    })
    own_client = client is None
    connection = client or httpx.Client(timeout=httpx.Timeout(30.0, connect=10.0), follow_redirects=False)
    completed, accepted, settlements = [], 0, []
    token = None
    pending = None
    failure = None
    status = "completed"
    journal_path = output / "journal.jsonl"
    with journal_path.open("x") as journal:
        journal_path.chmod(0o600)

        def write(value: dict) -> None:
            journal.write(json.dumps(value, allow_nan=False) + "\n")
            journal.flush()

        def request(endpoint: str, body: dict, hand_index: int) -> VerifiedState:
            nonlocal token, pending
            pending = {"hand_index": hand_index, "endpoint": endpoint,
                       "action": body.get("incr")}
            write({"kind": "request", **pending})
            payload = _post(connection, endpoint, body)
            secrets = tuple(item for item in (token, payload.get("token")) if isinstance(item, str))
            write({"kind": "response", "hand_index": hand_index, "body": _redact(payload, secrets)})
            state = verify_response(payload)
            if state.response.token is not None:
                token = state.response.token
            if token is None:
                raise SlumbotError("server did not issue a session token")
            return state

        try:
            for hand_index in range(hands):
                state = request("new_hand", {"token": token} if token else {}, hand_index)
                if (state.response.client_pos != hand_index % 2
                        or state.response.old_action not in (None, "")
                        or any(event.actor == state.response.client_pos for event in state.hand.public_history)):
                    raise SlumbotError("new hand has unexpected position or previous client actions")
                pending = None
                strategy = make_policy(policy, seed_for(seed, "slumbot-policy", hand_index), samples)
                while state.observation is not None:
                    action = strategy.act(state.observation)
                    encoded = encode_action(state, action)
                    write({"kind": "decision", "hand_index": hand_index,
                           "observation": asdict(state.observation), "action": asdict(action),
                           "info": strategy.last_info})
                    next_state = request("act", {"token": token, "incr": encoded}, hand_index)
                    verify_receipt(state, next_state, action)
                    pending = None
                    accepted += 1
                    state = next_state
                completed.append(state.response.winnings)
                settlements.append(state.settlement_check)
                write({"kind": "completed_hand", "hand_index": hand_index,
                       "client_pos": state.response.client_pos, "winnings": state.response.winnings,
                       "settlement_check": state.settlement_check})
                print(f"Hand {hand_index + 1}/{hands}: {state.response.winnings:+d} chips; total {sum(completed):+d}", flush=True)
        except (Exception, KeyboardInterrupt) as exc:
            status = "incomplete" if isinstance(exc, KeyboardInterrupt) else "failed"
            failure = {"type": type(exc).__name__,
                       "message": str(exc) if isinstance(exc, SlumbotError) else "benchmark stopped; see the private journal",
                       "unresolved_request": pending}
            write({"kind": "failure", **failure})
        finally:
            if own_client:
                connection.close()
    unchanged = environment() == original_environment
    if not unchanged:
        status = "invalidated"
    descriptive = summarize([value * 100 / CONFIG.big_blind for value in completed])
    descriptive["hands"] = descriptive.pop("pairs")
    report = {
        "schema_version": 1, "status": status, "requested_hands": hands,
        "completed_hands": len(completed), "accepted_actions": accepted,
        "net_chips": sum(completed), "net_big_blinds": sum(completed) / CONFIG.big_blind,
        "returns_chips": completed, "settlement_checks": settlements,
        "position_balanced": bool(completed) and len(completed) % 2 == 0,
        "descriptive_net_bb100": descriptive,
        "source_environment_unchanged": unchanged, "failure": failure,
        "playing_strength_established": False,
        "interpretation": "A bounded external AI benchmark; raw sequential deals and approximate intervals do not establish profitability or human-player strength.",
    }
    _private_json(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bounded anonymous Slumbot AI benchmark; 200BB, no money.")
    parser.add_argument("--hands", type=int, default=20)
    parser.add_argument("--policy", choices=POLICY_NAMES, default="range")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(args.output, args.hands, args.policy, args.samples, args.seed)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))
    return int(report["status"] != "completed")


if __name__ == "__main__":
    raise SystemExit(main())
