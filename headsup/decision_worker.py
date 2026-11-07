import hashlib
import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from random import Random
from time import monotonic, perf_counter, time
from typing import Annotated, Callable, Literal, TextIO

from pydantic import Field, TypeAdapter, ValidationError, model_validator

from .learning import OpponentProfile
from .opponent_memory import OpponentMemory, completion_fingerprint
from .policies import make_adaptive_policy, make_river_policy
from .protocol import History, Seat, WireAction, WireModel, WireObservation, validate_history
from .timing import ActionSchedule, ActionTiming

MAX_LINE_BYTES = 1_048_576
HandId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
EpochMs = Annotated[int, Field(gt=0, le=9_007_199_254_740_991)]


class Decide(WireModel):
    kind: Literal["decide"]
    hand_id: HandId
    observation: WireObservation
    observed_unix_ms: EpochMs
    deadline_unix_ms: EpochMs

    @model_validator(mode="after")
    def check_deadline(self) -> "Decide":
        if not 0 < self.deadline_unix_ms - self.observed_unix_ms <= 120_000:
            raise ValueError("deadline must be within 120 seconds after observation")
        return self


class Status(WireModel):
    kind: Literal["status"]
    hand_id: HandId
    turn_index: Annotated[int, Field(ge=0, le=4096)]


class Confirm(WireModel):
    kind: Literal["confirm"]
    hand_id: HandId
    history: History


class Finish(WireModel):
    kind: Literal["finish"]
    hand_id: HandId
    history: History
    hero_seat: Seat | None = None


class Ping(WireModel):
    kind: Literal["ping"]


Request = Annotated[Decide | Status | Confirm | Finish | Ping, Field(discriminator="kind")]
REQUEST = TypeAdapter(Request)


class WorkerError(ValueError):
    pass


@dataclass
class Pending:
    request: Decide
    action: WireAction | None
    info: dict
    rng_after: tuple
    schedule: ActionSchedule
    compute_ms: float


def _seed(master: int, *labels: object) -> int:
    value = json.dumps([master, *labels], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:16], "big")


class DecisionWorker:
    def __init__(self, seed: int, samples: int = 128, policy: str = "adaptive",
                 timing: ActionTiming = ActionTiming(), profile: OpponentProfile | None = None,
                 *, memory_store: OpponentMemory | None = None,
                 monotonic_clock: Callable[[], float] = monotonic,
                 wall_clock: Callable[[], float] = time) -> None:
        if type(seed) is not int or policy not in ("adaptive", "river"):
            raise ValueError("provide an integer seed and adaptive or river policy")
        if profile is not None and memory_store is not None:
            raise ValueError("use the saved opponent profile or an explicit profile, not both")
        self.memory_store = memory_store
        self.policy_name = policy
        self.learning_updates = 0
        self.factory = make_river_policy if policy == "river" else make_adaptive_policy
        saved = memory_store.profile if memory_store is not None else profile or OpponentProfile()
        self.profile = OpponentProfile.from_dict(saved.to_dict())
        self.factory(seed, samples, self.profile)
        self.seed, self.samples, self.timing = seed, samples, timing
        self.monotonic_clock, self.wall_clock = monotonic_clock, wall_clock
        self.clock_anchor = (monotonic_clock(), wall_clock())
        self.last_clock = self.clock_anchor[0]
        self.blocked = False
        self.hand_id: str | None = None
        self.policy = None
        self.reference: WireObservation | None = None
        self.history: History = ()
        self.pending: Pending | None = None
        self.completed: dict[str, str] = {}

    def _now(self) -> tuple[float, float]:
        now, wall = self.monotonic_clock(), self.wall_clock()
        origin, wall_origin = self.clock_anchor
        if (self.blocked or not all(math.isfinite(value) for value in (now, wall))
                or now < self.last_clock or abs((wall - wall_origin) - (now - origin)) > 0.25):
            self.blocked = True
            raise WorkerError("clock_changed_restart_on_new_hand")
        self.last_clock = now
        return now, wall

    def _same_hand(self, hand_id: str) -> None:
        if hand_id != self.hand_id:
            raise WorkerError("unknown_hand")

    def _history_extension(self, history: History) -> None:
        validate_history(history)
        if history[:len(self.history)] != self.history:
            raise WorkerError("history_changed")
        if any(event.actor == self.reference.actor for event in history[len(self.history):]):
            raise WorkerError("unconfirmed_own_action")
        if any(set(event.board).intersection(self.reference.hole) for event in history):
            raise WorkerError("board_contains_own_card")

    def _response(self, pending: Pending, now: float) -> dict:
        remaining = pending.schedule.remaining(now)
        ready = remaining == 0 and pending.action is not None
        request, schedule = pending.request, pending.schedule
        return {
            "status": "expired" if remaining is None else "ready" if ready else "waiting",
            "hand_id": request.hand_id, "turn_index": len(request.observation.history),
            "action": pending.action.model_dump() if ready else None,
            "wait_ms": None if remaining is None else remaining * 1000,
            "scheduled_delay_ms": (schedule.target_at - schedule.observed_at) * 1000,
            "send_at_unix_ms": request.observed_unix_ms + (schedule.target_at - schedule.observed_at) * 1000,
            "deadline_unix_ms": request.deadline_unix_ms,
            "reserve_ms": self.timing.reserve * 1000,
            "decision_ms": pending.compute_ms,
            "requires_fresh_state_before_send": True,
            "diagnostics": pending.info if ready else None,
        }

    def _decide(self, request: Decide, now: float, wall: float) -> dict:
        obs = request.observation
        if self.memory_store is not None:
            identity = self.memory_store.identity
            if (obs.big_blind != identity.big_blind or obs.rake_bps != 0 or obs.rake_cap != 0
                    or not obs.no_flop_no_drop):
                raise WorkerError("opponent_memory_rules_changed")
        if self.pending is not None:
            if request != self.pending.request:
                raise WorkerError("pending_turn_requires_confirmation")
            return self._response(self.pending, now)
        age = wall - request.observed_unix_ms / 1000
        if not 0 <= age <= 2:
            raise WorkerError("observation_not_fresh")
        if any(event.action.kind == "fold" for event in obs.history):
            raise WorkerError("decision_after_fold")
        deadline = now + request.deadline_unix_ms / 1000 - wall
        observed_at = now - age
        if self.hand_id is None:
            if (request.hand_id in self.completed
                    or self.memory_store is not None and self.memory_store.contains(request.hand_id)):
                raise WorkerError("completed_hand_reused")
            if (obs.street != "preflop" or any(event.actor == obs.actor for event in obs.history)
                    or len(obs.history) != 1 - obs.actor):
                raise WorkerError("worker_requires_new_clean_hand")
            if self.memory_store is not None:
                small, big = identity.small_blind, identity.big_blind
                first = obs.history[0] if obs.history else None
                paid = first.paid if first is not None else 0
                if (obs.bets != (big, small + paid) or obs.pot != small + big + paid
                        or first is not None and (first.pot_before != small + big
                                                  or first.to_call_before != big - small)):
                    raise WorkerError("opponent_memory_blinds_changed")
            candidate = self.factory(_seed(self.seed, "strategy", request.hand_id), self.samples, self.profile)
        else:
            self._same_hand(request.hand_id)
            self._history_extension(obs.history)
            previous = self.reference
            fixed = ("actor", "hole", "button", "big_blind", "rake_bps", "rake_cap", "no_flop_no_drop")
            if any(getattr(obs, key) != getattr(previous, key) for key in fixed):
                raise WorkerError("hand_identity_or_rules_changed")
            if obs.board[:len(previous.board)] != previous.board:
                raise WorkerError("board_changed")
            stacks, pot = list(previous.stacks), previous.pot
            for event in obs.history[len(previous.history):]:
                if event.pot_before != pot or event.paid > stacks[event.actor]:
                    raise WorkerError("public_payments_inconsistent")
                stacks[event.actor] -= event.paid
                pot += event.paid
            if tuple(stacks) != obs.stacks or pot != obs.pot:
                raise WorkerError("stacks_or_pot_disagree_with_history")
            if len(obs.history) == len(previous.history):
                raise WorkerError("turn_already_confirmed")
            candidate = self.policy
        schedule = self.timing.plan(observed_at, Random(_seed(self.seed, "timing", request.hand_id,
                                                            len(obs.history))), deadline)
        if schedule.remaining(now) is None:
            self.hand_id, self.policy, self.reference = request.hand_id, candidate, obs
            self.history = obs.history
            self.pending = Pending(request, None, {}, candidate.rng.getstate(), schedule, 0.0)
            return self._response(self.pending, now)
        before = candidate.rng.getstate()
        started = perf_counter()
        try:
            action = WireAction.model_validate(asdict(candidate.act(obs.to_domain())))
            after = candidate.rng.getstate()
        finally:
            candidate.rng.setstate(before)
        if not obs.accepts(action):
            raise WorkerError("policy_action_not_legal")
        compute_ms = (perf_counter() - started) * 1000
        finished, _ = self._now()
        pending = Pending(request, action, candidate.last_info.copy(), after, schedule, compute_ms)
        self.hand_id, self.policy, self.reference = request.hand_id, candidate, obs
        self.history, self.pending = obs.history, pending
        return self._response(pending, finished)

    def _confirm(self, request: Confirm) -> dict:
        self._same_hand(request.hand_id)
        validate_history(request.history)
        if self.pending is None:
            if request.history != self.history:
                raise WorkerError("no_pending_action")
            return {"status": "confirmed", "hand_id": request.hand_id,
                    "turn_index": len(self.reference.history)}
        pending = self.pending
        if pending.action is None:
            raise WorkerError("expired_turn_has_no_proposed_action")
        obs = pending.request.observation
        index = len(obs.history)
        if request.history[:index] != obs.history or len(request.history) <= index:
            raise WorkerError("confirmation_missing_expected_event")
        event = request.history[index]
        paid = (pending.action.amount - obs.bets[obs.actor] if pending.action.kind == "raise_to"
                else obs.to_call if pending.action.kind == "call" else 0)
        if ((event.actor, event.street, event.action, event.pot_before, event.to_call_before,
             event.paid, event.board) != (obs.actor, obs.street, pending.action, obs.pot, obs.to_call, paid, obs.board)):
            raise WorkerError("confirmation_action_mismatch")
        if any(item.actor == obs.actor for item in request.history[index + 1:]):
            raise WorkerError("confirmation_has_unmatched_own_actions")
        if any(set(item.board).intersection(obs.hole) for item in request.history):
            raise WorkerError("board_contains_own_card")
        self.policy.rng.setstate(pending.rng_after)
        self.history, self.pending = request.history, None
        return {"status": "confirmed", "hand_id": request.hand_id, "turn_index": index}

    def handle(self, request: Decide | Status | Confirm | Finish | Ping) -> dict:
        now, wall = self._now()
        if isinstance(request, Ping):
            return {"status": "worker_ready", "action": None, "pending": self.pending is not None,
                    "resume_supported": False, "learning_updates": self.learning_updates,
                    "policy": self.policy_name, "learning": self._learning_status()}
        if isinstance(request, Decide):
            return self._decide(request, now, wall)
        if isinstance(request, Finish):
            if self.memory_store is not None:
                if request.hero_seat is None:
                    raise WorkerError("completion_requires_hero_seat")
                fingerprint = completion_fingerprint(request.hand_id, request.hero_seat, request.history)
                previous = self.memory_store.completed_fingerprint(request.hand_id)
                if previous is not None:
                    if previous != fingerprint:
                        raise WorkerError("completed_history_changed")
                    return self._finished_response(request.hand_id, False)
                if self.hand_id is None:
                    return self._finish_without_action(request)
            elif request.hand_id in self.completed:
                fingerprint = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
                if self.completed[request.hand_id] != fingerprint:
                    raise WorkerError("completed_history_changed")
                return self._finished_response(request.hand_id, False)
        self._same_hand(request.hand_id)
        if isinstance(request, Status):
            if self.pending is None or request.turn_index != len(self.pending.request.observation.history):
                raise WorkerError("unknown_pending_turn")
            return self._response(self.pending, now)
        if isinstance(request, Confirm):
            return self._confirm(request)
        if request.hero_seat is not None and request.hero_seat != self.reference.actor:
            raise WorkerError("completion_seat_changed")
        if self.pending is not None and self.pending.action is not None:
            raise WorkerError("pending_turn_requires_confirmation")
        self._history_extension(request.history)
        stacks, pot = list(self.reference.stacks), self.reference.pot
        for event in request.history[len(self.reference.history):]:
            if event.pot_before != pot or event.paid > stacks[event.actor]:
                raise WorkerError("public_payments_inconsistent")
            stacks[event.actor] -= event.paid
            pot += event.paid
        last = request.history[-1] if request.history else None
        closed = last is not None and (
            last.action.kind == "fold"
            or last.action.kind == "call" and (last.street == "river" or min(stacks) == 0)
            or last.street == "river" and len(request.history) >= 2
            and all(event.action.kind == "check" for event in request.history[-2:])
            and request.history[-2].board == last.board and request.history[-2].actor != last.actor
        )
        if not closed:
            raise WorkerError("hand_has_not_finished_betting")
        updated = self._learn(request, self.reference.actor)
        if self.memory_store is None:
            self.completed[request.hand_id] = hashlib.sha256(request.model_dump_json().encode()).hexdigest()
        self.hand_id, self.policy, self.reference = None, None, None
        self.history, self.pending = (), None
        return self._finished_response(request.hand_id, updated)

    def _learning_status(self) -> dict:
        return self.memory_store.summary() if self.memory_store is not None else {"enabled": False}

    def _finished_response(self, hand_id: str, updated: bool) -> dict:
        return {"status": "finished", "hand_id": hand_id, "learning_updated": updated,
                "learning_updates": self.learning_updates, "resume_supported": False,
                "policy": self.policy_name, "learning": self._learning_status()}

    def _learn(self, request: Finish, hero_seat: int) -> bool:
        if self.memory_store is None:
            return False
        updated = self.memory_store.learn(request.hand_id, hero_seat, request.history)
        self.profile = OpponentProfile.from_dict(self.memory_store.profile.to_dict())
        self.learning_updates += int(updated)
        return updated

    def _finish_without_action(self, request: Finish) -> dict:
        identity = self.memory_store.identity
        event = request.history[0] if len(request.history) == 1 else None
        if (request.hero_seat != 0 or event is None or event.actor != 1
                or event.street != "preflop" or event.action.kind != "fold"
                or event.pot_before != identity.small_blind + identity.big_blind
                or event.to_call_before != identity.big_blind - identity.small_blind):
            raise WorkerError("worker_requires_observed_hand")
        return self._finished_response(request.hand_id, self._learn(request, 0))

    def close(self) -> None:
        if self.memory_store is not None:
            self.memory_store.close()


def _reject_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def run_worker(worker: DecisionWorker, input_stream: TextIO | None = None,
               output_stream: TextIO | None = None) -> int:
    source, output = input_stream or sys.stdin, output_stream or sys.stdout
    try:
        while line := source.readline(MAX_LINE_BYTES + 1):
            if len(line.encode()) > MAX_LINE_BYTES:
                output.write('{"status":"error","error":"request_too_large"}\n')
                output.flush()
                return 1
            fatal = False
            try:
                json.loads(line, object_pairs_hook=_reject_duplicates)
                request = REQUEST.validate_json(line)
                result = worker.handle(request)
            except WorkerError as error:
                result = {"status": "error", "error": str(error), "action": None}
            except (ValidationError, ValueError, RecursionError):
                result = {"status": "error", "error": "invalid_request", "action": None}
            except (sqlite3.Error, OSError):
                result = {"status": "error", "error": "memory_unavailable_restart_on_new_hand", "action": None}
                fatal = True
            output.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
            output.flush()
            if fatal:
                return 1
        return 0
    finally:
        worker.close()
