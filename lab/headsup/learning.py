"""Opponent response memory built only from completed hands' public actions."""

import hashlib
import json
from itertools import product
from math import isfinite

from scipy.stats import beta

from .models import Event

STREETS = ("preflop", "flop", "turn", "river")
MIN_RESPONSES = 24
MAX_COUNT = 1_000_000_000
RECENT_WINDOW_SIZE = 64
RECENT_HAND_HORIZON = 256
_CONTEXTS = frozenset(":".join(parts) for parts in product(
    STREETS, ("bb", "button"), ("first", "reraised"), ("small", "medium", "large"),
))


def response_context(street: str, actor: int, raises: int, pot: int, to_call: int) -> str:
    price = to_call / (pot + to_call)
    bucket = "small" if price <= 0.25 else "medium" if price <= 0.4 else "large"
    return f"{street}:{'button' if actor == 1 else 'bb'}:{'first' if raises == 1 else 'reraised'}:{bucket}"


class OpponentProfile:
    def __init__(self, memory: str = "lifetime") -> None:
        if memory not in ("lifetime", "recent"):
            raise ValueError("memory must be lifetime or recent")
        self.memory = memory
        self.hands_observed = 0
        self._responses: dict[str, dict[str, int]] = {}
        self._recent: dict[str, list[tuple[int, str]]] = {}

    def observe_hand(self, history: tuple[Event, ...], opponent_seat: int) -> None:
        if type(opponent_seat) is not int or opponent_seat not in (0, 1):
            raise ValueError("opponent_seat must be 0 or 1")
        if not isinstance(history, tuple) or any(not isinstance(event, Event) for event in history):
            raise ValueError("history must be a tuple of public Event values from a completed hand")
        updates: dict[str, dict[str, int]] = {}
        first_responses: dict[str, str] = {}
        raises = {street: 0 for street in STREETS}
        aggressor = {street: None for street in STREETS}
        for event in history:
            if (event.street not in STREETS or type(event.actor) is not int or event.actor not in (0, 1)
                    or type(event.pot_before) is not int or event.pot_before <= 0
                    or type(event.to_call_before) is not int or event.to_call_before < 0):
                raise ValueError("invalid public event")
            # Blind completion is not a response to a voluntary bet.
            if (event.actor == opponent_seat and event.to_call_before > 0
                    and aggressor[event.street] == 1 - opponent_seat):
                key = response_context(event.street, event.actor, raises[event.street],
                                       event.pot_before, event.to_call_before)
                action = "raise" if event.action.kind == "raise_to" else event.action.kind
                if action not in ("fold", "call", "raise"):
                    raise ValueError("a priced response must fold, call, or raise")
                counts = updates.setdefault(key, {"fold": 0, "call": 0, "raise": 0})
                counts[action] += 1
                first_responses.setdefault(key, action)
            if event.action.kind == "raise_to":
                raises[event.street] += 1
                aggressor[event.street] = event.actor
        candidate = self.to_dict()
        candidate["hands_observed"] += 1
        if self.memory == "recent":
            hand_index = candidate["hands_observed"]
            for key, action in first_responses.items():
                candidate["observations"].setdefault(key, []).append([hand_index, action])
            candidate["observations"] = {
                key: retained[-RECENT_WINDOW_SIZE:]
                for key, entries in candidate["observations"].items()
                if (retained := [entry for entry in entries
                                 if hand_index - entry[0] < RECENT_HAND_HORIZON])
            }
        else:
            for key, counts in updates.items():
                target = candidate["responses"].setdefault(key, {"fold": 0, "call": 0, "raise": 0})
                for action, count in counts.items():
                    target[action] += count
        checked = self.from_dict(candidate)
        self.hands_observed = checked.hands_observed
        self._responses = checked._responses
        self._recent = checked._recent

    def to_dict(self) -> dict:
        if self.memory == "recent":
            return {"schema_version": 2, "memory": "recent", "hands_observed": self.hands_observed,
                    "window_size": RECENT_WINDOW_SIZE, "hand_horizon": RECENT_HAND_HORIZON,
                    "observations": {key: [[index, action] for index, action in entries]
                                     for key, entries in sorted(self._recent.items())}}
        return {"schema_version": 1, "hands_observed": self.hands_observed,
                "responses": {key: dict(value) for key, value in sorted(self._responses.items())}}

    @classmethod
    def from_dict(cls, data: dict) -> "OpponentProfile":
        if isinstance(data, dict) and type(data.get("schema_version")) is int and data["schema_version"] == 3:
            return PopulationBackedProfile.from_dict(data)
        if isinstance(data, dict) and type(data.get("schema_version")) is int and data["schema_version"] == 2:
            return cls._from_recent(data)
        if (not isinstance(data, dict) or set(data) != {"schema_version", "hands_observed", "responses"}
                or type(data["schema_version"]) is not int or data["schema_version"] != 1):
            raise ValueError("unsupported opponent profile schema")
        hands = data["hands_observed"]
        if type(hands) is not int or not 0 <= hands <= MAX_COUNT:
            raise ValueError("hands_observed must be a bounded nonnegative integer")
        responses = data["responses"]
        if not isinstance(responses, dict) or any(key not in _CONTEXTS for key in responses):
            raise ValueError("invalid opponent response context")
        profile = cls()
        profile.hands_observed = hands
        for key, counts in responses.items():
            if not isinstance(counts, dict) or set(counts) != {"fold", "call", "raise"}:
                raise ValueError("response counts must contain fold, call, and raise")
            if any(type(count) is not int or not 0 <= count <= MAX_COUNT for count in counts.values()):
                raise ValueError("response counts must be bounded nonnegative integers")
            if sum(counts.values()) == 0:
                raise ValueError("empty response contexts must be omitted")
            profile._responses[key] = dict(counts)
        if sum(counts["fold"] for counts in profile._responses.values()) > hands:
            raise ValueError("an opponent can fold at most once per completed hand")
        if not hands and profile._responses:
            raise ValueError("response counts require completed hands")
        return profile

    @classmethod
    def _from_recent(cls, data: dict) -> "OpponentProfile":
        expected = {"schema_version", "memory", "hands_observed", "window_size", "hand_horizon", "observations"}
        if set(data) != expected or data["memory"] != "recent":
            raise ValueError("unsupported recent opponent profile schema")
        if (type(data["window_size"]) is not int or data["window_size"] != RECENT_WINDOW_SIZE
                or type(data["hand_horizon"]) is not int or data["hand_horizon"] != RECENT_HAND_HORIZON):
            raise ValueError("recent profile requires the supported window size and hand horizon")
        hands = data["hands_observed"]
        if type(hands) is not int or not 0 <= hands <= MAX_COUNT:
            raise ValueError("hands_observed must be a bounded nonnegative integer")
        observations = data["observations"]
        if not isinstance(observations, dict) or any(key not in _CONTEXTS for key in observations):
            raise ValueError("invalid opponent response context")
        profile = cls("recent")
        profile.hands_observed = hands
        folded_hands: set[int] = set()
        positions: dict[int, str] = {}
        hand_responses: dict[int, list[tuple[str, str]]] = {}
        for key, entries in observations.items():
            if not isinstance(entries, list) or not 1 <= len(entries) <= RECENT_WINDOW_SIZE:
                raise ValueError("recent contexts require one to 64 observations")
            previous = 0
            counts = {"fold": 0, "call": 0, "raise": 0}
            retained = []
            for entry in entries:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("recent observations must contain a hand index and action")
                index, action = entry
                if (type(index) is not int or not previous < index <= hands
                        or hands - index >= RECENT_HAND_HORIZON):
                    raise ValueError("recent hand indexes must be ordered, distinct, completed and unexpired")
                if type(action) is not str or action not in counts:
                    raise ValueError("recent actions must be fold, call, or raise")
                position = key.split(":")[1]
                if positions.setdefault(index, position) != position:
                    raise ValueError("an opponent has one position per completed hand")
                if action == "fold":
                    if index in folded_hands:
                        raise ValueError("an opponent can fold at most once per completed hand")
                    folded_hands.add(index)
                counts[action] += 1
                retained.append((index, action))
                hand_responses.setdefault(index, []).append((key, action))
                previous = index
            profile._recent[key] = retained
            profile._responses[key] = counts
        for responses in hand_responses.values():
            first_streets = [key.split(":")[0] for key, _ in responses if key.split(":")[2] == "first"]
            if len(set(first_streets)) != len(first_streets):
                raise ValueError("a hand has at most one first-bet response per street")
            calls = [key for key, action in responses if action == "call"]
            call_streets = [key.split(":")[0] for key in calls]
            if len(set(call_streets)) != len(call_streets):
                raise ValueError("a heads-up betting round has at most one priced call by an opponent")
            for call in calls:
                street, _, betting_round, _ = call.split(":")
                if betting_round == "first" and any(key != call and key.split(":")[0] == street
                                                    for key, _ in responses):
                    raise ValueError("a first-bet call ends the heads-up betting round")
            fold = next((key for key, action in responses if action == "fold"), None)
            if fold is None:
                continue
            fold_street, _, fold_round, _ = fold.split(":")
            for key, action in responses:
                street, _, _, _ = key.split(":")
                if (STREETS.index(street) > STREETS.index(fold_street)
                        or key != fold and street == fold_street
                        and (fold_round == "first" or action == "call")):
                    raise ValueError("a recent profile contains responses after a hand-ending fold")
        return profile

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    def estimate(self, context: str) -> dict:
        if context not in _CONTEXTS:
            raise ValueError("invalid opponent response context")
        counts = self._responses.get(context, {"fold": 0, "call": 0, "raise": 0})
        n = sum(counts.values())
        a, b = 2 + counts["fold"], 6 + counts["call"] + counts["raise"]
        lower, upper = beta.ppf((0.05, 0.95), a, b)
        return {"context": context, "responses": n, "counts": dict(counts),
                "fold_mean": a / (a + b), "fold_lower": float(lower), "fold_upper": float(upper),
                "ready": n >= MIN_RESPONSES, "prior": {"fold": 2, "continue": 6},
                "interval": "90% Beta posterior interval under a fixed-context Bernoulli model"}

    def summary(self) -> dict:
        return {"memory": self.memory,
                "window_size": RECENT_WINDOW_SIZE if self.memory == "recent" else None,
                "hand_horizon": RECENT_HAND_HORIZON if self.memory == "recent" else None,
                "observation_unit": "first eligible response per context per completed hand" if self.memory == "recent" else "eligible public response",
                "hands_observed": self.hands_observed, "profile_sha256": self.fingerprint,
                "total_responses": sum(sum(counts.values()) for counts in self._responses.values()),
                "ready_contexts": sum(sum(counts.values()) >= MIN_RESPONSES for counts in self._responses.values()),
                "minimum_responses": MIN_RESPONSES,
                "contexts": {key: self.estimate(key) for key in sorted(self._responses)},
                "scope": "Public fold/call/raise responses only; no private-card, showdown-strength, or bluff-frequency learning."}


class PopulationBackedProfile(OpponentProfile):
    def __init__(self, population: OpponentProfile, strength: float = 24,
                 memory: str = "lifetime", individual: OpponentProfile | None = None) -> None:
        if type(strength) not in (int, float) or not 1 <= strength <= 10000 or not isfinite(strength):
            raise ValueError("population strength must be finite and between 1 and 10000")
        if not isinstance(population, OpponentProfile) or population.to_dict().get("schema_version") != 1:
            raise ValueError("population must be a lifetime opponent profile")
        if memory not in ("lifetime", "recent"):
            raise ValueError("memory must be lifetime or recent")
        if individual is None:
            individual = OpponentProfile(memory)
        if (not isinstance(individual, OpponentProfile)
                or individual.to_dict().get("schema_version") not in (1, 2)
                or individual.memory != memory):
            raise ValueError("individual must be a plain profile with matching memory")
        self.memory = memory
        self.strength = float(strength)
        self._population = OpponentProfile.from_dict(population.to_dict())
        self._individual = OpponentProfile.from_dict(individual.to_dict())
        self._population_hash = self._population.fingerprint

    @property
    def hands_observed(self) -> int:
        return self._individual.hands_observed

    def observe_hand(self, history: tuple[Event, ...], opponent_seat: int) -> None:
        self._individual.observe_hand(history, opponent_seat)

    def to_dict(self) -> dict:
        return {"schema_version": 3, "hands_observed": self.hands_observed,
                "individual": self._individual.to_dict(), "population": self._population.to_dict(),
                "strength": self.strength}

    @classmethod
    def from_dict(cls, data: dict) -> "PopulationBackedProfile":
        expected = {"schema_version", "hands_observed", "individual", "population", "strength"}
        if (not isinstance(data, dict) or set(data) != expected
                or type(data["schema_version"]) is not int or data["schema_version"] != 3):
            raise ValueError("unsupported population-backed profile schema")
        hands = data["hands_observed"]
        if type(hands) is not int or not 0 <= hands <= MAX_COUNT:
            raise ValueError("hands_observed must be a bounded nonnegative integer")
        population, individual = data["population"], data["individual"]
        if (not isinstance(population, dict) or type(population.get("schema_version")) is not int
                or population["schema_version"] != 1):
            raise ValueError("population must use lifetime profile schema 1")
        if (not isinstance(individual, dict) or type(individual.get("schema_version")) is not int
                or individual["schema_version"] not in (1, 2)):
            raise ValueError("individual must use profile schema 1 or 2")
        own = OpponentProfile.from_dict(individual)
        if hands != own.hands_observed:
            raise ValueError("population-backed hand count must match individual hand count")
        return cls(OpponentProfile.from_dict(population), data["strength"], own.memory, own)

    def estimate(self, context: str) -> dict:
        own = self._individual.estimate(context)
        population = self._population.estimate(context)
        strength = min(self.strength, 8 + population["responses"])
        p = population["fold_mean"]
        prior_fold, prior_continue = strength * p, strength * (1 - p)
        counts = own["counts"]
        a = prior_fold + counts["fold"]
        b = prior_continue + counts["call"] + counts["raise"]
        lower, upper = beta.ppf((0.05, 0.95), a, b)
        return {**own, "fold_mean": a / (a + b), "fold_lower": float(lower), "fold_upper": float(upper),
                "prior": {"fold": prior_fold, "continue": prior_continue},
                "prior_strength": self.strength, "effective_prior_strength": strength,
                "population_responses": population["responses"], "population_fold_mean": p,
                "individual_weight": own["responses"] / (own["responses"] + strength),
                "population_sha256": self._population_hash,
                "interval": "90% Beta posterior conditional on the frozen population mean and chosen prior strength; excludes population and selection uncertainty"}

    def summary(self) -> dict:
        own = self._individual.summary()
        contexts = sorted(set(self._population.to_dict()["responses"]) | set(own["contexts"]))
        return {**own, "profile_type": "population_backed", "profile_sha256": self.fingerprint,
                "population_sha256": self._population_hash,
                "population_hands_observed": self._population.hands_observed,
                "population_total_responses": self._population.summary()["total_responses"],
                "prior_strength": self.strength,
                "contexts": {key: self.estimate(key) for key in contexts},
                "confidence_gate": "At least 24 matching individual responses; population support never satisfies this gate.",
                "scope": "Public fold/call/raise responses with a frozen population prior; intervals condition on the pool and chosen strength and exclude their uncertainty. Caller must exclude this individual's observations from the peer pool when evaluating historical data."}
