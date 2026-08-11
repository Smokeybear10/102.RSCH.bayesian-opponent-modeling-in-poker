import fcntl
import hashlib
import json
import sqlite3
from copy import deepcopy
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from .connection import load_population_prior
from .learning import MIN_RESPONSES, OpponentProfile, PopulationBackedProfile
from .protocol import MAX_CHIPS, History, WireModel

PlayerId = Annotated[str, Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")]
RoomUrl = Annotated[str, Field(max_length=256, pattern=r"^https://www\.pokernow\.com/games/[A-Za-z0-9_-]+$")]


class OpponentIdentity(WireModel):
    platform: Literal["pokernow"] = "pokernow"
    room_url: RoomUrl
    hero_id: PlayerId
    opponent_id: PlayerId
    small_blind: Annotated[int, Field(gt=0, le=MAX_CHIPS)]
    big_blind: Annotated[int, Field(gt=0, le=MAX_CHIPS)]

    @model_validator(mode="after")
    def check_identity(self) -> "OpponentIdentity":
        if self.hero_id == self.opponent_id or self.small_blind > self.big_blind:
            raise ValueError("opponent identity requires distinct players and valid blinds")
        return self


class SavedMemory(WireModel):
    schema_version: Literal[1] = 1
    identity: OpponentIdentity
    profile: dict[str, object]
    profile_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    population_prior_path: str | None = None


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _memory_key(identity: OpponentIdentity) -> str:
    return hashlib.sha256(_json(identity.model_dump()).encode()).hexdigest()


def _profile_summary(identity: OpponentIdentity, profile: OpponentProfile,
                     path: Path, population_prior_path: str | None) -> dict:
    detail = profile.summary()
    population_backed = isinstance(profile, PopulationBackedProfile)
    contexts = {}
    for key, estimate in detail["contexts"].items():
        if not estimate["responses"]:
            continue
        street, position, betting_round, price_bucket = key.split(":")
        contexts[key] = {
            **estimate, "street": street, "position": position,
            "betting_round": betting_round, "price_bucket": price_bucket,
            "rates": {action: count / estimate["responses"]
                      for action, count in estimate["counts"].items()},
            "prior_source": "frozen_population" if population_backed else "built_in",
        }
    return {
        "enabled": True, "identity": identity.model_dump(),
        "memory": profile.memory, "profile_hands": profile.hands_observed,
        "profile_sha256": profile.fingerprint,
        "total_responses": detail["total_responses"],
        "ready_contexts": detail["ready_contexts"], "minimum_responses": MIN_RESPONSES,
        "population_prior_enabled": population_backed,
        "population_prior_sha256": detail.get("population_sha256"),
        "population_prior_path": population_prior_path, "state_path": str(path),
        "behavior": {
            "observation_unit": detail["observation_unit"],
            "window_size": detail["window_size"], "hand_horizon": detail["hand_horizon"],
            "scope": detail["scope"],
            "confidence_gate": "At least 24 matching individual responses; readiness alone does not force a strategy adjustment.",
            "contexts": contexts,
        },
    }


def read_opponent_profiles(directory: Path, opponent_id: str | None = None,
                           room_url: str | None = None) -> list[dict]:
    if opponent_id is not None:
        TypeAdapter(PlayerId).validate_python(opponent_id, strict=True)
    if room_url is not None:
        TypeAdapter(RoomUrl).validate_python(room_url, strict=True)
    directory = directory.resolve()
    if not directory.exists():
        return []
    if not directory.is_dir():
        raise ValueError("opponent memory path must be a directory")
    profiles = []
    for path in sorted(directory.glob("*.sqlite3")):
        db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
        try:
            db.execute("BEGIN")
            row = db.execute("SELECT data FROM memory WHERE id=1").fetchone()
            completed = db.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
        finally:
            db.close()
        if row is None:
            raise ValueError(f"opponent memory is missing its saved profile: {path}")
        saved = SavedMemory.model_validate_json(row[0])
        profile = OpponentProfile.from_dict(saved.profile)
        if (path.stem != _memory_key(saved.identity)
                or profile.fingerprint != saved.profile_sha256
                or profile.hands_observed != completed):
            raise ValueError(f"saved opponent memory identity, fingerprint or counts disagree: {path}")
        if opponent_id is not None and saved.identity.opponent_id != opponent_id:
            continue
        if room_url is not None and saved.identity.room_url != room_url:
            continue
        profiles.append(_profile_summary(saved.identity, profile, path, saved.population_prior_path))
    return profiles


def completion_fingerprint(hand_id: str, hero_seat: int, history: History) -> str:
    public = {"hand_id": hand_id, "hero_seat": hero_seat,
              "history": [event.model_dump(mode="json") for event in history]}
    return hashlib.sha256(_json(public).encode()).hexdigest()


class OpponentMemory:
    def __init__(self, directory: Path, identity: OpponentIdentity, memory: str = "lifetime",
                 population_prior: Path | None = None) -> None:
        initial = OpponentProfile(memory)
        prior = load_population_prior(population_prior) if population_prior is not None else None
        if prior is not None:
            initial = PopulationBackedProfile(OpponentProfile.from_dict(prior["population"]),
                                              prior["strength"], memory)
        self.identity = identity
        self.population_prior_path = str(population_prior.resolve()) if population_prior else None
        directory = directory.resolve()
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        key = _memory_key(identity)
        self.path = directory / f"{key}.sqlite3"
        lock_path = directory / f"{key}.lock"
        self.lock = lock_path.open("a+")
        lock_path.chmod(0o600)
        self.db = None
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ValueError("opponent memory is already in use") from None
        try:
            self.db = sqlite3.connect(self.path)
            self.path.chmod(0o600)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS memory (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS completed (
                    hand_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL,
                    hero_seat INTEGER NOT NULL CHECK(hero_seat IN (0,1)), history TEXT NOT NULL
                );
            """)
            row = self.db.execute("SELECT data FROM memory WHERE id=1").fetchone()
            completed = self.db.execute("SELECT COUNT(*) FROM completed").fetchone()[0]
            if row is None:
                if completed:
                    raise ValueError("opponent memory is missing its saved profile")
                self.profile = initial
                with self.db:
                    self.db.execute("INSERT INTO memory VALUES (1, ?)", (self._saved(initial),))
            else:
                saved = SavedMemory.model_validate_json(row[0])
                self.profile = OpponentProfile.from_dict(saved.profile)
                if (saved.identity != identity or self.profile.memory != memory
                        or self.profile.fingerprint != saved.profile_sha256
                        or self.profile.hands_observed != completed):
                    raise ValueError("saved opponent memory identity, settings or counts disagree")
                existing = self.profile.to_dict()
                if prior is not None and (existing.get("schema_version") != 3
                        or existing["population"] != prior["population"]
                        or existing["strength"] != prior["strength"]):
                    raise ValueError("population prior differs from saved opponent memory")
                self.population_prior_path = self.population_prior_path or saved.population_prior_path
            self._refresh_summary()
        except BaseException:
            self.close()
            raise

    def _saved(self, profile: OpponentProfile) -> str:
        return SavedMemory(identity=self.identity, profile=profile.to_dict(),
                           profile_sha256=profile.fingerprint,
                           population_prior_path=self.population_prior_path).model_dump_json()

    def _refresh_summary(self) -> None:
        self._summary = _profile_summary(self.identity, self.profile, self.path,
                                         self.population_prior_path)

    def summary(self) -> dict:
        return deepcopy(self._summary)

    def completed_fingerprint(self, hand_id: str) -> str | None:
        row = self.db.execute("SELECT fingerprint FROM completed WHERE hand_id=?", (hand_id,)).fetchone()
        return row[0] if row else None

    def contains(self, hand_id: str) -> bool:
        return self.completed_fingerprint(hand_id) is not None

    def learn(self, hand_id: str, hero_seat: int, history: History) -> bool:
        fingerprint = completion_fingerprint(hand_id, hero_seat, history)
        previous = self.completed_fingerprint(hand_id)
        if previous is not None:
            if previous != fingerprint:
                raise ValueError("completed_history_changed")
            return False
        candidate = OpponentProfile.from_dict(self.profile.to_dict())
        candidate.observe_hand(tuple(event.to_domain() for event in history), 1 - hero_seat)
        with self.db:
            self.db.execute("INSERT INTO completed VALUES (?, ?, ?, ?)",
                            (hand_id, fingerprint, hero_seat,
                             _json([event.model_dump(mode="json") for event in history])))
            self.db.execute("UPDATE memory SET data=? WHERE id=1", (self._saved(candidate),))
        self.profile = candidate
        self._refresh_summary()
        return True

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
            self.db = None
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()
