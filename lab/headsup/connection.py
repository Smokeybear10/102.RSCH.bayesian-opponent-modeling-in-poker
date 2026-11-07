import fcntl
import json
import math
import secrets
import sqlite3
from dataclasses import asdict
from pathlib import Path
from time import monotonic, sleep
from typing import Annotated, Callable, Literal

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .arena import environment, seed_for
from .learning import OpponentProfile, PopulationBackedProfile
from .models import MAX_SAMPLES
from .policies import LEARNING_POLICY_NAMES, POLICY_NAMES, make_adaptive_policy, make_policy, make_river_policy
from .protocol import Command, Credentials, Receipt, TableState, WireAction

MAX_RESPONSE_BYTES = 1_048_576
MAX_PRIOR_BYTES = 1_048_576


class ConnectionFailure(ValueError):
    pass


class StaleState(Exception):
    pass


class _PopulationPriorArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Annotated[int, Field(ge=1, le=1)]
    kind: Literal["population-fold-prior"]
    population: dict[str, object]
    strength: Annotated[float, Field(ge=1, le=10000, allow_inf_nan=False)]
    population_sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
    training: object = None
    calibration: object = None


def _unique_json_fields(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON field")
        result[key] = value
    return result


def _finite_json_number(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("nonfinite JSON number")
    return number


def load_population_prior(path: Path) -> dict:
    with path.open("rb") as source:
        data = source.read(MAX_PRIOR_BYTES + 1)
    if len(data) > MAX_PRIOR_BYTES:
        raise ConnectionFailure("population prior exceeds the 1 MiB size limit")
    try:
        value = json.loads(data, object_pairs_hook=_unique_json_fields,
                           parse_float=_finite_json_number, parse_constant=_finite_json_number)
        artifact = _PopulationPriorArtifact.model_validate(value)
        if type(artifact.population.get("schema_version")) is not int or artifact.population["schema_version"] != 1:
            raise ValueError("population must use schema 1")
        population = OpponentProfile.from_dict(artifact.population)
        if population.fingerprint != artifact.population_sha256:
            raise ValueError("population fingerprint does not match")
    except (ValueError, RecursionError):
        raise ConnectionFailure("invalid population prior: expected schema 1, matching fingerprint, and finite strength from 1 to 10000") from None
    return {"population": population.to_dict(), "strength": artifact.strength,
            "population_sha256": population.fingerprint}


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), sort_keys=True, allow_nan=False)


def _tuples(value: object) -> object:
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


class ClientStore:
    def __init__(self, directory: Path, credentials: Credentials, settings: dict) -> None:
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
        self.directory = directory
        self.lock = (directory / "client.lock").open("a+")
        (directory / "client.lock").chmod(0o600)
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.lock.close()
            raise ConnectionFailure("another client is using this state directory") from None
        self.db = None
        try:
            self.db = sqlite3.connect(directory / "client.sqlite3")
            (directory / "client.sqlite3").chmod(0o600)
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), data TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS commands (
                    request_id TEXT PRIMARY KEY, hand_index INTEGER NOT NULL, command TEXT NOT NULL,
                    observation TEXT, info TEXT, rng_after TEXT,
                    status TEXT NOT NULL, receipt TEXT
                );
                CREATE INDEX IF NOT EXISTS commands_by_hand ON commands(hand_index, status);
                CREATE TABLE IF NOT EXISTS completed (
                    hand_index INTEGER PRIMARY KEY, state TEXT NOT NULL,
                    profile_before TEXT NOT NULL, profile_after TEXT NOT NULL
                );
            """)
            identity = {"schema_version": 1, "table_id": credentials.table_id,
                        "player_id": credentials.player_id, "url": credentials.url,
                        "settings": settings, "source_environment": environment()}
            row = self.db.execute("SELECT data FROM state WHERE id=1").fetchone()
            if row:
                self.data = json.loads(row[0])
                if self.data["identity"] != identity:
                    raise ConnectionFailure("saved client identity, settings or implementation changed; use its original environment")
                OpponentProfile.from_dict(self.data["profile"])
            else:
                prior = settings.get("population_prior")
                profile = (OpponentProfile(settings["memory"]) if prior is None else
                           PopulationBackedProfile(OpponentProfile.from_dict(prior["population"]),
                                                   strength=prior["strength"], memory=settings["memory"]))
                self.data = {"identity": identity, "master_seed": secrets.randbits(256),
                             "profile": profile.to_dict(),
                             "last_completed": -1, "last_state": None, "config": None,
                             "hand_index": None, "hand_profile": None, "rng": None,
                             "hole": None, "pending": None, "min_revision": 0,
                             "net_chips": 0, "hands_total": None}
                self.save()
        except BaseException:
            self.close()
            raise

    def save(self) -> None:
        self.db.execute("INSERT OR REPLACE INTO state VALUES (1, ?)", (_json(self.data),))
        self.db.commit()

    def _persist(self) -> None:
        self.db.execute("UPDATE state SET data=? WHERE id=1", (_json(self.data),))

    def pending(self) -> Command | None:
        request_id = self.data["pending"]
        if request_id is None:
            return None
        row = self.db.execute("SELECT command, status FROM commands WHERE request_id=?", (request_id,)).fetchone()
        if row is None or row[1] != "pending":
            raise ConnectionFailure("pending command journal is inconsistent")
        return Command.model_validate_json(row[0])

    def observe(self, state: TableState) -> None:
        identity = self.data["identity"]
        if state.table_id != identity["table_id"] or state.player_id != identity["player_id"]:
            raise ConnectionFailure("server returned a different table or player")
        if state.revision < self.data["min_revision"]:
            raise ConnectionFailure("server state predates an accepted command")
        config = state.config.model_dump(mode="json")
        if self.data["config"] is not None and config != self.data["config"]:
            raise ConnectionFailure("table rules changed during the match")
        if self.data["hands_total"] is not None and state.hands_total != self.data["hands_total"]:
            raise ConnectionFailure("match length changed")
        previous_data = self.data["last_state"]
        previous = TableState.model_validate_json(_json(previous_data)) if previous_data is not None else None
        if previous is None:
            if state.phase != "waiting" or state.hand_index != 0 or state.ready[state.player_id]:
                raise ConnectionFailure("this seat has already started; reconnect with its original client state directory")
        else:
            if state.revision < previous.revision or state.hand_index < previous.hand_index:
                raise ConnectionFailure("server state moved backwards")
            if state.revision == previous.revision and state != previous:
                raise ConnectionFailure("server changed state without advancing its revision")
            if state.hand_index == previous.hand_index and state.history[:len(previous.history)] != previous.history:
                raise ConnectionFailure("server rewrote the public action history")
            if (state.hand_index == previous.hand_index and previous.result is not None
                    and state.result != previous.result):
                raise ConnectionFailure("server changed an already observed settlement")
            if state.hand_index > previous.hand_index + 1:
                raise ConnectionFailure("server skipped a hand")
        completed = self.data["last_completed"]
        if state.hand_index > completed + 1:
            raise ConnectionFailure("a completed hand was missed; learning cannot continue safely")
        if state.hand_index < completed or (state.hand_index == completed and state.phase in ("waiting", "playing")):
            raise ConnectionFailure("a settled hand became active again")
        if state.hand_index != self.data["hand_index"]:
            self.data.update(hand_index=state.hand_index, hand_profile=self.data["profile"], rng=None, hole=None)
        if state.observation is not None:
            hole = list(state.observation.hole)
            if self.data["hole"] is not None and self.data["hole"] != hole:
                raise ConnectionFailure("own cards changed within the hand")
            self.data["hole"] = hole
        visible_board = (state.result.board if state.result is not None else
                         state.observation.board if state.observation is not None else
                         state.history[-1].board if state.history else ())
        if self.data["hole"] is not None and set(self.data["hole"]).intersection(visible_board):
            raise ConnectionFailure("public board contains one of this player's private cards")
        own_events = set()
        for command_json, observation_json in self.db.execute(
                "SELECT command, observation FROM commands WHERE hand_index=? "
                "AND status='accepted' AND observation IS NOT NULL", (state.hand_index,)):
            command = Command.model_validate_json(command_json)
            observation = json.loads(observation_json)
            index = len(observation["history"])
            if index >= len(state.history):
                raise ConnectionFailure("an accepted action is missing from the public history")
            event = state.history[index]
            if event.actor != state.hero_seat or event.action != command.action:
                raise ConnectionFailure("an accepted action differs from the public history")
            own_events.add(index)
        if any(event.actor == state.hero_seat and index not in own_events
               for index, event in enumerate(state.history)):
            raise ConnectionFailure("another client acted for this seat; resume its original client state")
        self.data.update(config=config, hands_total=state.hands_total,
                         last_state=state.model_dump(mode="json"))
        self.save()

    def learn(self, state: TableState) -> bool:
        if state.phase not in ("hand_complete", "complete") or state.result is None:
            raise ConnectionFailure("learning requires a completed hand")
        if state.hand_index == self.data["last_completed"]:
            return False
        if state.hand_index != self.data["last_completed"] + 1:
            raise ConnectionFailure("completed hand sequence is incomplete")
        profile = OpponentProfile.from_dict(self.data["profile"])
        profile.observe_hand(tuple(event.to_domain() for event in state.history), 1 - state.hero_seat)
        with self.db:
            self.db.execute("INSERT INTO completed VALUES (?, ?, ?, ?)",
                            (state.hand_index, state.model_dump_json(), _json(self.data["profile"]),
                             _json(profile.to_dict())))
            self.data.update(profile=profile.to_dict(), last_completed=state.hand_index,
                             net_chips=self.data["net_chips"] + state.result.net_returns[state.hero_seat])
            self._persist()
        return True

    def propose(self, state: TableState, action: WireAction | None = None,
                info: dict | None = None, rng_after: tuple | None = None) -> Command:
        if self.data["pending"] is not None:
            raise ConnectionFailure("cannot replace an unacknowledged command")
        command = Command(table_id=state.table_id, request_id=secrets.token_hex(16),
                          hand_index=state.hand_index, revision=state.revision,
                          kind="act" if action is not None else "ready", action=action)
        observation = state.observation.model_dump_json() if action is not None else None
        with self.db:
            self.db.execute("INSERT INTO commands VALUES (?, ?, ?, ?, ?, ?, 'pending', NULL)",
                            (command.request_id, command.hand_index, command.model_dump_json(), observation,
                             _json(info) if info is not None else None,
                             _json(rng_after) if rng_after is not None else None))
            self.data["pending"] = command.request_id
            self._persist()
        return command

    def acknowledge(self, command: Command, receipt: Receipt) -> None:
        if (receipt.table_id != command.table_id or receipt.player_id != self.data["identity"]["player_id"]
                or receipt.request_id != command.request_id or receipt.revision != command.revision + 1):
            raise ConnectionFailure("server returned an inconsistent command receipt")
        row = self.db.execute("SELECT rng_after FROM commands WHERE request_id=?", (command.request_id,)).fetchone()
        if self.data["pending"] != command.request_id or row is None:
            raise ConnectionFailure("receipt has no matching pending command")
        with self.db:
            self.db.execute("UPDATE commands SET status='accepted', receipt=? WHERE request_id=?",
                            (receipt.model_dump_json(), command.request_id))
            self.data.update(pending=None, min_revision=max(self.data["min_revision"], receipt.revision))
            if row[0] is not None:
                self.data["rng"] = json.loads(row[0])
            self._persist()

    def reject_stale(self, command: Command) -> None:
        if self.data["pending"] != command.request_id:
            raise ConnectionFailure("stale response has no matching pending command")
        with self.db:
            self.db.execute("UPDATE commands SET status='stale' WHERE request_id=?", (command.request_id,))
            self.data["pending"] = None
            self._persist()

    def decision(self, state: TableState, name: str, samples: int) -> tuple[WireAction, dict, tuple]:
        profile = OpponentProfile.from_dict(self.data["hand_profile"])
        seed = seed_for(self.data["master_seed"], "network-hand", state.table_id,
                        state.hand_index, state.player_id)
        if name in LEARNING_POLICY_NAMES:
            factory = make_river_policy if name == "river" else make_adaptive_policy
            policy = factory(seed, samples, profile)
        else:
            policy = make_policy(name, seed, samples)
        if self.data["rng"] is not None:
            policy.rng.setstate(_tuples(self.data["rng"]))
        action = WireAction.model_validate(asdict(policy.act(state.observation.to_domain())))
        if not state.observation.accepts(action):
            raise ConnectionFailure("bot proposed an illegal action; no action was sent")
        return action, policy.last_info, policy.rng.getstate()

    def report(self, status: str, error: str | None) -> dict:
        self.data = json.loads(self.db.execute("SELECT data FROM state WHERE id=1").fetchone()[0])
        accepted = self.db.execute("SELECT COUNT(*) FROM commands WHERE status='accepted' AND observation IS NOT NULL").fetchone()[0]
        report = {"schema_version": 1, "status": status, "table_id": self.data["identity"]["table_id"],
                  "player_id": self.data["identity"]["player_id"],
                  "completed_hands": self.data["last_completed"] + 1, "accepted_actions": accepted,
                  "net_chips": self.data["net_chips"], "pending_preserved": self.data["pending"] is not None,
                  "profile_hands": self.data["profile"]["hands_observed"],
                  "state_path": str(self.directory / "client.sqlite3"),
                  "profile_path": str(self.directory / "profile.json"), "error": error,
                  "opponent_type": "not verified; private table participants may be humans or bots",
                  "playing_strength_established": False}
        prior = self.data["identity"]["settings"].get("population_prior")
        report["population_prior_enabled"] = prior is not None
        if prior is not None:
            report.update(population_sha256=prior["population_sha256"], population_strength=prior["strength"])
        for name, value in (("report.json", report), ("profile.json", self.data["profile"])):
            temporary = self.directory / f"{name}.tmp"
            with temporary.open("w") as output:
                temporary.chmod(0o600)
                output.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
            temporary.replace(self.directory / name)
        return report

    def close(self) -> None:
        if self.db is not None:
            self.db.close()
        if not self.lock.closed:
            fcntl.flock(self.lock, fcntl.LOCK_UN)
            self.lock.close()


class TableConnection:
    def __init__(self, credentials: Credentials, retry_seconds: float,
                 progress: Callable[[str], None]) -> None:
        self.credentials = credentials
        self.retry_seconds = retry_seconds
        self.progress = progress
        self.http = httpx.Client(base_url=credentials.url, timeout=5.0,
                                 headers={"Authorization": f"Bearer {credentials.token}"},
                                 follow_redirects=False, trust_env=False)

    def request(self, method: str, path: str, command: Command | None = None) -> bytes:
        deadline = monotonic() + self.retry_seconds
        delay = 0.25
        warned = False
        while True:
            try:
                with self.http.stream(method, path, json=command.model_dump(mode="json") if command else None) as response:
                    data = bytearray()
                    for chunk in response.iter_bytes(chunk_size=65536):
                        data.extend(chunk)
                        if len(data) > MAX_RESPONSE_BYTES:
                            raise ConnectionFailure("table response exceeds the protocol size limit")
                    if response.status_code == 200:
                        return bytes(data)
                    if response.status_code in (401, 403):
                        raise ConnectionFailure("table rejected these seat credentials")
                    if response.status_code == 409 and command is not None:
                        try:
                            detail = json.loads(data).get("detail")
                            code = detail.get("code") if isinstance(detail, dict) else detail
                        except (ValueError, AttributeError):
                            code = None
                        if code == "stale_state":
                            raise StaleState
                        raise ConnectionFailure("table rejected a conflicting command; pending request retained")
                    if response.status_code not in (500, 502, 503, 504):
                        raise ConnectionFailure(f"table rejected the request (HTTP {response.status_code}); pending request retained")
            except httpx.RequestError:
                pass
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise ConnectionFailure("connection retry deadline reached; restart with the same client state directory")
            if not warned:
                self.progress("Connection interrupted; retrying the same request.")
                warned = True
            sleep(min(delay, remaining))
            delay = min(2.0, delay * 2)

    def state(self) -> TableState:
        try:
            return TableState.model_validate_json(self.request("GET", "/v1/state"))
        except ValidationError:
            raise ConnectionFailure("table returned an invalid state snapshot") from None

    def send(self, command: Command) -> Receipt:
        try:
            return Receipt.model_validate_json(self.request("POST", "/v1/command", command))
        except ValidationError:
            raise ConnectionFailure("table returned an invalid command receipt; pending request retained") from None

    def close(self) -> None:
        self.http.close()


def _human_action(state: TableState, progress: Callable[[str], None]) -> WireAction | None:
    observation = state.observation
    role = "button / small blind" if state.hero_seat == 1 else "big blind"
    progress(f"Hand {state.hand_index + 1} | {observation.street} | {role}")
    progress(f"Your cards: {' '.join(observation.hole)} | Board: {' '.join(observation.board) or '—'}")
    progress(f"Pot: {observation.pot} | Your stack: {observation.stacks[state.hero_seat]} | To call: {observation.to_call}")
    if observation.min_raise_to is not None:
        progress(f"Raise-to range: {observation.min_raise_to}..{observation.max_raise_to} (total on this street)")
    while True:
        words = input("Action [check/call/fold/r TOTAL/q]: ").strip().lower().split()
        if words == ["q"]:
            return None
        aliases = {"c": "call", "k": "check", "f": "fold", "r": "raise_to", "raise": "raise_to"}
        try:
            if not words:
                raise ValueError
            kind = aliases.get(words[0], words[0])
            if kind == "raise_to" and len(words) == 2:
                action = WireAction(kind="raise_to", amount=int(words[1]))
            elif len(words) == 1:
                action = WireAction(kind=kind)
            else:
                raise ValueError
            if not observation.accepts(action):
                raise ValueError
            return action
        except ValueError:
            progress("That action is unavailable. Use a legal check, call, fold or raise-to total.")


def run_client(credentials: Path, state_dir: Path, policy: str = "adaptive", samples: int = 128,
               memory: str = "lifetime", human: bool = False, poll_seconds: float = 0.25,
               retry_seconds: float = 30.0, progress: Callable[[str], None] = print,
               population_prior: Path | None = None) -> dict:
    if policy not in (*POLICY_NAMES, *LEARNING_POLICY_NAMES) or memory not in ("lifetime", "recent"):
        raise ValueError("unsupported policy or memory mode")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be 1..{MAX_SAMPLES}")
    if (not math.isfinite(poll_seconds) or not 0.01 <= poll_seconds <= 10
            or not math.isfinite(retry_seconds) or not 0 <= retry_seconds <= 300):
        raise ValueError("poll_seconds must be 0.01..10 and retry_seconds 0..300")
    if population_prior is not None and (human or policy not in LEARNING_POLICY_NAMES):
        raise ValueError("a population prior requires an adaptive or river bot client")
    prior = load_population_prior(population_prior) if population_prior is not None else None
    try:
        seat = Credentials.model_validate_json(credentials.read_bytes())
    except ValidationError:
        raise ConnectionFailure("invalid seat credential file") from None
    settings = {"policy": "human" if human else policy, "samples": samples, "memory": memory}
    if prior is not None:
        settings["population_prior"] = prior
    store = ClientStore(state_dir, seat, settings)
    connection = TableConnection(seat, retry_seconds, progress)
    status, error = "incomplete", None
    try:
        progress("Private play-money arena. Both seats may be operated by bots; no chips have monetary value.")
        progress(f"Using player {seat.player_id}; client state: {state_dir}")
        while True:
            pending = store.pending()
            if pending is not None:
                try:
                    receipt = connection.send(pending)
                except StaleState:
                    store.reject_stale(pending)
                else:
                    store.acknowledge(pending, receipt)
                continue
            state = connection.state()
            store.observe(state)
            if state.phase in ("hand_complete", "complete"):
                if store.learn(state):
                    progress(f"Hand {state.hand_index + 1}/{state.hands_total} complete: "
                             f"{state.result.net_returns[state.hero_seat]:+d} chips; "
                             f"opponent memory covers {store.data['profile']['hands_observed']} hands.")
                if state.phase == "complete":
                    status = "completed"
                    break
            if state.phase in ("waiting", "hand_complete") and not state.ready[state.player_id]:
                store.propose(state)
                continue
            if state.observation is not None:
                if human:
                    action = _human_action(state, progress)
                    if action is None:
                        break
                    store.propose(state, action, {"policy": "human"})
                else:
                    action, info, rng = store.decision(state, policy, samples)
                    store.propose(state, action, info, rng)
                continue
            sleep(poll_seconds)
    except (KeyboardInterrupt, EOFError):
        progress("Stopped. Restart with the same seat file and client state directory to resume.")
    except (ConnectionFailure, ValueError, OSError, sqlite3.Error) as exc:
        status = "failed"
        error = str(exc) if isinstance(exc, ConnectionFailure) else f"client {type(exc).__name__}; saved state retained"
        progress(error)
    finally:
        connection.close()
        try:
            report = store.report(status, error)
        finally:
            store.close()
    return report
