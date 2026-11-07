import hashlib
import json
import re
import secrets
import sqlite3
from base64 import b64encode
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from threading import RLock
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import ValidationError
from starlette.datastructures import Headers
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .arena import environment
from .engine import CARDS, Hand
from .models import Action, GameConfig
from .protocol import (
    MAX_EVENTS, MAX_HANDS, Command, Credentials, Receipt, TableState, WireConfig,
    WireEvent, checked_url, wire_observation, wire_result,
)


def _json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rules_identity() -> dict:
    root = Path(__file__).resolve().parent
    return {"pokerkit": version("pokerkit"),
            **{name: hashlib.sha256((root / name).read_bytes()).hexdigest()
               for name in ("engine.py", "models.py")}}


def _new_hand(config: GameConfig) -> dict:
    deck = list(CARDS)
    secrets.SystemRandom().shuffle(deck)
    hand = Hand(config, deck=tuple(deck))
    return {"deck": deck, "actions": []}


def _restore_hand(config: GameConfig, state: dict) -> Hand:
    hand = Hand(config, deck=tuple(state["hand"]["deck"]))
    for action in state["hand"]["actions"]:
        hand.act(Action(**action))
    return hand


def _write_private(path: Path, value: str) -> None:
    with path.open("x") as stream:
        path.chmod(0o600)
        stream.write(value)


def _error(code: str, message: str, status: int = 409) -> HTTPException:
    return HTTPException(status, {"code": code, "message": message})


class TableStore:
    def __init__(self, output: Path) -> None:
        self.output = output.resolve()
        self._lock = RLock()
        database = self.output / "table.sqlite3"
        if not database.is_file():
            raise ValueError("table.sqlite3 is missing; create a table before resuming")
        self.output.chmod(0o700)
        database.chmod(0o600)
        self._db = sqlite3.connect(database, check_same_thread=False)
        try:
            self._db.execute("PRAGMA busy_timeout = 5000")
            self._db.execute("PRAGMA synchronous = FULL")
            self._db.execute("PRAGMA journal_mode = WAL")
            self._initialize()
        except BaseException:
            self._db.close()
            raise

    def _initialize(self) -> None:
        metadata = self._db.execute("SELECT value FROM metadata WHERE id = 1").fetchone()
        if metadata is None:
            raise ValueError("table metadata is missing")
        self._metadata = json.loads(metadata[0])
        if type(self._metadata.get("schema_version")) is not int or self._metadata["schema_version"] != 1:
            raise ValueError("unsupported stored table schema")
        if self._metadata.get("rules_identity") != _rules_identity():
            raise ValueError("stored table rules differ from this engine; resume requires the original engine and PokerKit")
        self.table_id = self._metadata["table_id"]
        self.config = WireConfig.model_validate(self._metadata["config"]).to_domain()
        self.hands = self._metadata["hands_total"]
        self.base_url = checked_url(self._metadata["url"])
        try:
            self._credentials = tuple(Credentials.model_validate_json(_json(value))
                                      for value in self._metadata["credentials"])
        except (ValidationError, TypeError):
            raise ValueError("stored seat credentials are malformed") from None
        if (len(self._credentials) != 2
                or tuple(item.player_id for item in self._credentials) != (0, 1)
                or self._credentials[0].token == self._credentials[1].token
                or any(item.table_id != self.table_id or item.url != self.base_url
                       for item in self._credentials)):
            raise ValueError("stored seat credentials disagree with table metadata")
        for player in range(2):
            self.state(player)
        for credential in self._credentials:
            path = self.output / f"player-{credential.player_id}.json"
            if path.is_symlink():
                raise ValueError("seat credential files cannot be symbolic links")
            if not path.exists():
                _write_private(path, credential.model_dump_json(indent=2) + "\n")
                continue
            try:
                existing = Credentials.model_validate_json(path.read_bytes())
            except ValidationError:
                raise ValueError(f"{path.name} is malformed") from None
            if existing != credential:
                raise ValueError(f"{path.name} does not match the stored table credentials")
            path.chmod(0o600)

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def authenticate(self, authorization: str | None) -> int:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(401, "a seat bearer token is required", headers={"WWW-Authenticate": "Bearer"})
        supplied = authorization[7:]
        for credential in self._credentials:
            if secrets.compare_digest(supplied.encode("utf-8"), credential.token.encode("ascii")):
                return credential.player_id
        raise HTTPException(401, "invalid seat token", headers={"WWW-Authenticate": "Bearer"})

    def _load(self) -> dict:
        row = self._db.execute("SELECT value FROM state WHERE id = 1").fetchone()
        if row is None:
            raise ValueError("stored table state is missing")
        return json.loads(row[0])

    def _view(self, state: dict, player_id: int, hand: Hand | None = None) -> TableState:
        if type(player_id) is not int or player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        hand = hand or _restore_hand(self.config, state)
        seat = player_id ^ (state["hand_index"] % 2)
        observation = None
        if state["phase"] == "playing":
            current = hand.observation()
            if current.actor == seat:
                observation = wire_observation(current)
        result = wire_result(hand.result()) if state["phase"] in ("hand_complete", "complete") else None
        return TableState(
            table_id=self.table_id, player_id=player_id, opponent_id=1 - player_id,
            hero_seat=seat, hand_index=state["hand_index"], hands_total=self.hands,
            revision=state["revision"], phase=state["phase"], ready=tuple(state["ready"]),
            config=WireConfig.model_validate(asdict(self.config)), observation=observation,
            history=tuple(WireEvent.model_validate(asdict(event)) for event in hand.public_history),
            result=result,
        )

    def state(self, player_id: int) -> TableState:
        with self._lock:
            return self._view(self._load(), player_id)

    def command(self, player_id: int, command: Command) -> Receipt:
        if type(player_id) is not int or player_id not in (0, 1):
            raise ValueError("player_id must be 0 or 1")
        body = command.model_dump_json()
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                existing = self._db.execute(
                    "SELECT body, receipt FROM requests WHERE player_id = ? AND request_id = ?",
                    (player_id, command.request_id),
                ).fetchone()
                if existing is not None:
                    if existing[0] != body:
                        raise _error("request_conflict", "request_id was already used with a different command")
                    receipt = Receipt.model_validate_json(existing[1])
                    self._db.rollback()
                    return receipt
                state = self._load()
                if command.table_id != self.table_id:
                    raise _error("wrong_table", "command belongs to a different table")
                if command.hand_index != state["hand_index"] or command.revision != state["revision"]:
                    raise _error("stale_state", "stale hand or revision; fetch current state")
                hand = _restore_hand(self.config, state)
                if command.kind == "ready":
                    self._ready(state, player_id)
                    hand = _restore_hand(self.config, state)
                    if state["phase"] == "playing" and hand.finished:
                        self._finish_hand(state, hand)
                else:
                    self._act(state, hand, player_id, command)
                state["revision"] += 1
                self._view(state, 0, hand)
                self._view(state, 1, hand)
                receipt = Receipt(table_id=self.table_id, player_id=player_id,
                                  request_id=command.request_id, revision=state["revision"])
                self._db.execute("UPDATE state SET value = ? WHERE id = 1", (_json(state),))
                self._db.execute("INSERT INTO requests VALUES (?, ?, ?, ?)",
                                 (player_id, command.request_id, body, receipt.model_dump_json()))
                self._db.commit()
                return receipt
            except BaseException:
                self._db.rollback()
                raise

    def _ready(self, state: dict, player_id: int) -> None:
        if state["phase"] not in ("waiting", "hand_complete"):
            raise _error("not_ready", "ready is only available before play or after a completed hand")
        if state["ready"][player_id]:
            raise _error("not_ready", "this player is already ready")
        state["ready"][player_id] = True
        if not all(state["ready"]):
            return
        if state["phase"] == "waiting":
            state["phase"] = "playing"
            return
        if state["hand_index"] + 1 == self.hands:
            state["phase"] = "complete"
            return
        state["hand_index"] += 1
        state["phase"] = "playing"
        state["hand"] = _new_hand(self.config)

    def _act(self, state: dict, hand: Hand, player_id: int, command: Command) -> None:
        if state["phase"] != "playing":
            raise _error("not_ready", "the table is not awaiting an action")
        seat = player_id ^ (state["hand_index"] % 2)
        observation = hand.observation()
        if observation.actor != seat:
            raise _error("wrong_turn", "it is the other player's turn")
        if len(hand.public_history) >= MAX_EVENTS:
            raise _error("game_limit", "hand reached the supported public-history limit", 422)
        if command.action is None:
            raise _error("invalid_action", "an act command needs an action", 422)
        try:
            hand.act(command.action.to_domain())
        except ValueError as exc:
            raise _error("invalid_action", str(exc), 422) from exc
        state["hand"]["actions"].append(command.action.model_dump())
        if not hand.finished:
            return
        self._finish_hand(state, hand)

    def _finish_hand(self, state: dict, hand: Hand) -> None:
        state["phase"] = "hand_complete"
        state["ready"] = [False, False]
        record = {**hand.record(), "table_id": self.table_id, "hand_index": state["hand_index"],
                  "player_seats": [player ^ (state["hand_index"] % 2) for player in range(2)],
                  "mode": "private_play_money_connection"}
        self._db.execute("INSERT INTO hands VALUES (?, ?)", (state["hand_index"], _json(record)))

    def export_hands(self) -> Path:
        with self._lock:
            records = self._db.execute("SELECT record FROM hands ORDER BY hand_index").fetchall()
            destination = self.output / "hands.jsonl"
            temporary = self.output / f".hands-{uuid4().hex}.tmp"
            _write_private(temporary, "".join(record[0] + "\n" for record in records))
            temporary.replace(destination)
            return destination


def create_table(output: Path, config: GameConfig, hands: int, base_url: str) -> TableStore:
    wire_config = WireConfig.model_validate(asdict(config))
    if type(hands) is not int or not 1 <= hands <= MAX_HANDS:
        raise ValueError(f"hands must be an integer from 1 to {MAX_HANDS}")
    base_url = checked_url(base_url)
    output = output.resolve()
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    table_id = uuid4().hex
    credentials = [Credentials(table_id=table_id, player_id=player, url=base_url,
                               token=secrets.token_urlsafe(32)) for player in range(2)]
    metadata = {"schema_version": 1, "table_id": table_id, "config": wire_config.model_dump(),
                "hands_total": hands, "url": base_url,
                "rules_identity": _rules_identity(),
                "credentials": [item.model_dump() for item in credentials]}
    state = {"hand_index": 0, "revision": 0, "phase": "waiting", "ready": [False, False],
             "hand": _new_hand(config)}
    database = output / "table.sqlite3"
    _write_private(database, "")
    connection = sqlite3.connect(database)
    try:
        connection.executescript("""
            PRAGMA synchronous = FULL;
            CREATE TABLE metadata (id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT NOT NULL);
            CREATE TABLE state (id INTEGER PRIMARY KEY CHECK (id = 1), value TEXT NOT NULL);
            CREATE TABLE requests (
                player_id INTEGER NOT NULL, request_id TEXT NOT NULL, body TEXT NOT NULL,
                receipt TEXT NOT NULL, PRIMARY KEY (player_id, request_id)
            );
            CREATE TABLE hands (hand_index INTEGER PRIMARY KEY, record TEXT NOT NULL);
        """)
        connection.execute("INSERT INTO metadata VALUES (1, ?)", (_json(metadata),))
        connection.execute("INSERT INTO state VALUES (1, ?)", (_json(state),))
        connection.commit()
    finally:
        connection.close()
    manifest = {"schema_version": 1, "mode": "private_play_money_connection", "table_id": table_id,
                "created_utc": datetime.now(timezone.utc).isoformat(), "config": asdict(config),
                "hands_total": hands, "source_environment": environment(),
                "deck_generation": "secrets.SystemRandom; private host storage only",
                "recovery": "SQLite state, completed hands and command receipts commit atomically",
                "turn_timeout": None, "stakes": "play money only"}
    _write_private(output / "manifest.json", _json(manifest) + "\n")
    for credential in credentials:
        _write_private(output / f"player-{credential.player_id}.json", credential.model_dump_json(indent=2) + "\n")
    return TableStore(output)


def open_table(output: Path) -> TableStore:
    return TableStore(output)


class _CommandBodyLimit:
    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["method"] != "POST":
            await self.app(scope, receive, send)
            return
        length = Headers(scope=scope).get("content-length")
        if length is not None and (len(length) > 4 or not length.isascii()
                                   or not length.isdecimal() or int(length) > 4096):
            await JSONResponse({"detail": "command body exceeds 4096 bytes"}, status_code=413)(scope, receive, send)
            return
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            if len(body) + len(chunk) > 4096:
                await JSONResponse({"detail": "command body exceeds 4096 bytes"}, status_code=413)(scope, receive, send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break
        if length is not None and len(body) != int(length):
            await JSONResponse({"detail": "Content-Length disagrees with command body"}, status_code=400)(scope, receive, send)
            return
        delivered = False

        async def bounded_receive() -> Message:
            nonlocal delivered
            if delivered:
                return await receive()
            delivered = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        await self.app(scope, bounded_receive, send)


def create_app(store: TableStore) -> FastAPI:
    from .web import PAGE

    app = FastAPI(title="Heads-Up Poker Lab private table", docs_url=None,
                  redoc_url=None, openapi_url=None)
    app.add_middleware(_CommandBodyLimit)

    @app.middleware("http")
    async def prevent_caching(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    def player(authorization: str | None = Header(default=None)) -> int:
        return store.authenticate(authorization)

    @app.get("/", response_class=HTMLResponse)
    def browser_table() -> HTMLResponse:
        script_hashes = " ".join(
            f"'sha256-{b64encode(hashlib.sha256(script.encode()).digest()).decode()}'"
            for script in re.findall(r"<script>(.*?)</script>", PAGE, re.DOTALL)
        )
        return HTMLResponse(PAGE, headers={
            "Content-Security-Policy": (
                f"default-src 'none'; script-src {script_hashes}; style-src 'unsafe-inline'; "
                "connect-src 'self'; img-src data:; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
            ),
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
        })

    @app.get("/v1/state", response_model=TableState)
    def state(player_id: int = Depends(player)) -> TableState:
        return store.state(player_id)

    @app.post("/v1/command", response_model=Receipt)
    def command(body: Command, player_id: int = Depends(player)) -> Receipt:
        return store.command(player_id, body)

    return app
