import hashlib
import json
import re
import tempfile
import unittest
from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from fastapi.testclient import TestClient
from httpx import Response

from headsup.arena import replay
from headsup.models import GameConfig
from headsup.table import create_app, create_table, open_table


class PrivateTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.output = Path(self.temporary.name) / "table"
        self.config = GameConfig(rake_bps=2500, rake_cap=3)
        self.store = create_table(self.output, self.config, 2, "http://127.0.0.1:8765")
        self.addCleanup(lambda: self.store.close())
        self.credentials = [json.loads((self.output / f"player-{player}.json").read_text())
                            for player in range(2)]
        self.client = TestClient(create_app(self.store))
        self.addCleanup(lambda: self.client.close())
        self.request_number = 0

    def headers(self, player: int) -> dict:
        return {"Authorization": f"Bearer {self.credentials[player]['token']}"}

    def state(self, player: int = 0) -> dict:
        response = self.client.get("/v1/state", headers=self.headers(player))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def body(self, player: int, kind: str = "ready", action: dict | None = None) -> dict:
        state = self.state(player)
        self.request_number += 1
        body = {"table_id": state["table_id"], "request_id": f"{self.request_number:032x}",
                "hand_index": state["hand_index"], "revision": state["revision"], "kind": kind}
        if action is not None:
            body["action"] = action
        return body

    def post(self, player: int, body: dict) -> Response:
        return self.client.post("/v1/command", headers=self.headers(player), json=body)

    def command(self, player: int, kind: str = "ready", action: dict | None = None) -> dict:
        body = self.body(player, kind, action)
        response = self.post(player, body)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["revision"], body["revision"] + 1)
        return response.json()

    def ready_both(self) -> None:
        self.command(0)
        self.command(1)

    def acting_player(self) -> int:
        states = [self.state(player) for player in range(2)]
        acting = [player for player, state in enumerate(states) if state["observation"] is not None]
        self.assertEqual(len(acting), 1)
        return acting[0]

    def finish_passively(self) -> dict:
        for _ in range(10):
            if self.state()["phase"] != "playing":
                return self.state()
            player = self.acting_player()
            observation = self.state(player)["observation"]
            self.command(player, "act", {"kind": "call" if observation["to_call"] else "check"})
        self.fail("a call/check-only heads-up hand should settle within eight decisions")

    def restart(self) -> None:
        self.client.close()
        self.store.close()
        self.store = open_table(self.output)
        self.client = TestClient(create_app(self.store))

    def assert_error(self, response: Response, status: int, code: str) -> None:
        self.assertEqual(response.status_code, status, response.text)
        self.assertEqual(response.json()["detail"]["code"], code)

    def test_bearer_authentication_is_required_for_state_and_commands(self) -> None:
        command = self.body(0)
        for authorization in (None, "", "Basic placeholder", "Bearer invalid", "Bearer "):
            headers = {} if authorization is None else {"Authorization": authorization}
            with self.subTest(authorization=authorization):
                for response in (self.client.get("/v1/state", headers=headers),
                                 self.client.post("/v1/command", headers=headers, json=command)):
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["www-authenticate"], "Bearer")
                    self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(self.state()["revision"], 0)
        self.assertNotEqual(self.credentials[0]["token"], self.credentials[1]["token"])
        self.assertEqual([self.state(player)["player_id"] for player in range(2)], [0, 1])

    def test_both_players_must_be_ready_before_cards_or_actions_are_available(self) -> None:
        self.assertEqual(self.state()["phase"], "waiting")
        self.assertIsNone(self.state(1)["observation"])
        before = self.state()
        self.assert_error(self.post(1, self.body(1, "act", {"kind": "call"})), 409, "not_ready")
        self.assertEqual(self.state(), before)
        self.command(0)
        self.assertEqual((self.state()["phase"], self.state()["ready"]), ("waiting", [True, False]))
        self.assertIsNone(self.state(0)["observation"])
        self.assertIsNone(self.state(1)["observation"])
        self.assert_error(self.post(0, self.body(0)), 409, "not_ready")
        self.command(1)
        self.assertEqual((self.state()["phase"], self.state()["revision"]), ("playing", 2))
        self.assertEqual(self.acting_player(), 1)

    def test_only_acting_player_receives_own_hole_cards(self) -> None:
        self.ready_both()
        first = self.state(1)
        self.assertIsNone(self.state(0)["observation"])
        self.assertEqual((first["hero_seat"], first["observation"]["actor"]), (1, 1))
        first_cards = set(first["observation"]["hole"])
        self.assertEqual(len(first_cards), 2)
        self.command(1, "act", {"kind": "call"})
        second = self.state(0)
        self.assertIsNone(self.state(1)["observation"])
        self.assertEqual(second["observation"]["actor"], 0)
        second_cards = set(second["observation"]["hole"])
        self.assertEqual(len(first_cards | second_cards), 4)
        for player in range(2):
            state = self.state(player)
            self.assertFalse({"deck", "seed", "hole", "opponent_hole", "credentials", "token"} & state.keys())
            self.assertEqual(state["history"][0]["action"]["kind"], "call")
            self.assertNotIn("hole", json.dumps(state["history"]))
            self.assertEqual(state["player_id"], player)
            self.assertEqual(state["opponent_id"], 1 - player)

    def test_wrong_player_cannot_act_or_select_another_identity(self) -> None:
        self.ready_both()
        before = [self.state(player) for player in range(2)]
        self.assert_error(self.post(0, self.body(0, "act", {"kind": "fold"})), 409, "wrong_turn")
        impersonation = self.body(0, "act", {"kind": "fold"}) | {"player_id": 1}
        self.assertEqual(self.post(0, impersonation).status_code, 422)
        self.assertEqual([self.state(player) for player in range(2)], before)

    def test_stale_revision_hand_and_wrong_table_do_not_mutate_state(self) -> None:
        self.ready_both()
        before = self.state(1)
        for changes, code in (({"revision": 0}, "stale_state"),
                              ({"hand_index": 1}, "stale_state"),
                              ({"table_id": "f" * 32}, "wrong_table")):
            with self.subTest(changes=changes):
                command = self.body(1, "act", {"kind": "call"}) | changes
                self.assert_error(self.post(1, command), 409, code)
                self.assertEqual(self.state(1), before)

    def test_illegal_poker_actions_roll_back_and_request_can_be_corrected(self) -> None:
        self.ready_both()
        before = self.state(1)
        for action in ({"kind": "check"}, {"kind": "raise_to", "amount": 3},
                       {"kind": "raise_to", "amount": 201}):
            with self.subTest(action=action):
                command = self.body(1, "act", action)
                self.assert_error(self.post(1, command), 422, "invalid_action")
                self.assertEqual(self.state(1), before)
        command["action"] = {"kind": "call"}
        self.assertEqual(self.post(1, command).status_code, 200)
        self.assertEqual(len(self.state()["history"]), 1)

    def test_exact_request_replay_returns_original_receipt_without_reapplying(self) -> None:
        self.ready_both()
        command = self.body(1, "act", {"kind": "call"})
        receipt = self.post(1, command)
        self.assertEqual(receipt.status_code, 200)
        self.command(0, "act", {"kind": "check"})
        before = self.state(1)
        replayed = self.post(1, command)
        self.assertEqual(replayed.status_code, 200)
        self.assertEqual(replayed.json(), receipt.json())
        self.assertEqual(self.state(1), before)

    def test_changed_body_for_committed_request_is_a_conflict(self) -> None:
        command = self.body(0)
        self.assertEqual(self.post(0, command).status_code, 200)
        before = self.state()
        changed = command | {"revision": before["revision"]}
        self.assert_error(self.post(0, changed), 409, "request_conflict")
        self.assertEqual(self.state(), before)

    def test_request_identifiers_are_scoped_to_authenticated_player(self) -> None:
        first = self.body(0)
        self.assertEqual(self.post(0, first).status_code, 200)
        second = self.body(1) | {"request_id": first["request_id"]}
        response = self.post(1, second)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["player_id"], 1)
        self.assertEqual(self.state()["phase"], "playing")

    def test_simultaneous_duplicate_commands_commit_once(self) -> None:
        self.ready_both()
        command = self.body(1, "act", {"kind": "call"})
        with ThreadPoolExecutor(max_workers=4) as executor:
            responses = list(executor.map(lambda _: self.post(1, command), range(4)))
        self.assertEqual([response.status_code for response in responses], [200] * 4)
        self.assertTrue(all(response.json() == responses[0].json() for response in responses))
        self.assertEqual(self.state()["revision"], command["revision"] + 1)
        self.assertEqual(len(self.state()["history"]), 1)

    def test_concurrent_distinct_commands_for_one_revision_accept_only_one(self) -> None:
        self.ready_both()
        commands = [self.body(1, "act", {"kind": "call"}) for _ in range(2)]
        with ThreadPoolExecutor(max_workers=2) as executor:
            responses = list(executor.map(lambda command: self.post(1, command), commands))
        self.assertEqual(sorted(response.status_code for response in responses), [200, 409])
        rejected = next(response for response in responses if response.status_code == 409)
        self.assert_error(rejected, 409, "stale_state")
        self.assertEqual(self.state()["revision"], commands[0]["revision"] + 1)
        self.assertEqual(len(self.state()["history"]), 1)

    def test_restart_preserves_turn_cards_and_committed_receipt(self) -> None:
        self.ready_both()
        command = self.body(1, "act", {"kind": "call"})
        receipt = self.post(1, command).json()
        before = [self.state(player) for player in range(2)]
        self.restart()
        self.assertEqual([self.state(player) for player in range(2)], before)
        self.assertEqual(self.post(1, command).json(), receipt)
        self.assertEqual([self.state(player) for player in range(2)], before)
        self.command(0, "act", {"kind": "check"})
        self.assertEqual(self.state(0)["observation"]["street"], "flop")

    def test_settlement_position_rotation_match_completion_and_export_replay(self) -> None:
        self.ready_both()
        self.command(1, "act", {"kind": "fold"})
        first = self.state()
        self.assertEqual((first["phase"], first["ready"]), ("hand_complete", [False, False]))
        self.assertEqual(first["result"]["net_returns"], [1, -1])
        self.assertEqual(first["result"]["rake"], 0)
        self.assertIsNone(self.state(0)["observation"])
        self.assertIsNone(self.state(1)["observation"])
        self.command(0)
        self.assertEqual(self.state()["phase"], "hand_complete")
        self.command(1)
        self.assertEqual(self.state()["hand_index"], 1)
        self.assertEqual([self.state(player)["hero_seat"] for player in range(2)], [1, 0])
        self.assertEqual(self.acting_player(), 0)
        second = self.finish_passively()
        self.assertEqual(len(second["result"]["board"]), 5)
        self.assertEqual(second["result"]["rake"], 1)
        self.assertEqual(sum(second["result"]["net_returns"]) + second["result"]["rake"], 0)
        self.ready_both()
        complete = self.state()
        self.assertEqual((complete["phase"], complete["hand_index"]), ("complete", 1))
        self.assert_error(self.post(0, self.body(0)), 409, "not_ready")
        self.restart()
        self.assertEqual(self.state(), complete)
        exported = self.store.export_hands()
        records = [json.loads(line) for line in exported.read_text().splitlines()]
        self.assertEqual(len(records), 2)
        for index, record in enumerate(records):
            with self.subTest(hand=index):
                self.assertEqual(record["hand_index"], index)
                self.assertEqual(record["player_seats"], [player ^ index for player in range(2)])
                self.assertEqual(record["table_id"], complete["table_id"])
                actual = json.loads(json.dumps(asdict(replay(record))))
                self.assertEqual(actual, (first, second)[index]["result"])
        self.assertEqual(exported.stat().st_mode & 0o777, 0o600)
        before = exported.read_bytes()
        self.assertEqual(self.store.export_hands().read_bytes(), before)

    def test_html_uses_script_hashes_and_no_credentials_or_private_files_are_served(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["referrer-policy"], "no-referrer")
        policy = response.headers["content-security-policy"]
        for directive in ("default-src 'none'", "connect-src 'self'", "frame-ancestors 'none'",
                          "base-uri 'none'", "form-action 'none'"):
            self.assertIn(directive, policy)
        scripts = re.findall(r"<script>(.*?)</script>", response.text, re.DOTALL)
        self.assertTrue(scripts)
        for script in scripts:
            digest = b64encode(hashlib.sha256(script.encode()).digest()).decode()
            self.assertIn(f"'sha256-{digest}'", policy)
        self.assertNotIn("'unsafe-inline'", policy.split("script-src ")[1].split(";")[0])
        for credential in self.credentials:
            self.assertNotIn(credential["token"], response.text)
            self.assertNotIn(credential["token"], json.dumps(self.state(credential["player_id"])))
        for path in ("/docs", "/redoc", "/openapi.json", "/player-0.json", "/player-1.json",
                     "/table.sqlite3", "/manifest.json", "/hands.jsonl"):
            with self.subTest(path=path):
                self.assertEqual(self.client.get(path, headers=self.headers(0)).status_code, 404)

    def test_credentials_and_database_are_private_and_missing_credential_recovers(self) -> None:
        self.assertEqual(self.output.stat().st_mode & 0o777, 0o700)
        for name in ("table.sqlite3", "manifest.json", "player-0.json", "player-1.json"):
            self.assertEqual((self.output / name).stat().st_mode & 0o777, 0o600)
        (self.output / "player-0.json").unlink()
        self.restart()
        self.assertEqual(json.loads((self.output / "player-0.json").read_text()), self.credentials[0])
        self.assertEqual(self.state()["player_id"], 0)

    def test_body_size_and_content_length_are_bounded_before_parsing(self) -> None:
        before = self.state()
        for content, headers, status in (
                (b"x" * 4097, {}, 413),
                (b"{}", {"Content-Length": "4097"}, 413),
                (b"{}", {"Content-Length": "not-a-number"}, 413),
                (b"{}", {"Content-Length": "1"}, 400),
                (b"{}", {"Content-Length": "3"}, 400)):
            with self.subTest(length=headers.get("Content-Length"), size=len(content)):
                response = self.client.post("/v1/command", content=content,
                                            headers=self.headers(0) | headers)
                self.assertEqual(response.status_code, status, response.text)
        streamed = self.client.post("/v1/command", content=iter((b"x" * 2048, b"y" * 2049)),
                                    headers=self.headers(0))
        self.assertNotIn("content-length", streamed.request.headers)
        self.assertEqual(streamed.status_code, 413)
        self.assertEqual(self.state(), before)

    def test_strict_command_schema_rejects_coercion_unknown_fields_and_malformed_actions(self) -> None:
        before = self.state()
        invalid = ({"schema_version": True}, {"revision": "0"}, {"revision": False},
                   {"hand_index": 0.0}, {"revision": -1}, {"hand_index": 1000},
                   {"request_id": "not-an-id"}, {"table_id": "x" * 32},
                   {"kind": "withdraw"}, {"unexpected": "value"},
                   {"kind": "act"}, {"action": {"kind": "call"}},
                   {"kind": "act", "action": {"kind": "call", "amount": 1}},
                   {"kind": "act", "action": {"kind": "raise_to", "amount": 0}},
                   {"kind": "act", "action": {"kind": "raise_to", "amount": "4"}},
                   {"kind": "act", "action": {"kind": "fold", "amount": False}},
                   {"kind": "act", "action": {"kind": "fold", "player_id": 1}})
        for changes in invalid:
            with self.subTest(changes=changes):
                response = self.post(0, self.body(0) | changes)
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(self.state(), before)

    def test_malformed_json_and_wrong_body_shapes_do_not_change_state(self) -> None:
        before = self.state()
        for content in (b"{", b"null", b"[]", b'"command"'):
            with self.subTest(content=content):
                response = self.client.post("/v1/command", content=content,
                                            headers=self.headers(0) | {"Content-Type": "application/json"})
                self.assertEqual(response.status_code, 422, response.text)
                self.assertEqual(self.state(), before)


if __name__ == "__main__":
    unittest.main()
