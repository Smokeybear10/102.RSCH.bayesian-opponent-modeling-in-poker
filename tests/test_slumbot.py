import json
import stat
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from headsup.models import Action
from headsup.slumbot import BASE_URL, SlumbotError, encode_action, run_benchmark, verify_receipt, verify_response


def response(action: str = "b200", *, client: int = 0, board: list[str] | None = None,
             winnings: int | None = None, **extra: object) -> dict:
    payload = {"action": action, "client_pos": client, "hole_cards": ["Ac", "Ad"],
               "board": [] if board is None else board, **extra}
    if winnings is not None:
        payload["winnings"] = winnings
    return payload


class SlumbotStateTests(unittest.TestCase):
    def test_positions_match_engine_and_expose_only_hero_cards(self) -> None:
        big = verify_response(response())
        small = verify_response(response("", client=1))
        self.assertEqual((big.observation.actor, big.observation.button, big.observation.to_call), (0, 1, 100))
        self.assertEqual((small.observation.actor, small.observation.button, small.observation.to_call), (1, 1, 50))
        self.assertEqual(big.observation.hole, ("Ac", "Ad"))
        self.assertEqual(small.observation.hole, ("Ac", "Ad"))
        self.assertFalse(hasattr(big.observation, "opponent_hole"))
        self.assertFalse(hasattr(big.observation, "deck"))

    def test_street_total_bets_and_multistreet_reconstruction(self) -> None:
        state = verify_response(response("b200c/kk/kb400", board=["5s", "6d", "7h", "8c"]))
        obs = state.observation
        self.assertEqual((obs.street, obs.pot, obs.to_call, obs.bets, obs.stacks),
                         ("turn", 800, 400, (0, 400), (19800, 19400)))
        self.assertEqual(encode_action(state, Action("raise_to", 1000)), "b1000")
        self.assertEqual(state.hand.public_history[-1].paid, 400)
        self.assertEqual(obs.board, ("5s", "6d", "7h", "8c"))

    def test_optional_final_separator(self) -> None:
        for action in ("ck", "ck/"):
            state = verify_response(response(action, board=["2c", "3d", "4h"]))
            self.assertEqual((state.observation.street, state.observation.actor), ("flop", 0))

    def test_short_all_in_does_not_reopen_a_raise(self) -> None:
        initial = verify_response(response("b14000"))
        self.assertEqual(encode_action(initial, Action("raise_to", 20000)), "b20000")
        state = verify_response(response("b14000b20000", client=1))
        self.assertEqual(state.observation.to_call, 6000)
        self.assertIsNone(state.observation.min_raise_to)
        self.assertEqual(encode_action(state, Action("call")), "c")
        with self.assertRaises(SlumbotError):
            encode_action(state, Action("raise_to", 20000))

    def test_all_in_runout_and_revealed_showdown_settlement(self) -> None:
        for action in ("b20000c", "b20000c///"):
            state = verify_response(response(action, board=["2h", "3s", "4d", "7c", "9h"],
                                             winnings=20000, bot_hole_cards=["Kc", "Kd"]))
            self.assertIsNone(state.observation)
            self.assertEqual(state.settlement_check, "recomputed_showdown")
        with self.assertRaises(SlumbotError):
            verify_response(response("b20000c///", board=["2h", "3s", "4d", "7c", "9h"],
                                     winnings=-20000, bot_hole_cards=["Kc", "Kd"]))

    def test_tie_is_zero_net_winnings(self) -> None:
        state = verify_response(response("ck/kk/kk/kk", board=["As", "Ks", "Qs", "Js", "Ts"],
                                         winnings=0, bot_hole_cards=["2c", "3c"]))
        self.assertEqual(state.settlement_check, "recomputed_showdown")

    def test_fold_refund_and_unknown_showdown_verification_are_distinct(self) -> None:
        folded = verify_response(response("b200f", winnings=-100))
        self.assertEqual(folded.settlement_check, "recomputed_fold")
        unknown = verify_response(response("ck/kk/kk/kk", board=["2c", "3d", "4h", "7s", "9c"],
                                          winnings=100))
        self.assertIn("server_reported", unknown.settlement_check)
        with self.assertRaises(SlumbotError):
            verify_response(response("b200f", winnings=-200))
        with self.assertRaises(SlumbotError):
            verify_response(response("ck/kk/kk/kk", board=["2c", "3d", "4h", "7s", "9c"],
                                     winnings=150))

    def test_rejects_invalid_history_board_and_terminal_claims(self) -> None:
        payloads = [
            response("k"), response("b150"), response("b20001"), response("b0200"),
            response("b200c/"), response("b200ck", board=["2c", "3d", "4h"]),
            response("b200f/", winnings=-100), response("b200fc", winnings=-100),
            response("b20000c/", board=["2c", "3d", "4h", "7s", "9c"], winnings=20000),
            response("b20000c///", board=["2c", "3d", "4h", "7s", "9c"]),
            response("b200", winnings=100), response("b200", board=["Ac", "3d", "4h"]),
            response("b200", client=True), response("b200", bot_hole_cards=["Kc", "Kd"]),
            response("", client=0), response("c", client=1),
        ]
        for payload in payloads:
            with self.subTest(payload=payload), self.assertRaises(SlumbotError):
                verify_response(payload)

    def test_outgoing_actions_are_checked_without_mutating_prior_state(self) -> None:
        state = verify_response(response())
        for action in (Action("check"), Action("raise_to", 201), Action("raise_to", 20001)):
            with self.subTest(action=action), self.assertRaises(SlumbotError):
                encode_action(state, action)
        self.assertEqual(encode_action(state, Action("call")), "c")
        self.assertEqual(encode_action(state, Action("fold")), "f")
        self.assertEqual(len(state.hand.public_history), 1)

    def test_receipt_accepts_two_opponent_actions_across_street_boundary(self) -> None:
        before = verify_response(response("", client=1))
        after = verify_response(response("b200c/k", client=1, old_action="", board=["2c", "3d", "4h"]))
        verify_receipt(before, after, Action("raise_to", 200))
        self.assertEqual(after.observation.actor, 1)

    def test_receipt_rejects_wrong_raise_prefix_and_unsent_client_actions(self) -> None:
        before = verify_response(response("b200"))
        for action, old in (("b200b6000f", "b200"), ("b200b600f", "")):
            after = verify_response(response(action, old_action=old, winnings=200))
            with self.assertRaises(SlumbotError):
                verify_receipt(before, after, Action("raise_to", 600))
        before = verify_response(response("", client=1))
        after = verify_response(response("b200c/kk/k", client=1, old_action="", board=["2c", "3d", "4h", "7s"]))
        with self.assertRaises(SlumbotError):
            verify_receipt(before, after, Action("raise_to", 200))

    def test_receipt_rejects_hole_card_changes(self) -> None:
        before = verify_response(response("b200"))
        after_payload = response("b200b600f", old_action="b200", winnings=200)
        after_payload["hole_cards"] = ["Ah", "Ad"]
        with self.assertRaises(SlumbotError):
            verify_receipt(before, verify_response(after_payload), Action("raise_to", 600))


class SlumbotRunnerTests(unittest.TestCase):
    def test_mock_match_preserves_reports_and_redacts_rotating_tokens(self) -> None:
        tokens = ("private-session-token-1111", "private-session-token-2222")
        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            data = json.loads(request.content)
            requests.append((str(request.url), data))
            if len(requests) == 1:
                return httpx.Response(200, json=response("f", winnings=50, token=tokens[0]))
            if len(requests) == 2:
                self.assertEqual(data, {"token": tokens[0]})
                return httpx.Response(200, json=response("", client=1, token=tokens[1]))
            self.assertEqual(data, {"token": tokens[1], "incr": "f"})
            return httpx.Response(200, json=response("f", client=1, old_action="", winnings=-50,
                                                    metadata={"token": tokens[1], "echo": tokens[1]}))

        strategy = SimpleNamespace(act=lambda _: Action("fold"), last_info={"reason": "test"})
        with tempfile.TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            output = Path(temp) / "run"
            with patch("headsup.slumbot.make_policy", return_value=strategy):
                report = run_benchmark(output, 2, client=client)
            self.assertEqual((report["status"], report["completed_hands"], report["net_chips"]), ("completed", 2, 0))
            self.assertEqual(report["accepted_actions"], 1)
            self.assertEqual(report["descriptive_net_bb100"]["hands"], 2)
            self.assertNotIn("pairs", report["descriptive_net_bb100"])
            self.assertTrue(report["position_balanced"])
            self.assertEqual([url for url, _ in requests], [f"{BASE_URL}/new_hand", f"{BASE_URL}/new_hand", f"{BASE_URL}/act"])
            for path in output.iterdir():
                self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
                self.assertFalse(any(token in path.read_text() for token in tokens))
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)

    def test_ambiguous_post_is_never_retried_and_completed_hands_survive(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            if len(calls) == 1:
                return httpx.Response(200, json=response("f", winnings=50, token="private-session-token-1111"))
            raise httpx.ReadTimeout("may have been accepted", request=request)

        with tempfile.TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            output = Path(temp) / "run"
            report = run_benchmark(output, 2, client=client)
            self.assertEqual(len(calls), 2)
            self.assertEqual((report["status"], report["completed_hands"], report["net_chips"]), ("failed", 1, 50))
            self.assertFalse(report["position_balanced"])
            self.assertEqual(report["failure"]["unresolved_request"]["endpoint"], "new_hand")
            self.assertEqual(json.loads((output / "report.json").read_text())["returns_chips"], [50])

    def test_hand_bounds_reject_before_any_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            for hands in (0, 1, 3, 1002, True):
                with self.subTest(hands=hands), self.assertRaises(ValueError):
                    run_benchmark(Path(temp) / "run", hands)
            self.assertEqual(list(Path(temp).iterdir()), [])

    def test_policy_failure_preserves_completed_hands(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            payload = response("f", winnings=50) if len(calls) == 1 else response("", client=1)
            return httpx.Response(200, json={**payload, "token": "private-session-token-1111"})

        strategy = SimpleNamespace(act=lambda _: (_ for _ in ()).throw(RuntimeError("policy failed")))
        with tempfile.TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            with patch("headsup.slumbot.make_policy", return_value=strategy):
                report = run_benchmark(Path(temp) / "run", 2, client=client)
            self.assertEqual((report["status"], report["completed_hands"]), ("failed", 1))
            self.assertEqual(report["failure"]["type"], "RuntimeError")
            self.assertIsNone(report["failure"]["unresolved_request"])
            self.assertEqual(len(calls), 2)

    def test_duplicate_response_fields_are_rejected_without_retry(self) -> None:
        calls = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(request)
            return httpx.Response(200, content=b'{"action":"f","action":"b200"}')

        with tempfile.TemporaryDirectory() as temp, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            report = run_benchmark(Path(temp) / "run", 2, client=client)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
