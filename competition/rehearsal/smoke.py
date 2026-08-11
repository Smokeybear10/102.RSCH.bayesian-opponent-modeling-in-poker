import argparse
import importlib.util
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from time import perf_counter_ns
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", type=Path, required=True)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()
    sys.path.insert(0, str(args.agent.parent))
    spec = importlib.util.spec_from_file_location("candidate_agent", args.agent)
    module = importlib.util.module_from_spec(spec)
    started = perf_counter_ns()
    spec.loader.exec_module(module)
    import_us = (perf_counter_ns() - started) / 1000
    from sdk import fixture, legal_output

    results = []
    callback_us = []
    # Wiring assertions should not change their expected action when the OS
    # pauses this process. Both clock bindings share one simulated time domain.
    # The real-clock mode and wirecheck retain actual deadline enforcement.
    with ExitStack() as clocks:
        if args.deterministic:
            for name in ("competition.bridge", "competition.policy"):
                clocks.enter_context(patch(name + ".perf_counter_ns", return_value=0))
        started = perf_counter_ns()
        agent = module.Agent()
        init_us = (perf_counter_ns() - started) / 1000
        for mode in ("to", "increment", "additional"):
            raw = fixture(mode)
            started = perf_counter_ns()
            result = agent.act(raw)
            callback_us.append((perf_counter_ns() - started) / 1000)
            assert legal_output(raw, result), (mode, result)
            assert result["type"] == "raise", (
                "the connected policy fell back instead of choosing its expected value bet"
            )
            base = {"to": 0, "increment": 60, "additional": 20}[mode]
            assert result["amount"] + base == 348, (
                "SDK raise semantics changed the intended bet"
            )
            results.append(result)
        assert agent.bridge.events_seen == 1, "replayed snapshots double-counted history"
        fresh = fixture("to", "table-2")
        assert legal_output(fresh, agent.act(fresh))
        assert agent.bridge.table_resets == 2, "opponent state crossed a table boundary"
        assert agent.bridge.policy.opponents.seats[1].raises == 1
    assert "eval7" not in sys.modules and "pokerkit" not in sys.modules
    print(
        json.dumps(
            {
                "integration": "passed",
                "official_sdk": False,
                "raise_conventions": results,
                "table_resets": 2,
                "history_deduplicated": True,
                "third_party_runtime_imports": [],
                "cold_import_us": import_us,
                "agent_initialization_us": init_us,
                "callback_us": callback_us,
                "deadline_mode": "deterministic-wiring" if args.deterministic else "wall-clock",
                "timing_note": "Synthetic provided-equity SDK. Deterministic mode freezes policy/bridge budget clocks and includes mock overhead in wall timings; it tests wiring, not deadline performance. Real-clock mode enforces the budget. Neither certifies the official runtime or native policy.",
            }
        )
    )


if __name__ == "__main__":
    main()
