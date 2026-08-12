import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from .arena import Configuration, compare
from .policy import Parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Citadel Poker: local Hold'em preparation arena; not the unpublished competition Agent interface."
    )
    parser.add_argument(
        "--tables", type=int, default=32, help="tables per field and policy"
    )
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["mixed", "calling", "tight", "maniac", "random", "selfplay"],
        default=["mixed", "selfplay"],
    )
    parser.add_argument("--max-hands", type=int, default=200)
    parser.add_argument(
        "--payouts",
        type=float,
        nargs=8,
        default=[50, 30, 20, 0, 0, 0, 0, 0],
        help="practice payouts; the event payouts are not yet public",
    )
    parser.add_argument(
        "--blinds",
        type=int,
        nargs="+",
        default=[20, 40, 80, 160, 320, 640, 1280, 2560],
        help="practice big blinds, increasing every 25 hands",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-deadline",
        action="store_true",
        help="measure overruns without applying the check/fold fallback",
    )
    args = parser.parse_args()
    config = Configuration(
        payouts=tuple(args.payouts),
        blinds=tuple(args.blinds),
        max_hands=args.max_hands,
        enforce_deadline=not args.no_deadline,
    )
    config.validate()
    base = Parameters()
    variants = {
        "equity_only": replace(base, adapt=False, use_icm=False),
        "fixed_call_model": replace(base, learn_continuation=False),
        "tournament": base,
    }
    started = perf_counter()
    report = compare(
        variants, tuple(args.fields), args.tables, args.seed, args.workers, config
    )
    report["elapsed_seconds"] = perf_counter() - started
    report["limitations"] = [
        "The event variant, Agent interface, runtime, dependency availability, and permission to bring prewritten code remain unresolved.",
        "The policy is a heuristic, not a solved or trained equilibrium strategy.",
        "Equity uses bounded Monte Carlo and approximate opponent ranges; nonterminal decisions omit future betting and use a realization discount.",
        "ICM is applied directly only to heads-up terminal all-in calls; other decisions use chip-value heuristics.",
        "Local timing includes act() but excludes simulator-to-observation conversion; host scheduling and multiprocessing affect tail latency.",
        "PokerKit and eval7 are preparation dependencies, not confirmed event dependencies.",
    ]
    encoded = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
