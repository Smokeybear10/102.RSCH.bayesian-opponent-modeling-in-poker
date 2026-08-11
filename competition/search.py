import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

from .arena import Configuration, compare
from .policy import Parameters


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search practice policies, then evaluate frozen finalists on unused table seeds"
    )
    parser.add_argument("--tables", type=int, default=32)
    parser.add_argument("--validation-tables", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = Parameters()
    candidates = {
        "current": base,
        "equity_only": replace(base, adapt=False, use_icm=False),
        "icm_only": replace(base, adapt=False),
        "looser": replace(base, open_scale=1.2),
        "tighter": replace(base, open_scale=0.8),
        "thin_value": replace(base, value_edge=0.08, bet_fraction=0.5),
        "big_value": replace(base, bet_fraction=0.9),
        "pressure": replace(
            base, open_scale=1.15, value_edge=0.1, bet_fraction=0.45, call_edge=0.015
        ),
        "sticky": replace(base, call_edge=0),
        "patient": replace(base, call_edge=0.06, value_edge=0.2),
    }
    started = perf_counter()
    training = compare(
        candidates,
        ("mixed", "selfplay"),
        args.tables,
        args.seed,
        args.workers,
        Configuration(),
    )
    ranking = sorted(
        candidates,
        key=lambda name: (
            0.4 * training["fields"]["mixed"][name]["mean_points"]
            + 0.6 * training["fields"]["selfplay"][name]["mean_points"]
        ),
        reverse=True,
    )
    selected = ranking[0]
    print(f"Training complete: {ranking}; frozen nominee: {selected}", flush=True)
    nominees = {"current": base, "equity_only": candidates["equity_only"]}
    if candidates[selected] not in nominees.values():
        nominees[selected] = candidates[selected]
    validation = {}
    total_tables = 2 * len(candidates) * args.tables
    for index, field in enumerate(
        ("mixed", "selfplay", "calling", "tight", "maniac", "random")
    ):
        result = compare(
            nominees,
            (field,),
            args.validation_tables,
            args.seed + 1000003 * (index + 1),
            args.workers,
            Configuration(),
        )
        validation[field] = result
        total_tables += len(nominees) * args.validation_tables
        print(
            f"Fresh validation {field}: "
            + str(
                {
                    name: round(row["mean_points"], 3)
                    for name, row in result["fields"][field].items()
                }
            ),
            flush=True,
        )
    output = {
        "practice_only": True,
        "rules_source": "https://pokerbot.sequoiacap.com/rules.html",
        "selection_objective": "0.4 times mixed-field mean points plus 0.6 times selfplay-field mean points on training seeds only",
        "selected_before_validation": selected,
        "selected_parameters": asdict(candidates[selected]),
        "training_ranking": ranking,
        "training": training,
        "fresh_validation": validation,
        "total_tables": total_tables,
        "elapsed_seconds": perf_counter() - started,
        "limitations": [
            "All policies and fields are local Hold'em preparation; actual event rules, variant, payouts, runtime and entrant strategies are not yet available.",
            "Selecting a training winner does not establish it as better; examine the independent paired validation intervals before changing defaults.",
            "Reported 95% intervals are descriptive normal approximations without multiple-comparison corrections.",
            "Real-time sampling budgets and OS scheduling can change sampled actions even with fixed RNG seeds.",
            "No result in this report establishes an event win rate or profitability.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Saved {args.output}; {total_tables} tournaments", flush=True)


if __name__ == "__main__":
    main()
