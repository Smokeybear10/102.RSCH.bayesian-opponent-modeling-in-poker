import argparse
import json
from dataclasses import asdict
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter_ns, thread_time_ns

from .model import LegalActions, Observation, Seat
from .policy import Parameters, TournamentPolicy


def fixtures(seed: int, count: int) -> list[tuple[str, Observation]]:
    rng = Random(seed)
    result = []
    for n in (2, 3, 8):
        for street, size in enumerate((0, 3, 4, 5)):
            for hand in range(count):
                cards = rng.sample(range(52), 2 + size)
                seats = tuple(
                    Seat(
                        i,
                        1800 if i == 0 else 1760,
                        200 if i == 0 else 240,
                        20 if i == 0 else 60,
                    )
                    for i in range(n)
                )
                obs = Observation(
                    hand,
                    0,
                    n - 1,
                    street,
                    tuple(cards[:2]),
                    tuple(cards[2:]),
                    seats,
                    LegalActions(40, 100, 1820),
                    20,
                    tuple(float(n - i) for i in range(n)),
                    sum(s.committed for s in seats),
                )
                result.append((f"{n}p_street{street}", obs))
        for hand in range(count):
            cards = rng.sample(range(52), 7)
            seats = (Seat(0, 100, 0, 0), Seat(1, 0, 100, 100)) + tuple(
                Seat(i, 100, 0, 0, True) for i in range(2, n)
            )
            result.append(
                (
                    f"{n}p_terminal_icm",
                    Observation(
                        hand,
                        0,
                        n - 1,
                        3,
                        tuple(cards[:2]),
                        tuple(cards[2:]),
                        seats,
                        LegalActions(100),
                        20,
                        tuple(float(n - i) for i in range(n)),
                        100,
                    ),
                )
            )
    return result


def run(count: int, repeats: int, seed: int, parameters: Parameters) -> dict:
    if count < 1 or repeats < 1:
        raise ValueError("positive fixture and repetition counts are required")
    groups = {}
    for index, (group, obs) in enumerate(fixtures(seed, count)):
        for repeat in range(repeats):
            agent = TournamentPolicy(
                seed + index * 101 + repeat, parameters, len(obs.seats)
            )
            agent.opponents.begin_hand(obs.hand)
            agent.opponents.begin_street(obs.street)
            agent.opponents.ranges = [0.3] * len(obs.seats)
            agent.opponents.pressure = [2 if obs.street else 0] * len(obs.seats)
            started = perf_counter_ns()
            cpu_started = thread_time_ns()
            action = agent.act(obs)
            cpu_elapsed = thread_time_ns() - cpu_started
            elapsed = perf_counter_ns() - started
            if not obs.legal.accepts(action):
                raise RuntimeError(f"illegal benchmark action: {action}")
            groups.setdefault(group, []).append(
                (elapsed, agent.last_samples, cpu_elapsed)
            )
    output = {
        "practice_only": True,
        "seed": seed,
        "parameters": asdict(parameters),
        "fixture_note": "Cold per-agent range caches, 2/3/8 players, all streets, aggressive public-range hypotheses, and terminal all-in ICM states. Timings include act(), exclude Agent initialization and observation parsing, and are not official sandbox measurements.",
        "groups": {},
    }
    all_times = []
    for name, records in groups.items():
        times = sorted(t / 1000 for t, _, _ in records)
        cpu = sorted(t / 1000 for _, _, t in records)
        all_times.extend(times)
        output["groups"][name] = {
            "calls": len(times),
            "p50_us": times[len(times) // 2],
            "p99_us": times[round((len(times) - 1) * 0.99)],
            "max_us": max(times),
            "over_1ms": sum(t > 1000 for t in times),
            "mean_samples": mean(s for _, s, _ in records),
            "cpu_p99_us": cpu[round((len(cpu) - 1) * 0.99)],
            "cpu_max_us": max(cpu),
        }
    all_times.sort()
    output["total"] = {
        "calls": len(all_times),
        "p50_us": all_times[len(all_times) // 2],
        "p99_us": all_times[round((len(all_times) - 1) * 0.99)],
        "max_us": max(all_times),
        "over_1ms": sum(t > 1000 for t in all_times),
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cold-cache preparation policy latency stress test"
    )
    parser.add_argument("--fixtures", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(
        run(args.fixtures, args.repeats, args.seed, Parameters()), indent=2
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
