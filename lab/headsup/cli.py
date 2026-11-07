import argparse
import json
import sqlite3
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter_ns

from .arena import benchmark, environment, replay, seed_for, write_json
from .engine import Hand
from .models import MAX_SAMPLES, Action, GameConfig, Observation
from .policies import LEARNING_POLICY_NAMES, POLICY_NAMES, make_policy


def positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def sample_count(value: str) -> int:
    number = positive(value)
    if number > MAX_SAMPLES:
        raise argparse.ArgumentTypeError(f"must be at most {MAX_SAMPLES}")
    return number


def port_number(value: str) -> int:
    number = positive(value)
    if number > 65535:
        raise argparse.ArgumentTypeError("port must be 1..65535")
    return number


def table_hand_count(value: str) -> int:
    number = positive(value)
    if number > 1000:
        raise argparse.ArgumentTypeError("private matches are limited to 1000 hands")
    return number


def default_output(mode: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return Path("runs") / f"{mode}-{stamp}"


def config_from(args: argparse.Namespace) -> GameConfig:
    return GameConfig(args.small_blind, args.big_blind, args.stack_bb,
                      args.rake_bps, args.rake_cap, not args.rake_preflop)


def describe(observation: Observation, human: int) -> None:
    role = "button / small blind" if human == observation.button else "big blind"
    print(f"\n{observation.street.upper()} | You: {' '.join(observation.hole)} ({role})")
    print(f"Board: {' '.join(observation.board) or '—'} | Pot: {observation.pot} chips")
    print(f"Stacks: you {observation.stacks[human]}, bot {observation.stacks[1-human]}")
    print(f"Street bets: you {observation.bets[human]}, bot {observation.bets[1-human]}")
    legal = [f"call {observation.to_call}" if observation.to_call else "check"]
    if observation.can_fold:
        legal.append("fold")
    if observation.min_raise_to is not None:
        legal.append(f"raise_to {observation.min_raise_to}..{observation.max_raise_to}")
    print("Legal: " + " | ".join(legal))


def human_action() -> Action | None:
    words = input("Action [check/call/fold/r TOTAL/q]: ").strip().lower().split()
    if not words:
        raise ValueError("enter an action, or q to quit")
    if words == ["q"]:
        return None
    aliases = {"c": "call", "k": "check", "f": "fold", "r": "raise_to", "raise": "raise_to"}
    kind = aliases.get(words[0], words[0])
    if kind == "raise_to" and len(words) == 2:
        return Action("raise_to", int(words[1]))
    if len(words) != 1 or kind == "raise_to":
        raise ValueError("use check, call, fold, or r followed by the street-total bet")
    return Action(kind)


def play(output: Path, config: GameConfig, bot_name: str, hands: int, seed: int, samples: int) -> int:
    if bot_name not in POLICY_NAMES or type(hands) is not int or hands < 1:
        raise ValueError("choose a supported bot and a positive integer hand count")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES:
        raise ValueError(f"samples must be an integer from 1 to {MAX_SAMPLES}")
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", {"schema_version": 1, "mode": "human_local_play",
               "config": asdict(config), "bot": bot_name, "hands": hands, "seed": seed,
               "samples": samples, "environment": environment()})
    print("Heads-Up Poker Lab | local play-money simulation")
    print(f"Fresh {config.stack_bb} BB stacks each hand. Positions alternate. Amounts are chips.")
    print("raise_to means your TOTAL bet on this street, not additional chips.")
    print(f"Completed hands save to {output / 'hands.jsonl'}")
    completed = 0
    total_net = 0
    hand = None
    status = "completed"
    error = None
    decisions = []
    try:
        with (output / "hands.jsonl").open("x") as journal:
            for hand_index in range(hands):
                hand = None
                decisions = []
                human = 1 - hand_index % 2
                hand = Hand(config, seed_for(seed, "human-deck", hand_index))
                bot_seed = seed_for(seed, "human-opponent", hand_index)
                bot = make_policy(bot_name, bot_seed, samples)
                print(f"\n--- Hand {hand_index + 1}/{hands} ---")
                while not hand.finished:
                    observation = hand.observation()
                    if observation.actor == human:
                        describe(observation, human)
                        try:
                            action = human_action()
                            if action is None:
                                raise KeyboardInterrupt
                            hand.act(action)
                        except ValueError as exc:
                            print(f"Invalid action: {exc}")
                            continue
                        decisions.append({"actor": human, "action": asdict(action), "human": True})
                    else:
                        decision = {"actor": observation.actor, "street": observation.street, "accepted": False}
                        decisions.append(decision)
                        started = perf_counter_ns()
                        try:
                            action = bot.act(observation)
                        finally:
                            decision["latency_ns"] = perf_counter_ns() - started
                        decision.update(action=asdict(action), info=bot.last_info)
                        hand.act(action)
                        decision["accepted"] = True
                        print(f"Bot: {action.kind}" + (f" {action.amount}" if action.amount else ""))
                result = hand.result()
                record = {**hand.record(), "human_seat": human, "bot": bot_name,
                          "bot_seed": bot_seed, "samples": samples, "decisions": decisions}
                journal.write(json.dumps(record, allow_nan=False) + "\n")
                journal.flush()
                completed += 1
                total_net += result.net_returns[human]
                print(f"Board: {' '.join(result.board) or '—'}")
                print(f"Your net result: {result.net_returns[human]:+d} chips | Rake: {result.rake}")
                print(f"Session net: {total_net:+d} chips ({total_net/config.big_blind:+.1f} BB)")
    except (KeyboardInterrupt, EOFError):
        status = "incomplete"
        print("\nStopped. Completed hands are preserved; the unfinished hand is excluded.")
    except Exception as exc:
        status = "failed"
        error = {"type": type(exc).__name__, "message": str(exc)}
        print(f"Play failed: {exc}", file=sys.stderr)
    finally:
        if status != "completed" and hand is not None:
            write_json(output / "unfinished-hand.json", {**hand.snapshot(), "decisions": decisions})
        write_json(output / "report.json", {"status": status, "completed_hands": completed,
                   "human_net_chips": total_net, "error": error})
    return int(status == "failed")


def read_record(path: Path, hand_number: int) -> dict:
    if path.suffix == ".jsonl":
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if number == hand_number:
                    return json.loads(line)
        raise ValueError(f"hand {hand_number} is not present in {path}")
    if hand_number != 1:
        raise ValueError("--hand selects a row only for JSONL journals")
    return json.loads(path.read_text())


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="Heads-up Hold'em AI, private play-money tables and evaluation. Playing strength is unproven.")
    commands = root.add_subparsers(dest="command", required=True)
    human = commands.add_parser("play", help="play local heads-up hands against the baseline")
    human.add_argument("--bot", choices=POLICY_NAMES, default="range")
    human.add_argument("--hands", type=positive, default=10)
    arena = commands.add_parser("benchmark", help="run duplicate-deal, seat-swapped matches")
    arena.add_argument("--hero", choices=POLICY_NAMES, default="range")
    arena.add_argument("--opponents", nargs="+", choices=POLICY_NAMES, default=["calling", "tight", "aggressive", "equity"])
    arena.add_argument("--pairs", type=positive, default=50, help="deal pairs PER opponent; two hands per pair")
    learning = commands.add_parser("train", help="automatically learn opponents, save memory, and compare with the fixed baseline")
    learning.add_argument("--hands", type=positive, default=600, help="fresh training hands per known opponent")
    learning.add_argument("--warmup", type=positive, default=600, help="separate learning hands per held-out style")
    learning.add_argument("--pairs", type=positive, default=128, help="matched evaluation units per opponent; four hands each")
    learning.add_argument("--resume", type=Path, help="previous checkpoint.json; requires matching game settings and a new seed")
    learning.add_argument("--memory", choices=("lifetime", "recent"), default="lifetime",
                          help="recent uses bounded distinct-hand evidence with expiry; lifetime preserves the previous learner")
    historical = commands.add_parser("learn-hands", help="learn public responses from completed PHH hands and evaluate later hands offline")
    historical.add_argument("paths", nargs="+", type=Path, help="PHH/PHHS files or directories containing them")
    historical.add_argument("--train-fraction", type=float, default=0.7)
    historical.add_argument("--min-stack-bb", type=float, default=80, help="minimum starting effective stack in big blinds")
    historical.add_argument("--max-stack-bb", type=float, default=120, help="maximum starting effective stack in big blinds")
    historical.add_argument("--output", type=Path, help="new output directory for frozen profiles and evaluation")
    population_fit = commands.add_parser("fit-population", help="fit a frozen population prior and calibrate its weight on historical development hands")
    population_evaluation = commands.add_parser("evaluate-population", help="evaluate a fitted prior on new, later completed hands")
    population_evaluation.add_argument("checkpoint", type=Path, help="checkpoint.json from fit-population")
    for child in (population_fit, population_evaluation):
        child.add_argument("paths", nargs="+", type=Path, help="PHH/PHHS files or directories containing them")
        child.add_argument("--output", type=Path, help="new output directory")
    river_evaluation = commands.add_parser("evaluate-river", help="compare experimental river decisions with the adaptive baseline in matched local sessions")
    river_evaluation.add_argument("--population-prior", type=Path, required=True)
    river_evaluation.add_argument("--suite", choices=("original", "raise_sizes_v2"), default="original",
                                  help="raise_sizes_v2 adds varied river raises and a second hidden style switch")
    river_evaluation.add_argument("--sessions", type=positive, default=12)
    river_evaluation.add_argument("--hands", type=positive, default=64, help="hands per seat-swapped learning stream")
    adaptation = commands.add_parser("adaptation", help="automatically compare learning across independent sessions and opponent style changes")
    adaptation.add_argument("--sessions", type=positive, default=12, help="independent sessions per scenario")
    adaptation.add_argument("--hands", type=positive, default=384, help="hands per session per policy arm")
    adaptation.add_argument("--switch-at", type=positive, default=192, help="completed hands before changing synthetic opponent style")
    table = commands.add_parser("serve", help="host an authenticated private play-money table on loopback")
    table.add_argument("--hands", type=table_hand_count, default=100)
    table.add_argument("--port", type=port_number, help="8765 for a new table; saved port when resuming")
    table.add_argument("--resume", action="store_true", help="resume the table in --output with its saved rules and credentials")
    connection = commands.add_parser("connect", help="connect a bot or terminal player to our private table protocol")
    connection.add_argument("credentials", type=Path, help="private player-0.json or player-1.json seat file")
    connection.add_argument("--state", type=Path, required=True, help="private client directory; reuse it to reconnect")
    connection.add_argument("--bot", choices=(*LEARNING_POLICY_NAMES, *POLICY_NAMES), default="adaptive")
    connection.add_argument("--samples", type=sample_count, default=128)
    connection.add_argument("--memory", choices=("lifetime", "recent"), default="lifetime")
    connection.add_argument("--population-prior", type=Path, help="fitted population-prior.json for adaptive or river; reuse the same prior when reconnecting")
    connection.add_argument("--human", action="store_true", help="take actions from this terminal instead of the bot")
    worker = commands.add_parser("decision-worker", help="keep a private-test policy warm; read validated JSON lines from stdin")
    worker.add_argument("--seed", type=int, default=20261106)
    worker.add_argument("--bot", choices=LEARNING_POLICY_NAMES, default="adaptive")
    worker.add_argument("--samples", type=sample_count, default=128)
    worker.add_argument("--delay-min", type=float, default=0.3, help="minimum turn-to-action delay in seconds")
    worker.add_argument("--delay-max", type=float, default=1.1, help="maximum turn-to-action delay in seconds")
    worker.add_argument("--deadline-reserve", type=float, default=1.0, help="seconds reserved for submission and confirmation")
    worker.add_argument("--memory-dir", type=Path, help="persist completed-hand opponent learning in this directory")
    worker.add_argument("--room-url", help="pinned private PokerNow room for opponent memory")
    worker.add_argument("--hero-id", help="own PokerNow player ID")
    worker.add_argument("--opponent-id", help="opponent PokerNow player ID, not their display name")
    worker.add_argument("--small-blind", type=positive)
    worker.add_argument("--big-blind", type=positive)
    worker.add_argument("--memory", choices=("lifetime", "recent"), default="lifetime")
    worker.add_argument("--population-prior", type=Path, help="optional frozen prior for new opponent memory; saved priors reload automatically")
    profiles = commands.add_parser("opponent-profiles", help="inspect saved player identities and observed behavior without changing memory")
    profiles.add_argument("--memory-dir", type=Path, default=Path(__file__).resolve().parent.parent / "state" / "opponents")
    profiles.add_argument("--opponent-id", help="filter by exact player ID; display names are not identities")
    profiles.add_argument("--room-url", help="filter by exact PokerNow room URL")
    export = commands.add_parser("table-export", help="export private host records for local action replay")
    export.add_argument("path", type=Path, help="host table directory")
    for child in (human, arena, learning, adaptation, river_evaluation, table):
        if child is not table:
            child.add_argument("--seed", type=int, default=20261105)
            child.add_argument("--samples", type=sample_count, default=256)
        child.add_argument("--stack-bb", type=positive, default=100)
        child.add_argument("--small-blind", type=positive, default=1)
        child.add_argument("--big-blind", type=positive, default=2)
        child.add_argument("--rake-bps", type=int, default=0, help="simulated rake in basis points; 500 = 5%%")
        child.add_argument("--rake-cap", type=int, default=6, help="rake cap in integer chips, not BB")
        child.add_argument("--rake-preflop", action="store_true", help="also rake hands ending before the flop")
        child.add_argument("--output", type=Path, help="new output directory; existing paths are never overwritten")
    learning.set_defaults(seed=20261106)
    adaptation.set_defaults(seed=202611051101, samples=128)
    river_evaluation.set_defaults(seed=2026110607, samples=64, rake_bps=500)
    saved = commands.add_parser("replay", help="verify saved actions and settlement")
    saved.add_argument("path", type=Path)
    saved.add_argument("--hand", type=positive, default=1)
    saved.add_argument("--verify-policy", action="store_true", help="also reproduce bot decisions with current code/environment")
    return root


def main(argv: list[str] | None = None) -> int:
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        if args.command == "opponent-profiles":
            from .opponent_memory import read_opponent_profiles

            profiles = read_opponent_profiles(args.memory_dir, args.opponent_id, args.room_url)
            print(json.dumps({
                "profile_count": len(profiles), "profiles": profiles,
                "identity_scope": "Separate room, hero ID, opponent ID and blind settings; display names and IDs from different rooms are never merged.",
            }, indent=2, allow_nan=False))
            return 0
        if args.command in ("learn-hands", "fit-population", "evaluate-population"):
            for path in args.paths:
                if not path.exists():
                    raise ValueError(f"input does not exist: {path}")
                if not path.is_dir() and path.suffix.lower() not in (".phh", ".phhs"):
                    raise ValueError(f"expected a PHH/PHHS file or directory: {path}")
            paths = sorted({file.resolve() for path in args.paths
                            for file in (path.rglob("*") if path.is_dir() else [path])
                            if file.suffix.lower() in (".phh", ".phhs")})
            if not paths:
                raise ValueError("no PHH or PHHS files found")
            output = args.output or default_output(args.command)
            if args.command == "fit-population":
                from .population_learning import fit_population

                report = fit_population(paths, output)
                print(json.dumps({key: report[key] for key in ("status", "refit_hands", "refit_players", "strength", "candidate_brier")}, indent=2))
                print(f"Population prior: {output / 'population-prior.json'}")
                print(f"Evaluation checkpoint: {output / 'checkpoint.json'}")
                return int(report["status"] != "completed")
            if args.command == "evaluate-population":
                from .population_learning import evaluate_population

                report = evaluate_population(args.checkpoint, paths, output)
                print(json.dumps({"status": report["status"], "future_hands": report["future_hands"],
                                  "strength": report["strength"],
                                  "evaluation": {mode: result["all"] for mode, result in report["evaluation"].items()}}, indent=2))
                print(f"Population evaluation: {output / 'report.json'}")
                return int(report["status"] != "completed")
            from .human_learning import learn_hands

            report = learn_hands(paths, output, args.train_fraction, args.min_stack_bb, args.max_stack_bb)
            print(json.dumps({"status": report["status"], "split": report["split"],
                              "heldout": report["metrics"]["all"],
                              "default_policy_replaced": report["default_policy_replaced"]}, indent=2))
            print(f"Frozen opponent profiles: {output / 'profiles.json'}")
            print(f"Historical evaluation: {output / 'report.json'}")
            return int(report["status"] != "completed")
        if args.command == "decision-worker":
            from .decision_worker import DecisionWorker, run_worker
            from .opponent_memory import OpponentIdentity, OpponentMemory
            from .timing import ActionTiming

            timing = ActionTiming(args.delay_min, args.delay_max, args.deadline_reserve)
            identity_values = (args.room_url, args.hero_id, args.opponent_id, args.small_blind, args.big_blind)
            if args.memory_dir is None and (any(value is not None for value in identity_values)
                                            or args.population_prior is not None or args.memory != "lifetime"):
                raise ValueError("persistent opponent settings require --memory-dir")
            store = None
            if args.memory_dir is not None:
                if any(value is None for value in identity_values):
                    raise ValueError("--memory-dir requires --room-url, --hero-id, --opponent-id, --small-blind and --big-blind")
                identity = OpponentIdentity(room_url=args.room_url, hero_id=args.hero_id,
                                            opponent_id=args.opponent_id, small_blind=args.small_blind,
                                            big_blind=args.big_blind)
                store = OpponentMemory(args.memory_dir, identity, args.memory, args.population_prior)
            try:
                return run_worker(DecisionWorker(args.seed, args.samples, args.bot, timing, memory_store=store))
            finally:
                if store is not None:
                    store.close()
        if args.command == "connect":
            from .connection import run_client

            report = run_client(args.credentials, args.state, policy=args.bot, samples=args.samples,
                                memory=args.memory, human=args.human, population_prior=args.population_prior)
            print(json.dumps(report, indent=2))
            return int(report["status"] != "completed")
        if args.command == "table-export":
            from .table import open_table

            store = open_table(args.path)
            try:
                print(store.export_hands())
            finally:
                store.close()
            return 0
        if args.command == "serve":
            from urllib.parse import urlsplit

            import uvicorn

            from .table import create_app, create_table, open_table

            if args.resume and args.output is None:
                command_parser.error("serve --resume requires --output pointing to the existing table")
            output = (args.output or default_output("table")).resolve()
            port = args.port or 8765
            if args.resume:
                store = open_table(output)
                saved = urlsplit(store.base_url)
                port = saved.port or 80
                if saved.scheme != "http" or saved.hostname != "127.0.0.1" or (args.port and args.port != port):
                    store.close()
                    raise ValueError("resume must use the table's original loopback URL and port")
            else:
                store = create_table(output, config_from(args), args.hands, f"http://127.0.0.1:{port}")
            url = store.base_url
            print(f"Private play-money table: {url}", flush=True)
            print("Fresh stacks each hand; disconnects pause play. No casino connection.", flush=True)
            for player in (0, 1):
                print(f"Player {player} credentials: {output / f'player-{player}.json'}", flush=True)
            print("Share only that player's credential file, never the host directory.", flush=True)
            try:
                uvicorn.run(create_app(store), host="127.0.0.1", port=port,
                            access_log=False, log_level="warning")
            finally:
                try:
                    print(f"Completed-hand replay: {store.export_hands()}", flush=True)
                finally:
                    store.close()
            return 0
        if args.command == "replay":
            record = read_record(args.path, args.hand)
            result = replay(record, verify_policy=args.verify_policy)
            print(json.dumps({"verified": True, "policy_decisions_verified": args.verify_policy,
                              "result": asdict(result)}, indent=2))
            return 0
        config = config_from(args)
        output = args.output or default_output(args.command)
        if args.command == "play":
            return play(output, config, args.bot, args.hands, args.seed, args.samples)
        if args.command == "evaluate-river":
            from .river_evaluation import evaluate_river

            report = evaluate_river(output, config, args.population_prior, args.sessions, args.hands,
                                    args.seed, args.samples,
                                    lambda message: print(message, file=sys.stderr, flush=True), suite=args.suite)
            print(json.dumps(report, indent=2))
            print(f"Matched river comparison: {output / 'report.json'}")
            return int(report["status"] != "completed")
        if args.command == "train":
            from .training import train

            report = train(output, config, args.hands, args.warmup, args.pairs, args.seed,
                           args.samples, args.resume, lambda message: print(message, file=sys.stderr, flush=True), args.memory)
            print(json.dumps({key: report[key] for key in (
                "status", "completed_hands", "adapted_decisions", "positive_lift_evidence_on_tested_pool",
                "default_policy_replaced", "real_money_readiness_established",
            )}, indent=2))
            print(f"Full comparison: {output / 'report.json'}")
            print(f"Persistent opponent memory: {output / 'checkpoint.json'}")
            return int(report["status"] != "completed")
        if args.command == "adaptation":
            from .adaptation import evaluate_adaptation

            report = evaluate_adaptation(output, config, args.sessions, args.hands, args.switch_at,
                                         args.seed, args.samples,
                                         lambda message: print(message, file=sys.stderr, flush=True))
            print(json.dumps({key: report[key] for key in (
                "status", "completed_hands", "complete_matched_sessions", "adapted_decisions",
                "positive_aggregate_lift_over_lifetime", "default_policy_replaced",
            )}, indent=2))
            print(f"Full session comparison: {output / 'report.json'}")
            return int(report["status"] != "completed")
        report = benchmark(output, config, args.hero, tuple(args.opponents), args.pairs, args.seed, args.samples)
        print(json.dumps(report, indent=2))
        print(f"\nSaved manifest, hand journal and report to {output}")
        return int(report["status"] != "completed")
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        command_parser.error(str(exc))
    return 2
