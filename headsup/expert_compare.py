import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from zipfile import ZipFile

from .arena import environment, seed_for, write_json
from .engine import CARDS, STREETS, Hand
from .models import MAX_SAMPLES, Action, GameConfig, Observation
from .policies import POLICY_NAMES, make_policy

CONFIG = GameConfig(small_blind=50, big_blind=100, stack_bb=200, rake_cap=0)
SOURCE_URL = "https://www.deepstack.ai/downloads"
PROTOCOL_URL = "https://github.com/dmorrill10/project_acpc_server/blob/master/protocol.md"


@dataclass(frozen=True)
class HistoricalHand:
    member: str
    line: int
    number: int
    betting: str
    holes: tuple[tuple[str, str], tuple[str, str]]
    board: tuple[str, ...]
    returns: tuple[int, int]
    players: tuple[str, str]
    timestamp: str

    @property
    def human_seat(self) -> int:
        return 1 - self.players.index("DeepStack")

    @property
    def human(self) -> str:
        match = re.fullmatch(r"all_hands\.(.+)\.log", Path(self.member).name)
        return match[1] if match else self.players[self.human_seat]


def parse_record(line: str, member: str = "", line_number: int = 1) -> HistoricalHand:
    body, marker, timestamp = line.strip().partition(" # ")
    fields = body.split(":")
    if len(fields) != 6 or fields[0] != "STATE" or not re.fullmatch(r"[0-9]+", fields[1]):
        raise ValueError("invalid ACPC STATE fields")
    if not marker or not re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", timestamp):
        raise ValueError("missing or invalid hand timestamp")
    betting = fields[2]
    tokens = re.findall(r"r[1-9][0-9]*|[fc/]", betting)
    if not tokens or "".join(tokens) != betting or len(tokens) > 4096:
        raise ValueError("invalid ACPC betting string")
    card_streets = fields[3].split("/")
    holes = card_streets[0].split("|")
    if len(holes) != 2 or any(len(cards) != 4 for cards in holes):
        raise ValueError("both complete hole-card pairs are required")
    if len(card_streets) > 4 or any(len(cards) != size for cards, size in zip(card_streets[1:], (6, 2, 2))):
        raise ValueError("invalid board street lengths")
    all_cards = "".join(holes + card_streets[1:])
    cards = tuple(all_cards[i:i + 2] for i in range(0, len(all_cards), 2))
    if any(card not in CARDS for card in cards) or len(set(cards)) != len(cards):
        raise ValueError("unknown, invalid or repeated cards")
    payoffs = fields[4].split("|")
    if len(payoffs) != 2 or any(not re.fullmatch(r"-?[0-9]+", value) for value in payoffs):
        raise ValueError("invalid payoffs")
    returns = tuple(map(int, payoffs))
    if sum(returns) or any(abs(value) > CONFIG.starting_stack for value in returns):
        raise ValueError("payoffs are inconsistent with the game configuration")
    players = fields[5].split("|")
    if (len(players) != 2 or players.count("DeepStack") != 1
            or any(not name or len(name) > 256 or any(c.isspace() for c in name) for name in players)):
        raise ValueError("expected exactly one DeepStack player and one human identifier")
    return HistoricalHand(member, line_number, int(fields[1]), betting,
                          (cards[:2], cards[2:4]), cards[4:], returns, tuple(players), timestamp)


def _deck(record: HistoricalHand) -> tuple[str, ...]:
    deck = [None] * 52
    for slots, cards in zip(((0, 2), (1, 3), (5, 6, 7, 9, 11)), (*record.holes, record.board)):
        for slot, card in zip(slots, cards):
            deck[slot] = card
    known = {card for card in deck if card is not None}
    available = iter(card for card in CARDS if card not in known)
    return tuple(card if card is not None else next(available) for card in deck)


def reconstruct(record: HistoricalHand) -> list[tuple[Observation, Action]]:
    hand = Hand(CONFIG, deck=_deck(record))
    steps = []
    street = 0
    tokens = re.findall(r"r[1-9][0-9]*|[fc/]", record.betting)
    for index, token in enumerate(tokens):
        if hand.finished:
            if (hand.public_history[-1].action.kind == "fold"
                    or tokens[index:] != ["/"] * (3 - street)):
                raise ValueError("unexpected action or separator after settlement")
            break
        obs = hand.observation()
        if token == "/":
            if street >= 3 or obs.street != STREETS[street + 1]:
                raise ValueError("street separator disagrees with recorded actions")
            street += 1
            continue
        if obs.street != STREETS[street]:
            raise ValueError("missing street separator")
        if token.startswith("r"):
            previous_streets = CONFIG.starting_stack - obs.stacks[obs.actor] - obs.bets[obs.actor]
            action = Action("raise_to", int(token[1:]) - previous_streets)
        else:
            action = Action("fold" if token == "f" else "call" if obs.to_call else "check")
        steps.append((obs, action))
        hand.act(action)
    if not hand.finished:
        raise ValueError("historical hand is incomplete")
    if hand.result().board != record.board:
        raise ValueError("recorded board disagrees with replay")
    if hand.result().net_returns != record.returns:
        raise ValueError("recorded payoffs disagree with replay")
    return steps


def select_hands(archive: Path, hands_per_player: int,
                 players: tuple[str, ...] | None = None) -> tuple[list[HistoricalHand], dict]:
    if type(hands_per_player) is not int or not 1 <= hands_per_player <= 10000:
        raise ValueError("hands per player must be an integer from 1 to 10000")
    records = []
    members = {}
    notes = {}
    found = set()
    with ZipFile(archive) as source:
        for member in sorted(source.namelist()):
            match = re.fullmatch(r"all_hands\.(.+)\.log", Path(member).name)
            if "/ACPC/" not in member or not match or (players is not None and match[1] not in players):
                continue
            found.add(match[1])
            raw = source.read(member)
            members[member] = hashlib.sha256(raw).hexdigest()
            member_records = []
            for number, line in enumerate(raw.decode("utf-8-sig").splitlines(), 1):
                if not line.strip() or line.startswith("#"):
                    continue
                record = parse_record(line, member, number)
                if not re.fullmatch(re.escape(record.human) + r"(?:\.[0-9]+)?", record.players[record.human_seat]):
                    raise ValueError("human identity differs from the source file")
                member_records.append(record)
            annotation = member.split("/DeepStack_logs/", 1)[0] + f"/AIVAT_analysis/results.{match[1]}.csv"
            annotation_raw = source.read(annotation)
            members[annotation] = hashlib.sha256(annotation_raw).hexdigest()
            annotations = list(csv.reader(io.StringIO(annotation_raw.decode("utf-8-sig"))))
            if not annotations or [field.strip() for field in annotations[0]][-1] != "Notes" or len(annotations) != len(member_records) + 1:
                raise ValueError("annotation header or row count differs from the hand source")
            for record, row in zip(member_records, annotations[1:]):
                if len(row) < 19:
                    raise ValueError("incomplete annotation row")
                if (row[5].strip() != "".join(record.holes[record.human_seat])
                        or row[6].strip() != "".join(record.holes[1 - record.human_seat])
                        or row[10].strip() != ("bb" if record.human_seat == 0 else "sb")):
                    raise ValueError("annotation cards or position differ from the hand source")
                notes[record.member, record.line] = ",".join(row[18:]).strip()
            records.extend(member_records)
    if players is not None and (len(set(players)) != len(players) or set(players) != found):
        raise ValueError("requested players must be distinct and present in the archive")
    if not records:
        raise ValueError("archive has no ACPC records")
    identities = {(record.member, record.line) for record in records}
    if len(identities) != len(records):
        raise ValueError("duplicate archive member or record")
    groups = defaultdict(list)
    for record in records:
        groups[record.human].append(record)
    window = [record for human in sorted(groups)
              for record in sorted(groups[human], key=lambda r: (Decimal(r.timestamp), r.member, r.line))[:hands_per_player]]
    excluded = [{"member": r.member, "line": r.line, "note": notes[r.member, r.line]}
                for r in window if notes[r.member, r.line]]
    selected = [r for r in window if not notes[r.member, r.line]]
    selection = [{"member": r.member, "line": r.line, "hand_number": r.number,
                  "timestamp": r.timestamp, "human": r.human} for r in selected]
    return selected, {
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "member_sha256": members,
        "available_hands": len(records), "available_by_human": {key: len(value) for key, value in sorted(groups.items())},
        "selected_hands": len(selected), "selected_by_human": dict(Counter(r.human for r in selected)),
        "method": "First N timestamp-ordered complete records per canonical player file; ties by source line. Annotated records excluded without replacement. No selection by actions, cards or payouts.",
        "hands_per_player": hands_per_player, "selection": selection, "excluded_annotations": excluded,
        "requested_players": players,
    }


def _legal(action: Action, obs: Observation) -> bool:
    if action.kind == "fold":
        return obs.can_fold
    if action.kind in ("check", "call"):
        return (action.kind == "check") == (obs.to_call == 0)
    return (obs.min_raise_to is not None and obs.max_raise_to is not None
            and obs.min_raise_to <= action.amount <= obs.max_raise_to)


def metrics(rows: list[dict]) -> dict:
    confusion = Counter((row["human_action"]["kind"], row["bot_action"]["kind"]) for row in rows)
    sizes = [row for row in rows if row["human_action"]["kind"] == row["bot_action"]["kind"] == "raise_to"]
    agreement = sum(row["human_action"]["kind"] == row["bot_action"]["kind"] for row in rows)
    differences = [abs(row["human_action"]["amount"] - row["bot_action"]["amount"]) for row in sizes]
    return {
        "decisions": len(rows), "action_class_matches": agreement,
        "action_class_agreement": agreement / len(rows) if rows else None,
        "confusion_human_then_bot": {f"{human}->{bot}": count for (human, bot), count in sorted(confusion.items())},
        "both_raise_decisions": len(sizes),
        "mean_absolute_raise_to_difference_bb": sum(differences) / CONFIG.big_blind / len(sizes) if sizes else None,
        "mean_absolute_raise_to_difference_pot": sum(difference / max(1, row["observation"]["pot"] + row["observation"]["to_call"])
                                                     for difference, row in zip(differences, sizes)) / len(sizes) if sizes else None,
        "exact_action_matches": sum(row["human_action"] == row["bot_action"] for row in rows),
    }


def compare(archive: Path, output: Path, hands_per_player: int = 200,
            policies: tuple[str, ...] = ("range", "precision"), samples: int = 128,
            seed: int = 20261107, players: tuple[str, ...] | None = None) -> dict:
    if not policies or len(set(policies)) != len(policies) or any(policy not in POLICY_NAMES for policy in policies):
        raise ValueError("choose distinct supported policies")
    if type(samples) is not int or not 1 <= samples <= MAX_SAMPLES or type(seed) is not int:
        raise ValueError("invalid samples or seed")
    records, selection = select_hands(archive, hands_per_player, players)
    output.mkdir(parents=True, exist_ok=False)
    initial = environment()
    write_json(output / "manifest.json", {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_url": SOURCE_URL, "protocol_url": PROTOCOL_URL, "source_file": archive.name,
        "selection": selection, "config": asdict(CONFIG), "policies": policies,
        "samples": samples, "seed": seed, "environment": initial,
        "evaluation": "Frozen policies compare decisions at recorded human information sets; only recorded actions advance history. No online learning or counterfactual returns.",
        "seed_method": "Same deterministic seed across policies for each archive member/source line/action index/human seat.",
    })
    rows = defaultdict(list)
    completed = 0
    failure = None
    with (output / "decisions.jsonl").open("x") as journal:
        try:
            for record in records:
                steps = reconstruct(record)
                pending = []
                for index, (obs, historical) in enumerate(steps):
                    if obs.actor != record.human_seat:
                        continue
                    policy_seed = seed_for(seed, "expert-compare-v1", record.member, record.line, index, obs.actor)
                    for name in policies:
                        strategy = make_policy(name, policy_seed, samples)
                        action = strategy.act(obs)
                        if not _legal(action, obs):
                            raise ValueError("policy proposed an illegal action")
                        pending.append({
                            "member": record.member, "source_line": record.line, "hand_number": record.number,
                            "human": record.human, "human_seat": record.human_seat,
                            "action_index": index, "policy": name, "policy_seed": policy_seed,
                            "observation": asdict(obs), "human_action": asdict(historical),
                            "bot_action": asdict(action), "info": strategy.last_info,
                        })
                for row in pending:
                    journal.write(json.dumps(row, allow_nan=False) + "\n")
                    rows[row["policy"]].append(row)
                completed += 1
        except (ValueError, RuntimeError) as exc:
            failure = {"type": type(exc).__name__, "message": str(exc),
                       "member": record.member, "source_line": record.line}
    unchanged = environment() == initial and hashlib.sha256(archive.read_bytes()).hexdigest() == selection["archive_sha256"]
    report = {
        "schema_version": 1, "status": "completed" if failure is None and unchanged else "invalidated",
        "scope": "Descriptive comparison with recorded human decisions, not action quality, exploitability or winnings against humans.",
        "selected_hands": len(records), "validated_completed_hands": completed,
        "source_environment_and_archive_unchanged": unchanged, "failure": failure,
        "policies": {
            name: {"overall": metrics(rows[name]),
                   "by_street": {street: metrics([row for row in rows[name] if row["observation"]["street"] == street]) for street in STREETS},
                   "by_human": {human: metrics([row for row in rows[name] if row["human"] == human]) for human in sorted(selection["selected_by_human"])}}
            for name in policies
        },
        "limitations": [
            "One recorded professional action is a mixed-strategy sample, not an optimal-action label; agreement does not prove strength.",
            "Histories contain responses to DeepStack, not responses to this bot. Different bot actions have unknown counterfactual continuations.",
            "Historical winnings are used only to validate reconstruction and are never attributed to the compared policies.",
            "Only current actor cards, current public board and past actions enter each policy observation; opponent cards and future boards remain excluded.",
            "Rows within hands, matches and humans are correlated; these are descriptive summaries without independent-action significance claims.",
            "The fixed opening sample per human is not representative of all later play or current online populations.",
        ],
    }
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare frozen policies with recorded DeepStack-match human decisions.")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hands-per-player", type=int, default=200)
    parser.add_argument("--policies", choices=POLICY_NAMES, nargs="+", default=["range", "precision"])
    parser.add_argument("--players", nargs="+", help="Canonical archive player identifiers, such as laak.phil okearney.dara")
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261107)
    args = parser.parse_args()
    report = compare(args.archive, args.output, args.hands_per_player, tuple(args.policies), args.samples, args.seed,
                     tuple(args.players) if args.players is not None else None)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "completed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
