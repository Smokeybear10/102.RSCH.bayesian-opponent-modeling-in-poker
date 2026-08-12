import argparse
import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import zipfile
from math import isfinite
from pathlib import Path
from time import perf_counter

HERE = Path(__file__).parent
CORE = (
    "__init__.py",
    "cards.py",
    "equity.py",
    "sampling.py",
    "model.py",
    "icm.py",
    "opponents.py",
    "policy.py",
    "bridge.py",
)
IGNORE = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    "site-packages",
    "target",
    "dist",
}
LIMIT = 100_000_000


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(value, indent=2) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as f:
        temporary = Path(f.name)
        f.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def doctor(python: str) -> dict:
    script = "import importlib.util,json,platform,sys; print(json.dumps({'python':list(sys.version_info[:3]),'platform':platform.platform(),'eval7_available':importlib.util.find_spec('eval7') is not None}))"
    result = subprocess.run(
        [python, "-I", "-c", script],
        check=False,
        text=True,
        capture_output=True,
        timeout=15,
    )
    if result.returncode:
        raise RuntimeError(f"target Python probe failed: {result.stderr[-2000:]}")
    report = json.loads(result.stdout)
    report["core_python_supported"] = report["python"][:2] >= [3, 10]
    report["note"] = (
        "Installed does not mean permitted. Verify event dependencies and prewritten-code rules separately. A supplied equity backend avoids eval7."
    )
    return report


def intake(starter: Path, python: str) -> dict:
    root = starter.resolve()
    if not root.is_dir():
        raise ValueError("starter must be a local directory")
    files, interfaces, imports, evidence, errors = [], [], [], [], []
    for directory, dirs, names in os.walk(root, followlinks=False):
        dirs[:] = sorted(
            d
            for d in dirs
            if d not in IGNORE
            and not d.startswith(".")
            and not (Path(directory) / d).is_symlink()
        )
        for name in sorted(names):
            path = Path(directory) / name
            if (
                path.is_symlink()
                or name.startswith(".")
                or path.suffix
                not in {".py", ".json", ".toml", ".md", ".rst", ".txt", ".yaml", ".yml"}
            ):
                continue
            if len(files) >= 2000:
                raise ValueError(
                    "starter inspection exceeded 2000 files; select the actual kit directory"
                )
            if path.stat().st_size > 2_000_000:
                errors.append(
                    {
                        "file": str(path.relative_to(root)),
                        "error": "file exceeds 2MB inspection limit",
                    }
                )
                continue
            data = path.read_bytes()
            relative = str(path.relative_to(root))
            files.append(
                {
                    "file": relative,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
            )
            source = data.decode("utf-8", errors="replace")
            if path.suffix == ".py":
                try:
                    tree = ast.parse(source)
                except SyntaxError as exc:
                    errors.append(
                        {"file": relative, "line": exc.lineno, "error": exc.msg}
                    )
                    continue
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imports.extend(
                            {
                                "file": relative,
                                "module": alias.name,
                                "line": node.lineno,
                            }
                            for alias in node.names
                        )
                    elif isinstance(node, ast.ImportFrom):
                        imports.append(
                            {
                                "file": relative,
                                "module": "." * node.level + (node.module or ""),
                                "line": node.lineno,
                            }
                        )
                    if isinstance(node, ast.ClassDef):
                        methods = [
                            {
                                "name": method.name,
                                "arguments": ast.unparse(method.args),
                                "returns": ast.unparse(method.returns)
                                if method.returns
                                else None,
                                "line": method.lineno,
                            }
                            for method in node.body
                            if isinstance(
                                method, (ast.FunctionDef, ast.AsyncFunctionDef)
                            )
                        ]
                        fields = [
                            {
                                "name": ast.unparse(field.target),
                                "type": ast.unparse(field.annotation),
                            }
                            for field in node.body
                            if isinstance(field, ast.AnnAssign)
                        ]
                        interfaces.append(
                            {
                                "file": relative,
                                "class": node.name,
                                "bases": [ast.unparse(b) for b in node.bases],
                                "line": node.lineno,
                                "methods": methods,
                                "fields": fields,
                            }
                        )
            for line, text in enumerate(source.splitlines(), 1):
                if any(
                    term in text.lower()
                    for term in (
                        "payout",
                        "blind schedule",
                        "variant",
                        "raise_to",
                        "raise_by",
                        "deadline",
                        "millisecond",
                    )
                ):
                    evidence.append(
                        {"file": relative, "line": line, "text": text.strip()[:240]}
                    )
                    if len(evidence) >= 500:
                        break
    return {
        "starter": str(root),
        "executed_starter_code": False,
        "runtime": doctor(python),
        "files": files,
        "interfaces": interfaces,
        "imports": imports,
        "evidence": evidence[:500],
        "inspection_errors": errors,
        "mapping_to_complete": [
            "Confirm the revealed variant, deck, hand rankings, betting structure, and hole-card rules before choosing a strategy.",
            "Implement the real Agent class and method signatures in the starter's entry file.",
            "Map SDK observations to Frame/Observation: private hero cards, board, stable roster, button, stacks, total hand contributions, current-street bets, legal call cost, raise bounds, blinds, actual payouts.",
            "Declare raise amounts as street total, increment above the highest bet, or chips added by the actor; validate known examples in both directions.",
            "Supply stable table IDs and ordered current-hand public events, or disable adaptation explicitly if history is unavailable.",
            "Map canonical Action back to the SDK's actual action object; check it using the official simulator.",
            "If eval7 is unavailable, inject the SDK's equity service or SampledEquity(cards=SDK_CARD_VALUES, evaluate=SDK_RANKER). Normalize ranks so larger means stronger; supply an optional standard hand-category classifier only when its encoding is understood.",
            "Measure decode, history update, policy, and encode together in the provided runtime, including cold starts.",
            "Verify allowed dependencies, prewritten code, upload format, and what the 1ms measurement includes.",
        ],
        "integration_order": [
            "Get the starter's baseline running unchanged",
            "Read rules and resolve the mapping above",
            "Connect the adapter to the policy or provided equity backend",
            "Run official smoke tests and complete-game legality checks",
            "Benchmark the entire callback",
            "Package only the checked candidate; retain the last working artifact",
        ],
        "event_timeline": {
            "minutes_0_to_10": "Run starter baseline; identify variant, Agent interface, dependencies, scoring and timing",
            "minutes_10_to_25": "Connect the adapter; get a first accepted working bot",
            "minutes_25_to_45": "Add compatible strategy improvements and test actual tournament points",
            "minutes_45_to_60": "Freeze a validated candidate, verify the artifact, and submit with time to fix errors",
            "later_rounds": "Keep improving between rounds; preserve the accepted bot until a replacement passes the same checks",
        },
        "variant_limit": "The prepared strategy is standard no-limit Hold'em. Another revealed variant requires strategy and evaluator changes; an interface adapter alone is insufficient.",
    }


def _selected_files(root: Path, paths: tuple[str, ...]) -> dict[str, bytes]:
    selected = {}
    for relative in paths:
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError("candidate paths must stay inside the starter directory")
        target = root / name
        if (
            not target.exists()
            or target.is_symlink()
            or not target.resolve().is_relative_to(root)
        ):
            raise ValueError(f"missing or external candidate path: {relative}")
        entries = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in entries:
            if any(
                p in IGNORE or p.startswith(".env")
                for p in path.relative_to(root).parts
            ):
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError(
                    f"candidate includes a symlink or external path: {path}"
                )
            if path.is_file():
                selected[str(path.relative_to(root))] = path.read_bytes()
    return selected


def build(
    starter: Path,
    entry: str,
    includes: tuple[str, ...],
    output: Path,
    check: list[str],
    backend: str = "provided",
    context: str = "provided-starter",
    timeout: float = 120,
) -> dict:
    root = starter.resolve()
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("check timeout must be positive and finite")
    if not entry.endswith(".py") or not root.is_dir():
        raise ValueError(
            "supply a Python Agent entry file inside the starter directory"
        )
    if backend not in {"provided", "eval7", "starter"}:
        raise ValueError("backend must be provided, eval7, or starter")
    if not check or not any("{entry}" in arg or "{candidate}" in arg for arg in check):
        raise ValueError(
            "the simulator check must reference {entry} or {candidate}, so it tests the staged candidate"
        )
    core = () if backend == "starter" else CORE
    payload = {
        "competition/" + name: (HERE / name).read_bytes()
        for name in core + (("holdem.py",) if backend == "eval7" else ())
    }
    additions = _selected_files(root, (entry,) + includes)
    collisions = payload.keys() & additions.keys()
    if collisions:
        raise ValueError(
            f"starter files collide with prepared runtime: {sorted(collisions)}"
        )
    payload.update(additions)
    if sum(len(data) for data in payload.values()) > LIMIT:
        raise ValueError("candidate exceeds the 100MB uncompressed preparation limit")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="candidate-", dir=output.parent
    ) as temporary:
        candidate = Path(temporary)
        for name, data in payload.items():
            path = candidate / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)

        def verify_at(folder: Path, phase: str) -> dict:
            command = [
                arg.replace("{candidate}", str(folder))
                .replace("{entry}", str(folder / entry))
                .replace("{starter}", str(root))
                for arg in check
            ]
            environment = dict(os.environ)
            environment["PYTHONPATH"] = str(folder)
            started = perf_counter()
            try:
                result = subprocess.run(
                    command,
                    check=False,
                    cwd=folder,
                    env=environment,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                _write_json(
                    output.with_suffix(".failed.json"),
                    {
                        "passed": False,
                        "phase": phase,
                        "reason": "simulator check timed out",
                        "previous_artifact_preserved": True,
                        "command": check,
                    },
                )
                raise RuntimeError(
                    "simulator check timed out; the previous artifact was preserved"
                ) from None
            verification = {
                "phase": phase,
                "command_template": check,
                "returncode": result.returncode,
                "elapsed_seconds": perf_counter() - started,
                "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-12000:],
            }
            changed = [
                name
                for name, data in payload.items()
                if not (folder / name).is_file() or (folder / name).read_bytes() != data
            ]
            if result.returncode or changed:
                _write_json(
                    output.with_suffix(".failed.json"),
                    {
                        "passed": False,
                        "verification": verification,
                        "changed_source_files": changed,
                        "previous_artifact_preserved": True,
                    },
                )
                detail = (result.stderr.strip() or result.stdout.strip())[-1500:]
                raise RuntimeError(
                    f"{phase} check failed or modified candidate source; previous artifact preserved; see {output.with_suffix('.failed.json')}"
                    + (f"\nCheck output: {detail}" if detail else "")
                )
            return verification

        verification = verify_at(candidate, "staged-source")
        archive = candidate / "artifact.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in sorted(payload.items()):
                z.writestr(name, data)
        if archive.stat().st_size > LIMIT:
            raise ValueError("compressed candidate exceeds 100MB")
        extracted = candidate / "extracted"
        with zipfile.ZipFile(archive) as z:
            z.extractall(extracted)
        archive_verification = verify_at(extracted, "extracted-archive")
        manifest = {
            "checks_passed": True,
            "verification_context": context,
            "official_interface_assumed": False,
            "entry": entry,
            "backend": backend,
            "prepared_holdem_core_included": backend != "starter",
            "external_runtime_dependencies": (
                None if backend == "starter"
                else ["eval7==0.1.11"] if backend == "eval7"
                else []
            ),
            "dependency_inventory_note": "Starter dependency versions must be checked against the official runtime; external_import_roots is a static import inventory, not proof that dependencies are installed or allowed.",
            "external_import_roots": external_imports(payload),
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "archive_bytes": archive.stat().st_size,
            "files": [
                {
                    "path": name,
                    "bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
                for name, data in sorted(payload.items())
            ],
            "verification": verification,
            "archive_verification": archive_verification,
            "note": "The supplied command passed against exactly the staged source. Verify the event upload format and official runtime; a zip archive is not an assumed official submission format.",
        }
        if output.exists():
            digest = hashlib.sha256(output.read_bytes()).hexdigest()[:16]
            previous = output.parent / "previous" / f"{output.stem}-{digest}.zip"
            previous.parent.mkdir(parents=True, exist_ok=True)
            if not previous.exists():
                shutil.copy2(output, previous)
            manifest["previous_artifact"] = str(previous)
        archive.replace(output)
        _write_json(output.with_suffix(".manifest.json"), manifest)
        output.with_suffix(".failed.json").unlink(missing_ok=True)
    return manifest


def external_imports(payload: dict[str, bytes]) -> list[str]:
    local = {Path(name).parts[0].removesuffix(".py") for name in payload}
    roots = set()
    for name, source in payload.items():
        if not name.endswith(".py"):
            continue
        for node in ast.walk(ast.parse(source, filename=name)):
            if isinstance(node, ast.Import):
                roots.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
                roots.add(node.module.split(".")[0])
    return sorted(roots - local - sys.stdlib_module_names)


def rehearse(output: Path, deterministic: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    source = HERE / "rehearsal"
    report = intake(source, sys.executable)
    _write_json(output / "intake.json", report)
    command = [sys.executable, "-I", "-S", "{starter}/smoke.py", "--agent", "{entry}"]
    if deterministic:
        command.append("--deterministic")
    manifest = build(
        source,
        "agent.py",
        ("adapter.py", "sdk.py"),
        output / "rehearsal-agent.zip",
        command,
        context="synthetic-sdk-rehearsal",
    )
    summary = {
        "integration_rehearsal": "passed",
        "official_sdk": False,
        "strategy_strength_evaluated": False,
        "native_or_simulator_dependencies_required": False,
        "intake_interfaces_found": len(report["interfaces"]),
        "archive_bytes": manifest["archive_bytes"],
        "extracted_archive_verified": True,
        "details": json.loads(manifest["archive_verification"]["stdout"]),
    }
    _write_json(output / "rehearsal.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Citadel Poker: event-day starter intake, adapter rehearsal, and checked candidate packaging"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser(
        "intake", help="inspect a local starter without executing it"
    )
    inspect.add_argument("starter", type=Path)
    inspect.add_argument("--python", default=sys.executable)
    inspect.add_argument("--output", type=Path, required=True)
    practice = commands.add_parser(
        "rehearse",
        help="exercise a synthetic SDK and extracted runtime archive using only Python's standard library",
    )
    practice.add_argument("--output", type=Path, required=True)
    practice.add_argument(
        "--deterministic", action="store_true",
        help="freeze the mock SDK budget clocks for wiring checks; not a latency certification",
    )
    package = commands.add_parser(
        "build",
        help="stage source, run the supplied simulator check, then replace the candidate on success",
    )
    package.add_argument("starter", type=Path)
    package.add_argument("--entry", required=True)
    package.add_argument("--include", nargs="*", default=[])
    package.add_argument(
        "--backend",
        choices=["provided", "eval7", "starter"],
        default="provided",
        help="provided/eval7 include the prepared Hold'em core; starter packages only explicitly selected starter files for any variant",
    )
    package.add_argument("--output", type=Path, required=True)
    package.add_argument("--timeout", type=float, default=120)
    package.add_argument("--check", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()
    try:
        if args.command == "intake":
            result = intake(args.starter, args.python)
            _write_json(args.output, result)
            print(
                f"Inspected {len(result['files'])} files and {len(result['interfaces'])} interfaces. Report: {args.output}"
            )
            print(
                "Next: connect the real Agent entry and adapter, then run build with the official simulator check. A different variant needs strategy changes too."
            )
        elif args.command == "rehearse":
            print(json.dumps(rehearse(args.output, args.deterministic), indent=2))
        else:
            result = build(
                args.starter,
                args.entry,
                tuple(args.include),
                args.output,
                args.check,
                args.backend,
                timeout=args.timeout,
            )
            print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
