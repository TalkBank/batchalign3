#!/usr/bin/env python3
"""Compare batchalign3 output against the Jan 9 2026 legacy baseline.

Two workflows live behind one script:

1. Raw file-diff mode:
   run two CLIs side-by-side, collect output files, and compare them with an
   allowed-difference rewrite policy. Legacy baseline runners use their native
   `command input_dir output_dir` syntax.

2. Manifest benchmark mode:
   run curated benchmark cases from a JSON manifest, parse `.compare.csv`
   metrics, and classify current-vs-baseline differences as:
   - allowlisted baseline differences
   - other baseline improvements
   - unexpected regressions
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

REPO = Path(__file__).resolve().parents[1]
DEFAULT_POLICY = REPO / "test-fixtures" / "comparison-policies" / "morphotag-default.json"
DEFAULT_STOCK_MANIFEST = REPO / "test-fixtures" / "stock_batchalign" / "manifest.json"
DEFAULT_STOCK_ALLOWLIST = REPO / "test-fixtures" / "stock_batchalign" / "allowlist.json"
JAN9_BASELINE_CANDIDATES = (
    Path.home() / "bin" / "batchalign-jan84ad500",
    Path.home() / "bin" / "batchalign2-jan84ad500" / "batchalign",
    Path.home() / "bin" / "batchalign-pins" / "repos" / "batchalign2-jan84ad500" / "batchalign",
)

LOWER_IS_BETTER = {"wer", "insertions", "deletions"}
HIGHER_IS_BETTER = {"accuracy", "matches"}
TRACKED_BENCHMARK_METRICS = ("wer", "accuracy", "matches", "insertions", "deletions")

TargetSide = Literal["baseline", "current", "both"]


@dataclass(frozen=True)
class RewriteRule:
    id: str
    description: str
    target: TargetSide
    pattern: str
    replace: str
    line_prefixes: tuple[str, ...]

    @classmethod
    def from_json(cls, data: dict[str, object]) -> "RewriteRule":
        raw_line_prefixes = data.get("line_prefixes")
        line_prefixes = (
            tuple(str(v) for v in raw_line_prefixes)
            if isinstance(raw_line_prefixes, list)
            else ()
        )
        return cls(
            id=str(data["id"]),
            description=str(data.get("description", "")),
            target=str(data.get("target", "both")),  # type: ignore[arg-type]
            pattern=str(data["pattern"]),
            replace=str(data.get("replace", "")),
            line_prefixes=line_prefixes,
        )


@dataclass(frozen=True)
class ComparisonPolicy:
    description: str
    line_prefixes: tuple[str, ...]
    rewrites: tuple[RewriteRule, ...]
    source_path: Path | None

    @classmethod
    def empty(cls) -> "ComparisonPolicy":
        return cls("", (), (), None)


@dataclass(frozen=True)
class CommandRun:
    label: str
    executable: str
    argv: tuple[str, ...]
    returncode: int
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class StockCase:
    id: str
    description: str
    command: str
    input_path: Path
    gold_path: Path
    current_argv: tuple[str, ...]
    baseline_argv: tuple[str, ...]


class CliInvocationStyle(str, Enum):
    """How one runner expects its input and output paths on the CLI."""

    MODERN = "modern"
    LEGACY_DIRECTORY_IO = "legacy_directory_io"


@dataclass(frozen=True)
class PreparedBenchmarkCase:
    input_dir: Path
    output_dir: Path
    copied_input: Path

    def metrics_path(self) -> Path:
        return self.output_dir / f"{self.copied_input.stem}.compare.csv"


@dataclass(frozen=True)
class AllowlistEntry:
    case_id: str
    metric: str
    reason: str
    max_delta: float


def default_current_bin() -> str:
    env = os.environ.get("BATCHALIGN_COMPARE_CURRENT_BIN")
    if env:
        return env

    for candidate in (
        REPO / "target" / "debug" / "batchalign3",
        REPO / "target" / "release" / "batchalign3",
        REPO / ".venv" / "bin" / "batchalign3",
    ):
        if candidate.exists():
            return str(candidate)
    return "batchalign3"


def default_baseline_bin() -> str | None:
    for env_name in (
        "BATCHALIGN2_BASELINE_EXECUTABLE",
        "BATCHALIGN_COMPARE_BASELINE_BIN",
        "BATCHALIGN_BASELINE_EXECUTABLE",
    ):
        env = os.environ.get(env_name)
        if env:
            return env

    for candidate in JAN9_BASELINE_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return None


def resolve_executable(candidate: str) -> str | None:
    expanded = os.path.expanduser(candidate)
    if os.sep in expanded or expanded.startswith("."):
        path = Path(expanded)
        if path.exists():
            return str(path.resolve())
        return None
    return shutil.which(candidate)


def load_policy(path: Path | None) -> ComparisonPolicy:
    if path is None:
        return ComparisonPolicy.empty()

    data = json.loads(path.read_text())
    rewrites = tuple(RewriteRule.from_json(item) for item in data.get("rewrites", []))
    return ComparisonPolicy(
        description=str(data.get("description", "")),
        line_prefixes=tuple(str(v) for v in data.get("line_prefixes", [])),
        rewrites=rewrites,
        source_path=path,
    )


def load_stock_cases(path: Path) -> list[StockCase]:
    data = json.loads(path.read_text())
    base_dir = path.parent
    cases: list[StockCase] = []
    for raw in data.get("cases", []):
        current_argv = tuple(str(v) for v in raw.get("current_argv", raw.get("argv", [])))
        baseline_argv = tuple(str(v) for v in raw.get("baseline_argv", raw.get("argv", [])))
        cases.append(
            StockCase(
                id=str(raw["id"]),
                description=str(raw.get("description", "")),
                command=str(raw.get("command", "benchmark")),
                input_path=(base_dir / str(raw["audio_path"])).resolve(),
                gold_path=(base_dir / str(raw["gold_path"])).resolve(),
                current_argv=current_argv,
                baseline_argv=baseline_argv,
            )
        )
    return cases


def load_allowlist(path: Path | None) -> dict[tuple[str, str], AllowlistEntry]:
    if path is None or not path.exists():
        return {}

    data = json.loads(path.read_text())
    entries: dict[tuple[str, str], AllowlistEntry] = {}
    for raw in data.get("entries", []):
        entry = AllowlistEntry(
            case_id=str(raw["case_id"]),
            metric=str(raw["metric"]),
            reason=str(raw["reason"]),
            max_delta=float(raw["max_delta"]),
        )
        entries[(entry.case_id, entry.metric)] = entry
    return entries


def collect_matching_files(root: Path, pattern: str) -> dict[str, Path]:
    files: dict[str, Path] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob(pattern)):
        if path.is_file():
            files[str(path.relative_to(root))] = path
    return files


def select_lines(text: str, prefixes: tuple[str, ...]) -> list[str]:
    lines = text.splitlines()
    if not prefixes:
        return lines
    return [line for line in lines if line.startswith(prefixes)]


def normalize_lines(
    lines: list[str],
    policy: ComparisonPolicy,
    side: TargetSide,
) -> tuple[list[str], list[str]]:
    normalized: list[str] = []
    applied: list[str] = []
    seen = set()

    for line in lines:
        current = line
        for rule in policy.rewrites:
            if rule.target not in (side, "both"):
                continue
            if rule.line_prefixes and not current.startswith(rule.line_prefixes):
                continue
            updated, count = re.subn(rule.pattern, rule.replace, current)
            if count > 0 and rule.id not in seen:
                seen.add(rule.id)
                applied.append(rule.id)
            current = updated
        normalized.append(current)

    return normalized, applied


def run_command(
    *,
    label: str,
    executable: str,
    command: str,
    output_dir: Path,
    inputs: list[Path],
    common_args: list[str],
    extra_args: list[str],
    add_no_tui: bool,
    cli_style: CliInvocationStyle,
    workdir: Path,
    legacy_input_dir: Path | None = None,
) -> CommandRun:
    workdir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = [executable, command]
    if cli_style is CliInvocationStyle.MODERN:
        if add_no_tui:
            argv.append("--no-tui")
        argv.extend(common_args)
        argv.extend(extra_args)
        argv.extend(["--output", str(output_dir)])
        argv.extend(str(path) for path in inputs)
    else:
        legacy_root = legacy_input_dir or stage_legacy_inputs(inputs, workdir / f"{label}-input")
        argv.extend(common_args)
        argv.extend(extra_args)
        argv.extend([str(legacy_root), str(output_dir)])

    result = subprocess.run(
        argv,
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    stdout_path = workdir / f"{label}.stdout.log"
    stderr_path = workdir / f"{label}.stderr.log"
    stdout_path.write_text(result.stdout)
    stderr_path.write_text(result.stderr)

    return CommandRun(
        label=label,
        executable=executable,
        argv=tuple(argv),
        returncode=result.returncode,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )


def parse_metrics_csv(path: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lines = path.read_text().splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        metric, value = line.split(",", 1)
        metrics[metric] = float(value)
    return metrics


def stage_legacy_inputs(inputs: list[Path], input_dir: Path) -> Path:
    input_dir.mkdir(parents=True, exist_ok=True)
    seen_names: set[str] = set()
    for path in inputs:
        if not path.exists():
            raise FileNotFoundError(f"Missing legacy baseline input: {path}")
        destination = input_dir / path.name
        if destination.name in seen_names:
            raise ValueError(
                f"Legacy baseline staging would collide on {destination.name!r}; "
                "pass one input root at a time."
            )
        seen_names.add(destination.name)
        if path.is_dir():
            shutil.copytree(path, destination)
        else:
            shutil.copy2(path, destination)
    return input_dir


def plan_benchmark_case(case: StockCase, case_root: Path) -> PreparedBenchmarkCase:
    input_dir = case_root / "input"
    output_dir = case_root / "output"
    copied_input = input_dir / case.input_path.name
    return PreparedBenchmarkCase(input_dir=input_dir, output_dir=output_dir, copied_input=copied_input)


def prepare_benchmark_case(case: StockCase, case_root: Path) -> PreparedBenchmarkCase:
    if not case.input_path.exists():
        raise FileNotFoundError(f"Missing benchmark input: {case.input_path}")
    if not case.gold_path.exists():
        raise FileNotFoundError(f"Missing benchmark gold file: {case.gold_path}")

    prepared = plan_benchmark_case(case, case_root)
    prepared.input_dir.mkdir(parents=True, exist_ok=True)
    prepared.output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(case.input_path, prepared.copied_input)
    shutil.copy2(case.gold_path, prepared.input_dir / f"{case.input_path.stem}.cha")
    return prepared


def compare_metric(metric: str, current: float, baseline: float) -> tuple[str, float] | None:
    if metric in LOWER_IS_BETTER:
        if current < baseline:
            return "improved", baseline - current
        if current > baseline:
            return "regressed", current - baseline
        return None

    if metric in HIGHER_IS_BETTER:
        if current > baseline:
            return "improved", current - baseline
        if current < baseline:
            return "regressed", baseline - current
        return None

    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare batchalign3 and a Jan 9 2026 legacy batchalign/batchalignhk "
            "baseline using either raw output file diffing or a benchmark manifest."
        )
    )
    parser.add_argument("inputs", nargs="*", help="Input files or directories for raw file-diff mode")
    parser.add_argument(
        "--command",
        default="morphotag",
        help="Command to run in raw file-diff mode (default: morphotag)",
    )
    parser.add_argument(
        "--manifest",
        help="JSON manifest for benchmark comparison mode",
    )
    parser.add_argument(
        "--allowlist",
        help="JSON allowlist for known baseline differences in benchmark mode",
    )
    parser.add_argument(
        "--current-bin",
        "--current-executable",
        dest="current_bin",
        help="Current batchalign3 executable (default: best local batchalign3 candidate)",
    )
    parser.add_argument(
        "--baseline-bin",
        "--baseline-executable",
        dest="baseline_bin",
        help=(
            "Jan 9 2026 legacy `batchalign` or `batchalignhk` baseline executable. "
            "Defaults to "
            "BATCHALIGN2_BASELINE_EXECUTABLE / BATCHALIGN_COMPARE_BASELINE_BIN / "
            "BATCHALIGN_BASELINE_EXECUTABLE or known maintainer-local pinned "
            "wrappers if present."
        ),
    )
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY) if DEFAULT_POLICY.exists() else None,
        help="JSON policy file describing allowed-difference rewrites for raw file-diff mode",
    )
    parser.add_argument(
        "--report",
        help="Write a JSON report to this path",
    )
    parser.add_argument(
        "--run-root",
        help="Directory for benchmark-mode case artifacts",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run benchmark manifest mode with only the current executable",
    )
    parser.add_argument(
        "--compare-glob",
        default="*.cha",
        help="Glob used under each output tree when collecting files (default: *.cha)",
    )
    parser.add_argument(
        "--compare-prefix",
        action="append",
        default=[],
        help="Only compare lines with this prefix in raw file-diff mode (repeatable)",
    )
    parser.add_argument(
        "--common-arg",
        action="append",
        default=[],
        help="Extra CLI arg to pass to both commands in raw file-diff mode (repeatable)",
    )
    parser.add_argument(
        "--current-arg",
        action="append",
        default=[],
        help="Extra CLI arg to pass only to batchalign3 (repeatable)",
    )
    parser.add_argument(
        "--baseline-arg",
        action="append",
        default=[],
        help="Extra CLI arg to pass only to the Jan 9 legacy baseline (repeatable)",
    )
    parser.add_argument(
        "--no-current-no-tui",
        action="store_true",
        help="Do not automatically add --no-tui to the batchalign3 command",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="Keep temporary comparison artifacts even on success",
    )
    return parser.parse_args()


def run_file_compare_mode(args: argparse.Namespace, current_bin: str, baseline_bin: str) -> int:
    inputs = [Path(value).expanduser().resolve() for value in args.inputs]
    missing_inputs = [str(path) for path in inputs if not path.exists()]
    if missing_inputs:
        print(f"Missing inputs: {missing_inputs}", file=sys.stderr)
        return 1

    policy_path = Path(args.policy).expanduser().resolve() if args.policy else None
    if policy_path and not policy_path.exists():
        print(f"Policy file not found: {policy_path}", file=sys.stderr)
        return 1
    policy = load_policy(policy_path)
    compare_prefixes = tuple(args.compare_prefix) if args.compare_prefix else policy.line_prefixes

    workdir = Path(tempfile.mkdtemp(prefix="batchalign-compare-"))
    current_out = workdir / "current"
    baseline_out = workdir / "baseline"
    current_out.mkdir()
    baseline_out.mkdir()
    diffs_dir = workdir / "diffs"
    diffs_dir.mkdir()

    keep_workdir = args.keep_workdir

    try:
        current_run = run_command(
            label="current",
            executable=current_bin,
            command=args.command,
            output_dir=current_out,
            inputs=inputs,
            common_args=list(args.common_arg),
            extra_args=list(args.current_arg),
            add_no_tui=not args.no_current_no_tui,
            cli_style=CliInvocationStyle.MODERN,
            workdir=workdir,
        )
        baseline_run = run_command(
            label="baseline",
            executable=baseline_bin,
            command=args.command,
            output_dir=baseline_out,
            inputs=inputs,
            common_args=list(args.common_arg),
            extra_args=list(args.baseline_arg),
            add_no_tui=False,
            cli_style=CliInvocationStyle.LEGACY_DIRECTORY_IO,
            workdir=workdir,
        )

        failures = [run for run in (current_run, baseline_run) if run.returncode != 0]
        if failures:
            keep_workdir = True

        current_files = collect_matching_files(current_out, args.compare_glob)
        baseline_files = collect_matching_files(baseline_out, args.compare_glob)
        all_relative_paths = sorted(set(current_files) | set(baseline_files))

        exact_matches = 0
        allowed_differences = 0
        unexpected_differences = 0
        missing_outputs = 0
        file_reports: list[dict[str, object]] = []

        for relative in all_relative_paths:
            current_path = current_files.get(relative)
            baseline_path = baseline_files.get(relative)
            if current_path is None or baseline_path is None:
                missing_outputs += 1
                file_reports.append(
                    {
                        "relative_path": relative,
                        "status": "missing_output",
                        "current_present": current_path is not None,
                        "baseline_present": baseline_path is not None,
                        "diff_path": None,
                        "applied_rules": [],
                    }
                )
                continue

            current_lines = select_lines(current_path.read_text(), compare_prefixes)
            baseline_lines = select_lines(baseline_path.read_text(), compare_prefixes)

            if current_lines == baseline_lines:
                exact_matches += 1
                file_reports.append(
                    {
                        "relative_path": relative,
                        "status": "exact",
                        "current_present": True,
                        "baseline_present": True,
                        "diff_path": None,
                        "applied_rules": [],
                    }
                )
                continue

            normalized_current, current_rules = normalize_lines(current_lines, policy, "current")
            normalized_baseline, baseline_rules = normalize_lines(baseline_lines, policy, "baseline")
            applied_rules = sorted(set(current_rules + baseline_rules))

            diff_lines = list(
                difflib.unified_diff(
                    normalized_baseline,
                    normalized_current,
                    fromfile=f"baseline/{relative}",
                    tofile=f"current/{relative}",
                    lineterm="",
                )
            )
            diff_path = diffs_dir / f"{relative}.diff"
            diff_path.parent.mkdir(parents=True, exist_ok=True)
            diff_path.write_text("\n".join(diff_lines) + ("\n" if diff_lines else ""))

            if normalized_current == normalized_baseline:
                allowed_differences += 1
                status = "allowed_difference"
            else:
                unexpected_differences += 1
                status = "unexpected_difference"
                keep_workdir = True

            file_reports.append(
                {
                    "relative_path": relative,
                    "status": status,
                    "current_present": True,
                    "baseline_present": True,
                    "diff_path": str(diff_path),
                    "applied_rules": applied_rules,
                }
            )

        report = {
            "mode": "file-diff",
            "command": args.command,
            "current_bin": current_bin,
            "baseline_bin": baseline_bin,
            "policy": {
                "path": str(policy.source_path) if policy.source_path else None,
                "description": policy.description,
                "line_prefixes": list(compare_prefixes),
                "rewrite_ids": [rule.id for rule in policy.rewrites],
            },
            "runs": {
                "current": {
                    "argv": list(current_run.argv),
                    "returncode": current_run.returncode,
                    "stdout_path": str(current_run.stdout_path),
                    "stderr_path": str(current_run.stderr_path),
                },
                "baseline": {
                    "argv": list(baseline_run.argv),
                    "returncode": baseline_run.returncode,
                    "stdout_path": str(baseline_run.stdout_path),
                    "stderr_path": str(baseline_run.stderr_path),
                },
            },
            "summary": {
                "exact_matches": exact_matches,
                "allowed_differences": allowed_differences,
                "unexpected_differences": unexpected_differences,
                "missing_outputs": missing_outputs,
                "kept_workdir": keep_workdir,
                "workdir": str(workdir) if keep_workdir else None,
            },
            "files": file_reports,
        }

        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        print(
            "Comparison summary: "
            f"exact={exact_matches}, "
            f"allowed={allowed_differences}, "
            f"unexpected={unexpected_differences}, "
            f"missing={missing_outputs}"
        )
        if args.report:
            print(f"Report: {Path(args.report).expanduser().resolve()}")
        if keep_workdir:
            print(f"Artifacts kept at: {workdir}")

        if failures or unexpected_differences or missing_outputs:
            return 1
        return 0
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir)


def run_manifest_mode(
    args: argparse.Namespace,
    current_bin: str,
    baseline_bin: str | None,
) -> int:
    manifest_path = Path(args.manifest).expanduser().resolve()
    if not manifest_path.exists():
        print(f"Manifest file not found: {manifest_path}", file=sys.stderr)
        return 1

    allowlist_path: Path | None
    if args.allowlist:
        allowlist_path = Path(args.allowlist).expanduser().resolve()
    elif DEFAULT_STOCK_ALLOWLIST.exists():
        allowlist_path = DEFAULT_STOCK_ALLOWLIST
    else:
        allowlist_path = None
    if allowlist_path and not allowlist_path.exists():
        print(f"Allowlist file not found: {allowlist_path}", file=sys.stderr)
        return 1

    cases = load_stock_cases(manifest_path)
    allowlist = load_allowlist(allowlist_path)

    run_root = (
        Path(args.run_root).expanduser().resolve()
        if args.run_root
        else Path(tempfile.mkdtemp(prefix="batchalign-stock-compare-"))
    )
    run_root.mkdir(parents=True, exist_ok=True)
    keep_run_root = args.keep_workdir or args.run_root is not None
    failures = False
    case_reports: list[dict[str, object]] = []

    print(f"Current runner: {current_bin}")
    if args.skip_baseline:
        print("Baseline runner: skipped")
    else:
        print(f"Baseline runner: {baseline_bin}")

    try:
        for case in cases:
            case_report: dict[str, object] = {
                "id": case.id,
                "description": case.description,
                "command": case.command,
            }
            print(f"\nCase {case.id}")

            current_case_root = run_root / "current" / case.id
            try:
                current_prepared = prepare_benchmark_case(case, current_case_root)
            except FileNotFoundError as exc:
                print(f"  {'status':<12}: current setup failed")
                print(f"    - {exc}")
                case_report["status"] = "current setup failed"
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            current_run = run_command(
                label="current",
                executable=current_bin,
                command=case.command,
                output_dir=current_prepared.output_dir,
                inputs=[current_prepared.copied_input],
                common_args=list(case.current_argv),
                extra_args=list(args.current_arg),
                add_no_tui=not args.no_current_no_tui,
                cli_style=CliInvocationStyle.MODERN,
                workdir=current_case_root,
            )
            if current_run.returncode != 0:
                print(f"  {'status':<12}: current failed")
                print(f"    - see {current_run.stderr_path}")
                case_report["status"] = "current failed"
                case_report["current_returncode"] = current_run.returncode
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            current_metrics_path = current_prepared.metrics_path()
            if not current_metrics_path.exists():
                print(f"  {'status':<12}: current missing metrics")
                case_report["status"] = "current missing metrics"
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            current_metrics = parse_metrics_csv(current_metrics_path)
            case_report["current_metrics"] = current_metrics
            print(
                f"  {'current':<12}: "
                f"wer={current_metrics.get('wer', float('nan')):.4f} "
                f"accuracy={current_metrics.get('accuracy', float('nan')):.4f}"
            )

            if args.skip_baseline:
                print(f"  {'status':<12}: baseline skipped")
                case_report["status"] = "baseline skipped"
                case_reports.append(case_report)
                continue

            assert baseline_bin is not None
            baseline_case_root = run_root / "baseline" / case.id
            try:
                baseline_prepared = prepare_benchmark_case(case, baseline_case_root)
            except FileNotFoundError as exc:
                print(f"  {'status':<12}: baseline setup failed")
                print(f"    - {exc}")
                case_report["status"] = "baseline setup failed"
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            baseline_run = run_command(
                label="baseline",
                executable=baseline_bin,
                command=case.command,
                output_dir=baseline_prepared.output_dir,
                inputs=[],
                common_args=list(case.baseline_argv),
                extra_args=list(args.baseline_arg),
                add_no_tui=False,
                cli_style=CliInvocationStyle.LEGACY_DIRECTORY_IO,
                workdir=baseline_case_root,
                legacy_input_dir=baseline_prepared.input_dir,
            )
            if baseline_run.returncode != 0:
                print(f"  {'status':<12}: baseline failed")
                print(f"    - see {baseline_run.stderr_path}")
                case_report["status"] = "baseline failed"
                case_report["baseline_returncode"] = baseline_run.returncode
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            baseline_metrics_path = baseline_prepared.metrics_path()
            if not baseline_metrics_path.exists():
                print(f"  {'status':<12}: baseline missing metrics")
                case_report["status"] = "baseline missing metrics"
                failures = True
                keep_run_root = True
                case_reports.append(case_report)
                continue

            baseline_metrics = parse_metrics_csv(baseline_metrics_path)
            case_report["baseline_metrics"] = baseline_metrics
            print(
                f"  {'baseline':<12}: "
                f"wer={baseline_metrics.get('wer', float('nan')):.4f} "
                f"accuracy={baseline_metrics.get('accuracy', float('nan')):.4f}"
            )

            allowlisted_messages: list[str] = []
            baseline_difference_messages: list[str] = []
            regression_messages: list[str] = []

            for metric in TRACKED_BENCHMARK_METRICS:
                if metric not in current_metrics or metric not in baseline_metrics:
                    continue
                comparison = compare_metric(metric, current_metrics[metric], baseline_metrics[metric])
                if comparison is None:
                    continue
                direction, delta = comparison
                if direction == "improved":
                    entry = allowlist.get((case.id, metric))
                    if entry and delta <= entry.max_delta + 1e-12:
                        allowlisted_messages.append(
                            f"{metric} improved by {delta:.4f} ({entry.reason})"
                        )
                    else:
                        baseline_difference_messages.append(f"{metric} improved by {delta:.4f}")
                elif direction == "regressed":
                    regression_messages.append(f"{metric} regressed by {delta:.4f}")

            if allowlisted_messages:
                print("  allowlisted baseline differences:")
                for message in allowlisted_messages:
                    print(f"    - {message}")

            if baseline_difference_messages:
                print("  baseline differences:")
                for message in baseline_difference_messages:
                    print(f"    - {message}")

            if regression_messages:
                print("  unexpected regressions:")
                for message in regression_messages:
                    print(f"    - {message}")
                case_report["status"] = "unexpected regressions"
                failures = True
                keep_run_root = True
            else:
                case_report["status"] = "ok"

            case_report["allowlisted_differences"] = allowlisted_messages
            case_report["baseline_differences"] = baseline_difference_messages
            case_report["unexpected_regressions"] = regression_messages
            case_reports.append(case_report)

        report = {
            "mode": "benchmark-manifest",
            "manifest": str(manifest_path),
            "allowlist": str(allowlist_path) if allowlist_path else None,
            "current_bin": current_bin,
            "baseline_bin": baseline_bin,
            "baseline_skipped": args.skip_baseline,
            "run_root": str(run_root) if keep_run_root else None,
            "cases": case_reports,
        }
        if args.report:
            report_path = Path(args.report).expanduser().resolve()
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

        if keep_run_root:
            print(f"\nArtifacts kept at: {run_root}")

        return 1 if failures else 0
    finally:
        if not keep_run_root:
            shutil.rmtree(run_root)


def main() -> int:
    args = parse_args()

    if args.manifest and args.inputs:
        print("Cannot combine --manifest with positional inputs.", file=sys.stderr)
        return 1
    if not args.manifest and not args.inputs:
        print("Provide positional inputs for raw mode or --manifest for benchmark mode.", file=sys.stderr)
        return 1

    current_candidate = args.current_bin or default_current_bin()
    current_bin = resolve_executable(current_candidate)
    if current_bin is None:
        print(f"Current executable not found: {current_candidate}", file=sys.stderr)
        return 1

    needs_baseline = not (args.manifest and args.skip_baseline)
    baseline_bin: str | None = None
    if needs_baseline:
        baseline_candidate = args.baseline_bin or default_baseline_bin()
        if baseline_candidate is None:
            print(
                "Baseline executable not found. Pass --baseline-bin or "
                "BATCHALIGN2_BASELINE_EXECUTABLE pointing at a Jan 9 2026 "
                "legacy batchalign/batchalignhk (`84ad500`) runner.",
                file=sys.stderr,
            )
            return 1
        baseline_bin = resolve_executable(baseline_candidate)
        if baseline_bin is None:
            print(f"Baseline executable not found: {baseline_candidate}", file=sys.stderr)
            return 1

    if args.manifest:
        return run_manifest_mode(args, current_bin, baseline_bin)
    assert baseline_bin is not None
    return run_file_compare_mode(args, current_bin, baseline_bin)


if __name__ == "__main__":
    raise SystemExit(main())
