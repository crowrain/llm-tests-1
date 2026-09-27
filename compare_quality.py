#!/usr/bin/env python3
"""Side-by-side A/B comparison of two llm-tests-1 summaries.

Reads the summary-<model>.json files the harnesses write and answers the
question the harnesses exist for: which deployment answered better, and how
fast? It first verifies that profile, fixtures, recorded selection and request options match,
then prints accuracy over scored responses alongside request accuracy, truncation
and error counts, and median prefill/decode throughput.

Usage:
  python3 compare_quality.py runs/2026-09-27
  python3 compare_quality.py runs/day-a runs/day-b
  python3 compare_quality.py runs/a/summary-model-a.json runs/b/summary-model-b.json
  python3 compare_quality.py runs/2026-09-27 --output comparison.json

With one argument it must be an output directory containing exactly two
summaries (compared in file-name order); with two, each is a summary file or
a directory with exactly one summary. Deltas are second minus first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


Pair = tuple[str, dict[str, Any]]


def _load_summary(path: Path) -> Pair:
    summary = json.loads(path.read_text())
    name = summary.get("label") or summary.get("model") or path.name.removeprefix("summary-").removesuffix(".json")
    return name, summary


def compatibility_issues(s_a: dict[str, Any], s_b: dict[str, Any]) -> list[str]:
    """Explain why two summaries are not proven to cover the same work."""
    identity_a = s_a.get("run_identity")
    identity_b = s_b.get("run_identity")
    if not isinstance(identity_a, dict) or not isinstance(identity_b, dict):
        return ["one or both summaries have no run_identity (they predate compatibility metadata)"]
    issues = []
    required = {
        "schema_version",
        "profile",
        "fixtures_sha256",
        "selection_sha256",
        "case_ids",
        "request_options",
    }
    for side, identity in (("first", identity_a), ("second", identity_b)):
        missing = sorted(required - identity.keys())
        if missing:
            issues.append(f"{side} run_identity is missing {', '.join(missing)}")
    if issues:
        return issues
    if identity_a["schema_version"] != 1 or identity_b["schema_version"] != 1:
        issues.append(
            "unsupported run_identity schema "
            f"({identity_a['schema_version']!r}, {identity_b['schema_version']!r})"
        )
    for key, description in (
        ("profile", "harness profile"),
        ("fixtures_sha256", "full fixture set"),
        ("selection_sha256", "scored case selection"),
        ("request_options", "request and concurrency options"),
    ):
        if identity_a.get(key) != identity_b.get(key):
            issues.append(f"{description} differs ({identity_a.get(key)!r} != {identity_b.get(key)!r})")
    return issues


def quality_accuracy(summary: dict[str, Any]) -> Any:
    """Accuracy over actual model responses; transport failures are availability, not quality."""
    if "accuracy_excluding_errors" in summary:
        return summary.get("accuracy_excluding_errors")
    return summary.get("accuracy")


def resolve(paths: list[Path]) -> tuple[Pair, Pair]:
    """Resolve one or two paths into exactly two (model, summary) pairs."""
    if len(paths) == 1:
        directory = paths[0]
        if not directory.is_dir():
            raise SystemExit("a single argument must be an output directory containing two summaries")
        summaries = sorted(directory.glob("summary-*.json"))
        if len(summaries) != 2:
            raise SystemExit(f"expected exactly two summary-*.json files in {directory}, found {len(summaries)}")
        return _load_summary(summaries[0]), _load_summary(summaries[1])
    pairs: list[Pair] = []
    for path in paths:
        if path.is_file():
            pairs.append(_load_summary(path))
        elif path.is_dir():
            summaries = sorted(path.glob("summary-*.json"))
            if len(summaries) != 1:
                raise SystemExit(f"expected exactly one summary-*.json file in {path}, found {len(summaries)}")
            pairs.append(_load_summary(summaries[0]))
        else:
            raise SystemExit(f"no such file or directory: {path}")
    return pairs[0], pairs[1]


def _pct(value: Any) -> str:
    return "–" if not isinstance(value, (int, float)) else f"{100 * value:.1f}%"


def _pp(a: Any, b: Any) -> str:
    """Delta in percentage points (second minus first)."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
        return "–"
    return f"{100 * (b - a):+.1f} pp"


def _rel(a: Any, b: Any) -> str:
    """Relative delta in percent (second minus first, relative to the first)."""
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or a <= 0:
        return "–"
    return f"{100 * (b - a) / a:+.1f}%"


def _num(value: Any, digits: int = 1) -> str:
    return "–" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def _count(value: Any) -> str:
    return "–" if not isinstance(value, int) else str(value)


def _signed(a: Any, b: Any) -> str:
    if not isinstance(a, int) or not isinstance(b, int):
        return "–"
    return f"{b - a:+d}"


def render(name_a: str, s_a: dict[str, Any], name_b: str, s_b: dict[str, Any]) -> str:
    width = max(len(name_a), len(name_b))
    lines = [f"{name_a} vs {name_b}", ""]

    def row(label: str, a: str, b: str, delta: str = "–") -> str:
        return f"  {label:<16} {a:<{width}} {b:<{width}} {delta}"

    lines.append("Overall")
    acc_a, acc_b = quality_accuracy(s_a), quality_accuracy(s_b)
    lines.append(row("scored accuracy", _pct(acc_a), _pct(acc_b), _pp(acc_a, acc_b)))
    raw_a, raw_b = s_a.get("accuracy"), s_b.get("accuracy")
    lines.append(row("request accuracy", _pct(raw_a), _pct(raw_b), _pp(raw_a, raw_b)))
    lines.append(
        row(
            "correct",
            f"{_count(s_a.get('correct'))}/{_count(s_a.get('total'))}",
            f"{_count(s_b.get('correct'))}/{_count(s_b.get('total'))}",
        )
    )
    trunc_a, trunc_b = s_a.get("truncated"), s_b.get("truncated")
    lines.append(row("truncated", _count(trunc_a), _count(trunc_b), _signed(trunc_a, trunc_b)))
    err_a, err_b = s_a.get("errors"), s_b.get("errors")
    lines.append(row("errors", _count(err_a), _count(err_b), _signed(err_a, err_b)))
    lines.append("")
    lines.append("By category (accuracy, Δ = second − first)")
    categories = sorted(set(s_a.get("by_category", {})) | set(s_b.get("by_category", {})))
    for category in categories:
        cat_a = s_a.get("by_category", {}).get(category)
        cat_b = s_b.get("by_category", {}).get(category)
        cat_acc_a = quality_accuracy(cat_a) if isinstance(cat_a, dict) else None
        cat_acc_b = quality_accuracy(cat_b) if isinstance(cat_b, dict) else None
        lines.append(row(category, _pct(cat_acc_a), _pct(cat_acc_b), _pp(cat_acc_a, cat_acc_b)))
    lines.append("")
    lines.append("Speed (medians)")
    dec_a, dec_b = s_a.get("median_decode_tps"), s_b.get("median_decode_tps")
    lines.append(row("decode tps", _num(dec_a), _num(dec_b), _rel(dec_a, dec_b)))
    pre_a, pre_b = s_a.get("median_prefill_tps"), s_b.get("median_prefill_tps")
    lines.append(row("prefill tps", _num(pre_a), _num(pre_b), _rel(pre_a, pre_b)))
    return "\n".join(lines)


def as_json(name_a: str, s_a: dict[str, Any], name_b: str, s_b: dict[str, Any]) -> dict[str, Any]:
    def subset(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            key: summary.get(key)
            for key in (
                "accuracy",
                "accuracy_excluding_errors",
                "correct",
                "total",
                "truncated",
                "errors",
                "answer_empty",
                "median_decode_tps",
                "median_prefill_tps",
            )
        }

    def delta(a: Any, b: Any) -> Any:
        return (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    def side(name: str, summary: dict[str, Any]) -> dict[str, Any]:
        return {
            "name": name,
            "model": summary.get("model") or name,
            "label": summary.get("label") or name,
            **subset(summary),
        }

    by_category: dict[str, Any] = {}
    categories = sorted(set(s_a.get("by_category", {})) | set(s_b.get("by_category", {})))
    for category in categories:
        cat_a = s_a.get("by_category", {}).get(category, {})
        cat_b = s_b.get("by_category", {}).get(category, {})
        acc_a = quality_accuracy(cat_a)
        acc_b = quality_accuracy(cat_b)
        by_category[category] = {"a": acc_a, "b": acc_b, "delta": delta(acc_a, acc_b)}
    return {
        "a": side(name_a, s_a),
        "b": side(name_b, s_b),
        "delta_accuracy": delta(quality_accuracy(s_a), quality_accuracy(s_b)),
        "delta_request_accuracy": delta(s_a.get("accuracy"), s_b.get("accuracy")),
        "by_category": by_category,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two llm-tests-1 summaries side by side.")
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="one output directory with two summaries, or two summary files / directories",
    )
    parser.add_argument("--output", type=Path, help="also write the comparison as JSON to this path")
    parser.add_argument(
        "--allow-incompatible",
        action="store_true",
        help="compare even when profile/fixture/selection/request identity is missing or differs",
    )
    args = parser.parse_args()
    if not 1 <= len(args.paths) <= 2:
        raise SystemExit("pass one output directory or exactly two summaries")
    (name_a, s_a), (name_b, s_b) = resolve(args.paths)
    issues = compatibility_issues(s_a, s_b)
    if issues and not args.allow_incompatible:
        details = "; ".join(issues)
        raise SystemExit(
            f"refusing a non-like-for-like comparison: {details}. "
            "Use --allow-incompatible only if this difference is intentional"
        )
    print(render(name_a, s_a, name_b, s_b), flush=True)
    if args.output:
        args.output.write_text(json.dumps(as_json(name_a, s_a, name_b, s_b), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
