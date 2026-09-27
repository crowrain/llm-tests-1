#!/usr/bin/env python3
"""Side-by-side A/B comparison of two llm-tests-1 summaries.

Reads the summary-<model>.json files the harnesses write and answers the
question the harnesses exist for: which deployment answered better, and how
fast? Prints overall and per-category accuracy with deltas, truncation and
error counts, and median prefill/decode throughput.

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
    model = summary.get("model") or path.name.removeprefix("summary-").removesuffix(".json")
    return model, summary


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
    lines.append(row("accuracy", _pct(s_a.get("accuracy")), _pct(s_b.get("accuracy")), _pp(s_a.get("accuracy"), s_b.get("accuracy"))))
    lines.append(row("correct", f"{_count(s_a.get('correct'))}/{_count(s_a.get('total'))}", f"{_count(s_b.get('correct'))}/{_count(s_b.get('total'))}"))
    lines.append(row("truncated", _count(s_a.get("truncated")), _count(s_b.get("truncated")), _signed(s_a.get("truncated"), s_b.get("truncated"))))
    lines.append(row("errors", _count(s_a.get("errors")), _count(s_b.get("errors")), _signed(s_a.get("errors"), s_b.get("errors"))))
    lines.append("")
    lines.append("By category (accuracy, Δ = second − first)")
    categories = sorted(set(s_a.get("by_category", {})) | set(s_b.get("by_category", {})))
    for category in categories:
        cat_a = s_a.get("by_category", {}).get(category)
        cat_b = s_b.get("by_category", {}).get(category)
        acc_a = cat_a.get("accuracy") if isinstance(cat_a, dict) else None
        acc_b = cat_b.get("accuracy") if isinstance(cat_b, dict) else None
        lines.append(row(category, _pct(acc_a), _pct(acc_b), _pp(acc_a, acc_b)))
    lines.append("")
    lines.append("Speed (medians)")
    lines.append(row("decode tps", _num(s_a.get("median_decode_tps")), _num(s_b.get("median_decode_tps")), _rel(s_a.get("median_decode_tps"), s_b.get("median_decode_tps"))))
    lines.append(row("prefill tps", _num(s_a.get("median_prefill_tps")), _num(s_b.get("median_prefill_tps")), _rel(s_a.get("median_prefill_tps"), s_b.get("median_prefill_tps"))))
    return "\n".join(lines)


def as_json(name_a: str, s_a: dict[str, Any], name_b: str, s_b: dict[str, Any]) -> dict[str, Any]:
    def subset(summary: dict[str, Any]) -> dict[str, Any]:
        return {
            key: summary.get(key)
            for key in ("accuracy", "correct", "total", "truncated", "errors", "answer_empty", "median_decode_tps", "median_prefill_tps")
        }

    def delta(a: Any, b: Any) -> Any:
        return (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None

    by_category: dict[str, Any] = {}
    categories = sorted(set(s_a.get("by_category", {})) | set(s_b.get("by_category", {})))
    for category in categories:
        acc_a = s_a.get("by_category", {}).get(category, {}).get("accuracy")
        acc_b = s_b.get("by_category", {}).get(category, {}).get("accuracy")
        by_category[category] = {"a": acc_a, "b": acc_b, "delta": delta(acc_a, acc_b)}
    return {
        "a": {"model": name_a, **subset(s_a)},
        "b": {"model": name_b, **subset(s_b)},
        "delta_accuracy": delta(s_a.get("accuracy"), s_b.get("accuracy")),
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
    args = parser.parse_args()
    if not 1 <= len(args.paths) <= 2:
        raise SystemExit("pass one output directory or exactly two summaries")
    (name_a, s_a), (name_b, s_b) = resolve(args.paths)
    print(render(name_a, s_a, name_b, s_b), flush=True)
    if args.output:
        args.output.write_text(json.dumps(as_json(name_a, s_a, name_b, s_b), ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
