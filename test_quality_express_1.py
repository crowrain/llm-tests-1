#!/usr/bin/env python3
"""Bounded, reproducible A/B quality test: the express profile (72 fast cases).

Thin entry point: dataset fetching, answer extraction, scoring, the HTTP layer and
summary reporting live in quality_common.py. The express profile spreads fewer,
cheaper samples across each test split, uses a short timeout/retry budget and
aborts the run on a failed request instead of recording it.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import quality_common as qc


def make_cases() -> list[dict[str, Any]]:
    # Spread fixed samples across each test split instead of taking one contiguous shard.
    gsm = qc.get_rows("openai/gsm8k", "main", "test", [i * 43 for i in range(30)])
    arc = qc.get_rows("allenai/ai2_arc", "ARC-Challenge", "test", [i * 37 for i in range(30)])
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(gsm):
        answer = row["answer"].rsplit("####", 1)[-1].strip()
        cases.append(
            {
                "id": f"gsm8k_{index:02d}",
                "category": "gsm8k",
                "prompt": row["question"] + "\n\nEnd the final answer with `#### <number>`.",
                "expected": answer,
                "max_tokens": 768,
            }
        )
    for index, row in enumerate(arc):
        labels = row["choices"]["label"]
        choices = row["choices"]["text"]
        rendered = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
        cases.append(
            {
                "id": f"arc_{index:02d}",
                "category": "arc_challenge",
                "prompt": row["question"] + "\n\n" + rendered + "\n\nChoose one option. End with `Answer: X`.",
                "expected": row["answerKey"].strip().upper(),
                "max_tokens": 384,
            }
        )
    for item in qc.INSTRUCTION_CASES:
        cases.append({**item, "category": "instruction_json", "max_tokens": 256})
    return cases


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080")
    parser.add_argument("--make-fixtures", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / "fixtures.json"
    if args.make_fixtures or not fixture_path.exists():
        fixtures = make_cases()
        fixture_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
    else:
        fixtures = json.loads(fixture_path.read_text())

    results_path = args.output_dir / f"results-{qc.model_filename(args.model)}.jsonl"
    started = time.monotonic()
    with results_path.open("w", encoding="utf-8") as output:
        # Quick profile: short timeout/retry budget, and a failed request aborts the run
        # rather than being recorded — for 72 cheap cases a dead endpoint is best found now.
        records = qc.run_cases(
            args.base_url,
            args.model,
            fixtures,
            output,
            timeout=900,
            retry_delay=4.0,
            tolerate_errors=False,
        )
    summary = qc.summarize(records)
    summary.update({"model": args.model, "elapsed_seconds": time.monotonic() - started})
    (args.output_dir / f"summary-{qc.model_filename(args.model)}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
