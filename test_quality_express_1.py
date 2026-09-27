#!/usr/bin/env python3
"""Bounded, reproducible A/B quality test: the express profile (72 fast cases).

Thin entry point: dataset fetching, answer extraction, scoring, the HTTP layer and
summary reporting live in quality_common.py. The express profile spreads fewer,
cheaper samples across each test split, uses a short timeout/retry budget and
aborts the run on a failed request instead of recording it.
"""

from __future__ import annotations

from typing import Any

import quality_common as qc


def make_cases() -> list[dict[str, Any]]:
    # Spread fixed samples across each test split instead of taking one contiguous shard.
    gsm = qc.get_rows("openai/gsm8k", "main", "test", [i * 43 for i in range(30)])
    arc = qc.get_rows("allenai/ai2_arc", "ARC-Challenge", "test", [i * 37 for i in range(30)])
    cases: list[dict[str, Any]] = []
    for index, row in enumerate(gsm):
        cases.append(
            {
                "id": f"gsm8k_{index:02d}",
                "category": "gsm8k",
                "prompt": qc.gsm8k_prompt(row["question"]),
                "expected": qc.gsm8k_expected(row["answer"]),
                "max_tokens": 768,
            }
        )
    for index, row in enumerate(arc):
        cases.append(
            {
                "id": f"arc_{index:02d}",
                "category": "arc_challenge",
                "prompt": qc.choice_prompt(row["question"], row["choices"]["label"], row["choices"]["text"]),
                "expected": row["answerKey"].strip().upper(),
                "max_tokens": 384,
            }
        )
    for item in qc.INSTRUCTION_CASES:
        cases.append({**item, "category": "instruction_json", "max_tokens": 256})
    return cases


if __name__ == "__main__":
    # Quick profile: short timeout/retry budget, and a failed request aborts the run
    # rather than being recorded — for 72 cheap cases a dead endpoint is best found now.
    qc.main(
        make_cases,
        timeout=900,
        retry_delay=4.0,
        tolerate_errors=False,
        profile="express",
        description="Express profile: 72 fast cases; a failed request aborts the run.",
    )
