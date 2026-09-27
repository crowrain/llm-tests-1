#!/usr/bin/env python3
"""Reproducible, bounded A/B test: the expanded profile (266 cases).

Thin entry point: dataset fetching, answer extraction, scoring, the HTTP layer and
summary reporting live in quality_common.py. The expanded profile adds knowledge
(MMLU) and long-context needle cases, uses a long timeout/retry budget and records
a failed request instead of aborting the run. It deliberately measures the deployed
model *and* serving profile.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

import quality_common as qc


MMLU_SUBJECTS = [
    "college_computer_science",
    "high_school_mathematics",
    "logical_fallacies",
    "professional_medicine",
    "high_school_world_history",
]

# Bumping the salt reissues every archive; old fixtures.json files keep their own labels.
NEEDLE_SALT = "ninfer-needle-v2"


def needle_key(char_budget: int, index: int, bump: int = 0) -> str:
    """Opaque, reproducible label for one archive record.

    Hashed rather than an arithmetic sequence: a linear generator lets the model read two
    neighbouring records, infer the step and *compute* the answer instead of retrieving it,
    which is the opposite of what a needle test measures.
    """
    seed = f"{NEEDLE_SALT}:{char_budget}:{index}:{bump}".encode()
    return f"K{int.from_bytes(hashlib.sha256(seed).digest()[:5], 'big') % 100_000_000:08d}Z"


def make_long_case(label: str, char_budget: int, fraction: float) -> dict[str, Any]:
    prefix = "ARCHIVE START. Each record is independent; labels are opaque and must be copied exactly.\n"
    filler = " provenance cedar quartz lantern delta meadow orbit ember cobalt harbor."
    lines: list[str] = []
    seen: set[str] = set()
    length = len(prefix)
    while length < char_budget:
        index = len(lines)
        bump = 0
        key = needle_key(char_budget, index)
        while key in seen:  # exactly one record may answer the question
            bump += 1
            key = needle_key(char_budget, index, bump)
        seen.add(key)
        line = f"Record {index:06d}: storage label {key};" + filler * 3 + "\n"
        lines.append(line)
        length += len(line)
    target = min(len(lines) - 1, max(0, int(len(lines) * fraction)))
    expected = re.search(r"label (K\d{8}Z)", lines[target]).group(1)
    prompt = (
        prefix + "".join(lines) + "ARCHIVE END.\n\n"
        f"What is the storage label of Record {target:06d}? Search the archive. "
        "End with exactly `Key: <label>` and no alternative label."
    )
    # 256 truncated the deepest archive mid-answer; the 180k tiebreak needed 282 tokens.
    return {
        "id": f"needle_{label}",
        "category": f"long_context_{label}",
        "prompt": prompt,
        "expected": expected,
        "max_tokens": 768,
        "approx_chars": len(prompt),
        "target_record": target,
    }


def make_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    gsm = qc.get_rows("openai/gsm8k", "main", "test", [i * 13 for i in range(100)])
    for index, row in enumerate(gsm):
        cases.append(
            {
                "id": f"gsm8k_{index:03d}",
                "category": "gsm8k",
                "prompt": qc.gsm8k_prompt(row["question"]),
                "expected": qc.gsm8k_expected(row["answer"]),
                "max_tokens": 1024,
            }
        )
    arc = qc.get_rows("allenai/ai2_arc", "ARC-Challenge", "test", [i * 11 for i in range(100)])
    for index, row in enumerate(arc):
        cases.append(
            {
                "id": f"arc_{index:03d}",
                "category": "arc_challenge",
                "prompt": qc.choice_prompt(row["question"], row["choices"]["label"], row["choices"]["text"]),
                "expected": row["answerKey"].strip().upper(),
                "max_tokens": 768,
            }
        )
    for subject in MMLU_SUBJECTS:
        rows = qc.get_rows("cais/mmlu", subject, "test", [i * 7 for i in range(10)])
        for index, row in enumerate(rows):
            # MMLU ships choices as a bare list and the answer as its index.
            choices = row["choices"]
            cases.append(
                {
                    "id": f"mmlu_{subject}_{index:02d}",
                    "category": "mmlu",
                    "subject": subject,
                    "prompt": qc.choice_prompt(row["question"], qc.letter_labels(len(choices)), choices),
                    "expected": chr(65 + int(row["answer"])),
                    "max_tokens": 768,
                }
            )
    cases.extend({**item, "category": "instruction_json", "max_tokens": 256} for item in qc.INSTRUCTION_CASES)
    cases.extend(
        [
            make_long_case("16k", 64_000, 0.19),
            make_long_case("64k", 256_000, 0.51),
            make_long_case("128k", 512_000, 0.73),
            make_long_case("180k", 720_000, 0.87),
        ]
    )
    return cases


if __name__ == "__main__":
    # Long profile: generous timeout/retry budget for 180k-context cases, and a failed
    # request is recorded instead of aborting the run.
    qc.main(
        make_cases,
        timeout=1800,
        retry_delay=5.0,
        tolerate_errors=True,
        profile="expanded",
        description=(
            "Expanded profile: 266 cases including MMLU and long-context needles; "
            "a failed request is recorded."
        ),
    )
