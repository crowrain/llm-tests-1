#!/usr/bin/env python3
"""Reproducible, bounded A/B test for the production NInfer and llama.cpp profiles.

It uses fixed public test-set samples plus deterministic needle-in-a-haystack
prompts.  It deliberately measures the deployed model *and* serving profile.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import time
import urllib.parse
import urllib.request
from fractions import Fraction
from pathlib import Path
from typing import Any


API_BASE = "https://datasets-server.huggingface.co/rows"
SYSTEM = (
    "Follow the user instruction exactly. Think carefully, then give a concise final answer. "
    "Do not mention this evaluation."
)
MMLU_SUBJECTS = [
    "college_computer_science",
    "high_school_mathematics",
    "logical_fallacies",
    "professional_medicine",
    "high_school_world_history",
]

INSTRUCTION_CASES: list[dict[str, Any]] = [
    {"id": "if_01", "prompt": "Return only valid JSON, with no Markdown: an object whose keys are `largest`, `smallest`, and `sum`, for the integers [17, -4, 29, 0].", "expected": {"largest": 29, "smallest": -4, "sum": 42}},
    {"id": "if_02", "prompt": "Return only valid JSON, with no Markdown: sort these strings by length ascending and then alphabetically: [\"pear\", \"fig\", \"apple\", \"plum\"]. Use key `ordered`.", "expected": {"ordered": ["fig", "pear", "plum", "apple"]}},
    {"id": "if_03", "prompt": "Return only valid JSON, with no Markdown: from [3, 3, 7, 2, 7, 7, 5], return the distinct values in first-appearance order under key `values` and their count under key `count`.", "expected": {"values": [3, 7, 2, 5], "count": 4}},
    {"id": "if_04", "prompt": "Return only valid JSON, with no Markdown: convert 2 hours, 17 minutes, and 9 seconds to seconds. Use exactly one numeric key named `seconds`.", "expected": {"seconds": 8229}},
    {"id": "if_05", "prompt": "Return only valid JSON, with no Markdown: extract the unique uppercase letters from `aBcdEFBaG`, sorted alphabetically, under key `letters`.", "expected": {"letters": ["B", "E", "F", "G"]}},
    {"id": "if_06", "prompt": "Return only valid JSON, with no Markdown: a rectangle has width 13 and height 8. Return its area and perimeter with exactly the keys `area` and `perimeter`.", "expected": {"area": 104, "perimeter": 42}},
    {"id": "if_07", "prompt": "Return only valid JSON, with no Markdown: for the word `mississippi`, count each character. Use an object under key `counts` with keys m, i, s, p.", "expected": {"counts": {"m": 1, "i": 4, "s": 4, "p": 2}}},
    {"id": "if_08", "prompt": "Return only valid JSON, with no Markdown: calculate the median and arithmetic mean of [2, 5, 9, 12]. Use exactly keys `median` and `mean`.", "expected": {"median": 7, "mean": 7}},
    {"id": "if_09", "prompt": "Return only valid JSON, with no Markdown: a price of 250 is discounted by 12% and then taxed by 10%. Return the final price under key `final_price`.", "expected": {"final_price": 242}},
    {"id": "if_10", "prompt": "Return only valid JSON, with no Markdown: reverse the words in `red green blue yellow` and return them as an array under key `words`.", "expected": {"words": ["yellow", "blue", "green", "red"]}},
    {"id": "if_11", "prompt": "Return only valid JSON, with no Markdown: list the prime numbers from 10 through 25 inclusive under key `primes`.", "expected": {"primes": [11, 13, 17, 19, 23]}},
    {"id": "if_12", "prompt": "Return only valid JSON, with no Markdown: convert the Roman numeral XLIV to an integer under key `value`.", "expected": {"value": 44}},
]


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode({"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length})
    error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API_BASE}?{query}", timeout=45) as response:
                return [item["row"] for item in json.load(response)["rows"]]
        except Exception as exc:
            error = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"dataset request failed after retries: {error}")


def get_rows(dataset: str, config: str, split: str, offsets: list[int]) -> list[dict[str, Any]]:
    chunks: dict[int, list[dict[str, Any]]] = {}
    for start in sorted({offset // 100 * 100 for offset in offsets}):
        chunks[start] = fetch_rows(dataset, config, split, start, 100)
    return [chunks[offset // 100 * 100][offset % 100] for offset in offsets]


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
    return {"id": f"needle_{label}", "category": f"long_context_{label}", "prompt": prompt, "expected": expected, "max_tokens": 768, "approx_chars": len(prompt), "target_record": target}


def make_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    gsm = get_rows("openai/gsm8k", "main", "test", [i * 13 for i in range(100)])
    for index, row in enumerate(gsm):
        cases.append({"id": f"gsm8k_{index:03d}", "category": "gsm8k", "prompt": row["question"] + "\n\nEnd the final answer with `#### <number>`.", "expected": row["answer"].rsplit("####", 1)[-1].strip(), "max_tokens": 1024})
    arc = get_rows("allenai/ai2_arc", "ARC-Challenge", "test", [i * 11 for i in range(100)])
    for index, row in enumerate(arc):
        choices = "\n".join(f"{label}. {text}" for label, text in zip(row["choices"]["label"], row["choices"]["text"]))
        cases.append({"id": f"arc_{index:03d}", "category": "arc_challenge", "prompt": row["question"] + "\n\n" + choices + "\n\nChoose one option. End with `Answer: X`.", "expected": row["answerKey"].strip().upper(), "max_tokens": 768})
    for subject in MMLU_SUBJECTS:
        rows = get_rows("cais/mmlu", subject, "test", [i * 7 for i in range(10)])
        for index, row in enumerate(rows):
            choices = "\n".join(f"{chr(65 + n)}. {choice}" for n, choice in enumerate(row["choices"]))
            cases.append({"id": f"mmlu_{subject}_{index:02d}", "category": "mmlu", "subject": subject, "prompt": row["question"] + "\n\n" + choices + "\n\nChoose one option. End with `Answer: X`.", "expected": chr(65 + int(row["answer"])), "max_tokens": 768})
    cases.extend({**item, "category": "instruction_json", "max_tokens": 256} for item in INSTRUCTION_CASES)
    cases.extend([make_long_case("16k", 64_000, 0.19), make_long_case("64k", 256_000, 0.51), make_long_case("128k", 512_000, 0.73), make_long_case("180k", 720_000, 0.87)])
    return cases


# Some builds do not split reasoning into `reasoning_content` and instead close it inline
# with a model-specific tag: `</think>`, `</ifm|think>`, `</ifm|think_fast>`.
THINK_CLOSE = re.compile(r"</[^<>]*think[^<>]*>", re.I)
THINK_OPEN = re.compile(r"<[^<>/]*think[^<>]*>", re.I)
FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n?|\n?```")
# ARC answer keys are letters in most rows and digits in some, so accept both.
ANSWER_CHOICE = re.compile(r"(?:answer|option)\s*(?:is\s*)?[:\-]?\s*\(?([A-E]|[1-5])\)?(?![\w.])", re.I)
NEEDLE_KEY = re.compile(r"(?:key|label)\s*:\s*(K\d{8}Z)", re.I)
# Tried in order: the requested `#### n` marker, an explicitly stated answer, then any
# trailing number. Each fallback is looser, so a marked answer always wins over stray digits.
GSM_PATTERNS = (
    re.compile(r"####\s*([-+]?[$\d,.]+(?:/\d+)?)"),
    re.compile(r"(?:answer|total)\s*(?:is|:)\s*\$?\s*([-+]?[\d,.]+(?:/\d+)?)", re.I),
    re.compile(r"(?<![\w.])[-+]?[$\d,.]+(?:/\d+)?(?![\w.])"),
)


def strip_reasoning(text: str) -> str:
    """Drop chain-of-thought the model emitted into `content` and keep the answer after it."""
    closes = list(THINK_CLOSE.finditer(text))
    if closes:
        return text[closes[-1].end() :]
    # Thinking opened and never closed: the answer was never reached.
    return "" if THINK_OPEN.search(text) else text


def iter_balanced_objects(text: str) -> list[str]:
    """Return every brace-balanced `{...}` span, ignoring braces inside string literals."""
    spans: list[str] = []
    depth, start, in_string, escaped = 0, -1, False, False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0:
                spans.append(text[start : index + 1])
    return spans


def parse_json_answer(text: str) -> Any:
    """Return the last brace-balanced JSON object in `text`.

    Scanning for balance and taking the newest candidate keeps a JSON snippet quoted inside
    the reasoning ("Provide {...}") from being glued onto the real answer, which a plain
    first-`{`-to-last-`}` slice would do.
    """
    text = FENCE.sub("", strip_reasoning(text)).strip()
    for candidate in reversed(iter_balanced_objects(text)):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return json.loads(text)


def normalize_number(value: str) -> str:
    return value.replace(",", "").replace("$", "").strip().rstrip(".")


def numbers_equal(actual: str, expected: str) -> bool:
    """Compare numerically so `6.00` matches `6` and `1/2` matches `0.5`."""
    if actual == expected:
        return True
    try:
        return Fraction(actual) == Fraction(expected)
    except (ValueError, ZeroDivisionError):
        return False


def score(case: dict[str, Any], content: str) -> tuple[bool, str | None]:
    category = case["category"]
    # Reasoning is stripped for every category: a label or digit the model weighed mid-thought
    # must not outrank the answer it settled on.
    content = strip_reasoning(content)
    if category == "gsm8k":
        actual = ""
        for pattern in GSM_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                actual = normalize_number(matches[-1])
                break
        return numbers_equal(actual, normalize_number(str(case["expected"]))), actual or None
    if category in {"arc_challenge", "mmlu"}:
        matches = ANSWER_CHOICE.findall(content)
        actual = matches[-1].upper() if matches else ""
        return actual == str(case["expected"]).strip().upper(), actual or None
    if category.startswith("long_context_"):
        matches = NEEDLE_KEY.findall(content)
        actual = matches[-1].upper() if matches else ""
        return actual == case["expected"], actual or None
    try:
        actual = parse_json_answer(content)
    except Exception:
        return False, None
    return actual == case["expected"], json.dumps(actual, ensure_ascii=False, sort_keys=True)


def request(base_url: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    payload = {"model": model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case["prompt"]}], "temperature": 0, "top_p": 1, "seed": 20260926, "reasoning_effort": "medium", "max_completion_tokens": case["max_tokens"]}
    req = urllib.request.Request(f"{base_url.rstrip('/')}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=1800) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"inference request failed: {error}")


def median_timing(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r.get("timings", {}).get(key) for r in records]
    values = [value for value in values if isinstance(value, (int, float)) and value > 0]
    return statistics.median(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"total": len(records), "correct": sum(r["correct"] for r in records), "by_category": {}}
    result["accuracy"] = result["correct"] / result["total"] if result["total"] else 0
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        result["by_category"][category] = {"total": len(rows), "correct": sum(r["correct"] for r in rows), "accuracy": sum(r["correct"] for r in rows) / len(rows), "truncated": sum(1 for r in rows if r.get("truncated")), "median_prefill_tps": median_timing(rows, "prompt_per_second"), "median_decode_tps": median_timing(rows, "predicted_per_second"), "median_elapsed_seconds": statistics.median(r["elapsed_seconds"] for r in rows)}
    # Separate "ran out of tokens before answering" from "answered wrongly": the first is a
    # budget setting, the second is the model.
    result["truncated"] = sum(1 for r in records if r.get("truncated"))
    result["answer_empty"] = sum(1 for r in records if r.get("answer_empty"))
    result["wrong_and_truncated"] = sum(1 for r in records if r.get("truncated") and not r["correct"])
    result["wrong_and_complete"] = sum(1 for r in records if not r.get("truncated") and not r["correct"])
    result["median_prefill_tps"] = median_timing(records, "prompt_per_second")
    result["median_decode_tps"] = median_timing(records, "predicted_per_second")
    return result


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
        fixture_path.write_text(json.dumps(fixtures, ensure_ascii=False) + "\n")
    else:
        fixtures = json.loads(fixture_path.read_text())
    results_path = args.output_dir / f"results-{args.model}.jsonl"
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with results_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(fixtures, 1):
            began = time.monotonic()
            try:
                response = request(args.base_url, args.model, case)
                choice = response.get("choices", [{}])[0]
                message = choice.get("message", {})
                content = message.get("content") or ""
                correct, parsed = score(case, content)
                error = None
            except Exception as exc:
                response, choice, message, content, correct, parsed, error = {}, {}, {}, "", False, None, repr(exc)
            # A reasoning model can burn the whole budget before it states an answer; that is a
            # different failure from answering wrongly, so record it apart.
            finish_reason = choice.get("finish_reason")
            record = {"id": case["id"], "category": case["category"], "subject": case.get("subject"), "expected": case["expected"], "parsed": parsed, "correct": correct, "content": content, "reasoning_content": message.get("reasoning_content"), "finish_reason": finish_reason, "truncated": finish_reason == "length", "answer_empty": not strip_reasoning(content).strip(), "usage": response.get("usage"), "timings": response.get("timings", {}), "elapsed_seconds": time.monotonic() - began, "error": error, "approx_chars": case.get("approx_chars")}
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            print(f"{args.model} {index}/{len(fixtures)} {case['category']} {'OK' if correct else 'FAIL'} elapsed={time.monotonic() - started:.0f}s", flush=True)
    summary = summarize(records)
    summary.update({"model": args.model, "elapsed_seconds": time.monotonic() - started})
    (args.output_dir / f"summary-{args.model}.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
