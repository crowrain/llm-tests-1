#!/usr/bin/env python3
"""Bounded, reproducible A/B quality test for two OpenAI-compatible models."""

from __future__ import annotations

import argparse
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

INSTRUCTION_CASES: list[dict[str, Any]] = [
    {
        "id": "if_01",
        "prompt": "Return only valid JSON, with no Markdown: an object whose keys are `largest`, `smallest`, and `sum`, for the integers [17, -4, 29, 0].",
        "expected": {"largest": 29, "smallest": -4, "sum": 42},
    },
    {
        "id": "if_02",
        "prompt": "Return only valid JSON, with no Markdown: sort these strings by length ascending and then alphabetically: [\"pear\", \"fig\", \"apple\", \"plum\"]. Use key `ordered`.",
        "expected": {"ordered": ["fig", "pear", "plum", "apple"]},
    },
    {
        "id": "if_03",
        "prompt": "Return only valid JSON, with no Markdown: from [3, 3, 7, 2, 7, 7, 5], return the distinct values in first-appearance order under key `values` and their count under key `count`.",
        "expected": {"values": [3, 7, 2, 5], "count": 4},
    },
    {
        "id": "if_04",
        "prompt": "Return only valid JSON, with no Markdown: convert 2 hours, 17 minutes, and 9 seconds to seconds. Use exactly one numeric key named `seconds`.",
        "expected": {"seconds": 8229},
    },
    {
        "id": "if_05",
        "prompt": "Return only valid JSON, with no Markdown: extract the unique uppercase letters from `aBcdEFBaG`, sorted alphabetically, under key `letters`.",
        "expected": {"letters": ["B", "E", "F", "G"]},
    },
    {
        "id": "if_06",
        "prompt": "Return only valid JSON, with no Markdown: a rectangle has width 13 and height 8. Return its area and perimeter with exactly the keys `area` and `perimeter`.",
        "expected": {"area": 104, "perimeter": 42},
    },
    {
        "id": "if_07",
        "prompt": "Return only valid JSON, with no Markdown: for the word `mississippi`, count each character. Use an object under key `counts` with keys m, i, s, p.",
        "expected": {"counts": {"m": 1, "i": 4, "s": 4, "p": 2}},
    },
    {
        "id": "if_08",
        "prompt": "Return only valid JSON, with no Markdown: calculate the median and arithmetic mean of [2, 5, 9, 12]. Use exactly keys `median` and `mean`.",
        "expected": {"median": 7, "mean": 7},
    },
    {
        "id": "if_09",
        "prompt": "Return only valid JSON, with no Markdown: a price of 250 is discounted by 12% and then taxed by 10%. Return the final price under key `final_price`.",
        "expected": {"final_price": 242},
    },
    {
        "id": "if_10",
        "prompt": "Return only valid JSON, with no Markdown: reverse the words in `red green blue yellow` and return them as an array under key `words`.",
        "expected": {"words": ["yellow", "blue", "green", "red"]},
    },
    {
        "id": "if_11",
        "prompt": "Return only valid JSON, with no Markdown: list the prime numbers from 10 through 25 inclusive under key `primes`.",
        "expected": {"primes": [11, 13, 17, 19, 23]},
    },
    {
        "id": "if_12",
        "prompt": "Return only valid JSON, with no Markdown: convert the Roman numeral XLIV to an integer under key `value`.",
        "expected": {"value": 44},
    },
]


def fetch_rows(dataset: str, config: str, split: str, offset: int, length: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {"dataset": dataset, "config": config, "split": split, "offset": offset, "length": length}
    )
    last_error: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{API_BASE}?{query}", timeout=45) as response:
                return [item["row"] for item in json.load(response)["rows"]]
        except Exception as error:
            last_error = error
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"dataset request failed after retries: {last_error}")


def get_rows(dataset: str, config: str, split: str, offsets: list[int]) -> list[dict[str, Any]]:
    chunks: dict[int, list[dict[str, Any]]] = {}
    for start in sorted({offset // 100 * 100 for offset in offsets}):
        chunks[start] = fetch_rows(dataset, config, split, start, 100)
    return [chunks[offset // 100 * 100][offset % 100] for offset in offsets]


def make_cases() -> list[dict[str, Any]]:
    # Spread fixed samples across each test split instead of taking one contiguous shard.
    gsm = get_rows("openai/gsm8k", "main", "test", [i * 43 for i in range(30)])
    arc = get_rows("allenai/ai2_arc", "ARC-Challenge", "test", [i * 37 for i in range(30)])
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
    for item in INSTRUCTION_CASES:
        cases.append({**item, "category": "instruction_json", "max_tokens": 256})
    return cases


# Some builds do not split reasoning into `reasoning_content` and instead close it with a
# model-specific tag: `</think>`, `</ifm|think>`, `</ifm|think_fast>`.
THINK_CLOSE = re.compile(r"</[^<>]*think[^<>]*>", re.I)
THINK_OPEN = re.compile(r"<[^<>/]*think[^<>]*>", re.I)
FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n?|\n?```")


def strip_reasoning(text: str) -> str:
    """Drop chain-of-thought the model emitted into `content` and keep the answer after it."""
    closes = list(THINK_CLOSE.finditer(text))
    if closes:
        return text[closes[-1].end() :]
    if THINK_OPEN.search(text):
        # Thinking opened and never closed: the answer was never reached.
        return ""
    return text


def iter_balanced_objects(text: str) -> list[str]:
    """Return every brace-balanced `{...}` span, ignoring braces inside string literals."""
    spans: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escaped = False
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


# ARC answer keys are letters in most rows and digits in some, so accept both.
ARC_ANSWER = re.compile(r"(?:answer|option)\s*(?:is\s*)?[:\-]?\s*\(?([A-E]|[1-5])\)?(?![\w.])", re.I)
# Tried in order: the requested `#### n` marker, then an explicitly stated answer, then any
# trailing number. Each fallback is looser, so a marked answer always wins over stray digits.
GSM_PATTERNS = (
    re.compile(r"####\s*([-+]?[$\d,.]+(?:/\d+)?)"),
    re.compile(r"(?:answer|total)\s*(?:is|:)\s*\$?\s*([-+]?[\d,.]+(?:/\d+)?)", re.I),
    re.compile(r"(?<![\w.])[-+]?[$\d,.]+(?:/\d+)?(?![\w.])"),
)


def score(case: dict[str, Any], content: str) -> tuple[bool, str | None]:
    content = strip_reasoning(content)
    if case["category"] == "gsm8k":
        actual = ""
        for pattern in GSM_PATTERNS:
            matches = pattern.findall(content)
            if matches:
                actual = normalize_number(matches[-1])
                break
        return numbers_equal(actual, normalize_number(str(case["expected"]))), actual or None
    if case["category"] == "arc_challenge":
        matches = ARC_ANSWER.findall(content)
        actual = matches[-1].upper() if matches else ""
        return actual == str(case["expected"]).strip().upper(), actual or None
    try:
        actual_json = parse_json_answer(content)
    except Exception:
        return False, None
    return actual_json == case["expected"], json.dumps(actual_json, ensure_ascii=False, sort_keys=True)


def request(base_url: str, model: str, case: dict[str, Any]) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": case["prompt"]},
        ],
        "temperature": 0,
        "top_p": 1,
        "seed": 20260926,
        "reasoning_effort": "medium",
        "max_completion_tokens": case["max_tokens"],
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=900) as response:
                return json.load(response)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(4)
    raise AssertionError("unreachable")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    categories = sorted({record["category"] for record in records})
    result: dict[str, Any] = {"total": len(records), "correct": sum(r["correct"] for r in records)}
    result["accuracy"] = result["correct"] / result["total"] if result["total"] else 0
    result["by_category"] = {}
    for category in categories:
        rows = [r for r in records if r["category"] == category]
        result["by_category"][category] = {
            "total": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
        }
    # Separate "ran out of tokens before answering" from "answered wrongly": the first is a
    # budget setting, the second is the model.
    result["truncated"] = sum(1 for r in records if r.get("truncated"))
    result["answer_empty"] = sum(1 for r in records if r.get("answer_empty"))
    result["wrong_and_truncated"] = sum(1 for r in records if r.get("truncated") and not r["correct"])
    result["wrong_and_complete"] = sum(1 for r in records if not r.get("truncated") and not r["correct"])
    decode = [r["timings"].get("predicted_per_second") for r in records if r.get("timings", {}).get("predicted_per_second")]
    if decode:
        result["median_decode_tps"] = statistics.median(decode)
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
        fixture_path.write_text(json.dumps(fixtures, ensure_ascii=False, indent=2) + "\n")
    else:
        fixtures = json.loads(fixture_path.read_text())

    results_path = args.output_dir / f"results-{args.model}.jsonl"
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    with results_path.open("w", encoding="utf-8") as output:
        for index, case in enumerate(fixtures, 1):
            response = request(args.base_url, args.model, case)
            choice = response.get("choices", [{}])[0]
            message = choice.get("message", {})
            content = message.get("content") or ""
            correct, parsed = score(case, content)
            finish_reason = choice.get("finish_reason")
            record = {
                "id": case["id"],
                "category": case["category"],
                "expected": case["expected"],
                "parsed": parsed,
                "correct": correct,
                "content": content,
                "reasoning_content": message.get("reasoning_content"),
                "finish_reason": finish_reason,
                # A reasoning model can burn the whole budget before it states an answer;
                # that is a different failure from answering wrongly, so record it apart.
                "truncated": finish_reason == "length",
                "answer_empty": not strip_reasoning(content).strip(),
                "usage": response.get("usage"),
                "timings": response.get("timings", {}),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            records.append(record)
            print(
                f"{args.model} {index}/{len(fixtures)} {case['category']} "
                f"{'OK' if correct else 'FAIL'} elapsed={time.monotonic() - started:.0f}s",
                flush=True,
            )
    summary = summarize(records)
    summary.update({"model": args.model, "elapsed_seconds": time.monotonic() - started})
    (args.output_dir / f"summary-{args.model}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
