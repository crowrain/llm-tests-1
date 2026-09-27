#!/usr/bin/env python3
"""Shared logic for the llm-tests-1 quality harnesses.

Dataset fetching, answer extraction, scoring, the HTTP layer and summary
reporting live here so the two entry points (express and expanded) stay thin
and cannot drift apart. They differ only in which cases they select, their
timeout/retry budget, and whether a failed request aborts the run.
"""

from __future__ import annotations

import json
import re
import statistics
import time
import urllib.parse
import urllib.request
from fractions import Fraction
from typing import Any, TextIO


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


# --- Dataset fetching -------------------------------------------------------


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


# --- Answer extraction ------------------------------------------------------

# Some builds do not split reasoning into `reasoning_content` and instead close it inline
# with a model-specific tag: `</think>`, `</ifm|think>`, `</ifm|think_fast>`.
THINK_CLOSE = re.compile(r"</[^<>]*think[^<>]*>", re.I)
THINK_OPEN = re.compile(r"<[^<>/]*think[^<>]*>", re.I)
FENCE = re.compile(r"```[a-zA-Z0-9_+-]*[ \t]*\n?|\n?```")
# ARC/MMLU answer keys are letters in most rows and digits in some, so accept both. A trailing
# sentence period is common ("Answer: A.") and must not disqualify the match.
ANSWER_CHOICE = re.compile(r"(?:answer|option)\s*(?:is\s*)?[:\-]?\s*\(?([A-E]|[1-5])\)?\.?(?!\w)", re.I)
NEEDLE_KEY = re.compile(r"(?:key|label)\s*:\s*(K\d{8}Z)", re.I)
# Tried in order: the requested `#### n` marker, then an explicitly stated answer, then any
# trailing number. Each fallback is looser, so a marked answer always beats a stray digit.
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


# A JSON answer that carries a number as a string ({"seconds": "8229"}) is still the right
# value, so numeric strings are converted before comparing.
_INT_RE = re.compile(r"[-+]?\d+")
_FLOAT_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+)(?:[eE][-+]?\d+)?")


def normalize_json_values(value: Any) -> Any:
    """Recursively convert numeric strings to numbers so `42` and `"42"` compare equal."""
    if isinstance(value, str):
        text = value.strip()
        if _INT_RE.fullmatch(text):
            return int(text)
        if _FLOAT_RE.fullmatch(text):
            return float(text)
        return value
    if isinstance(value, list):
        return [normalize_json_values(item) for item in value]
    if isinstance(value, dict):
        return {key: normalize_json_values(item) for key, item in value.items()}
    return value


# --- Scoring -----------------------------------------------------------------


def score(case: dict[str, Any], content: str) -> tuple[bool, str | None]:
    """Score one case; return (correct, parsed-answer-or-None).

    Reasoning is stripped for every category: a label or digit the model weighed mid-thought
    must not outrank the answer it settled on.
    """
    content = strip_reasoning(content)
    category = case["category"]
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
        actual_json = parse_json_answer(content)
    except Exception:
        return False, None
    return (
        normalize_json_values(actual_json) == normalize_json_values(case["expected"]),
        json.dumps(actual_json, ensure_ascii=False, sort_keys=True),
    )


# --- HTTP layer ----------------------------------------------------------------


def request(
    base_url: str,
    model: str,
    case: dict[str, Any],
    *,
    timeout: int = 1800,
    retry_delay: float = 5.0,
) -> dict[str, Any]:
    """POST one case to /v1/chat/completions, retrying on any failure.

    ``timeout`` bounds each attempt in seconds; ``retry_delay`` is multiplied by the
    attempt number (1, 2) before sleeping.
    """
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
    error: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            if attempt == 2:
                raise RuntimeError(f"inference request failed: {error}") from exc
            time.sleep(retry_delay * (attempt + 1))
    raise AssertionError("unreachable")


# --- Reporting ------------------------------------------------------------------


def model_filename(model: str) -> str:
    """Keep output names to a single path component: model ids may contain '/' or spaces."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", model)


def median_timing(records: list[dict[str, Any]], key: str) -> float | None:
    values = [r.get("timings", {}).get(key) for r in records]
    values = [value for value in values if isinstance(value, (int, float)) and value > 0]
    return statistics.median(values) if values else None


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "total": len(records),
        "correct": sum(r["correct"] for r in records),
        "by_category": {},
    }
    result["accuracy"] = result["correct"] / result["total"] if result["total"] else 0
    for category in sorted({r["category"] for r in records}):
        rows = [r for r in records if r["category"] == category]
        result["by_category"][category] = {
            "total": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "median_prefill_tps": median_timing(rows, "prompt_per_second"),
            "median_decode_tps": median_timing(rows, "predicted_per_second"),
            "median_elapsed_seconds": statistics.median(r["elapsed_seconds"] for r in rows),
        }
    # Separate "ran out of tokens before answering" from "answered wrongly": the first is a
    # budget setting, the second is the model. A request that failed after retries is neither,
    # so it is counted apart instead of dragging accuracy down silently.
    result["errors"] = sum(1 for r in records if r.get("error"))
    clean = [r for r in records if not r.get("error")]
    result["accuracy_excluding_errors"] = sum(r["correct"] for r in clean) / len(clean) if clean else 0
    result["truncated"] = sum(1 for r in records if r.get("truncated"))
    result["answer_empty"] = sum(1 for r in clean if r.get("answer_empty"))
    result["wrong_and_truncated"] = sum(1 for r in clean if r.get("truncated") and not r["correct"])
    result["wrong_and_complete"] = sum(1 for r in clean if not r.get("truncated") and not r["correct"])
    result["median_prefill_tps"] = median_timing(records, "prompt_per_second")
    result["median_decode_tps"] = median_timing(records, "predicted_per_second")
    return result


# --- Run loop ---------------------------------------------------------------------


def run_cases(
    base_url: str,
    model: str,
    cases: list[dict[str, Any]],
    output: TextIO,
    *,
    timeout: int = 1800,
    retry_delay: float = 5.0,
    tolerate_errors: bool = True,
) -> list[dict[str, Any]]:
    """Send each case to the endpoint, score the answer and stream one JSONL record per
    case to ``output``. Returns all records.

    With ``tolerate_errors`` a request that fails after retries is recorded with ``error``
    and the run continues; otherwise the first such failure aborts the run.
    """
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for index, case in enumerate(cases, 1):
        began = time.monotonic()
        content = ""
        reasoning_content: Any = None
        finish_reason: Any = None
        usage: Any = None
        timings: dict[str, Any] = {}
        correct = False
        parsed: Any = None
        error: str | None = None
        try:
            response = request(base_url, model, case, timeout=timeout, retry_delay=retry_delay)
        except Exception as exc:
            if not tolerate_errors:
                raise
            error = repr(exc)
            response = {}
        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning_content = message.get("reasoning_content")
        finish_reason = choice.get("finish_reason")
        usage = response.get("usage")
        timings = response.get("timings", {})
        if error is None:
            # A request failure must not be scored: there is no answer to misread.
            correct, parsed = score(case, content)
        # A reasoning model can burn the whole budget before it states an answer; that is a
        # different failure from answering wrongly, so record it apart.
        record = {
            "id": case["id"],
            "category": case["category"],
            "subject": case.get("subject"),
            "expected": case["expected"],
            "parsed": parsed,
            "correct": correct,
            "content": content,
            "reasoning_content": reasoning_content,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "answer_empty": not strip_reasoning(content).strip(),
            "usage": usage,
            "timings": timings,
            "elapsed_seconds": time.monotonic() - began,
            "error": error,
            "approx_chars": case.get("approx_chars"),
        }
        output.write(json.dumps(record, ensure_ascii=False) + "\n")
        output.flush()
        records.append(record)
        print(
            f"{model} {index}/{len(cases)} {case['category']} "
            f"{'OK' if correct else 'FAIL'} elapsed={time.monotonic() - started:.0f}s",
            flush=True,
        )
    return records
