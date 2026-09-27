#!/usr/bin/env python3
"""Shared logic for the llm-tests-1 quality harnesses.

Dataset fetching, answer extraction, scoring, the HTTP layer and summary
reporting live here so the two entry points (express and expanded) stay thin
and cannot drift apart. They differ only in which cases they select, their
timeout/retry budget, and whether a failed request aborts the run.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import json
import re
import statistics
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, TextIO


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
# Markdown emphasis and backticks routinely wrap the value a model was told to end with
# ("Answer: **A**", "Key: `K...`"). They carry no meaning here, so every marker-based pattern
# tolerates them on either side of the value; without that the answer reads as absent.
EMPHASIS = r"""[\s*_`"']*"""
# ARC/MMLU answer keys are letters in most rows and digits in some, so accept both. A trailing
# sentence period is common ("Answer: A.") and must not disqualify the match.
ANSWER_CHOICE = re.compile(
    rf"(?:answer|option)\s*(?:is\s*)?[:\-]?\s*{EMPHASIS}\(?([A-E]|[1-5])\)?{EMPHASIS}\.?(?!\w)", re.I
)
# The colon is required on purpose: archive records read "storage label K...;" with no colon,
# so it is what separates the model's own answer line from a record it merely echoed back.
NEEDLE_KEY = re.compile(rf"(?:key|label)\s*:\s*{EMPHASIS}(K\d{{8}}Z)", re.I)
# Tried in order: the requested `#### n` marker, then an explicitly stated answer, then any
# trailing number. Each fallback is looser, so a marked answer always beats a stray digit.
GSM_PATTERNS = (
    re.compile(r"####\s*([-+]?[$\d,.]+(?:/\d+)?)"),
    re.compile(r"(?:answer|total)\s*(?:is|:)\s*\$?\s*([-+]?[\d,.]+(?:/\d+)?)", re.I),
    # The lookahead forces at least one digit into the match. Without it the character class
    # also matches a punctuation-only token — the "." closing "**42**." — which, being the
    # last match, won and discarded the real number, scoring a right answer as no answer.
    re.compile(r"(?<![\w.])[-+]?(?=[$,.]*\d)[$\d,.]+(?:/\d+)?(?![\w.])"),
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

# Keep-alive connections are reused across requests so a 266-case run pays the TCP
# handshake once per endpoint instead of 266 times. They live in thread-local storage:
# http.client is not thread-safe, so --concurrency workers each own their own pool.
_LOCAL = threading.local()


def _connections() -> dict[tuple, "http.client.HTTPConnection"]:
    if not hasattr(_LOCAL, "conns"):
        _LOCAL.conns = {}
    return _LOCAL.conns


def _post_json(url: str, data: bytes, *, timeout: float) -> bytes:
    """POST ``data`` to ``url`` and return the response body, reusing the keep-alive
    connection for (scheme, host, port). A failed exchange drops the connection so the
    caller's next attempt reconnects. HTTP errors (status >= 400) are raised, not
    returned: the body is not a completion."""
    parts = urllib.parse.urlsplit(url)
    key = (parts.scheme, parts.hostname, parts.port)
    path = parts.path or "/"
    if parts.query:
        path = f"{path}?{parts.query}"
    conn = _connections().get(key)
    if conn is None:
        if parts.scheme == "https":
            conn = http.client.HTTPSConnection(parts.hostname, parts.port, timeout=timeout)
        else:
            conn = http.client.HTTPConnection(parts.hostname, parts.port, timeout=timeout)
        _connections()[key] = conn
    # The socket timeout is fixed at connect time, so apply the per-request bound to the
    # live socket (reuse) or to the connection's default (first connect).
    if conn.sock is None:
        conn.timeout = timeout
    else:
        conn.sock.settimeout(timeout)
    try:
        conn.request("POST", path, body=data, headers={"Content-Type": "application/json"})
        response = conn.getresponse()
        body = response.read()
    except Exception:
        _connections().pop(key, None)
        raise
    if response.status >= 400:
        _connections().pop(key, None)
        raise RuntimeError(f"HTTP {response.status} from {url}: {body[:500]!r}")
    return body


def request(
    base_url: str,
    model: str,
    case: dict[str, Any],
    *,
    timeout: int = 1800,
    retry_delay: float = 5.0,
    send_seed: bool = True,
    send_reasoning_effort: bool = True,
) -> dict[str, Any]:
    """POST one case to /v1/chat/completions, retrying on any failure.

    ``timeout`` bounds each attempt in seconds; ``retry_delay`` is multiplied by the
    attempt number (1, 2) before sleeping. ``send_seed`` / ``send_reasoning_effort``
    gate the optional sampler fields for servers that reject unknown payload keys
    (some llama.cpp / vLLM builds answer 400 to them).
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": case["prompt"]},
        ],
        "temperature": 0,
        "top_p": 1,
        "max_completion_tokens": case["max_tokens"],
    }
    if send_seed:
        payload["seed"] = 20260926
    if send_reasoning_effort:
        payload["reasoning_effort"] = "medium"
    data = json.dumps(payload).encode()
    error: Exception | None = None
    for attempt in range(3):
        try:
            return json.loads(_post_json(f"{base_url.rstrip('/')}/v1/chat/completions", data, timeout=timeout))
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
    # A get() default only covers a *missing* key: a server (or an older results file being
    # resumed) may carry "timings" as an explicit null, which the default would let through.
    values = [r["timings"].get(key) for r in records if isinstance(r.get("timings"), dict)]
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
        # Records from runs older than this field may lack it (e.g. a --resume of an old
        # results file); a missing median is not an error.
        elapsed = [r["elapsed_seconds"] for r in rows if isinstance(r.get("elapsed_seconds"), (int, float))]
        result["by_category"][category] = {
            "total": len(rows),
            "correct": sum(r["correct"] for r in rows),
            "accuracy": sum(r["correct"] for r in rows) / len(rows),
            "truncated": sum(1 for r in rows if r.get("truncated")),
            "median_prefill_tps": median_timing(rows, "prompt_per_second"),
            "median_decode_tps": median_timing(rows, "predicted_per_second"),
            "median_elapsed_seconds": statistics.median(elapsed) if elapsed else None,
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


def _process_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    *,
    timeout: int,
    retry_delay: float,
    tolerate_errors: bool,
    send_seed: bool,
    send_reasoning_effort: bool,
) -> dict[str, Any]:
    """Request one case and score the answer. Pure worker: no file I/O, safe to run in
    threads. Returns the record for the case."""
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
        response = request(
            base_url,
            model,
            case,
            timeout=timeout,
            retry_delay=retry_delay,
            send_seed=send_seed,
            send_reasoning_effort=send_reasoning_effort,
        )
        # A response carrying no usable choice holds no answer, so it is a failed request
        # rather than an empty one: raised here, inside the try, so the mode below decides
        # (record it or abort). Reaching for [0] after the try crashed even a tolerant run.
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise RuntimeError(f"response carried no usable choice: {json.dumps(response)[:300]}")
    except Exception as exc:
        if not tolerate_errors:
            raise
        error = repr(exc)
        response = {}
    # Either the try validated choices[0] as a dict, or it left `response` empty.
    choice = response.get("choices", [{}])[0]
    # `or {}` rather than a get() default: these keys may be present and explicitly null.
    message = choice.get("message") or {}
    content = message.get("content") or ""
    reasoning_content = message.get("reasoning_content")
    finish_reason = choice.get("finish_reason")
    usage = response.get("usage")
    timings = response.get("timings") or {}
    if error is None:
        # A request failure must not be scored: there is no answer to misread.
        correct, parsed = score(case, content)
    # A reasoning model can burn the whole budget before it states an answer; that is a
    # different failure from answering wrongly, so record it apart.
    return {
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


def _emit(output: TextIO, model: str, done: int, total: int, case: dict[str, Any], record: dict[str, Any], started: float) -> None:
    """Stream one record and a progress line. Crash-safe: each record lands on its own
    flushed line, which is what makes --resume possible."""
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()
    print(
        f"{model} {done}/{total} {case['category']} "
        f"{'OK' if record['correct'] else 'FAIL'} elapsed={time.monotonic() - started:.0f}s",
        flush=True,
    )


def run_cases(
    base_url: str,
    model: str,
    cases: list[dict[str, Any]],
    output: TextIO,
    *,
    timeout: int = 1800,
    retry_delay: float = 5.0,
    tolerate_errors: bool = True,
    skip: frozenset[str] = frozenset(),
    send_seed: bool = True,
    send_reasoning_effort: bool = True,
    concurrency: int = 1,
) -> list[dict[str, Any]]:
    """Send each case to the endpoint, score the answer and stream one JSONL record per
    case to ``output``. Returns the records of the cases actually sent.

    With ``tolerate_errors`` a request that fails after retries is recorded with ``error``
    and the run continues; otherwise the first such failure aborts the run. Cases whose id
    is in ``skip`` are not sent (used by ``--resume`` to continue an interrupted run).
    With ``concurrency`` greater than one, cases are sent by parallel worker threads and
    records stream in completion order; per-case wall-clock (elapsed_seconds) then
    overlaps, so the run measures throughput under load, not per-case latency.
    """
    pending = [case for case in cases if case["id"] not in skip]
    if skip:
        print(f"{model} resuming: {len(skip)} of {len(cases)} cases already recorded, skipping them", flush=True)
    if concurrency > 1:
        print(f"{model} concurrency={concurrency}: requests overlap, per-case elapsed_seconds is a load number, not a latency", flush=True)
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    worker_kwargs = dict(
        timeout=timeout,
        retry_delay=retry_delay,
        tolerate_errors=tolerate_errors,
        send_seed=send_seed,
        send_reasoning_effort=send_reasoning_effort,
    )
    if concurrency <= 1:
        for case in pending:
            record = _process_case(base_url, model, case, **worker_kwargs)
            _emit(output, model, len(records) + 1, len(pending), case, record, started)
            records.append(record)
    else:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        futures = {pool.submit(_process_case, base_url, model, case, **worker_kwargs): case for case in pending}
        try:
            for future in as_completed(futures):
                case = futures[future]
                record = future.result()  # propagates worker exceptions (abort mode)
                _emit(output, model, len(records) + 1, len(pending), case, record, started)
                records.append(record)
        finally:
            # Do not wait for in-flight requests on the way out (abort mode would hang).
            pool.shutdown(wait=False, cancel_futures=True)
    return records


def run_cases_interleaved(
    base_urls: list[str],
    models: list[str],
    cases: list[dict[str, Any]],
    outputs: dict[str, TextIO],
    *,
    timeout: int,
    retry_delay: float,
    tolerate_errors: bool,
    skip_by_model: dict[str, frozenset[str]],
    send_seed: bool = True,
    send_reasoning_effort: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """Strict A/B interleave: for every case, ask each model in turn, one request in
    flight at a time. Each model's answers thus span the same time window, so machine
    drift (heat, cache) cannot systematically favour one side, and no run distorts the
    other's throughput. Records stream to each model's own output in request order.
    """
    records_by_model: dict[str, list[dict[str, Any]]] = {model: [] for model in models}
    pending_by_model = {model: [case for case in cases if case["id"] not in skip_by_model[model]] for model in models}
    for model in models:
        if skip_by_model[model]:
            print(f"{model} resuming: {len(skip_by_model[model])} of {len(cases)} cases already recorded, skipping them", flush=True)
    started = time.monotonic()
    for case in cases:
        for model, base_url in zip(models, base_urls):
            if case["id"] in skip_by_model[model]:
                continue
            record = _process_case(
                base_url,
                model,
                case,
                timeout=timeout,
                retry_delay=retry_delay,
                tolerate_errors=tolerate_errors,
                send_seed=send_seed,
                send_reasoning_effort=send_reasoning_effort,
            )
            _emit(outputs[model], model, len(records_by_model[model]) + 1, len(pending_by_model[model]), case, record, started)
            records_by_model[model].append(record)
    return records_by_model


# --- CLI ------------------------------------------------------------------------

# Bumped if the fixture file layout changes; old runs keep their own files, so pinned cases
# (and their needle labels) stay comparable. Version 2 added the `profile` header.
FIXTURES_VERSION = 2


def write_fixtures(
    fixture_path: Path, fixtures: list[dict[str, Any]], profile: str | None = None
) -> None:
    """Write the pinned cases behind a version and profile header, so a stale or foreign cache
    is refused on the next read instead of being reused as if it were ours."""
    header: dict[str, Any] = {"version": FIXTURES_VERSION}
    if profile:
        header["profile"] = profile
    fixture_path.write_text(json.dumps({**header, "cases": fixtures}, ensure_ascii=False, indent=2) + "\n")


def load_fixtures(fixture_path: Path, profile: str | None = None) -> list[dict[str, Any]]:
    """Read pinned cases, refusing a file this build cannot honestly interpret.

    Both headers are now checked rather than merely written. The version must be one this
    build knows: a newer file may lay its cases out differently, and reading it anyway would
    compare the wrong work. The profile must match the harness asking, because both profiles
    cache under the same `fixtures.json` name — without this check, running the expanded
    harness in a directory an express run had created silently re-ran the 72 express cases
    and wrote them out as an expanded result. Pre-versioning files were bare case arrays and
    still load.
    """
    data = json.loads(fixture_path.read_text())
    if isinstance(data, list):
        return data
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError(f"unrecognized fixtures.json format: {fixture_path}")
    version = data.get("version")
    if version is not None and (not isinstance(version, int) or version > FIXTURES_VERSION):
        raise SystemExit(
            f"{fixture_path} declares fixtures version {version!r}, but this build understands "
            f"at most {FIXTURES_VERSION}; update the harness or use a different --output-dir"
        )
    found = data.get("profile")
    if profile and isinstance(found, str) and found != profile:
        raise SystemExit(
            f"{fixture_path} holds {found!r} fixtures but this is the {profile!r} harness; the "
            f"two profiles pin different cases, so give each profile its own --output-dir (or "
            f"--make-fixtures to rebuild, discarding the {found!r} comparison)"
        )
    if profile and found is None:
        print(
            f"warning: {fixture_path.name} carries no profile header (written by an older "
            f"build); assuming its cases are {profile!r}",
            flush=True,
        )
    return data["cases"]


def resume_records(results_path: Path) -> list[dict[str, Any]]:
    """Load records from a previous (possibly interrupted) run, safe for appending.

    A torn line left by a crash mid-write is dropped and the file truncated to the last
    complete record, so appending starts on a line boundary; the torn case simply runs
    again. Lines that parse but miss the fields the summary needs are treated the same
    way, so their cases are re-run instead of crashing the summary.
    """
    if not results_path.exists():
        return []
    lines = results_path.read_text(encoding="utf-8").splitlines(keepends=True)
    records: list[dict[str, Any]] = []
    kept: list[str] = []
    for line in lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and {"id", "category", "correct"} <= record.keys():
            records.append(record)
            kept.append(line)
    if len(kept) != len(lines):
        results_path.write_text("".join(kept), encoding="utf-8")
    return records


def select_cases(
    cases: list[dict[str, Any]],
    categories: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Filter cases by category (exact name or prefix) and/or cap the count, keeping order.

    ``categories`` is a comma-separated list, e.g. ``long_context`` selects all four
    needle archives; a token that matches no fixture category is an error so a typo
    cannot silently run an empty or wrong subset.
    """
    selected = cases
    if categories:
        tokens = [token.strip() for token in categories.split(",") if token.strip()]
        if not tokens:
            raise SystemExit("--categories is empty")
        present = {case["category"] for case in cases}
        for token in tokens:
            if not any(category == token or category.startswith(token) for category in present):
                raise SystemExit(f"no fixture category matches {token!r}; known: {', '.join(sorted(present))}")
        selected = [case for case in cases if any(case["category"] == token or case["category"].startswith(token) for token in tokens)]
    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be at least 1")
        selected = selected[:limit]
    if not selected:
        raise SystemExit("the selected subset is empty")
    return selected


def main(
    make_cases: Callable[[], list[dict[str, Any]]],
    *,
    timeout: int,
    retry_delay: float,
    tolerate_errors: bool,
    description: str,
    profile: str | None = None,
) -> None:
    """Shared entry point: build or reuse fixtures, run the cases, write results + summary.

    ``profile`` names the calling harness and is stamped into (and checked against)
    ``fixtures.json``, so one profile cannot reuse the other's pinned cases.

    ``--model`` may list several comma-separated model ids (with matching ``--base-url``
    entries) for a strict A/B interleave; a single model is the classic run.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", required=True, help="model id, or comma-separated ids for an A/B interleave")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="endpoint, or comma-separated endpoints matching --model")
    parser.add_argument("--make-fixtures", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip cases already recorded in results-<model>.jsonl and append to that file",
    )
    parser.add_argument(
        "--categories",
        help="comma-separated fixture categories (exact name or prefix) to run, e.g. 'long_context,mmlu'",
    )
    parser.add_argument("--limit", type=int, help="run at most this many of the selected cases")
    parser.add_argument(
        "--no-reasoning-effort",
        action="store_true",
        help="do not send reasoning_effort (for servers that reject unknown payload fields)",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="do not send seed (for servers that reject unknown payload fields)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="parallel requests for a single-model run (overlaps per-case latency)",
    )
    args = parser.parse_args()
    if args.resume and args.make_fixtures:
        raise SystemExit("--resume cannot be combined with --make-fixtures")
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    models = [model.strip() for model in args.model.split(",") if model.strip()]
    if not models:
        raise SystemExit("--model is empty")
    base_urls = [url.strip() for url in args.base_url.split(",") if url.strip()]
    if len(base_urls) == 1 and len(models) > 1:
        base_urls = base_urls * len(models)
    if len(base_urls) != len(models):
        raise SystemExit("comma-separated --base-url entries must match the --model count (or be a single shared URL)")
    if len(models) > 1 and args.concurrency > 1:
        raise SystemExit("--concurrency is not supported with multiple models: the interleave is sequential by design")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fixture_path = args.output_dir / "fixtures.json"
    if args.make_fixtures or not fixture_path.exists():
        all_fixtures = make_cases()
        write_fixtures(fixture_path, all_fixtures, profile)
    else:
        all_fixtures = load_fixtures(fixture_path, profile)
    fixtures = select_cases(all_fixtures, categories=args.categories, limit=args.limit)
    if len(fixtures) != len(all_fixtures):
        parts = [f"running {len(fixtures)} of {len(all_fixtures)} fixtures"]
        if args.categories:
            parts.append(f"categories={args.categories}")
        if args.limit:
            parts.append(f"limit={args.limit}")
        print(f"{' / '.join(models)} {' '.join(parts)}", flush=True)

    # Per-model resume state. The guard compares against the FULL fixture set, so a
    # --categories subset is a narrower view of the same fixtures, not a fixture change.
    states: dict[str, dict[str, Any]] = {}
    for model in models:
        results_path = args.output_dir / f"results-{model_filename(model)}.jsonl"
        records = resume_records(results_path) if args.resume else []
        seen_ids = {record["id"] for record in records}
        unknown = sorted(seen_ids - {case["id"] for case in all_fixtures})
        if unknown:
            raise SystemExit(
                f"{results_path.name} contains {len(unknown)} case(s) not in the current fixtures "
                f"(e.g. {', '.join(unknown[:3])}); the fixtures changed — run without --resume "
                f"or delete the results file"
            )
        states[model] = {"path": results_path, "records": records, "seen": frozenset(seen_ids)}
    mode = "a" if args.resume else "w"
    started = time.monotonic()
    with contextlib.ExitStack() as stack:
        outputs = {model: stack.enter_context(states[model]["path"].open(mode, encoding="utf-8")) for model in models}
        if len(models) == 1:
            model = models[0]
            states[model]["records"] = states[model]["records"] + run_cases(
                base_urls[0],
                model,
                fixtures,
                outputs[model],
                timeout=timeout,
                retry_delay=retry_delay,
                tolerate_errors=tolerate_errors,
                skip=states[model]["seen"],
                send_seed=not args.no_seed,
                send_reasoning_effort=not args.no_reasoning_effort,
                concurrency=args.concurrency,
            )
        else:
            print(f"interleaving {len(models)} models: each case goes to every model in turn, one request at a time", flush=True)
            new_records = run_cases_interleaved(
                base_urls,
                models,
                fixtures,
                outputs,
                timeout=timeout,
                retry_delay=retry_delay,
                tolerate_errors=tolerate_errors,
                skip_by_model={model: states[model]["seen"] for model in models},
                send_seed=not args.no_seed,
                send_reasoning_effort=not args.no_reasoning_effort,
            )
            for model in models:
                states[model]["records"] = states[model]["records"] + new_records[model]
    # The summary covers every record, resumed ones included.
    for model in models:
        summary = summarize(states[model]["records"])
        summary.update({"model": model, "elapsed_seconds": time.monotonic() - started})
        (args.output_dir / f"summary-{model_filename(model)}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False), flush=True)
