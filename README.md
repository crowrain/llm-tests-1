# llm-tests-1

[![tests](https://github.com/crowrain/llm-tests-1/actions/workflows/tests.yml/badge.svg)](https://github.com/crowrain/llm-tests-1/actions/workflows/tests.yml)

Two reproducible A/B quality harnesses for OpenAI-compatible inference endpoints
(llama.cpp / llama-swap, vLLM, NInfer, or anything else that serves
`/v1/chat/completions`).

They exist to answer one question honestly: **given the same fixed cases, which
deployment answers better and how fast?** Cases are pinned by dataset offset, the
sampler is greedy with a fixed seed, and fixtures are cached to disk, so two runs
compare the same work.

## The two harnesses

| | `test_quality_express_1.py` | `test_quality_expanded_1.py` |
|---|---|---|
| Cases | 72 | 266 |
| GSM8K | 30 | 100 |
| ARC-Challenge | 30 | 100 |
| MMLU | — | 50 (5 subjects × 10) |
| JSON instruction following | 12 | 12 |
| Needle-in-a-haystack | — | 4 (16k / 64k / 128k / 180k) |
| Timeout per request | 900 s | 1800 s |
| Retry delay / attempts | 4 s / 3 | 5 s / 3 |
| Survives a failed request | no | yes |

Use **express** for a quick read between two candidates, **expanded** when the
answer matters: it adds knowledge (MMLU) and long context, and it records an
error instead of aborting the run.

Both entry points are thin: dataset fetching, the prompt wording, answer extraction,
scoring, the HTTP layer, the run loop and summary reporting live in `quality_common.py`,
so the two harnesses cannot drift apart. They differ only in case selection, request
budget and failure tolerance — asked the same question, both profiles produce the same
prompt, which a test asserts.

## Requirements

- Python 3.9+, standard library only — no dependencies to install. CI runs the
  test suite on the supported range's ends: 3.9 and 3.14.
- Network access to `datasets-server.huggingface.co` to build fixtures the first
  time. After that `fixtures.json` is reused and the run is offline apart from the
  endpoint itself.
- An endpoint serving `/v1/chat/completions`. If it requires a bearer token (vLLM
  started with `--api-key`, a hosted provider, a proxy in front of llama.cpp), set
  `$OPENAI_API_KEY` or pass `--api-key`.

## Usage

```bash
# first model: builds fixtures.json in the output directory
python3 test_quality_express_1.py \
  --model my-model-a \
  --output-dir runs/2026-09-27 \
  --base-url http://127.0.0.1:8080

# second model: reuses the same fixtures, so the comparison is like-for-like
python3 test_quality_express_1.py \
  --model my-model-b \
  --output-dir runs/2026-09-27

# a crashed or interrupted run continues where it stopped
python3 test_quality_express_1.py --model my-model-a --output-dir runs/2026-09-27 --resume

# only the long-context archives (or cap any run with --limit 10)
python3 test_quality_expanded_1.py --model my-model-b --output-dir runs/2026-09-27 \
  --categories long_context

# strict A/B interleave: both models answer every case, one request at a time
python3 test_quality_express_1.py --model my-model-a,my-model-b \
  --base-url http://127.0.0.1:8080,http://127.0.0.1:8081 --output-dir runs/2026-09-27

# an endpoint that wants a bearer token
OPENAI_API_KEY=sk-... python3 test_quality_express_1.py \
  --model my-model-a --output-dir runs/2026-09-27 --base-url https://endpoint.example
```

`test_quality_expanded_1.py` takes the same flags. Flag reference:

| Flag | Meaning |
|---|---|
| `--model` | model id (required). Comma-separated ids switch the run to a strict A/B interleave. |
| `--base-url` | endpoint, default `http://127.0.0.1:8080`. Comma-separated entries must match the `--model` count, or a single URL is shared by all models. |
| `--api-key` | bearer token for endpoints that require one. Defaults to `$OPENAI_API_KEY`. Follows the same comma-separated rule as `--base-url`, so each side of an interleave can carry its own key. |
| `--output-dir` | directory for `fixtures.json`, `results-<model>.jsonl`, `summary-<model>.json` (created if missing). |
| `--make-fixtures` | force a rebuild of `fixtures.json`. Cannot be combined with `--resume`. |
| `--resume` | skip cases already recorded in `results-<model>.jsonl` and append to the file; a torn final line left by a crash is dropped. Refuses to run if the recorded cases no longer match the current fixtures. |
| `--overwrite` | replace an existing `results-<model>.jsonl` instead of refusing to run. Without it, a run that would discard recorded cases stops and names both ways forward. |
| `--categories` | run a subset by category — exact names or prefixes, comma-separated (`long_context` selects all four needle archives). A token matching nothing is an error, so a typo cannot silently run an empty subset. |
| `--limit N` | cap the run to the first N selected cases (fixture order). Combines with `--resume`: already-recorded cases stay skipped. |
| `--concurrency N` | send N cases in parallel (single model only, default 1). Per-case wall-clock then overlaps, so elapsed medians become load numbers rather than latencies — the run prints a note. Each worker thread keeps its own keep-alive connection; the endpoint should handle concurrent connections (standard for inference servers). |
| `--no-reasoning-effort` | omit `reasoning_effort` from the payload (servers that reject unknown keys). |
| `--no-seed` | omit `seed` from the payload (same). |

### A/B interleave

Listing several models interleaves them strictly: case 1 → model a, case 1 →
model b, case 2 → model a, … — one request in flight at a time. Each model's
answers span the same time window, so machine drift (heat, cache) cannot
systematically favour one side, and no run distorts the other's throughput. Each
model gets its own results/summary files; `--resume` works per model.
`--concurrency` is rejected with multiple models by design.

### Authentication

An endpoint that wants a bearer token gets one from `$OPENAI_API_KEY`, or from `--api-key`
if it is given (the flag wins). Prefer the environment variable: a command line is visible
to other users through `ps`. The key is sent as `Authorization: Bearer <key>` and goes
nowhere else — not into the progress log, `results-<model>.jsonl` or `summary-<model>.json`.
Keys themselves must not contain a comma, since that is how per-model entries are split.

### Request payload

Every request is sent with `temperature=0`, `top_p=1`, `seed=20260926` and
`reasoning_effort="medium"` (the last two unless disabled above); per-case
`max_completion_tokens` come from the fixture. Requests reuse a keep-alive TCP
connection per endpoint, so a run pays the handshake once, not once per case; a
dead or closed connection is dropped and reconnected on retry.

### Output

```
runs/2026-09-27/
  fixtures.json              the pinned cases (version + profile header + cases), reused by later models
  results-<model>.jsonl      one record per case, full content kept
  summary-<model>.json       accuracy by category, throughput, truncation counts
```

A run never discards recorded cases by accident. `results-<model>.jsonl` is replaced only
on a fresh run or with `--overwrite`; otherwise the harness refuses and points at `--resume`.
That matters for the obvious follow-up — "let me just re-check the long-context cases" with
`--categories long_context` — which would otherwise truncate a finished run's records and
overwrite its summary with the subset.

Model ids that contain path separators or spaces are flattened to `_` in the
output file names (`org/model` → `results-org_model.jsonl`).

Fixtures carry a `version` and a `profile` header, and both are checked on read. A
`version` newer than the harness understands is refused rather than guessed at, and a
profile mismatch is refused too: both harnesses cache under the same `fixtures.json`
name, so give each profile its own `--output-dir` (or pass `--make-fixtures`, which
rebuilds and discards the other profile's comparison). Pre-versioning bare-array
`fixtures.json` files are still read, and a file written before the profile header
existed loads with a warning.

## Comparison

`compare_quality.py` answers the question the harnesses exist for — which
deployment answered better, and how fast:

```bash
# both models ran into one directory (compared in file-name order)
python3 compare_quality.py runs/2026-09-27

# or two summary files / directories, with an optional JSON report
python3 compare_quality.py runs/a/summary-model-a.json runs/b/summary-model-b.json \
  --output comparison.json
```

It prints overall and per-category accuracy with the delta (second model minus
first), truncation and error counts, and median prefill/decode throughput.

## Scoring

Extraction is deliberately defensive, because a benchmark that misreads a correct
answer is worse than no benchmark. Four failure modes are handled:

**Reasoning leaking into `content`.** Some builds do not split chain-of-thought
into `reasoning_content` and instead close it inline with a model-specific tag
(`</think>`, `</ifm|think>`, `</ifm|think_fast>`). Everything up to the last such
tag is dropped before scoring, for every category — a value the model weighed
mid-thought must not outrank the one it settled on. If thinking opened and never
closed, the answer never arrived and the case fails.

**JSON glued together from two places.** A first-`{`-to-last-`}` slice breaks when
the reasoning quotes a snippet of its own answer (`Provide {"seconds":8229}.`)
before emitting the real one — the slice spans both and parses as nothing. Instead
the text is scanned for brace-balanced candidates, respecting string literals and
escapes, and the last one wins.

**Markdown wrapped around the value.** A model that was told to end with `Answer: X`
often writes `Answer: **A**`, `` Key: `K...` `` or `The answer is **42**.` — the marker
is there, but the value sits behind emphasis. Every marker-based pattern therefore
tolerates `*`, `_`, backticks and quotes on either side of the value. For the needle
cases the colon itself stays mandatory: archive records read `storage label K...;`
without one, so the colon is what tells the model's own answer line apart from a record
it merely echoed back.

**Formatting that is not a real difference.** Numbers compare by value, so `6.00`
matches `6` and `1/2` matches `0.5`. Multiple-choice accepts both letter and digit
answer keys, since ARC-Challenge uses each in different rows, and tolerates a
trailing sentence period (`Answer: A.` counts as `A`). JSON answers also compare
by value, so a number the model emits as a string (`"8229"`) still matches `8229`.
GSM8K tries the requested `#### n` marker first, then an explicitly stated answer,
then any trailing number — a marked answer always beats a stray digit.

### Truncation is not wrongness

A reasoning model can spend its whole token budget before it states an answer.
That is a budget setting, not a quality result, so it is reported separately:

- per record — `finish_reason`, `truncated`, `answer_empty`
- per summary — `truncated`, `answer_empty`, `wrong_and_truncated`,
  `wrong_and_complete`
- per category — `truncated`, in both harnesses

If `wrong_and_truncated` is high, raise `max_tokens` before drawing conclusions
about the model.

### Failed requests are not wrong answers

A response that carries no usable choice (`"choices": []`, or a null entry) counts as
a failed request too, not as an empty answer: there is no completion in it to score.

In `test_quality_expanded_1.py` a request that still fails after retries is
recorded with `error` instead of aborting the run, and the summary counts it
apart: `errors`, plus `accuracy_excluding_errors`, with error records excluded
from `answer_empty`, `wrong_and_truncated` and `wrong_and_complete`. A request
failure is an infrastructure event, not a data point about the model.

## Notes on the long-context cases

The needle archives are generated locally and deterministically — no dataset
fetch. Each record carries an opaque label `K########Z`, and the question asks for
the label of one record roughly 19 % / 51 % / 73 % / 87 % of the way in.

Labels are a SHA-256 of `salt:char_budget:index`, **not** an arithmetic sequence.
That matters: with a linear generator a capable model reads two neighbouring
records, infers the step and computes the answer instead of retrieving it, which
measures arithmetic rather than retrieval. Labels are also forced unique, so
exactly one record answers the question.

Changing `NEEDLE_SALT` reissues every archive. Existing `fixtures.json` files keep
their own labels, so old runs stay comparable.

## Tests

The scoring, extraction, summary, run-loop, HTTP-layer and CLI (fixtures, resume,
subset selection, payload flags, concurrency, interleave) helpers are covered by
81 stdlib-only regression tests (no endpoint and no network needed — the request
layer is mocked, including the markdown, degenerate-response and fixture-header
regressions above):

```bash
python3 -m unittest discover -s tests -t . -v
```

`-s tests` is worth keeping: a bare `discover` also matches the harness entry points, whose
names start with `test_`, and imports them. Nothing breaks today, but it means a dataset
fetch that ever moved to module level would reach the network from a test run.

They also run automatically on every push to `main` and on pull requests via
GitHub Actions (Python 3.9 and 3.14, `fail-fast: false`).

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for the full history of changes, each entry
linked to its commit.

## License

MIT — see [LICENSE](LICENSE).

The harnesses fetch public evaluation sets (GSM8K, ARC-Challenge, MMLU) at runtime
through the HuggingFace datasets server; those datasets carry their own licenses
and are not redistributed here.
