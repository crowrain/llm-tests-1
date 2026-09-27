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
| Prefill / decode throughput | yes | yes |
| Per-category timings | yes | yes |
| Timeout / retry budget | 900 s / 4 s × 3 | 1800 s / 5 s × 3 |
| Survives a failed request | no | yes |

Use **express** for a quick read between two candidates, **expanded** when the
answer matters: it adds knowledge (MMLU) and long context, and it records an
error instead of aborting the run.

Both entry points are thin: dataset fetching, answer extraction, scoring, the
HTTP layer and summary reporting live in `quality_common.py`, so the two
harnesses cannot drift apart. They differ only in case selection, request
budget and failure tolerance.

## Requirements

- Python 3.9+, standard library only — no dependencies to install. Tested on 3.12
  and 3.14.
- Network access to `datasets-server.huggingface.co`, to build fixtures the first
  time. After that `fixtures.json` is reused and the run is offline apart from the
  endpoint itself.
- An endpoint serving `/v1/chat/completions`.

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
```

`--make-fixtures` forces a rebuild. `--resume` skips cases already recorded in
`results-<model>.jsonl` (a torn final line left by a crash is dropped) and rebuilds
the summary over all records; it refuses to run if the recorded cases no longer
match the current fixtures, and cannot be combined with `--make-fixtures`.
`--categories` restricts the run to fixture categories — exact names or prefixes,
comma-separated (`long_context` selects all four needle archives); a token that
matches nothing is an error, so a typo cannot silently run an empty subset.
`--limit N` caps the run to the first N selected cases (fixture order). Both
combine with `--resume`: already-recorded cases stay skipped.
`--concurrency N` sends N cases in parallel (single model only) — useful for
long runs; per-case wall-clock then overlaps, so elapsed medians become load
numbers rather than latencies. The endpoint should handle concurrent
connections (standard for inference servers); each worker thread keeps its
own keep-alive connection.
Listing several comma-separated models (with matching `--base-url` entries, or
one shared URL) interleaves them strictly: each case goes to every model in
turn, one request in flight at a time, so machine drift (heat, cache) cannot
favour one side and no run distorts the other's throughput. Each model gets its
own results/summary files.
`--base-url` defaults to `http://127.0.0.1:8080`. `test_quality_expanded_1.py`
takes the same flags.

Every request is sent with `temperature=0`, `top_p=1`, `seed=20260926` and
`reasoning_effort="medium"`; per-case `max_completion_tokens` come from the fixture.
Servers that reject unknown payload fields (some llama.cpp / vLLM builds) can be
served with `--no-reasoning-effort` and/or `--no-seed`. Requests reuse a
keep-alive TCP connection per endpoint (one per worker thread), so a run pays
the handshake once, not once per case.

### Output

```
runs/2026-09-27/
  fixtures.json              the pinned cases (version header + cases), reused by later models
  results-<model>.jsonl      one record per case, full content kept
  summary-<model>.json       accuracy by category, throughput, truncation counts
```

Model ids that contain path separators or spaces are flattened to `_` in the output
file names (`org/model` → `results-org_model.jsonl`). Fixtures written today carry a
small `version` header; pre-versioning bare-array `fixtures.json` files are still read.

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
answer is worse than no benchmark. Three failure modes are handled:

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

The scoring, extraction, summary, run-loop and CLI (fixtures, resume) helpers are
covered by stdlib-only regression tests (no endpoint and no network needed — the
request layer is mocked):

```bash
python3 -m unittest discover -v
```

They also run automatically on every push to `main` and on pull requests via
GitHub Actions (Python 3.9 and 3.14).

## License

MIT — see [LICENSE](LICENSE).

The harnesses fetch public evaluation sets (GSM8K, ARC-Challenge, MMLU) at runtime
through the HuggingFace datasets server; those datasets carry their own licenses
and are not redistributed here.
