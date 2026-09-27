# llm-tests-1

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
| Prefill throughput | — | yes |
| Per-category timings | — | yes |
| Survives a failed request | no | yes |

Use **express** for a quick read between two candidates, **expanded** when the
answer matters: it adds knowledge (MMLU), long context, prefill numbers, and it
records an error instead of aborting the run.

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
```

`--make-fixtures` forces a rebuild. `--base-url` defaults to
`http://127.0.0.1:8080`. `test_quality_expanded_1.py` takes the same four flags.

Every request is sent with `temperature=0`, `top_p=1`, `seed=20260926` and
`reasoning_effort="medium"`; per-case `max_completion_tokens` come from the fixture.

### Output

```
runs/2026-09-27/
  fixtures.json              the pinned cases, reused by later models
  results-<model>.jsonl      one record per case, full content kept
  summary-<model>.json       accuracy by category, throughput, truncation counts
```

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
answer keys, since ARC-Challenge uses each in different rows. GSM8K tries the
requested `#### n` marker first, then an explicitly stated answer, then any
trailing number — a marked answer always beats a stray digit.

### Truncation is not wrongness

A reasoning model can spend its whole token budget before it states an answer.
That is a budget setting, not a quality result, so it is reported separately:

- per record — `finish_reason`, `truncated`, `answer_empty`
- per summary — `truncated`, `answer_empty`, `wrong_and_truncated`,
  `wrong_and_complete`
- per category — `truncated`, in `test_quality_expanded_1.py` only

If `wrong_and_truncated` is high, raise `max_tokens` before drawing conclusions
about the model.

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

## License

MIT — see [LICENSE](LICENSE).

The harnesses fetch public evaluation sets (GSM8K, ARC-Challenge, MMLU) at runtime
through the HuggingFace datasets server; those datasets carry their own licenses
and are not redistributed here.
