# Changelog

All notable changes since the upstream baseline [`6518f0a`] (harness rename,
2026-09-27). Each entry names the commit that introduced it, so every change
can be checked out and diffed.

[`6518f0a`]: https://github.com/crowrain/llm-tests-1/commit/6518f0a

## 2026-09-27

### Bug fixes

- Multiple-choice scoring no longer rejects a trailing sentence period:
  `Answer: A.` counts as `A` (`2fcd776`).
- JSON answers are compared by value: numeric strings match numbers
  (`"8229"` == `8229`) (`2fcd776`).
- Failed requests are no longer silently scored as wrong: summaries report
  `errors` and `accuracy_excluding_errors`, and error records are excluded
  from the `answer_empty` / `wrong_and_*` counters (`2fcd776`).
- Model ids containing `/` (e.g. `org/model`) no longer break output
  filenames; non-safe characters become `_` (`2fcd776`).
- HTTP status >= 400 is raised with the response body instead of being
  returned and parsed as a completion (`902bb32`).

### Added

- **A/B interleave** (`902bb32`): `--model a,b --base-url u1,u2` runs both
  models over the same cases in strict alternation (case 1 → a, case 1 → b,
  case 2 → …), one request in flight at a time, so machine drift cannot
  favour one side and no run distorts the other's throughput. Per-model
  `results-*.jsonl` / `summary-*.json` files; `--resume` works per model.
- **`--concurrency N`** (`902bb32`): parallel requests for single-model
  runs. Default 1 keeps sequential semantics; records stream in completion
  order; per-case wall-clock then measures load, not latency (warned in
  the log). Not combined with multi-model mode.
- **HTTP keep-alive** (`902bb32`): connections are reused per endpoint and
  per worker thread, so a 266-case run pays the TCP handshake once; stale
  connections are dropped and reconnected on retry. Per-request timeout is
  applied to the live socket.
- **`--resume`** (`5c1b1e8`): continue an interrupted run — already-recorded
  cases are skipped, a torn final line is dropped, the summary is rebuilt
  over all records; refuses to run if the fixtures changed or with
  `--make-fixtures`.
- **`fixtures.json` version header** (`5c1b1e8`); pre-versioning bare-array
  files still load.
- **`compare_quality.py`** (`5c1b1e8`): A/B delta report from one run
  directory holding two summaries, or from two summary files/dirs; optional
  `--output` JSON.
- **`--categories` / `--limit`** (`f6d7682`): run a subset by category
  (exact name or prefix, comma-separated) and/or cap the count; a token
  matching no category is an error, so a typo cannot silently run an empty
  subset.
- **`--no-reasoning-effort` / `--no-seed`** (`b7a295c`): omit optional
  payload fields for servers that reject unknown keys (some llama.cpp /
  vLLM builds); the default payload is unchanged.
- **GitHub Actions CI** (`08c4518`): `python -m unittest discover` on every
  push to `main` and on pull requests, matrix of Python 3.9 and 3.14 (the
  documented supported range), fail-fast off.

### Changed

- **`quality_common.py`** (`30a5a09`): all shared logic — dataset fetching,
  answer extraction, scoring, HTTP layer, summarization, the per-case run
  loop, and the CLI — moved out of the two entry points, which are now thin
  wrappers (`make_cases` + profile parameters). Express records gained
  `elapsed_seconds`, `subject`, `error` and `approx_chars`; both profiles
  report the same per-category summary fields.
- CLI handling (flags, fixtures, resume, results + summary) unified in
  `quality_common.main` (`5c1b1e8`).
- The run loop, the concurrent path and the interleaved path share one
  worker (`_process_case`) and one output helper (`_emit`) (`902bb32`).
- README documents the new flags, the interleave mode, resume behaviour and
  the CI workflow.

### Tests

- 65 offline, stdlib-only `unittest` cases (grown from 20 in `2fcd776`):
  scoring, extraction, summarization, filenames, the run loop and CLI
  integration with a mocked request layer, resume, subset selection,
  payload flags, keep-alive connection reuse and error handling,
  concurrency, and strict interleave.
