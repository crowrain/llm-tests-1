"""Regression tests for the shared harness logic in quality_common.py.

Run from the repository root with:  python3 -m unittest discover -v
"""

import io
import json
import sys
import tempfile
import time
import unittest
from functools import partial
from pathlib import Path
from unittest import mock

import compare_quality as cq
import quality_common as qc


class AnswerKeyTests(unittest.TestCase):
    """Multiple-choice extraction, including the trailing-period cases that used to fail."""

    def test_trailing_period_counts(self):
        self.assertEqual(qc.ANSWER_CHOICE.findall("The reasoning is done. Answer: A.")[-1], "A")

    def test_no_period_still_counts(self):
        self.assertEqual(qc.ANSWER_CHOICE.findall("The reasoning is done. Answer: A")[-1], "A")

    def test_lowercase_key_with_period(self):
        self.assertEqual(qc.ANSWER_CHOICE.findall("The answer is b.")[-1].upper(), "B")

    def test_digit_key_with_period(self):
        self.assertEqual(qc.ANSWER_CHOICE.findall("option 3. Answer: 3.")[-1], "3")

    def test_last_key_wins(self):
        self.assertEqual(qc.ANSWER_CHOICE.findall("Answer: A. Wait, no. Answer: B.")[-1], "B")

    def test_word_after_letter_is_not_a_key(self):
        # "Apple" starts with a valid option letter but is a word, not a key.
        self.assertEqual(qc.ANSWER_CHOICE.findall("The answer is Apple."), [])

    def test_score_accepts_period_form(self):
        case = {"id": "a1", "category": "arc_challenge", "expected": "A"}
        correct, actual = qc.score(case, "Reasoning... Answer: A.")
        self.assertTrue(correct, msg="expected 'Answer: A.' to be correct")
        self.assertEqual(actual, "A")

    def test_needle_key_is_case_insensitive_marker(self):
        case = {"id": "n1", "category": "long_context_16k", "expected": "K00000001Z"}
        correct, actual = qc.score(case, "I found it. key: K00000001Z")
        self.assertTrue(correct)
        self.assertEqual(actual, "K00000001Z")


class JsonScoringTests(unittest.TestCase):
    """JSON answers compare by value: numeric strings match numbers."""

    def test_numeric_string_matches_number(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"seconds": 8229}}
        self.assertTrue(qc.score(case, '{"seconds": "8229"}')[0])

    def test_numeric_strings_in_list_match(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"values": [3, 7, 2, 5]}}
        self.assertTrue(qc.score(case, '{"values": ["3", "7", "2", "5"]}')[0])

    def test_float_string_matches_number(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"mean": 7.5}}
        self.assertTrue(qc.score(case, '{"mean": "7.5"}')[0])

    def test_non_numeric_string_still_fails(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"seconds": 8229}}
        self.assertFalse(qc.score(case, '{"seconds": "8229 seconds"}')[0])

    def test_wrong_structure_still_fails(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"ordered": ["fig", "pear", "plum", "apple"]}}
        self.assertFalse(qc.score(case, '{"ordered": ["pear", "fig", "plum", "apple"]}')[0])


class ExtractionTests(unittest.TestCase):
    """Existing extraction behaviour the README documents, pinned as regression guards."""

    def test_closed_think_keeps_the_tail(self):
        self.assertEqual(qc.strip_reasoning("<think>hmm 41</think>42"), "42")

    def test_unclosed_think_yields_nothing(self):
        self.assertEqual(qc.strip_reasoning("<think>hmm 42"), "")

    def test_last_balanced_object_wins(self):
        text = 'Thinking... Provide {"seconds": 1}. Real: {"seconds": 8229}'
        self.assertEqual(qc.parse_json_answer(text), {"seconds": 8229})

    def test_code_fence_is_ignored(self):
        self.assertEqual(qc.parse_json_answer('```json\n{"value": 44}\n```'), {"value": 44})

    def test_numbers_compare_by_value(self):
        self.assertTrue(qc.numbers_equal("6.00", "6"))
        self.assertTrue(qc.numbers_equal("1/2", "0.5"))
        self.assertFalse(qc.numbers_equal("5", "6"))


class SummaryTests(unittest.TestCase):
    """A failed request is counted apart, not as a wrong or empty answer."""

    def make_records(self):
        return [
            {"id": "a", "category": "gsm8k", "correct": True, "truncated": False,
             "answer_empty": False, "timings": {}, "elapsed_seconds": 1.0},
            {"id": "b", "category": "gsm8k", "correct": False, "truncated": False,
             "answer_empty": False, "timings": {}, "elapsed_seconds": 2.0},
            {"id": "c", "category": "gsm8k", "correct": False, "truncated": False,
             "answer_empty": True, "timings": {}, "elapsed_seconds": 3.0,
             "error": "RuntimeError('boom')"},
        ]

    def test_failed_request_is_counted_apart(self):
        summary = qc.summarize(self.make_records())
        self.assertEqual(summary["total"], 3)
        self.assertAlmostEqual(summary["accuracy"], 1 / 3)
        self.assertEqual(summary["errors"], 1)
        self.assertAlmostEqual(summary["accuracy_excluding_errors"], 0.5)
        self.assertEqual(summary["answer_empty"], 0)
        self.assertEqual(summary["wrong_and_complete"], 1)
        self.assertEqual(summary["wrong_and_truncated"], 0)

    def test_no_errors_means_unchanged_semantics(self):
        records = self.make_records()[:2]
        summary = qc.summarize(records)
        self.assertEqual(summary["errors"], 0)
        self.assertAlmostEqual(summary["accuracy_excluding_errors"], 0.5)
        self.assertEqual(summary["answer_empty"], 0)

    def test_per_category_stats_present(self):
        summary = qc.summarize(self.make_records())
        stats = summary["by_category"]["gsm8k"]
        self.assertEqual(stats["total"], 3)
        self.assertEqual(stats["truncated"], 0)
        self.assertAlmostEqual(stats["median_elapsed_seconds"], 2.0)
        self.assertIsNone(stats["median_prefill_tps"])
        self.assertIsNone(stats["median_decode_tps"])


class FilenameTests(unittest.TestCase):
    def test_model_name_is_a_single_path_component(self):
        self.assertEqual(qc.model_filename("org/model q8-0"), "org_model_q8-0")
        self.assertEqual(qc.model_filename("plain-model"), "plain-model")


class RunCasesTests(unittest.TestCase):
    """The shared run loop, with the request layer mocked: no network needed."""

    CASES = [{"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32}]
    RESPONSE = {
        "choices": [
            {"message": {"content": "Answer: A.", "reasoning_content": "thinking"}, "finish_reason": "stop"}
        ],
        "usage": {"completion_tokens": 10},
        "timings": {"predicted_per_second": 50.0},
    }

    def test_ok_record_is_scored_and_streamed(self):
        output = io.StringIO()
        with mock.patch.object(qc, "request", return_value=self.RESPONSE):
            records = qc.run_cases("http://x", "model", self.CASES, output)
        self.assertTrue(records[0]["correct"])
        self.assertIsNone(records[0]["error"])
        self.assertFalse(records[0]["truncated"])
        self.assertFalse(records[0]["answer_empty"])
        line = json.loads(output.getvalue().strip())
        self.assertEqual(line["id"], "c1")
        self.assertEqual(line["finish_reason"], "stop")
        self.assertEqual(line["parsed"], "A")

    def test_failed_request_is_recorded(self):
        with mock.patch.object(qc, "request", side_effect=RuntimeError("boom")):
            records = qc.run_cases("http://x", "model", self.CASES, io.StringIO())
        self.assertEqual(records[0]["error"], "RuntimeError('boom')")
        self.assertFalse(records[0]["correct"])
        self.assertIsNone(records[0]["parsed"])

    def test_failed_request_aborts_when_intolerated(self):
        with mock.patch.object(qc, "request", side_effect=RuntimeError("boom")):
            with self.assertRaises(RuntimeError):
                qc.run_cases("http://x", "model", self.CASES, io.StringIO(), tolerate_errors=False)


class FixturesTests(unittest.TestCase):
    def fixtures(self):
        return [{"id": "gsm8k_00", "category": "gsm8k", "prompt": "q", "expected": "42", "max_tokens": 16}]

    def test_roundtrip_with_version_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            qc.write_fixtures(path, self.fixtures())
            data = json.loads(path.read_text())
            self.assertEqual(data["version"], qc.FIXTURES_VERSION)
            self.assertEqual(qc.load_fixtures(path), self.fixtures())

    def test_pre_versioning_bare_array_still_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            path.write_text(json.dumps(self.fixtures()) + "\n")
            self.assertEqual(qc.load_fixtures(path), self.fixtures())

    def test_unrecognized_format_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            path.write_text('{"version": 99}')
            with self.assertRaises(ValueError):
                qc.load_fixtures(path)


class ResumeTests(unittest.TestCase):
    def test_torn_final_line_is_dropped_and_file_truncated(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "results-model.jsonl"
            path.write_text('{"id": "a", "category": "gsm8k", "correct": true}\n{"id": "b", "correct": fa')
            records = qc.resume_records(path)
            self.assertEqual([r["id"] for r in records], ["a"])
            self.assertEqual(path.read_text(), '{"id": "a", "category": "gsm8k", "correct": true}\n')

    def test_missing_file_yields_no_records(self):
        self.assertEqual(qc.resume_records(Path("/nonexistent/results-model.jsonl")), [])

    def test_run_cases_skips_recorded_ids(self):
        cases = [
            {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
            {"id": "c2", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
        ]
        with mock.patch.object(qc, "request", return_value=RunCasesTests.RESPONSE) as fake:
            records = qc.run_cases("http://x", "m", cases, io.StringIO(), skip=frozenset({"c1"}))
        self.assertEqual([r["id"] for r in records], ["c2"])
        self.assertEqual(fake.call_count, 1)

    def test_summarize_tolerates_records_without_elapsed_seconds(self):
        # --resume of a results file written before elapsed_seconds existed.
        record = {"id": "a", "category": "gsm8k", "correct": True, "truncated": False,
                  "answer_empty": False, "timings": {}}
        summary = qc.summarize([record])
        self.assertEqual(summary["by_category"]["gsm8k"]["median_elapsed_seconds"], None)


class CompareTests(unittest.TestCase):
    def make_summary(self, model: str, accuracy: float, decode_tps: float) -> dict:
        return {
            "model": model,
            "total": 10,
            "correct": int(round(accuracy * 10)),
            "accuracy": accuracy,
            "truncated": 1,
            "errors": 0,
            "median_prefill_tps": None,
            "median_decode_tps": decode_tps,
            "by_category": {
                "gsm8k": {
                    "total": 10,
                    "correct": int(round(accuracy * 10)),
                    "accuracy": accuracy,
                    "truncated": 1,
                    "median_prefill_tps": None,
                    "median_decode_tps": decode_tps,
                    "median_elapsed_seconds": 1.0,
                }
            },
        }

    def test_render_reports_models_and_deltas(self):
        text = cq.render(
            "model-a", self.make_summary("model-a", 0.8, 50.0),
            "model-b", self.make_summary("model-b", 0.7, 40.0),
        )
        self.assertIn("model-a vs model-b", text)
        self.assertIn("-10.0 pp", text)  # gsm8k: 70% − 80%
        self.assertIn("-20.0%", text)  # decode: (40 − 50) / 50

    def test_resolve_single_directory_with_two_summaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            (tmp_path / "summary-model-a.json").write_text(json.dumps(self.make_summary("model-a", 0.8, 50.0)))
            (tmp_path / "summary-model-b.json").write_text(json.dumps(self.make_summary("model-b", 0.7, 40.0)))
            (a, b) = cq.resolve([tmp_path])
            self.assertEqual(a[0], "model-a")
            self.assertEqual(b[0], "model-b")

    def test_resolve_two_summary_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            file_a = tmp_path / "summary-model-a.json"
            file_b = tmp_path / "summary-model-b.json"
            file_a.write_text(json.dumps(self.make_summary("model-a", 0.8, 50.0)))
            file_b.write_text(json.dumps(self.make_summary("model-b", 0.7, 40.0)))
            (a, b) = cq.resolve([file_a, file_b])
            self.assertEqual((a[0], b[0]), ("model-a", "model-b"))

    def test_single_directory_with_one_summary_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "summary-model-a.json").write_text(json.dumps(self.make_summary("model-a", 0.8, 50.0)))
            with self.assertRaises(SystemExit):
                cq.resolve([Path(tmp)])

    def test_as_json_deltas(self):
        data = cq.as_json(
            "model-a", self.make_summary("model-a", 0.8, 50.0),
            "model-b", self.make_summary("model-b", 0.7, 40.0),
        )
        self.assertAlmostEqual(data["delta_accuracy"], -0.1)
        self.assertAlmostEqual(data["by_category"]["gsm8k"]["delta"], -0.1)
        self.assertEqual(data["a"]["model"], "model-a")


class RequestPayloadTests(unittest.TestCase):
    """The optional sampler fields are sent by default and gateable for strict servers."""

    CASE = {"id": "c1", "category": "arc_challenge", "expected": "A", "prompt": "q", "max_tokens": 32}

    def capture_payload(self, **kwargs):
        captured = {}

        def fake_post(url, data, timeout=None, api_key=None):
            captured["data"] = data
            return b'{"choices": []}'

        with mock.patch.object(qc, "_post_json", side_effect=fake_post):
            qc.request("http://x", "m", self.CASE, **kwargs)
        return json.loads(captured["data"])

    def test_defaults_send_seed_and_reasoning_effort(self):
        payload = self.capture_payload()
        self.assertEqual(payload["seed"], 20260926)
        self.assertEqual(payload["reasoning_effort"], "medium")
        self.assertEqual(payload["temperature"], 0)
        self.assertEqual(payload["top_p"], 1)
        self.assertEqual(payload["max_completion_tokens"], 32)

    def test_seed_can_be_omitted(self):
        payload = self.capture_payload(send_seed=False)
        self.assertNotIn("seed", payload)
        self.assertEqual(payload["reasoning_effort"], "medium")

    def test_reasoning_effort_can_be_omitted(self):
        payload = self.capture_payload(send_reasoning_effort=False)
        self.assertNotIn("reasoning_effort", payload)
        self.assertEqual(payload["seed"], 20260926)

    def test_run_cases_passes_payload_flags(self):
        cases = [{"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32}]
        answer = {"choices": [{"message": {"content": "Answer: A."}, "finish_reason": "stop"}]}
        with mock.patch.object(qc, "request", return_value=answer) as fake:
            qc.run_cases(
                "http://x", "m", cases, io.StringIO(), send_seed=False, send_reasoning_effort=False
            )
        self.assertFalse(fake.call_args.kwargs["send_seed"])
        self.assertFalse(fake.call_args.kwargs["send_reasoning_effort"])


class KeepAliveTests(unittest.TestCase):
    """The HTTP layer reuses one keep-alive connection per endpoint, per thread."""

    CASE = {"id": "c1", "category": "arc_challenge", "expected": "A", "prompt": "q", "max_tokens": 32}
    BASE = "http://127.0.0.1:8080"

    class FakeResponse:
        def __init__(self, body=b'{"choices": []}', status=200):
            self.body = body
            self.status = status

        def read(self):
            return self.body

    class FakeConn:
        instances: list = []

        def __init__(self, host, port, timeout=None):
            self.failed = False
            self.sock = None  # not connected yet: _post_json sets conn.timeout
            self.response = KeepAliveTests.FakeResponse()
            KeepAliveTests.FakeConn.instances.append(self)

        def request(self, method, path, body=None, headers=None, timeout=None):
            if self.failed:
                raise ConnectionResetError("stale keep-alive connection")

        def getresponse(self):
            return self.response

    def setUp(self):
        self.FakeConn.instances = []
        qc._LOCAL.conns = {}
        patcher = mock.patch.object(qc.http.client, "HTTPConnection", self.FakeConn)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_connection_is_reused_across_requests(self):
        for _ in range(3):
            self.assertEqual(qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0.01), {"choices": []})
        self.assertEqual(len(self.FakeConn.instances), 1)

    def test_stale_connection_is_replaced_on_retry(self):
        qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0.01)
        self.FakeConn.instances[0].failed = True  # server closed the idle keep-alive
        self.assertEqual(qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0.01), {"choices": []})
        self.assertEqual(len(self.FakeConn.instances), 2)  # dropped and reconnected

    def test_http_error_is_reported_not_returned(self):
        class Fake400(KeepAliveTests.FakeConn):
            def __init__(self, host, port, timeout=None):
                super().__init__(host, port, timeout)
                self.response = KeepAliveTests.FakeResponse(b'{"error": "unknown field: seed"}', status=400)

        with mock.patch.object(qc.http.client, "HTTPConnection", Fake400):
            with self.assertRaises(RuntimeError) as ctx:
                qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0.01)
        self.assertIn("HTTP 400", str(ctx.exception))


class ConcurrencyTests(unittest.TestCase):
    def ok_response(self, case):
        return {
            "choices": [{"message": {"content": f"Answer: {case['expected']}."}, "finish_reason": "stop"}],
            "usage": None,
            "timings": {},
        }

    def test_all_cases_run_and_streamed(self):
        cases = [{"id": f"c{i}", "category": "arc_challenge", "expected": "A", "max_tokens": 32} for i in range(6)]

        def fake_request(base_url, model, case, **kwargs):
            time.sleep(0.01)
            return self.ok_response(case)

        output = io.StringIO()
        with mock.patch.object(qc, "request", side_effect=fake_request):
            records = qc.run_cases("http://x", "m", cases, output, concurrency=3)
        self.assertEqual({r["id"] for r in records}, {c["id"] for c in cases})
        self.assertEqual(len(output.getvalue().splitlines()), 6)
        self.assertTrue(all(json.loads(line)["correct"] for line in output.getvalue().splitlines()))

    def test_errors_are_recorded_under_concurrency(self):
        cases = [
            {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
            {"id": "c2", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
        ]

        def fake_request(base_url, model, case, **kwargs):
            if case["id"] == "c2":
                raise RuntimeError("boom")
            return self.ok_response(case)

        with mock.patch.object(qc, "request", side_effect=fake_request):
            records = qc.run_cases("http://x", "m", cases, io.StringIO(), concurrency=2)
        by_id = {r["id"]: r for r in records}
        self.assertTrue(by_id["c1"]["correct"])
        self.assertEqual(by_id["c2"]["error"], "RuntimeError('boom')")


class InterleaveTests(unittest.TestCase):
    """Strict A/B interleave: each case goes to every model in turn, one request at a time."""

    CASES = [
        {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
        {"id": "c2", "category": "arc_challenge", "expected": "B", "max_tokens": 32},
    ]
    MODELS = ["model-a", "model-b"]
    URLS = ["http://a", "http://b"]

    def fake_request(self, calls, base_url, model, case, **kwargs):
        calls.append((model, case["id"]))
        return {
            "choices": [{"message": {"content": f"Answer: {case['expected']}."}, "finish_reason": "stop"}],
            "usage": None,
            "timings": {},
        }

    def test_strict_alternation_and_per_model_outputs(self):
        calls: list = []
        outputs = {"model-a": io.StringIO(), "model-b": io.StringIO()}
        with mock.patch.object(qc, "request", side_effect=partial(self.fake_request, calls)):
            records = qc.run_cases_interleaved(
                self.URLS, self.MODELS, self.CASES, outputs,
                timeout=1, retry_delay=0.01, tolerate_errors=True,
                skip_by_model={"model-a": frozenset(), "model-b": frozenset()},
            )
        self.assertEqual(calls, [("model-a", "c1"), ("model-b", "c1"), ("model-a", "c2"), ("model-b", "c2")])
        self.assertEqual({r["id"] for r in records["model-a"]}, {"c1", "c2"})
        self.assertEqual({r["id"] for r in records["model-b"]}, {"c1", "c2"})
        self.assertTrue(all(r["correct"] for r in records["model-a"] + records["model-b"]))

    def test_skip_is_per_model(self):
        calls: list = []
        outputs = {model: io.StringIO() for model in self.MODELS}
        with mock.patch.object(qc, "request", side_effect=partial(self.fake_request, calls)):
            qc.run_cases_interleaved(
                self.URLS, self.MODELS, self.CASES, outputs,
                timeout=1, retry_delay=0.01, tolerate_errors=True,
                skip_by_model={"model-a": frozenset({"c1"}), "model-b": frozenset()},
            )
        self.assertEqual(calls, [("model-b", "c1"), ("model-a", "c2"), ("model-b", "c2")])


class SelectCasesTests(unittest.TestCase):
    def cases(self):
        return [
            {"id": f"g{i}", "category": "gsm8k"} for i in range(3)
        ] + [
            {"id": f"n_{label}", "category": f"long_context_{label}"} for label in ("16k", "64k")
        ] + [{"id": "if_01", "category": "instruction_json"}]

    def test_no_filters_returns_all_in_order(self):
        self.assertEqual(
            [c["id"] for c in qc.select_cases(self.cases())],
            ["g0", "g1", "g2", "n_16k", "n_64k", "if_01"],
        )

    def test_exact_category(self):
        self.assertEqual([c["id"] for c in qc.select_cases(self.cases(), categories="gsm8k")], ["g0", "g1", "g2"])

    def test_prefix_matches_all_needle_archives(self):
        self.assertEqual(
            [c["category"] for c in qc.select_cases(self.cases(), categories="long_context")],
            ["long_context_16k", "long_context_64k"],
        )

    def test_multiple_categories(self):
        self.assertEqual(len(qc.select_cases(self.cases(), categories="gsm8k, instruction_json")), 4)

    def test_unknown_category_is_an_error(self):
        with self.assertRaises(SystemExit):
            qc.select_cases(self.cases(), categories="gg")

    def test_empty_categories_is_an_error(self):
        with self.assertRaises(SystemExit):
            qc.select_cases(self.cases(), categories=" , ")

    def test_limit_caps_in_fixture_order(self):
        self.assertEqual([c["id"] for c in qc.select_cases(self.cases(), limit=2)], ["g0", "g1"])

    def test_limit_applies_after_categories(self):
        selected = qc.select_cases(self.cases(), categories="long_context,gsm8k", limit=2)
        self.assertEqual([c["id"] for c in selected], ["g0", "g1"])

    def test_zero_limit_is_an_error(self):
        with self.assertRaises(SystemExit):
            qc.select_cases(self.cases(), limit=0)


class SharedMainTests(unittest.TestCase):
    """The shared CLI (fixtures, resume, results + summary), fully offline."""

    def fixtures(self):
        return [
            {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
            {"id": "c2", "category": "arc_challenge", "expected": "B", "max_tokens": 32},
        ]

    def test_resume_continues_and_summary_covers_all_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # First "run": c1 completed, then a crash left a torn line for c2.
            qc.write_fixtures(out_dir / "fixtures.json", self.fixtures())
            (out_dir / "results-m.jsonl").write_text(
                '{"id": "c1", "category": "arc_challenge", "correct": true, "truncated": false, '
                '"answer_empty": false, "timings": {}, "elapsed_seconds": 1.0}\n'
                '{"id": "c2", "correct": fa'
            )

            def fake_request(base_url, model, case, **kwargs):
                if case["id"] == "c1":
                    raise AssertionError("c1 must be skipped on resume")
                return {
                    "choices": [
                        {"message": {"content": "Answer: B.", "reasoning_content": None},
                         "finish_reason": "stop"}
                    ],
                    "usage": None,
                    "timings": {},
                }

            argv = ["prog", "--model", "m", "--output-dir", str(out_dir), "--resume"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                qc, "request", side_effect=fake_request
            ):
                qc.main(lambda: self.fixtures(), timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

            text = (out_dir / "results-m.jsonl").read_text()
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            self.assertEqual([r["id"] for r in lines], ["c1", "c2"])
            summary = json.loads((out_dir / "summary-m.json").read_text())
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["correct"], 2)

    def test_resume_rejects_make_fixtures(self):
        argv = ["prog", "--model", "m", "--output-dir", "/tmp/none", "--resume", "--make-fixtures"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                qc.main(lambda: [], timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

    def test_main_runs_only_selected_categories(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fixtures = [
                {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
                {"id": "c2", "category": "long_context_16k", "expected": "K00000001Z", "max_tokens": 32},
            ]
            sent: list[str] = []

            def fake_request(base_url, model, case, **kwargs):
                sent.append(case["id"])
                return {
                    "choices": [
                        {"message": {"content": "Answer: A. Key: K00000001Z",
                                     "reasoning_content": None},
                         "finish_reason": "stop"}
                    ],
                    "usage": None,
                    "timings": {},
                }

            argv = ["prog", "--model", "m", "--output-dir", str(out_dir),
                    "--categories", "long_context"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                qc, "request", side_effect=fake_request
            ):
                qc.main(lambda: fixtures, timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")
            self.assertEqual(sent, ["c2"])
            summary = json.loads((out_dir / "summary-m.json").read_text())
            self.assertEqual(summary["total"], 1)

    def test_resume_with_category_filter_keeps_prior_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            # c1 was already recorded in a previous run; resume restricted to arc_challenge.
            qc.write_fixtures(out_dir / "fixtures.json", self.fixtures())
            (out_dir / "results-m.jsonl").write_text(
                '{"id": "c1", "category": "arc_challenge", "correct": true, "truncated": false, '
                '"answer_empty": false, "timings": {}, "elapsed_seconds": 1.0}\n'
            )

            def fake_request(base_url, model, case, **kwargs):
                self.assertEqual(case["id"], "c2")
                return {
                    "choices": [
                        {"message": {"content": "Answer: B.", "reasoning_content": None},
                         "finish_reason": "stop"}
                    ],
                    "usage": None,
                    "timings": {},
                }

            argv = ["prog", "--model", "m", "--output-dir", str(out_dir),
                    "--resume", "--categories", "arc_challenge"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                qc, "request", side_effect=fake_request
            ):
                qc.main(lambda: self.fixtures(), timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

            text = (out_dir / "results-m.jsonl").read_text()
            lines = [json.loads(line) for line in text.splitlines() if line.strip()]
            self.assertEqual([r["id"] for r in lines], ["c1", "c2"])
            summary = json.loads((out_dir / "summary-m.json").read_text())
            self.assertEqual(summary["total"], 2)

    def test_main_interleaves_two_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            fixtures = [
                {"id": "c1", "category": "arc_challenge", "expected": "A", "max_tokens": 32},
                {"id": "c2", "category": "arc_challenge", "expected": "B", "max_tokens": 32},
            ]
            calls: list = []

            def fake_request(base_url, model, case, **kwargs):
                calls.append((base_url, model, case["id"]))
                return {
                    "choices": [{"message": {"content": f"Answer: {case['expected']}."}, "finish_reason": "stop"}],
                    "usage": None,
                    "timings": {},
                }

            argv = ["prog", "--model", "a,b", "--base-url", "http://a,http://b", "--output-dir", str(out_dir)]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(qc, "request", side_effect=fake_request):
                qc.main(lambda: fixtures, timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")
            # Strict alternation: case 1 to both, then case 2 to both, one request at a time.
            self.assertEqual(
                calls,
                [("http://a", "a", "c1"), ("http://b", "b", "c1"),
                 ("http://a", "a", "c2"), ("http://b", "b", "c2")],
            )
            for model in ("a", "b"):
                lines = (out_dir / f"results-{model}.jsonl").read_text().splitlines()
                self.assertEqual(len(lines), 2)
                summary = json.loads((out_dir / f"summary-{model}.json").read_text())
                self.assertEqual(summary["model"], model)
                self.assertEqual(summary["total"], 2)
                self.assertEqual(summary["correct"], 2)

    def test_main_rejects_concurrency_with_multiple_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            argv = ["prog", "--model", "a,b", "--output-dir", str(tmp), "--concurrency", "2"]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    qc.main(lambda: [], timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

    def test_main_omits_optional_payload_fields_when_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            answer = {"choices": [{"message": {"content": "Answer: A."}, "finish_reason": "stop"}]}
            argv = ["prog", "--model", "m", "--output-dir", str(tmp),
                    "--no-seed", "--no-reasoning-effort"]
            with mock.patch.object(sys, "argv", argv), mock.patch.object(
                qc, "request", return_value=answer
            ) as fake:
                qc.main(
                    lambda: self.fixtures(), timeout=1, retry_delay=0.01,
                    tolerate_errors=True, description="t",
                )
            self.assertFalse(fake.call_args.kwargs["send_seed"])
            self.assertFalse(fake.call_args.kwargs["send_reasoning_effort"])

    def test_resume_rejects_changed_fixtures(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            cases = self.fixtures()[:1]
            (out_dir / "fixtures.json").write_text(json.dumps({"version": qc.FIXTURES_VERSION, "cases": cases}))
            (out_dir / "results-m.jsonl").write_text('{"id": "old1", "category": "gsm8k", "correct": true}\n')
            with mock.patch.object(sys, "argv", ["prog", "--model", "m", "--output-dir", str(out_dir), "--resume"]):
                with self.assertRaises(SystemExit):
                    qc.main(lambda: cases, timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")


class MarkdownAroundAnswerTests(unittest.TestCase):
    """Emphasis and backticks between the marker and the value.

    Every case here used to score a *correct* answer as no answer at all, because each
    marker-based pattern demanded the value sit immediately after the separator. Models
    that format their output tripped all three extractors.
    """

    def test_gsm8k_bold_answer_with_trailing_period(self):
        for text in ("The answer is **42**.", "#### **42**.", "Final: (42).", "Result: \\(42\\)."):
            with self.subTest(text=text):
                self.assertEqual(qc.score({"category": "gsm8k", "expected": "42"}, text), (True, "42"))

    def test_gsm8k_punctuation_only_token_never_wins(self):
        # The bug: "." matched the fallback class and, being last, replaced the real number.
        self.assertEqual(qc.GSM_PATTERNS[2].findall("The answer is **42**."), ["42"])

    def test_gsm8k_plain_forms_still_score(self):
        for text, expected in (("#### 42", "42"), ("The answer is $42.", "42"), ("we get 42.", "42")):
            with self.subTest(text=text):
                self.assertEqual(qc.score({"category": "gsm8k", "expected": "42"}, text), (True, expected))

    def test_multiple_choice_emphasis(self):
        for text in ("Answer: **A**", "Answer: **A**.", "Answer: `A`", "Answer: __A__", 'Answer: "A"'):
            with self.subTest(text=text):
                self.assertEqual(qc.score({"category": "arc_challenge", "expected": "A"}, text), (True, "A"))

    def test_needle_key_emphasis(self):
        expected = "K12345678Z"
        for text in (f"Key: **{expected}**", f"Key: `{expected}`", f"label: **{expected}**."):
            with self.subTest(text=text):
                self.assertEqual(
                    qc.score({"category": "long_context_16k", "expected": expected}, text), (True, expected)
                )

    def test_needle_still_requires_the_colon(self):
        """An echoed archive record reads "storage label K...;" with no colon.

        The colon is what tells the model's own answer line apart from a record it copied
        back, so tolerating emphasis must not tolerate a missing separator.
        """
        echoed = "Record 000123: storage label K99999999Z; provenance cedar quartz."
        self.assertEqual(qc.score({"category": "long_context_16k", "expected": "K12345678Z"}, echoed), (False, None))


class NoUsableChoiceTests(unittest.TestCase):
    """A response that carries no choice is an infrastructure failure, not an empty answer."""

    def case(self):
        return {"id": "c1", "category": "gsm8k", "prompt": "q", "expected": "42", "max_tokens": 16}

    def process(self, response, *, tolerate_errors=True):
        with mock.patch.object(qc, "request", return_value=response):
            return qc._process_case(
                "http://x", "m", self.case(), timeout=1, retry_delay=0,
                tolerate_errors=tolerate_errors, send_seed=True, send_reasoning_effort=True,
            )

    def test_empty_choices_list_is_recorded_not_raised(self):
        # Used to raise IndexError from outside the try, aborting even a tolerant run.
        record = self.process({"choices": []})
        self.assertIsNotNone(record["error"])
        self.assertFalse(record["correct"])

    def test_missing_and_null_choices_are_recorded(self):
        for response in ({}, {"choices": None}, {"choices": [None]}, {"choices": "nope"}):
            with self.subTest(response=response):
                self.assertIsNotNone(self.process(response)["error"])

    def test_no_usable_choice_aborts_when_intolerated(self):
        with self.assertRaises(RuntimeError):
            self.process({"choices": []}, tolerate_errors=False)

    def test_null_message_and_timings_do_not_crash(self):
        record = self.process({"choices": [{"message": None, "finish_reason": "stop"}], "timings": None})
        self.assertIsNone(record["error"])
        self.assertEqual(record["content"], "")
        self.assertEqual(record["timings"], {})

    def test_summarize_tolerates_null_timings(self):
        # An older results file being resumed can carry the key explicitly null.
        records = [{"id": "a", "category": "gsm8k", "correct": True, "timings": None}]
        self.assertIsNone(qc.summarize(records)["median_decode_tps"])


class FixtureHeaderTests(unittest.TestCase):
    """The version and profile headers are enforced, not merely written."""

    def fixtures(self):
        return [{"id": "gsm8k_00", "category": "gsm8k", "prompt": "q", "expected": "42", "max_tokens": 16}]

    def write(self, path, **header):
        path.write_text(json.dumps({**header, "cases": self.fixtures()}))

    def test_profile_is_written_and_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            qc.write_fixtures(path, self.fixtures(), "expanded")
            self.assertEqual(json.loads(path.read_text())["profile"], "expanded")
            self.assertEqual(qc.load_fixtures(path, "expanded"), self.fixtures())

    def test_foreign_profile_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            qc.write_fixtures(path, self.fixtures(), "express")
            with self.assertRaises(SystemExit) as caught:
                qc.load_fixtures(path, "expanded")
            self.assertIn("express", str(caught.exception))

    def test_future_version_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            self.write(path, version=qc.FIXTURES_VERSION + 1, profile="expanded")
            with self.assertRaises(SystemExit):
                qc.load_fixtures(path, "expanded")

    def test_older_header_without_profile_loads_with_a_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fixtures.json"
            self.write(path, version=1)
            with mock.patch("builtins.print") as printed:
                self.assertEqual(qc.load_fixtures(path, "expanded"), self.fixtures())
            self.assertIn("no profile header", printed.call_args[0][0])

    def test_expanded_harness_refuses_an_express_fixture_directory(self):
        """The whole point: reusing the other profile's pinned cases must not be silent."""
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            qc.write_fixtures(out_dir / "fixtures.json", self.fixtures(), "express")
            argv = ["prog", "--model", "m", "--output-dir", str(out_dir)]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    qc.main(
                        lambda: self.fixtures(), timeout=1, retry_delay=0,
                        tolerate_errors=True, description="t", profile="expanded",
                    )


class OverwriteGuardTests(unittest.TestCase):
    """A run that would discard recorded cases has to say so first.

    Without --resume the results file is opened "w". A narrower follow-up run into the
    directory of a finished run -- "let me just re-check the long-context cases" -- used to
    truncate that run's records and overwrite its summary with the subset, silently.
    """

    def fixtures(self):
        return [
            {"id": "c1", "category": "gsm8k", "prompt": "q", "expected": "1", "max_tokens": 16},
            {"id": "c2", "category": "long_context_16k", "prompt": "q", "expected": "K1", "max_tokens": 16},
        ]

    def answer(self, *args, **kwargs):
        return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}], "timings": {}}

    def run_main(self, out_dir, *extra):
        argv = ["prog", "--model", "m", "--output-dir", str(out_dir), *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(qc, "request", side_effect=self.answer):
            qc.main(
                lambda: self.fixtures(), timeout=1, retry_delay=0,
                tolerate_errors=True, description="t", profile="p",
            )

    def recorded(self, out_dir):
        text = (out_dir / "results-m.jsonl").read_text()
        return [json.loads(line)["id"] for line in text.splitlines() if line.strip()]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        qc.write_fixtures(self.out_dir / "fixtures.json", self.fixtures(), "p")

    def test_fresh_run_is_unaffected(self):
        self.run_main(self.out_dir)
        self.assertEqual(self.recorded(self.out_dir), ["c1", "c2"])

    def test_subset_rerun_is_refused_and_records_survive(self):
        self.run_main(self.out_dir)
        with self.assertRaises(SystemExit) as caught:
            self.run_main(self.out_dir, "--categories", "long_context")
        self.assertIn("--resume", str(caught.exception))
        self.assertIn("--overwrite", str(caught.exception))
        # The point of the guard: nothing was lost.
        self.assertEqual(self.recorded(self.out_dir), ["c1", "c2"])
        self.assertEqual(json.loads((self.out_dir / "summary-m.json").read_text())["total"], 2)

    def test_overwrite_replaces_deliberately(self):
        self.run_main(self.out_dir)
        self.run_main(self.out_dir, "--categories", "long_context", "--overwrite")
        self.assertEqual(self.recorded(self.out_dir), ["c2"])

    def test_resume_still_appends(self):
        self.run_main(self.out_dir, "--categories", "gsm8k")
        self.run_main(self.out_dir, "--resume")
        self.assertEqual(self.recorded(self.out_dir), ["c1", "c2"])

    def test_resume_and_overwrite_are_rejected_together(self):
        with self.assertRaises(SystemExit):
            self.run_main(self.out_dir, "--resume", "--overwrite")

    def test_an_empty_results_file_does_not_block(self):
        (self.out_dir / "results-m.jsonl").write_text("\n")
        self.run_main(self.out_dir)
        self.assertEqual(self.recorded(self.out_dir), ["c1", "c2"])

    def test_guard_is_per_model(self):
        self.run_main(self.out_dir)
        argv = ["prog", "--model", "other", "--output-dir", str(self.out_dir)]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(qc, "request", side_effect=self.answer):
            qc.main(
                lambda: self.fixtures(), timeout=1, retry_delay=0,
                tolerate_errors=True, description="t", profile="p",
            )
        self.assertEqual(self.recorded(self.out_dir), ["c1", "c2"])


class ApiKeyTests(unittest.TestCase):
    """Bearer-token auth: sent when given, absent when not, and never echoed anywhere."""

    CASE = {"id": "c1", "category": "arc_challenge", "expected": "A", "prompt": "q", "max_tokens": 32}
    BASE = "http://127.0.0.1:8080"
    KEY = "sk-secret-value"

    class FakeConn:
        headers: list = []

        def __init__(self, host, port, timeout=None):
            self.sock = None

        def request(self, method, path, body=None, headers=None, timeout=None):
            ApiKeyTests.FakeConn.headers.append(dict(headers or {}))

        def getresponse(self):
            return ApiKeyTests.FakeResponse()

    class FakeResponse:
        status = 200

        def read(self):
            return b'{"choices": [{"message": {"content": "Answer: A."}, "finish_reason": "stop"}]}'

    def setUp(self):
        self.FakeConn.headers = []
        qc._LOCAL.conns = {}
        patcher = mock.patch.object(qc.http.client, "HTTPConnection", self.FakeConn)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_key_is_sent_as_a_bearer_token(self):
        qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0, api_key=self.KEY)
        self.assertEqual(self.FakeConn.headers[0]["Authorization"], f"Bearer {self.KEY}")

    def test_no_key_means_no_authorization_header(self):
        qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0)
        self.assertNotIn("Authorization", self.FakeConn.headers[0])
        self.assertEqual(self.FakeConn.headers[0]["Content-Type"], "application/json")

    def test_empty_key_is_treated_as_absent(self):
        qc.request(self.BASE, "m", self.CASE, timeout=1, retry_delay=0, api_key="")
        self.assertNotIn("Authorization", self.FakeConn.headers[0])

    def test_run_cases_forwards_the_key(self):
        answer = {"choices": [{"message": {"content": "Answer: A."}}]}
        with mock.patch.object(qc, "request", return_value=answer) as fake:
            qc.run_cases(self.BASE, "m", [dict(self.CASE)], io.StringIO(), api_key=self.KEY)
        self.assertEqual(fake.call_args.kwargs["api_key"], self.KEY)

    def test_key_never_reaches_the_record(self):
        """A key in a record would be copied into results-*.jsonl and shared with the run."""
        with mock.patch.object(qc, "request", return_value={"choices": [{"message": {"content": "Answer: A."}}]}):
            out = io.StringIO()
            qc.run_cases(self.BASE, "m", [dict(self.CASE)], out, api_key=self.KEY)
        self.assertNotIn(self.KEY, out.getvalue())


class PerModelFlagTests(unittest.TestCase):
    """--base-url and --api-key line up with --model the same way."""

    def test_single_entry_is_shared(self):
        self.assertEqual(qc.per_model("u", ["a", "b"], "--base-url", "URL"), ["u", "u"])

    def test_matching_count_is_kept_in_order(self):
        self.assertEqual(qc.per_model("u1,u2", ["a", "b"], "--base-url", "URL"), ["u1", "u2"])

    def test_mismatched_count_is_an_error(self):
        with self.assertRaises(SystemExit):
            qc.per_model("u1,u2,u3", ["a", "b"], "--base-url", "URL")

    def test_whitespace_is_trimmed(self):
        self.assertEqual(qc.per_model(" u1 , u2 ", ["a", "b"], "--api-key", "key"), ["u1", "u2"])


class ApiKeyCliTests(unittest.TestCase):
    """The CLI resolves the key from --api-key, then $OPENAI_API_KEY, then not at all."""

    def fixtures(self):
        return [{"id": "c1", "category": "gsm8k", "prompt": "q", "expected": "1", "max_tokens": 16}]

    def run_main(self, out_dir, *extra, env=None):
        seen = {}

        def fake_request(base_url, model, case, **kwargs):
            seen["api_key"] = kwargs.get("api_key")
            return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}]}

        argv = ["prog", "--model", "m", "--output-dir", str(out_dir), *extra]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(qc, "request", side_effect=fake_request), \
                mock.patch.dict(qc.os.environ, env or {}, clear=False):
            qc.main(
                lambda: self.fixtures(), timeout=1, retry_delay=0,
                tolerate_errors=True, description="t", profile="p",
            )
        return seen["api_key"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out_dir = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(lambda: qc.os.environ.pop("OPENAI_API_KEY", None))
        qc.os.environ.pop("OPENAI_API_KEY", None)

    def test_flag_wins(self):
        key = self.run_main(self.out_dir, "--api-key", "sk-flag", env={"OPENAI_API_KEY": "sk-env"})
        self.assertEqual(key, "sk-flag")

    def test_environment_is_the_fallback(self):
        self.assertEqual(self.run_main(self.out_dir, env={"OPENAI_API_KEY": "sk-env"}), "sk-env")

    def test_no_key_configured_sends_none(self):
        self.assertIsNone(self.run_main(self.out_dir))

    def test_empty_environment_value_sends_none(self):
        self.assertIsNone(self.run_main(self.out_dir, env={"OPENAI_API_KEY": "   "}))

    def test_key_count_must_match_the_model_count(self):
        argv = ["prog", "--model", "a,b", "--output-dir", str(self.out_dir), "--api-key", "k1,k2,k3"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit):
                qc.main(
                    lambda: self.fixtures(), timeout=1, retry_delay=0,
                    tolerate_errors=True, description="t", profile="p",
                )

    def test_interleave_gives_each_model_its_own_key(self):
        qc.write_fixtures(self.out_dir / "fixtures.json", self.fixtures(), "p")
        seen = {}

        def fake_request(base_url, model, case, **kwargs):
            seen[model] = kwargs.get("api_key")
            return {"choices": [{"message": {"content": "#### 1"}, "finish_reason": "stop"}]}

        argv = [
            "prog", "--model", "a,b", "--base-url", "http://a,http://b",
            "--output-dir", str(self.out_dir), "--api-key", "k-a,k-b",
        ]
        with mock.patch.object(sys, "argv", argv), mock.patch.object(qc, "request", side_effect=fake_request):
            qc.main(
                lambda: self.fixtures(), timeout=1, retry_delay=0,
                tolerate_errors=True, description="t", profile="p",
            )
        self.assertEqual(seen, {"a": "k-a", "b": "k-b"})


class SharedCaseBuilderTests(unittest.TestCase):
    """The wording of a question is one fact, shared by both profiles.

    It used to be copy-pasted into each entry point, so editing one copy would move that
    profile's cases and leave the other's alone: the two harnesses would stop being
    comparable while every test still passed.
    """

    GSM_ROW = {"question": "Janet has 3 apples and buys 4 more. How many?", "answer": "3+4=7\n#### 7"}
    ARC_ROW = {
        "question": "Which planet is largest?",
        "choices": {"label": ["A", "B", "C", "D"], "text": ["Mars", "Jupiter", "Venus", "Earth"]},
        "answerKey": "b",
    }
    MMLU_ROW = {"question": "What is 2+2?", "choices": ["3", "4", "5", "6"], "answer": 1}

    def fake_rows(self, dataset, config, split, offsets):
        row = self.GSM_ROW if "gsm8k" in dataset else self.ARC_ROW if "ai2_arc" in dataset else self.MMLU_ROW
        return [dict(row) for _ in offsets]

    def build(self):
        # The import sits *inside* the mock on purpose. Nothing else in the suite imports the
        # entry points, so this is the first import of them, and a dataset fetch that ever
        # moved to module level would run right here -- caught by the fake rather than
        # reaching the network from a test run.
        with mock.patch.object(qc, "get_rows", side_effect=self.fake_rows):
            import test_quality_expanded_1 as expanded
            import test_quality_express_1 as express

            return express.make_cases(), expanded.make_cases()

    def test_gsm8k_prompt_wording(self):
        self.assertEqual(qc.gsm8k_prompt("2+2?"), "2+2?\n\nEnd the final answer with `#### <number>`.")

    def test_gsm8k_expected_takes_the_marker_tail(self):
        self.assertEqual(qc.gsm8k_expected("a=1\nb=2\n#### 18"), "18")
        self.assertEqual(qc.gsm8k_expected("#### 1,000"), "1,000")

    def test_choice_prompt_renders_one_option_per_line(self):
        self.assertEqual(
            qc.choice_prompt("Which?", ["A", "B"], ["x", "y"]),
            "Which?\n\nA. x\nB. y\n\nChoose one option. End with `Answer: X`.",
        )

    def test_letter_labels_for_bare_choice_lists(self):
        self.assertEqual(qc.letter_labels(4), ["A", "B", "C", "D"])

    def test_the_instruction_suffixes_match_what_the_scorers_look_for(self):
        # The prompt asks for `#### n` and `Answer: X`; the scorer must accept exactly that.
        self.assertTrue(qc.score({"category": "gsm8k", "expected": "7"}, "#### 7")[0])
        self.assertTrue(qc.score({"category": "arc_challenge", "expected": "B"}, "Answer: B")[0])
        self.assertIn("#### <number>", qc.GSM8K_INSTRUCTION)
        self.assertIn("Answer: X", qc.CHOICE_INSTRUCTION)

    def test_both_profiles_ask_the_same_question_for_the_same_row(self):
        express_cases, expanded_cases = self.build()
        for category in ("gsm8k", "arc_challenge"):
            first = next(c for c in express_cases if c["category"] == category)
            second = next(c for c in expanded_cases if c["category"] == category)
            with self.subTest(category=category):
                self.assertEqual(first["prompt"], second["prompt"])
                self.assertEqual(first["expected"], second["expected"])

    def test_profile_sizes_stay_as_documented(self):
        express_cases, expanded_cases = self.build()
        self.assertEqual(len(express_cases), 72)
        self.assertEqual(len(expanded_cases), 266)

    def test_mmlu_options_are_lettered_from_a_bare_list(self):
        _, expanded_cases = self.build()
        mmlu = next(c for c in expanded_cases if c["category"] == "mmlu")
        self.assertIn("A. 3\nB. 4\nC. 5\nD. 6", mmlu["prompt"])
        self.assertEqual(mmlu["expected"], "B")


if __name__ == "__main__":
    unittest.main()
