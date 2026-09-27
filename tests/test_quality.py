"""Regression tests for the shared harness logic in quality_common.py.

Run from the repository root with:  python3 -m unittest discover -v
"""

import io
import json
import sys
import tempfile
import time
import unittest
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
            {"id": "a", "category": "gsm8k", "correct": True, "truncated": False, "answer_empty": False, "timings": {}, "elapsed_seconds": 1.0},
            {"id": "b", "category": "gsm8k", "correct": False, "truncated": False, "answer_empty": False, "timings": {}, "elapsed_seconds": 2.0},
            {"id": "c", "category": "gsm8k", "correct": False, "truncated": False, "answer_empty": True, "timings": {}, "elapsed_seconds": 3.0, "error": "RuntimeError('boom')"},
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
        record = {"id": "a", "category": "gsm8k", "correct": True, "truncated": False, "answer_empty": False, "timings": {}}
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
        text = cq.render("model-a", self.make_summary("model-a", 0.8, 50.0), "model-b", self.make_summary("model-b", 0.7, 40.0))
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
        data = cq.as_json("model-a", self.make_summary("model-a", 0.8, 50.0), "model-b", self.make_summary("model-b", 0.7, 40.0))
        self.assertAlmostEqual(data["delta_accuracy"], -0.1)
        self.assertAlmostEqual(data["by_category"]["gsm8k"]["delta"], -0.1)
        self.assertEqual(data["a"]["model"], "model-a")


class RequestPayloadTests(unittest.TestCase):
    """The optional sampler fields are sent by default and gateable for strict servers."""

    CASE = {"id": "c1", "category": "arc_challenge", "expected": "A", "prompt": "q", "max_tokens": 32}

    def capture_payload(self, **kwargs):
        captured = {}

        def fake_post(url, data, timeout=None):
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
        with mock.patch.object(qc, "request", return_value={"choices": [{"message": {"content": "Answer: A."}, "finish_reason": "stop"}]}) as fake:
            qc.run_cases("http://x", "m", cases, io.StringIO(), send_seed=False, send_reasoning_effort=False)
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
        with mock.patch.object(qc, "request", side_effect=lambda base_url, model, case, **kw: self.fake_request(calls, base_url, model, case, **kw)):
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
        with mock.patch.object(qc, "request", side_effect=lambda base_url, model, case, **kw: self.fake_request(calls, base_url, model, case, **kw)):
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
        self.assertEqual([c["id"] for c in qc.select_cases(self.cases())], ["g0", "g1", "g2", "n_16k", "n_64k", "if_01"])

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
                '{"id": "c1", "category": "arc_challenge", "correct": true, "truncated": false, "answer_empty": false, "timings": {}, "elapsed_seconds": 1.0}\n'
                '{"id": "c2", "correct": fa'
            )

            def fake_request(base_url, model, case, **kwargs):
                if case["id"] == "c1":
                    raise AssertionError("c1 must be skipped on resume")
                return {
                    "choices": [{"message": {"content": "Answer: B.", "reasoning_content": None}, "finish_reason": "stop"}],
                    "usage": None,
                    "timings": {},
                }

            with mock.patch.object(sys, "argv", ["prog", "--model", "m", "--output-dir", str(out_dir), "--resume"]), mock.patch.object(
                qc, "request", side_effect=fake_request
            ):
                qc.main(lambda: self.fixtures(), timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

            lines = [json.loads(line) for line in (out_dir / "results-m.jsonl").read_text().splitlines() if line.strip()]
            self.assertEqual([r["id"] for r in lines], ["c1", "c2"])
            summary = json.loads((out_dir / "summary-m.json").read_text())
            self.assertEqual(summary["total"], 2)
            self.assertEqual(summary["correct"], 2)

    def test_resume_rejects_make_fixtures(self):
        with mock.patch.object(sys, "argv", ["prog", "--model", "m", "--output-dir", "/tmp/none", "--resume", "--make-fixtures"]):
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
                    "choices": [{"message": {"content": "Answer: A. Key: K00000001Z", "reasoning_content": None}, "finish_reason": "stop"}],
                    "usage": None,
                    "timings": {},
                }

            with mock.patch.object(sys, "argv", ["prog", "--model", "m", "--output-dir", str(out_dir), "--categories", "long_context"]), mock.patch.object(
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
                '{"id": "c1", "category": "arc_challenge", "correct": true, "truncated": false, "answer_empty": false, "timings": {}, "elapsed_seconds": 1.0}\n'
            )

            def fake_request(base_url, model, case, **kwargs):
                self.assertEqual(case["id"], "c2")
                return {
                    "choices": [{"message": {"content": "Answer: B.", "reasoning_content": None}, "finish_reason": "stop"}],
                    "usage": None,
                    "timings": {},
                }

            with mock.patch.object(sys, "argv", ["prog", "--model", "m", "--output-dir", str(out_dir), "--resume", "--categories", "arc_challenge"]), mock.patch.object(
                qc, "request", side_effect=fake_request
            ):
                qc.main(lambda: self.fixtures(), timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")

            lines = [json.loads(line) for line in (out_dir / "results-m.jsonl").read_text().splitlines() if line.strip()]
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
            self.assertEqual(calls, [("http://a", "a", "c1"), ("http://b", "b", "c1"), ("http://a", "a", "c2"), ("http://b", "b", "c2")])
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
            with mock.patch.object(
                sys, "argv", ["prog", "--model", "m", "--output-dir", str(tmp), "--no-seed", "--no-reasoning-effort"]
            ), mock.patch.object(qc, "request", return_value={"choices": [{"message": {"content": "Answer: A."}, "finish_reason": "stop"}]}) as fake:
                qc.main(lambda: self.fixtures(), timeout=1, retry_delay=0.01, tolerate_errors=True, description="t")
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


if __name__ == "__main__":
    unittest.main()
