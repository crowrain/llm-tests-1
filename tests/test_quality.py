"""Regression tests for the shared harness logic in quality_common.py.

Run from the repository root with:  python3 -m unittest discover -v
"""

import io
import json
import sys
import tempfile
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
