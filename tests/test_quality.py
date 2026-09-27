"""Regression tests for the shared harness logic in quality_common.py.

Run from the repository root with:  python3 -m unittest discover -v
"""

import io
import json
import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
