"""Regression tests for the shared scoring, extraction and summary logic.

Run from the repository root with:  python3 -m unittest discover -v
"""

import unittest

import test_quality_expanded_1 as expanded
import test_quality_express_1 as express

MODULES = (express, expanded)


def answer_pattern(module):
    return module.ARC_ANSWER if hasattr(module, "ARC_ANSWER") else module.ANSWER_CHOICE


class AnswerKeyTests(unittest.TestCase):
    """Multiple-choice extraction, including the trailing-period cases that used to fail."""

    def test_trailing_period_counts(self):
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("The reasoning is done. Answer: A.")[-1], "A")

    def test_no_period_still_counts(self):
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("The reasoning is done. Answer: A")[-1], "A")

    def test_lowercase_key_with_period(self):
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("The answer is b.")[-1].upper(), "B")

    def test_digit_key_with_period(self):
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("option 3. Answer: 3.")[-1], "3")

    def test_last_key_wins(self):
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("Answer: A. Wait, no. Answer: B.")[-1], "B")

    def test_word_after_letter_is_not_a_key(self):
        # "Apple" starts with a valid option letter but is a word, not a key.
        for module in MODULES:
            self.assertEqual(answer_pattern(module).findall("The answer is Apple."), [])

    def test_score_accepts_period_form(self):
        case = {"id": "a1", "category": "arc_challenge", "expected": "A"}
        for module in MODULES:
            correct, actual = module.score(case, "Reasoning... Answer: A.")
            self.assertTrue(correct, msg=f"{module.__name__}: expected 'Answer: A.' to be correct")
            self.assertEqual(actual, "A")


class JsonScoringTests(unittest.TestCase):
    """JSON answers compare by value: numeric strings match numbers."""

    def test_numeric_string_matches_number(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"seconds": 8229}}
        for module in MODULES:
            self.assertTrue(module.score(case, '{"seconds": "8229"}')[0])

    def test_numeric_strings_in_list_match(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"values": [3, 7, 2, 5]}}
        for module in MODULES:
            self.assertTrue(module.score(case, '{"values": ["3", "7", "2", "5"]}')[0])

    def test_float_string_matches_number(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"mean": 7.5}}
        for module in MODULES:
            self.assertTrue(module.score(case, '{"mean": "7.5"}')[0])

    def test_non_numeric_string_still_fails(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"seconds": 8229}}
        for module in MODULES:
            self.assertFalse(module.score(case, '{"seconds": "8229 seconds"}')[0])

    def test_wrong_structure_still_fails(self):
        case = {"id": "x", "category": "instruction_json", "expected": {"ordered": ["fig", "pear", "plum", "apple"]}}
        for module in MODULES:
            self.assertFalse(module.score(case, '{"ordered": ["pear", "fig", "plum", "apple"]}')[0])


class ExtractionTests(unittest.TestCase):
    """Existing extraction behaviour the README documents, pinned as regression guards."""

    def test_closed_think_keeps_the_tail(self):
        for module in MODULES:
            self.assertEqual(module.strip_reasoning("<think>hmm 41</think>42"), "42")

    def test_unclosed_think_yields_nothing(self):
        for module in MODULES:
            self.assertEqual(module.strip_reasoning("<think>hmm 42"), "")

    def test_last_balanced_object_wins(self):
        text = 'Thinking... Provide {"seconds": 1}. Real: {"seconds": 8229}'
        for module in MODULES:
            self.assertEqual(module.parse_json_answer(text), {"seconds": 8229})

    def test_code_fence_is_ignored(self):
        for module in MODULES:
            self.assertEqual(module.parse_json_answer('```json\n{"value": 44}\n```'), {"value": 44})

    def test_numbers_compare_by_value(self):
        for module in MODULES:
            self.assertTrue(module.numbers_equal("6.00", "6"))
            self.assertTrue(module.numbers_equal("1/2", "0.5"))
            self.assertFalse(module.numbers_equal("5", "6"))


class SummaryTests(unittest.TestCase):
    """A failed request is counted apart, not as a wrong or empty answer."""

    def make_records(self):
        return [
            {"id": "a", "category": "gsm8k", "correct": True, "truncated": False, "answer_empty": False, "timings": {}, "elapsed_seconds": 1.0},
            {"id": "b", "category": "gsm8k", "correct": False, "truncated": False, "answer_empty": False, "timings": {}, "elapsed_seconds": 2.0},
            {"id": "c", "category": "gsm8k", "correct": False, "truncated": False, "answer_empty": True, "timings": {}, "elapsed_seconds": 3.0, "error": "RuntimeError('boom')"},
        ]

    def test_failed_request_is_counted_apart(self):
        for module in MODULES:
            summary = module.summarize(self.make_records())
            self.assertEqual(summary["total"], 3)
            self.assertAlmostEqual(summary["accuracy"], 1 / 3)
            self.assertEqual(summary["errors"], 1)
            self.assertAlmostEqual(summary["accuracy_excluding_errors"], 0.5)
            self.assertEqual(summary["answer_empty"], 0)
            self.assertEqual(summary["wrong_and_complete"], 1)
            self.assertEqual(summary["wrong_and_truncated"], 0)

    def test_no_errors_means_unchanged_semantics(self):
        for module in MODULES:
            records = self.make_records()[:2]
            summary = module.summarize(records)
            self.assertEqual(summary["errors"], 0)
            self.assertAlmostEqual(summary["accuracy_excluding_errors"], 0.5)
            self.assertEqual(summary["answer_empty"], 0)


class FilenameTests(unittest.TestCase):
    def test_model_name_is_a_single_path_component(self):
        for module in MODULES:
            self.assertEqual(module.model_filename("org/model q8-0"), "org_model_q8-0")
            self.assertEqual(module.model_filename("plain-model"), "plain-model")


if __name__ == "__main__":
    unittest.main()
