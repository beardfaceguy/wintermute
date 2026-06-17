"""
Tests for MC answer parsers in eval/benchmarks/.

These are pure functions with no external dependencies. The Claude/GPQA bug
(verbose response not extracting a letter) is the canonical failure mode these
tests guard against.
"""


# ---------------------------------------------------------------------------
# GPQA / MMLU parser
# ---------------------------------------------------------------------------

from eval.benchmarks.gpqa import _parse as gpqa_parse
from eval.benchmarks.mmlu import _parse_answer as mmlu_parse


class TestGPQAParser:
    def test_single_letter(self):
        assert gpqa_parse("A") == "A"
        assert gpqa_parse("B") == "B"
        assert gpqa_parse("C") == "C"
        assert gpqa_parse("D") == "D"

    def test_single_letter_lowercase(self):
        assert gpqa_parse("b") == "B"

    def test_letter_with_trailing_newline(self):
        assert gpqa_parse("A\n") == "A"

    def test_verbose_answer_is_phrase(self):
        # The Claude bug: model writes prose instead of a single letter
        assert gpqa_parse("The answer is B.") == "B"
        assert gpqa_parse("The answer is: C") == "C"
        assert gpqa_parse("I believe the answer is A here.") == "A"

    def test_verbose_answer_is_correct(self):
        assert gpqa_parse("B is correct.") == "B"
        assert gpqa_parse("D is correct") == "D"

    def test_bolded_answer(self):
        assert gpqa_parse("The answer is **B**") == "B"

    def test_last_letter_fallback(self):
        # Long response ending with a letter
        assert (
            gpqa_parse("Using the uncertainty principle, we compute the energy width. Therefore D.")
            == "D"
        )

    def test_empty_response(self):
        assert gpqa_parse("") is None

    def test_no_letter(self):
        assert gpqa_parse("I don't know") is None

    def test_full_sentence_no_answer_phrase(self):
        # Long text with no A/B/C/D at all
        assert gpqa_parse("The formula is E=mc squared.") is None


class TestMMLUParser:
    def test_single_letter(self):
        assert mmlu_parse("A") == "A"
        assert mmlu_parse("D") == "D"

    def test_letter_with_punctuation(self):
        assert mmlu_parse("B.") == "B"
        assert mmlu_parse("(C)") == "C"

    def test_verbose(self):
        assert mmlu_parse("The correct answer is C.") == "C"

    def test_empty(self):
        assert mmlu_parse("") is None


# ---------------------------------------------------------------------------
# ARC parser
# ---------------------------------------------------------------------------

from eval.benchmarks.arc import _parse as arc_parse


class TestARCParser:
    def test_single_letter(self):
        assert arc_parse("A") == "A"
        assert arc_parse("C") == "C"

    def test_verbose(self):
        assert arc_parse("The answer is B.") == "B"

    def test_none_on_no_match(self):
        assert arc_parse("I'm not sure") is None


# ---------------------------------------------------------------------------
# HellaSwag parser (numeric 0-3)
# ---------------------------------------------------------------------------

from eval.benchmarks.hellaswag import _parse as hellaswag_parse


class TestHellaSwagParser:
    def test_single_digit(self):
        assert hellaswag_parse("0") == 0
        assert hellaswag_parse("3") == 3

    def test_digit_in_sentence(self):
        assert hellaswag_parse("I choose option 2.") == 2

    def test_none_on_no_match(self):
        assert hellaswag_parse("no idea") is None

    def test_out_of_range_ignored(self):
        # "4" is not a valid HellaSwag option
        assert hellaswag_parse("4") is None


# ---------------------------------------------------------------------------
# WinoGrande parser (1 or 2)
# ---------------------------------------------------------------------------

from eval.benchmarks.winogrande import _parse as wino_parse


class TestWinoGrandeParser:
    def test_option_one(self):
        assert wino_parse("1") == "1"

    def test_option_two(self):
        assert wino_parse("2") == "2"

    def test_in_sentence(self):
        assert wino_parse("Option 1 is correct.") == "1"

    def test_none_on_no_match(self):
        assert wino_parse("neither") is None


# ---------------------------------------------------------------------------
# GSM8K answer extractor
# ---------------------------------------------------------------------------

from eval.benchmarks.gsm8k import _extract_answer, _normalize


class TestGSM8KExtractor:
    def test_hashtag_format(self):
        assert _extract_answer("Step 1: ... Step 2: ... #### 42") == "42"

    def test_hashtag_with_comma(self):
        assert _extract_answer("#### 1,234") == "1234"

    def test_fallback_last_number(self):
        assert _extract_answer("The total cost is 99.50") == "99.50"

    def test_empty(self):
        assert _extract_answer("no numbers here") is None

    def test_normalize_int(self):
        assert _normalize("42") == _normalize("42.0")

    def test_normalize_comma(self):
        assert _normalize("1,234") == _normalize("1234")


# ---------------------------------------------------------------------------
# MATH boxed extractor
# ---------------------------------------------------------------------------

from eval.benchmarks.math_bench import _extract as math_extract
from eval.benchmarks.math_bench import _normalize as math_normalize


class TestMATHExtractor:
    def test_simple_boxed(self):
        assert math_extract(r"The answer is \boxed{42}") == "42"

    def test_expression_boxed(self):
        # Regex [^}]+ stops at first } so nested braces only capture partial
        assert math_extract(r"\boxed{42}") == "42"
        assert math_extract(r"\boxed{x+1}") == "x+1"

    def test_no_boxed(self):
        assert math_extract("just a number 42") is None

    def test_normalize_strips_dollar(self):
        assert math_normalize("$42$") == math_normalize("42")

    def test_normalize_float(self):
        assert math_normalize("42.0") == math_normalize("42")
