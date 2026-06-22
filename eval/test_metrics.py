"""
Tests for eval/metrics.py

Run with:
    python -m pytest eval/test_metrics.py -v
"""
from eval.metrics import parse_answer, values_match, score_task


# ---------------------------------------------------------------------------
# parse_answer
# ---------------------------------------------------------------------------

def test_parse_single_tag():
    result = parse_answer("The answer is @mean_fare[34.65]")
    assert result == {"mean_fare": "34.65"}


def test_parse_multiple_tags():
    result = parse_answer("@mean[1.0] and @sd[0.5]")
    assert result == {"mean": "1.0", "sd": "0.5"}


def test_parse_no_tags():
    assert parse_answer("No tags here at all") == {}


def test_parse_tag_in_prose():
    # Agent wraps tag in explanation — should still extract it
    result = parse_answer("After computing, the result is @correlation_coefficient[0.21] as required.")
    assert result == {"correlation_coefficient": "0.21"}


def test_parse_strips_whitespace():
    result = parse_answer("@mean_fare[ 34.65 ]")
    assert result["mean_fare"] == "34.65"


def test_parse_duplicate_tag_last_wins():
    # If agent repeats a tag, last value wins
    result = parse_answer("@mean_fare[34.65] ... actually @mean_fare[34.66]")
    assert result["mean_fare"] == "34.66"


def test_parse_string_value():
    result = parse_answer("@relationship_type[linear]")
    assert result == {"relationship_type": "linear"}


# ---------------------------------------------------------------------------
# values_match
# ---------------------------------------------------------------------------

def test_exact_numeric_match():
    assert values_match("34.65", "34.65") is True


def test_trailing_zero_numeric_match():
    # "34.650" and "34.65" should be equal numerically
    assert values_match("34.650", "34.65") is True


def test_numeric_mismatch():
    assert values_match("34.64", "34.65") is False


def test_string_match_case_insensitive():
    assert values_match("linear", "Linear") is True
    assert values_match("NONLINEAR", "nonlinear") is True


def test_string_mismatch():
    assert values_match("linear", "nonlinear") is False


def test_integer_matches_float():
    # Ground truth "2081990" should match extracted "2081990.0"
    assert values_match("2081990.0", "2081990") is True

def test_comma_separated_numeric_list():
    # DABench sometimes puts multi-value answers in one tag: "0.00, 1.00, 0.06"
    assert values_match("0.0, 1.0, 0.0629", "0.00, 1.00, 0.0629") is True

def test_comma_separated_integer_list():
    assert values_match("314, 577", "314, 577") is True

def test_comma_separated_wrong_length():
    assert values_match("1.0, 2.0", "1.0, 2.0, 3.0") is False

def test_comma_separated_wrong_value():
    assert values_match("0.0, 2.0, 0.0629", "0.00, 1.00, 0.0629") is False


# ---------------------------------------------------------------------------
# score_task
# ---------------------------------------------------------------------------

def test_score_pass_single_answer():
    result = score_task("@mean_fare[34.65]", [["mean_fare", "34.65"]])
    assert result["passed"] is True
    assert result["failure_mode"] is None


def test_score_pass_multi_answer():
    answer = "@mean[1.00] and @sd[0.50]"
    gt = [["mean", "1.0"], ["sd", "0.5"]]
    result = score_task(answer, gt)
    assert result["passed"] is True


def test_score_fail_format_not_followed():
    # Agent gave the right number in prose but no @tag
    result = score_task("The mean fare is 34.65", [["mean_fare", "34.65"]])
    assert result["passed"] is False
    assert result["failure_mode"] == "format_not_followed"


def test_score_fail_wrong_value():
    result = score_task("@mean_fare[99.99]", [["mean_fare", "34.65"]])
    assert result["passed"] is False
    assert result["failure_mode"] == "wrong_value"


def test_score_fail_max_iterations():
    answer = "[agent hit max iterations without finishing]"
    result = score_task(answer, [["mean_fare", "34.65"]])
    assert result["passed"] is False
    assert result["failure_mode"] == "max_iterations"


def test_score_fail_partial_format():
    # Agent produced one tag but not the other
    answer = "@mean[1.00]"
    gt = [["mean", "1.0"], ["sd", "0.5"]]
    result = score_task(answer, gt)
    assert result["passed"] is False
    assert result["failure_mode"] == "partial_format"


def test_score_all_or_nothing_one_wrong():
    # Both tags present but one value wrong — entire task fails
    answer = "@mean[1.00] @sd[9.99]"
    gt = [["mean", "1.0"], ["sd", "0.5"]]
    result = score_task(answer, gt)
    assert result["passed"] is False
    assert result["failure_mode"] == "wrong_value"


def test_score_details_populated():
    answer = "@mean_fare[34.65]"
    gt = [["mean_fare", "34.65"]]
    result = score_task(answer, gt)
    assert len(result["details"]) == 1
    assert result["details"][0]["match"] is True
    assert result["details"][0]["variable"] == "mean_fare"
