"""
Tests for eval/trajectory.py

Run with:
    python -m pytest eval/test_trajectory.py -v
"""
import json, tempfile
from pathlib import Path
from eval.trajectory import classify_error, classify_failure_location, parse_trajectory, score_trajectories


# ---------------------------------------------------------------------------
# classify_error
# ---------------------------------------------------------------------------

def test_classify_name_error():
    assert classify_error("NameError: name 'df' is not defined") == "name_error"

def test_classify_module_not_found():
    assert classify_error("ModuleNotFoundError: No module named 'scipy'") == "module_not_found"

def test_classify_timeout():
    assert classify_error("[error] Code timed out after 30 seconds.") == "timeout"

def test_classify_stderr_with_traceback():
    # [stderr] with a Traceback is a real error
    assert classify_error("[stderr]\nTraceback (most recent call last):\n  ...") == "other_error"

def test_classify_stderr_with_named_error():
    assert classify_error("[stderr]\nRuntimeError: something went wrong") == "other_error"

def test_classify_stderr_warning_only_not_an_error():
    # sklearn / numpy routinely print RuntimeWarnings to stderr — not a failure
    warn = "[stderr]\n/path/to/sklearn.py:279: RuntimeWarning: overflow encountered in matmul"
    assert classify_error(warn) is None

def test_classify_clean_output():
    assert classify_error("4.0\n") is None
    assert classify_error("Mean: 34.65") is None


# ---------------------------------------------------------------------------
# classify_failure_location — precedence rules
# ---------------------------------------------------------------------------

def _make_msgs_with_tool_results(results: list) -> list:
    """Helper: build a minimal message list with given tool result strings."""
    msgs = [{"role": "user", "content": "task"}]
    for r in results:
        msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})
        msgs.append({"role": "user",      "content": [{"type": "tool_result", "content": r}]})
    # Last assistant turn ends on tool_use (no text) — simulates max_iterations
    msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})
    return msgs

def test_precedence_format_only_over_stuck():
    # Even if trajectory looks stuck, format_not_followed takes priority
    msgs = _make_msgs_with_tool_results(["NameError", "NameError", "NameError"])
    assert classify_failure_location(msgs, failure_mode="format_not_followed") == "format_only"

def test_precedence_format_only_partial():
    msgs = _make_msgs_with_tool_results(["ok", "ok"])
    assert classify_failure_location(msgs, failure_mode="partial_format") == "format_only"

def test_precedence_wrong_final_answer():
    msgs = _make_msgs_with_tool_results(["ok", "ok"])
    assert classify_failure_location(msgs, failure_mode="wrong_value") == "wrong_final_answer"

NE = "NameError: name 'df' is not defined"   # realistic error string

def test_stuck_heuristic_two_of_three_errors():
    # Last 3 tool results: 2 errors → stuck
    results = ["ok", "ok", NE, NE, "ok"]
    msgs = _make_msgs_with_tool_results(results)
    assert classify_failure_location(msgs, failure_mode="max_iterations") == "stuck"

def test_stuck_heuristic_three_of_three_errors():
    results = ["ok", NE, NE, NE]
    msgs = _make_msgs_with_tool_results(results)
    assert classify_failure_location(msgs, failure_mode="max_iterations") == "stuck"

def test_cut_short_heuristic_one_of_three_errors():
    # Last 3 tool results: only 1 error → cut_short
    results = ["ok", "ok", "ok", NE, "ok", "ok"]
    msgs = _make_msgs_with_tool_results(results)
    assert classify_failure_location(msgs, failure_mode="max_iterations") == "cut_short"

def test_cut_short_heuristic_no_errors():
    results = ["ok", "ok", "ok", "ok", "ok"]
    msgs = _make_msgs_with_tool_results(results)
    assert classify_failure_location(msgs, failure_mode="max_iterations") == "cut_short"


# ---------------------------------------------------------------------------
# parse_trajectory
# ---------------------------------------------------------------------------

def _passed_trajectory():
    """Minimal passing trajectory: 2 tool calls, no errors, ends with text."""
    return [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "tool_use"}]},
        {"role": "user",      "content": [{"type": "tool_result", "content": "df loaded"}]},
        {"role": "assistant", "content": [{"type": "tool_use"}]},
        {"role": "user",      "content": [{"type": "tool_result", "content": "Mean: 34.65"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "@mean[34.65]"}]},
    ]

def _error_then_recover_trajectory():
    """Agent hits a NameError then fixes it and finishes."""
    return [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "tool_use"}]},
        {"role": "user", "content": [{"type": "tool_result",
                                      "content": "NameError: name 'df' is not defined"}]},
        {"role": "assistant", "content": [{"type": "tool_use"}]},
        {"role": "user", "content": [{"type": "tool_result", "content": "Mean: 34.65"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "@mean[34.65]"}]},
    ]

def _max_iterations_stuck_trajectory():
    """Last 3 tool results all have NameErrors — agent was stuck."""
    msgs = [{"role": "user", "content": "task"}]
    ne = "NameError: name 'df' is not defined"
    for content in ["ok", ne, ne, ne]:
        msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})
        msgs.append({"role": "user", "content": [{"type": "tool_result", "content": content}]})
    # Ends on a tool_use — hit max
    msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})
    return msgs

def test_parse_passed_task():
    r = parse_trajectory(_passed_trajectory())
    assert r["n_tool_calls"] == 2
    assert r["n_errors"] == 0
    assert r["hit_max_iterations"] is False
    assert r["error_recovery"] is False
    assert r["failure_location"] is None

def test_parse_error_recovery():
    r = parse_trajectory(_error_then_recover_trajectory())
    assert r["n_tool_calls"] == 2
    assert r["n_errors"] == 1
    assert r["error_types"] == ["name_error"]
    assert r["error_recovery"] is True
    assert r["hit_max_iterations"] is False
    assert r["failure_location"] is None  # task passed

def test_parse_max_iterations_stuck():
    r = parse_trajectory(_max_iterations_stuck_trajectory(),
                         failure_mode="max_iterations")
    assert r["hit_max_iterations"] is True
    assert r["failure_location"] == "stuck"

def test_parse_error_rate():
    r = parse_trajectory(_error_then_recover_trajectory())
    assert r["error_rate"] == 0.5   # 1 error out of 2 calls

# ---------------------------------------------------------------------------
# score_trajectories — integration test
# This was the missing test that allowed a bug where failure_mode was never
# populated from the verifiable eval (score_trajectories loaded level from
# the results file but left failure_mode=None for all tasks).
# ---------------------------------------------------------------------------

def test_score_trajectories_populates_failure_location(tmp_path):
    """failure_mode from verifiable eval must flow into failure_location."""
    # Build a minimal trajectory: agent ran out of steps (hit max, last 3 had errors)
    ne = "NameError: name 'df' is not defined"
    traj_msgs = [{"role": "user", "content": "task"}]
    for content in ["ok", ne, ne, ne]:
        traj_msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})
        traj_msgs.append({"role": "user", "content": [{"type": "tool_result", "content": content}]})
    traj_msgs.append({"role": "assistant", "content": [{"type": "tool_use"}]})  # no text — hit max

    traj_file = tmp_path / "task_42.json"
    traj_file.write_text(json.dumps({"task_id": 42, "question": "q", "trajectory": traj_msgs}))

    # Build a matching results JSON with failure_mode that verifiable eval will compute
    results = {
        "results": [{
            "task_id": 42, "level": "hard",
            "agent_answer": "[agent hit max iterations without finishing]",
            "ground_truth": [["mean", "1.0"]],
            "iterations": 5, "cost_usd": 0.05,
            "format": "@mean[value]", "question": "q",
            "constraints": "", "concepts": [], "file_name": "test.csv",
        }]
    }
    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps(results))

    summary = score_trajectories(tmp_path, results_file)
    task = next(t for t in summary["tasks"] if t["task_id"] == 42)

    # failure_location must be set (not None) — this is what the bug left unset
    assert task["failure_location"] is not None
    # Last 3 tool results all had NameErrors → stuck
    assert task["failure_location"] == "stuck"


def test_parse_no_tool_calls():
    # Edge case: agent gives a direct text answer with no tool calls
    msgs = [
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": [{"type": "text", "text": "answer"}]},
    ]
    r = parse_trajectory(msgs)
    assert r["n_tool_calls"] == 0
    assert r["error_rate"] == 0
