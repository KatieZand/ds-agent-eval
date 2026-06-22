"""
Trajectory evaluation for DABench agent runs.

Covers deterministic mechanics only: step counts, error types, error recovery,
and failure location. Path-quality assessment (approach soundness, whether the
agent reasoned vs. got lucky) requires judgment and is handled in the LLM-judge
phase.

Failure-location label precedence (each failed task gets exactly one label):
  1. format_only        — task completed but agent never produced @var[value] tags
  2. wrong_final_answer — task completed, tags present, values wrong
  3. stuck              — hit max iterations; last 3 tool results had >=2 errors
                          (heuristic: agent was looping or unable to make progress)
  4. cut_short          — hit max iterations; last 3 tool results had <2 errors
                          (heuristic: agent was computing meaningfully but ran out of steps)

Labels 1 and 2 come from verifiable eval (pass in failure_mode from metrics.py).
Labels 3 and 4 are determined from the trajectory alone.

Usage:
    python eval/trajectory.py <traj_dir> [results_json]
"""
import re
import json
import sys
from pathlib import Path
from collections import Counter


# ---------------------------------------------------------------------------
# Error detection
# ---------------------------------------------------------------------------

ERROR_PATTERNS = [
    ("timeout",          r"\[error\] Code timed out"),
    ("name_error",       r"NameError: name '"),
    ("module_not_found", r"ModuleNotFoundError: No module named"),
    ("attribute_error",  r"AttributeError:"),
    ("value_error",      r"ValueError:"),
    ("type_error",       r"TypeError:"),
    ("key_error",        r"KeyError:"),
    ("index_error",      r"IndexError:"),
    # [stderr] alone is not enough — sklearn/numpy routinely print RuntimeWarnings
    # to stderr without failing. We only count it as an error if it also contains
    # "Traceback" (an unhandled exception) or "Error:" (a named exception type).
    ("other_error",      r"\[stderr\][\s\S]*(Traceback|Error:)"),
]


def classify_error(tool_result_text: str):
    """
    Return error category string for a tool result, or None if no error.

    Stderr output that contains only warnings (e.g. sklearn RuntimeWarning)
    is NOT classified as an error — the code ran successfully despite the warning.

    >>> classify_error("4.0")
    >>> classify_error("NameError: name 'df' is not defined")
    'name_error'
    >>> classify_error("ModuleNotFoundError: No module named 'scipy'")
    'module_not_found'
    >>> classify_error("[error] Code timed out after 30 seconds.")
    'timeout'
    >>> classify_error("[stderr]\\nTraceback (most recent call last):\\n  ...")
    'other_error'
    >>> classify_error("[stderr]\\n/path/to/sklearn.py:10: RuntimeWarning: overflow")
    """
    for label, pattern in ERROR_PATTERNS:
        if re.search(pattern, tool_result_text):
            return label
    return None


# ---------------------------------------------------------------------------
# Failure location
# ---------------------------------------------------------------------------

def _last_n_tool_results(messages: list, n: int = 3) -> list:
    """Return the content strings of the last N tool_result blocks."""
    results = []
    for msg in messages:
        if msg["role"] == "user" and isinstance(msg["content"], list):
            for block in msg["content"]:
                if block.get("type") == "tool_result":
                    results.append(block.get("content", ""))
    return results[-n:]


def classify_failure_location(messages: list, failure_mode: str = None) -> str:
    """
    Assign one of four failure-location labels to a failed task.

    Precedence (see module docstring):
      1. format_only          if failure_mode in {format_not_followed, partial_format}
      2. wrong_final_answer   if failure_mode == wrong_value
      3. stuck / cut_short    determined from trajectory (max_iterations case)

    The stuck/cut_short split heuristic:
      Look at the last 3 tool results. If >=2 of them contain errors, label
      "stuck" (agent was looping or unable to progress). Otherwise "cut_short"
      (agent was still computing usefully but ran out of steps).
    """
    # Priority 1 & 2: cases where the agent finished but the answer was wrong
    if failure_mode in ("format_not_followed", "partial_format"):
        return "format_only"
    if failure_mode == "wrong_value":
        return "wrong_final_answer"

    # Priority 3 & 4: max_iterations — determine from trajectory shape
    last_results = _last_n_tool_results(messages, n=3)
    n_errors_in_last = sum(1 for r in last_results if classify_error(r) is not None)

    # Heuristic: >=2 of the last 3 tool results had errors → agent was stuck
    if n_errors_in_last >= 2:
        return "stuck"
    return "cut_short"


# ---------------------------------------------------------------------------
# Single-task trajectory parsing
# ---------------------------------------------------------------------------

def parse_trajectory(messages: list, failure_mode: str = None) -> dict:
    """
    Extract trajectory metrics from the message history for one task.

    Args:
        messages:     Full message list from the trajectory JSON.
        failure_mode: Failure category from verifiable eval (metrics.py),
                      or None if the task passed / no verifiable result available.

    Returns:
        n_tool_calls (int):        total code snippets executed
        n_errors (int):            tool results that returned an error
        error_rate (float):        n_errors / n_tool_calls
        error_types (list[str]):   error category per error occurrence
        error_recovery (bool):     had errors but still finished with an answer
        hit_max_iterations (bool): last assistant turn ended mid-tool-call
        failure_location (str|None): one of the four labels, or None if task passed

    >>> msgs = [
    ...   {"role": "user", "content": "task"},
    ...   {"role": "assistant", "content": [{"type": "tool_use"}]},
    ...   {"role": "user", "content": [{"type": "tool_result", "content": "4"}]},
    ...   {"role": "assistant", "content": [{"type": "text", "text": "@mean[4]"}]},
    ... ]
    >>> r = parse_trajectory(msgs)
    >>> r["n_tool_calls"], r["n_errors"], r["hit_max_iterations"]
    (1, 0, False)
    >>> r["failure_location"] is None
    True
    """
    n_tool_calls = 0
    n_errors     = 0
    error_types  = []

    for msg in messages:
        content = msg["content"]
        if not isinstance(content, list):
            continue

        if msg["role"] == "assistant":
            for block in content:
                if block.get("type") == "tool_use":
                    n_tool_calls += 1

        elif msg["role"] == "user":
            for block in content:
                if block.get("type") == "tool_result":
                    err = classify_error(block.get("content", ""))
                    if err:
                        n_errors += 1
                        error_types.append(err)

    # Did the agent finish with a text answer or end on a tool call?
    last_assistant = next(
        (msg for msg in reversed(messages) if msg["role"] == "assistant"), None
    )
    hit_max = False
    if last_assistant and isinstance(last_assistant["content"], list):
        blocks   = last_assistant["content"]
        has_text = any(b.get("type") == "text" for b in blocks)
        has_tool = any(b.get("type") == "tool_use" for b in blocks)
        hit_max  = has_tool and not has_text

    error_recovery    = (n_errors > 0) and (not hit_max)
    failure_location  = (
        classify_failure_location(messages, failure_mode)
        if (failure_mode is not None or hit_max)
        else None
    )

    return {
        "n_tool_calls":      n_tool_calls,
        "n_errors":          n_errors,
        "error_rate":        round(n_errors / n_tool_calls, 2) if n_tool_calls else 0,
        "error_types":       error_types,
        "error_recovery":    error_recovery,
        "hit_max_iterations": hit_max,
        "failure_location":  failure_location,
    }


# ---------------------------------------------------------------------------
# Scoring a full trajectory directory
# ---------------------------------------------------------------------------

def score_trajectories(traj_dir, results_path=None) -> dict:
    """
    Score all trajectory files in a directory.

    When results_path is provided, runs the verifiable eval to get failure_mode
    per task so failure_location can be set correctly for completed-but-failed tasks.
    """
    from eval.metrics import score_results as verifiable_score

    traj_dir   = Path(traj_dir)
    traj_files = sorted(traj_dir.glob("task_*.json"))

    # Load level and failure_mode from verifiable eval if a results file is provided
    task_meta = {}
    if results_path:
        raw = json.loads(Path(results_path).read_text())
        for r in raw["results"]:
            task_meta[r["task_id"]] = {"level": r["level"], "failure_mode": None}

        # Run verifiable eval to get failure_mode per task
        scored = verifiable_score(results_path)
        for t in scored["tasks"]:
            if t["task_id"] in task_meta:
                task_meta[t["task_id"]]["failure_mode"] = t["failure_mode"]

    tasks = []
    for f in traj_files:
        traj_data    = json.loads(f.read_text())
        task_id      = traj_data["task_id"]
        meta         = task_meta.get(task_id, {})
        failure_mode = meta.get("failure_mode")

        metrics = parse_trajectory(traj_data["trajectory"], failure_mode=failure_mode)
        tasks.append({
            "task_id": task_id,
            "level":   meta.get("level", "unknown"),
            **metrics,
        })

    # Step-count distributions per difficulty (min / median / max)
    def step_distribution(subset):
        counts = sorted(t["n_tool_calls"] for t in subset)
        if not counts:
            return {}
        n = len(counts)
        median = counts[n // 2] if n % 2 else (counts[n//2 - 1] + counts[n//2]) / 2
        return {
            "n":      n,
            "min":    counts[0],
            "median": median,
            "max":    counts[-1],
            "mean":   round(sum(counts) / n, 1),
        }

    def aggregate(subset):
        if not subset:
            return {}
        n = len(subset)
        return {
            "n_tasks":           n,
            "step_distribution": step_distribution(subset),
            "pct_with_errors":   round(sum(1 for t in subset if t["n_errors"] > 0) / n * 100, 1),
            "pct_recovered":     round(sum(1 for t in subset if t["error_recovery"]) / n * 100, 1),
            "pct_hit_max":       round(sum(1 for t in subset if t["hit_max_iterations"]) / n * 100, 1),
        }

    levels = ["easy", "medium", "hard"]
    return {
        "traj_dir":              str(traj_dir),
        "overall":               aggregate(tasks),
        "by_level":              {lvl: aggregate([t for t in tasks if t["level"] == lvl])
                                  for lvl in levels},
        "error_type_counts":     dict(Counter(
                                     et for t in tasks for et in t["error_types"]
                                 ).most_common()),
        "failure_location_counts": dict(Counter(
                                     t["failure_location"] for t in tasks
                                     if t["failure_location"]
                                 ).most_common()),
        "tasks":                 tasks,
    }


# ---------------------------------------------------------------------------
# CLI report
# ---------------------------------------------------------------------------

def print_report(summary: dict):
    print(f"\nTrajectory dir: {summary['traj_dir']}")
    print("=" * 60)

    o = summary["overall"]
    sd = o.get("step_distribution", {})
    print(f"\nOVERALL ({o['n_tasks']} tasks)")
    print(f"  Step counts  min={sd.get('min')}  median={sd.get('median')}  "
          f"max={sd.get('max')}  mean={sd.get('mean')}")
    print(f"  Tasks with errors:  {o['pct_with_errors']}%")
    print(f"  Error recovery:     {o['pct_recovered']}%")
    print(f"  Hit max iterations: {o['pct_hit_max']}%")

    print("\nStep-count distribution by difficulty:")
    for lvl in ["easy", "medium", "hard"]:
        s = summary["by_level"].get(lvl, {})
        if not s:
            continue
        sd = s.get("step_distribution", {})
        print(f"  {lvl:<8}  min={sd.get('min')}  median={sd.get('median')}  "
              f"max={sd.get('max')}  (n={s['n_tasks']})")

    print("\nError types:")
    for etype, count in summary["error_type_counts"].items():
        print(f"  {etype}: {count}")

    print("\nFailure locations:")
    for loc, count in summary["failure_location_counts"].items():
        print(f"  {loc}: {count}")

    print("\nPer-task:")
    for t in sorted(summary["tasks"], key=lambda x: x["task_id"]):
        fl   = f"  [{t['failure_location']}]" if t["failure_location"] else ""
        errs = f" [{','.join(t['error_types'])}]" if t["error_types"] else ""
        rec  = " (recovered)" if t["error_recovery"] else ""
        print(f"  Task {t['task_id']:3d} ({t['level']:<6}) "
              f"calls={t['n_tool_calls']}  errors={t['n_errors']}{errs}{rec}{fl}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/trajectory.py <traj_dir> [results_json]")
        sys.exit(1)
    summary = score_trajectories(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    print_report(summary)
