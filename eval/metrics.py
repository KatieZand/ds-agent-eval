"""
Verifiable evaluation for DABench tasks.

DABench specifies answer format as @variable[value] tags.
Example: "The mean fare is @mean_fare[34.65]"

Scoring is all-or-nothing: a task passes only if ALL required variables are
present in the agent's answer AND each value matches ground truth.

Usage:
    python eval/metrics.py results/dev_20260618_153514.json
"""
import re
import json
import sys
from pathlib import Path


def parse_answer(text: str) -> dict:
    """
    Extract all @variable[value] tags from the agent's answer text.

    Returns a dict mapping variable name -> extracted value (raw string).
    If the same variable appears more than once, the last value wins.

    >>> parse_answer("The answer is @mean_fare[34.65]")
    {'mean_fare': '34.65'}
    >>> parse_answer("@mean[1.0] and @sd[0.5]")
    {'mean': '1.0', 'sd': '0.5'}
    >>> parse_answer("No tags here at all")
    {}
    """
    pattern = r"@(\w+)\[([^\]]+)\]"
    return {var: val.strip() for var, val in re.findall(pattern, text)}


def values_match(extracted: str, expected: str) -> bool:
    """
    Compare two values flexibly.

    Tries numeric comparison first so "34.650" matches "34.65".
    For comma-separated lists (e.g. "0.0, 1.0, 0.06"), splits and
    compares each element numerically in order.
    Falls back to case-insensitive string comparison for categoricals.

    >>> values_match("34.65", "34.65")
    True
    >>> values_match("34.650", "34.65")
    True
    >>> values_match("34.64", "34.65")
    False
    >>> values_match("linear", "Linear")
    True
    >>> values_match("linear", "nonlinear")
    False
    >>> values_match("0.0, 1.0, 0.0629", "0.00, 1.00, 0.0629")
    True
    >>> values_match("314, 577", "314, 577")
    True
    """
    extracted = extracted.strip()
    expected  = expected.strip()

    # Comma-separated list — split and compare element by element
    if "," in expected:
        ext_parts = [v.strip() for v in extracted.split(",")]
        exp_parts = [v.strip() for v in expected.split(",")]
        if len(ext_parts) != len(exp_parts):
            return False
        return all(values_match(e, x) for e, x in zip(ext_parts, exp_parts))

    try:
        return float(extracted) == float(expected)
    except ValueError:
        return extracted.lower() == expected.lower()


def score_task(agent_answer: str, ground_truth: list) -> dict:
    """
    Score a single task using all-or-nothing logic.

    Args:
        agent_answer:  Raw text from the agent's final response.
        ground_truth:  List of [variable_name, expected_value] pairs.

    Returns a dict with:
        passed (bool):       True only if ALL variables matched.
        extracted (dict):    Variables the agent produced.
        details (list):      Per-variable match results.
        failure_mode (str):  Reason for failure, or None if passed.
                             One of: max_iterations, format_not_followed,
                             partial_format, wrong_value.

    >>> score_task("@mean_fare[34.65]", [["mean_fare", "34.65"]])["passed"]
    True
    >>> score_task("The answer is 34.65", [["mean_fare", "34.65"]])["failure_mode"]
    'format_not_followed'
    >>> score_task("@mean_fare[99.99]", [["mean_fare", "34.65"]])["failure_mode"]
    'wrong_value'
    >>> score_task("[agent hit max iterations without finishing]", [["mean_fare", "34.65"]])["failure_mode"]
    'max_iterations'
    """
    extracted = parse_answer(agent_answer)

    details   = []
    all_match = True

    for var, expected in ground_truth:
        if var not in extracted:
            details.append({"variable": var, "expected": expected,
                            "extracted": None, "match": False,
                            "reason": "missing — variable not found in answer"})
            all_match = False
        elif not values_match(extracted[var], expected):
            details.append({"variable": var, "expected": expected,
                            "extracted": extracted[var], "match": False,
                            "reason": "value mismatch"})
            all_match = False
        else:
            details.append({"variable": var, "expected": expected,
                            "extracted": extracted[var], "match": True,
                            "reason": "correct"})

    failure_mode = None
    if not all_match:
        missing = [d for d in details if d["extracted"] is None]
        if agent_answer.strip().startswith("[agent hit max iterations"):
            failure_mode = "max_iterations"
        elif len(missing) == len(ground_truth):
            failure_mode = "format_not_followed"
        elif missing:
            failure_mode = "partial_format"
        else:
            failure_mode = "wrong_value"

    return {"passed": all_match, "extracted": extracted,
            "details": details, "failure_mode": failure_mode}


def score_results(results_path) -> dict:
    """
    Score all tasks in a run results JSON file.

    Returns a summary with per-task scores and aggregate stats by difficulty.
    """
    data  = json.loads(Path(results_path).read_text())
    tasks = data["results"]

    scored = []
    for r in tasks:
        result = score_task(r["agent_answer"], r["ground_truth"])
        scored.append({
            "task_id":      r["task_id"],
            "level":        r["level"],
            "concepts":     r["concepts"],
            "passed":       result["passed"],
            "failure_mode": result["failure_mode"],
            "details":      result["details"],
            "iterations":   r["iterations"],
            "cost_usd":     r["cost_usd"],
        })

    def aggregate(subset):
        total  = len(subset)
        passed = sum(1 for t in subset if t["passed"])
        failure_modes = {}
        for t in subset:
            if not t["passed"]:
                fm = t["failure_mode"] or "unknown"
                failure_modes[fm] = failure_modes.get(fm, 0) + 1
        return {
            "total": total, "passed": passed, "failed": total - passed,
            "pass_rate": round(passed / total, 3) if total else 0,
            "failure_modes": failure_modes,
        }

    levels = ["easy", "medium", "hard"]
    return {
        "results_file": str(results_path),
        "overall":      aggregate(scored),
        "by_level":     {lvl: aggregate([t for t in scored if t["level"] == lvl])
                         for lvl in levels},
        "tasks":        scored,
    }


def print_report(summary: dict):
    """Print a human-readable evaluation report to stdout."""
    print(f"\nResults file: {summary['results_file']}")
    print("=" * 60)

    o = summary["overall"]
    print(f"\nOVERALL: {o['passed']}/{o['total']} passed ({o['pass_rate']*100:.0f}%)")

    print("\nBy difficulty:")
    for level, stats in summary["by_level"].items():
        bar = "█" * stats["passed"] + "░" * stats["failed"]
        print(f"  {level:<8} {stats['passed']}/{stats['total']}  {bar}")

    print("\nFailure modes:")
    all_failures = {}
    for t in summary["tasks"]:
        if not t["passed"]:
            fm = t["failure_mode"] or "unknown"
            all_failures[fm] = all_failures.get(fm, 0) + 1
    for fm, count in sorted(all_failures.items(), key=lambda x: -x[1]):
        print(f"  {fm}: {count}")

    print("\nPer-task breakdown:")
    for t in summary["tasks"]:
        status = "✓" if t["passed"] else "✗"
        fm = f"  [{t['failure_mode']}]" if not t["passed"] else ""
        print(f"  {status} Task {t['task_id']:3d} ({t['level']:<6}) "
              f"{t['iterations']} iter  ${t['cost_usd']:.4f}{fm}")
        if not t["passed"]:
            for d in t["details"]:
                if not d["match"]:
                    got = d["extracted"] if d["extracted"] is not None else "(missing)"
                    print(f"      @{d['variable']}: expected={d['expected']}  got={got}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval/metrics.py <results_json_path>")
        sys.exit(1)

    results_path = Path(sys.argv[1])
    summary = score_results(results_path)
    print_report(summary)

    # Save scored output next to the results file with a "metrics_" prefix
    out_path = results_path.parent / f"metrics_{results_path.name}"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")
