"""
Failure taxonomy for the DS agent comparative evaluation.

ONE shared taxonomy applied to ALL models. Failures are categorised on four axes:

  1. failure_location  — WHERE in the task lifecycle the failure occurred
                         (reused from trajectory.py; precedence rules already defined)
  2. mechanism         — WHY the answer is wrong or task not completed
  3. source            — heuristic attribution: who is primarily responsible
  4. quality           — reserved for LLM-judge scores (None until that phase)

SOURCE IS A HEURISTIC ATTRIBUTION, NOT OBJECTIVE FACT.
Some failures have both agent and scaffolding components — we assign the
dominant one and document the reasoning in source_note.

NOTE ON CODE ERRORS:
  code_error_unrecovered counts zero here. Zero ≠ no code errors.
  Recovered code errors (NameError, ModuleNotFoundError, etc.) are fully
  tracked per-task in trajectory eval. This category only applies when
  unrecovered runtime errors directly caused a task to fail.

Mechanism precedence (first matching rule wins per failed task):
  1. iteration_limit        — max iterations hit; task never completed
  2. output_format_failure  — failure_location == format_only
  3. [manual override]      — MANUAL_MECHANISMS dict overrides all heuristics
  4. rounding_artifact      — all wrong values within ROUNDING_THRESHOLD
                              (rule out our own tolerance artifacts FIRST,
                               before attributing anything to agent or benchmark)
  5. task_ambiguity         — both models fail same task with overlapping wrong values
  6. constraint_violation   — single-model failure (heuristic candidate; refine via
                               manual overrides after auditing the trajectory)
  7. numeric_mismatch       — residual: cross-model fail, values differ between models
                              (AUDIT THIS BUCKET — if it keeps growing, it may be
                               hiding an unnamed category)
  8. code_error_unrecovered — unrecovered runtime errors caused the failure

Usage:
    python eval/taxonomy.py <metrics_sonnet.json> <metrics_haiku.json> \\
                            <traj_dir_sonnet>     <traj_dir_haiku>
"""
import json
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

MECHANISMS = [
    "iteration_limit",        # max iterations hit; task not completed
    "output_format_failure",  # format_only: agent finished but no @tags produced
    "rounding_artifact",      # difference < ROUNDING_THRESHOLD — eval tolerance issue
    "task_ambiguity",         # both models fail same task same way → benchmark signal
    "constraint_violation",   # agent violated explicit, inspectable task constraint
    "numeric_mismatch",       # correct approach, wrong number, one model only (AUDIT)
    "code_error_unrecovered", # unrecovered runtime errors caused failure (zero now)
]

SOURCES = ["agent_induced", "scaffolding_induced"]

# Quality axis: reserved for LLM judge. Schema is set now so adding values later
# is a populate-only change, not a structural one.
QUALITY_VALUES = [None, "sound_approach", "flawed_approach", "lucky_correct"]

# Numeric tolerance for rounding_artifact: checked BEFORE task_ambiguity and
# numeric_mismatch to rule out our own eval precision artifacts first.
ROUNDING_THRESHOLD = 0.01


# ---------------------------------------------------------------------------
# Manual overrides
# ---------------------------------------------------------------------------
# Automated heuristics classify most failures. These dicts override for cases
# where the heuristic is wrong or the trajectory needs human inspection.
# Key: (task_id, model_tag)

MANUAL_MECHANISMS: dict = {
    # Task 28, Haiku: single-model fail — Haiku violated the encoding constraint
    # that Sonnet handled correctly. Trajectory confirms different encoding choice.
    (28, "haiku"): "constraint_violation",

    # Task 124, Haiku: completed (no max_iter) but wrong significance result.
    # Numeric approach was plausible; statistical conclusion was wrong.
    (124, "haiku"): "numeric_mismatch",
}

MANUAL_SOURCES: dict = {}


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _numeric_diff(extracted: str, expected: str):
    """Return abs(extracted - expected) if both parse as float, else None."""
    try:
        return abs(float(extracted) - float(expected))
    except (ValueError, TypeError):
        return None


def classify_mechanism(
    task_id: int,
    model_tag: str,
    failure_location: str,
    hit_max_iterations: bool,
    task_details: list,
    cross_model_fail: bool,
    cross_model_same_values: bool,
) -> str:
    """Assign one mechanism label following the precedence in the module docstring."""

    # 1. Iteration limit
    if hit_max_iterations:
        return "iteration_limit"

    # 2. Format failure
    if failure_location == "format_only":
        return "output_format_failure"

    # 3. Manual override
    if (task_id, model_tag) in MANUAL_MECHANISMS:
        return MANUAL_MECHANISMS[(task_id, model_tag)]

    wrong = [d for d in task_details if not d["match"] and d["extracted"] is not None]

    # 4. Rounding artifact — checked BEFORE task_ambiguity and numeric_mismatch.
    # If all wrong values are within tolerance, this is our eval precision, not
    # an agent or benchmark error.
    if wrong:
        diffs = [_numeric_diff(d["extracted"], d["expected"]) for d in wrong]
        if all(diff is not None and diff < ROUNDING_THRESHOLD for diff in diffs):
            return "rounding_artifact"

    # 5. Task ambiguity — both models fail same task with same wrong values.
    # Signals the benchmark, not the agent.
    if cross_model_fail and cross_model_same_values:
        return "task_ambiguity"

    # 6. Constraint violation — single-model failure.
    # Heuristic: only one model got it wrong → model-specific misinterpretation.
    # Confirm by reading the trajectory before trusting this label.
    if not cross_model_fail:
        return "constraint_violation"

    # 7. Numeric mismatch — residual. Cross-model fail but values differ.
    # AUDIT THIS: if it keeps growing, it likely hides a missing category.
    return "numeric_mismatch"


def classify_source(task_id: int, model_tag: str, mechanism: str) -> tuple:
    """
    Assign (source, note) — a heuristic attribution of dominant responsibility.

    Returns (source_str, note_str).
    """
    if (task_id, model_tag) in MANUAL_SOURCES:
        return MANUAL_SOURCES[(task_id, model_tag)], "manual override"

    if mechanism == "iteration_limit":
        return (
            "scaffolding_induced",
            "dominant: our 10-step cap is the proximate cause; "
            "agent efficiency is a contributing factor",
        )
    if mechanism == "code_error_unrecovered":
        return (
            "scaffolding_induced",
            "NameError from subprocess isolation is our design; "
            "other unrecovered errors would be agent_induced",
        )
    return "agent_induced", "model reasoning or interpretation caused the wrong output"


# ---------------------------------------------------------------------------
# Build taxonomy
# ---------------------------------------------------------------------------

def build_taxonomy(metrics_files: list, traj_dirs: list) -> dict:
    """
    Build the shared failure taxonomy from multiple model result files.

    Args:
        metrics_files: list of paths to metrics_*.json (one per model run)
        traj_dirs:     matching trajectory directories

    Returns a taxonomy dict with failure records and cross-model signals.
    """
    # Load metrics per model
    all_metrics = []
    model_tags  = []
    for mf in metrics_files:
        mf   = Path(mf)
        data = json.loads(mf.read_text())
        # Prefer the "model" field in the JSON; fall back to filename parsing
        if "model" in data:
            model_tag = data["model"].split("-")[1]   # "claude-sonnet-4-6" → "sonnet"
        else:
            parts     = mf.stem.split("_")
            known     = {"sonnet", "haiku", "opus"}
            model_tag = next(
                (p for p in reversed(parts) if p in known),
                next((p for p in reversed(parts)
                      if not p.isdigit() and len(p) < 10), "unknown")
            )
        all_metrics.append((model_tag, data))
        model_tags.append(model_tag)

    # Group failures by task_id across models
    failures_by_task: dict = {}
    for model_tag, data in all_metrics:
        for task in data["tasks"]:
            if not task["passed"]:
                tid = task["task_id"]
                failures_by_task.setdefault(tid, {})[model_tag] = task

    # Classify each failure
    records = []
    for task_id, model_results in sorted(failures_by_task.items()):
        cross_model_fail = len(model_results) > 1

        # Check whether models produced the same wrong extracted values
        cross_model_same_values = False
        if cross_model_fail:
            value_sets = [
                frozenset(
                    (d["variable"], d["extracted"])
                    for d in r["details"]
                    if not d["match"] and d["extracted"] is not None
                )
                for r in model_results.values()
            ]
            cross_model_same_values = len(value_sets) >= 2 and bool(
                value_sets[0] & value_sets[1]
            )

        for model_tag, task_result in sorted(model_results.items()):
            # metrics.py stores failure_mode; trajectory.py stores failure_location.
            # Map metrics failure_mode → failure_location for consistent labelling.
            failure_mode = task_result.get("failure_mode") or ""
            _fm_map = {
                "wrong_value":         "wrong_final_answer",
                "max_iterations":      "cut_short",   # simplified; trajectory eval
                                                       # distinguishes cut_short/stuck
                "format_not_followed": "format_only",
                "partial_format":      "format_only",
            }
            failure_location = _fm_map.get(failure_mode, failure_mode or "unknown")
            hit_max          = failure_mode == "max_iterations"

            mechanism = classify_mechanism(
                task_id=task_id,
                model_tag=model_tag,
                failure_location=failure_location,
                hit_max_iterations=hit_max,
                task_details=task_result["details"],
                cross_model_fail=cross_model_fail,
                cross_model_same_values=cross_model_same_values,
            )
            source, source_note = classify_source(task_id, model_tag, mechanism)

            records.append({
                "task_id":          task_id,
                "model":            model_tag,
                "failure_location": failure_location,
                "mechanism":        mechanism,
                "source":           source,
                "source_note":      source_note,
                "quality":          None,   # reserved for LLM judge
                "cross_model_fail": cross_model_fail,
                "cross_model_same_values": cross_model_same_values,
            })

    return {
        "model_tags": model_tags,
        "failures":   records,
        "n_failures": {t: sum(1 for r in records if r["model"] == t)
                       for t in model_tags},
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(taxonomy: dict):
    model_tags = taxonomy["model_tags"]
    failures   = taxonomy["failures"]
    col        = 12

    print("\n" + "=" * 72)
    print("FAILURE TAXONOMY — COMPARATIVE EVALUATION")
    print("=" * 72)

    print(f"\nTotal failures: " +
          " | ".join(f"{t}={taxonomy['n_failures'][t]}" for t in model_tags))

    # --- Comparison table ---
    print("\n" + "-" * 72)
    print("COMPARISON TABLE  (zeros are findings)")
    print(f"  ‼ code_error_unrecovered=0 means no unrecovered errors.")
    print(f"    Recovered code errors are tracked in trajectory eval.\n")

    # Only show (mechanism, source) pairs that are taxonomically valid
    valid_pairs = [
        ("iteration_limit",        "scaffolding_induced"),
        ("output_format_failure",  "agent_induced"),
        ("rounding_artifact",      "agent_induced"),
        ("task_ambiguity",         "agent_induced"),
        ("constraint_violation",   "agent_induced"),
        ("numeric_mismatch",       "agent_induced"),
        ("code_error_unrecovered", "scaffolding_induced"),
        ("code_error_unrecovered", "agent_induced"),
    ]

    counts: dict = {pair: {t: 0 for t in model_tags} for pair in valid_pairs}
    for r in failures:
        pair = (r["mechanism"], r["source"])
        if pair in counts:
            counts[pair][r["model"]] += 1

    header = f"  {'mechanism':<28} {'source':<22}" + \
             "".join(f"{t:>{col}}" for t in model_tags)
    print(header)
    print("  " + "-" * (50 + col * len(model_tags)))

    for pair in valid_pairs:
        row_counts = counts[pair]
        vals = "".join(f"{row_counts[t]:>{col}}" for t in model_tags)
        print(f"  {pair[0]:<28} {pair[1]:<22}{vals}")

    # --- Per-task detail ---
    print("\n" + "-" * 72)
    print("PER-TASK FAILURES\n")
    for tid in sorted({r["task_id"] for r in failures}):
        task_records = [r for r in failures if r["task_id"] == tid]
        print(f"  Task {tid}:")
        for r in sorted(task_records, key=lambda x: x["model"]):
            print(f"    [{r['model']:<8}]  "
                  f"location={r['failure_location']:<22} "
                  f"mechanism={r['mechanism']:<25} "
                  f"source={r['source']}")

    # --- Cross-model observations ---
    print("\n" + "-" * 72)
    print("CROSS-MODEL OBSERVATIONS\n")

    by_task: dict = {}
    for r in failures:
        by_task.setdefault(r["task_id"], []).append(r)

    shared   = {tid: rs for tid, rs in by_task.items() if len(rs) > 1}
    specific = {tid: rs for tid, rs in by_task.items() if len(rs) == 1}

    print(f"  Shared failures (both models): {len(shared)}")
    for tid, rs in sorted(shared.items()):
        mechs = {r["model"]: r["mechanism"] for r in rs}
        flip  = "  ← MODE FLIP" if len(set(mechs.values())) > 1 else ""
        detail = " | ".join(f"{m}:{v}" for m, v in sorted(mechs.items()))
        print(f"    Task {tid}: {detail}{flip}")

    print(f"\n  Model-specific failures: {len(specific)}")
    for tid, rs in sorted(specific.items()):
        r = rs[0]
        print(f"    Task {tid}: [{r['model']}] {r['mechanism']}")

    print(f"\n  Source breakdown:")
    for source in SOURCES:
        for tag in model_tags:
            n = sum(1 for r in failures if r["model"] == tag and r["source"] == source)
            print(f"    {tag:<12} {source}: {n}")

    unrecovered = [r for r in failures if r["mechanism"] == "code_error_unrecovered"]
    print(f"\n  code_error_unrecovered: {len(unrecovered)}")
    print("  (Zero ≠ no code errors — recovered errors tracked in trajectory eval.)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 5:
        print("Usage: python eval/taxonomy.py "
              "<metrics_sonnet.json> <metrics_haiku.json> "
              "<traj_dir_sonnet> <traj_dir_haiku>")
        sys.exit(1)

    taxonomy = build_taxonomy([sys.argv[1], sys.argv[2]],
                              [sys.argv[3], sys.argv[4]])
    print_report(taxonomy)

    out = Path(sys.argv[1]).parent / "taxonomy.json"
    out.write_text(json.dumps(taxonomy, indent=2))
    print(f"\nSaved: {out}")
