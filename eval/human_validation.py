"""
Human validation workflow for the LLM judge.

Two-phase blind annotation to measure judge-human agreement via weighted Cohen's
kappa and ordinal Krippendorff's alpha, reported per scoring dimension (OQ / TQ).

Blinding guarantees:
  - Judge scores and rationales are stripped from the annotation file.
  - Model identity is replaced with a neutral "Agent X / Agent Y" label.
  - TQ items do not show the correctness verdict, GT values, or the final answer block.
  - Tasks are shuffled independently in each section and assigned opaque IDs.
  - The key file (ID → task_id + model) is written separately; do not open until done.

Stratification note: all currently judged tasks are "hard" difficulty.
Stratification is therefore by model × pass/fail outcome only. Flag this
limitation if you report by-difficulty breakdowns.

Usage:

    # Phase 1 — generate annotation files (run once; fill in the .md before phase 2)
    PYTHONPATH=. python eval/human_validation.py generate \\
        results/judge_hard_dev_all_haiku_20260622_152459.json \\
        results/judge_hard_dev_all_sonnet_combined.json \\
        [--n 6] [--seed 42] [--out-dir results]

    # Phase 2 — compute agreement after scoring
    PYTHONPATH=. python eval/human_validation.py join \\
        results/annotation_blind_<TIMESTAMP>.md \\
        results/annotation_key_<TIMESTAMP>.json \\
        results/judge_hard_dev_all_haiku_20260622_152459.json \\
        results/judge_hard_dev_all_sonnet_combined.json
"""

import json
import random
import re
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path

from eval.llm_judge import format_trajectory


# ---------------------------------------------------------------------------
# Rubric text (kept here as the single source of truth for annotation files)
# ---------------------------------------------------------------------------

_OQ_RUBRIC = """\
  0 = Wrong approach, answer missing, or doesn't address the question
  1 = Right approach but wrong answer (missed validation), OR correct but purely
      mechanical — no validation, no interpretation
  2 = Correct answer (within rounding tolerance) + at least some validation or
      interpretation. Correctness is required to score 2 or higher.
  3 = Correct + validates assumptions + handles edge cases + interprets result"""

_TQ_RUBRIC = """\
GATE 1 — COMPLETION (check first):
  Did the agent produce a final answer in @var[value] format?
  NO → score is capped at 1, regardless of process quality.
  YES → apply scores 1–3 below based on process quality.

  0 = Disorganized: no coherent strategy; thrashing; circular reasoning;
      errors with no recovery.

  1 = Functional: produced a final answer via a notably flawed, lucky, or
      circuitous path (e.g., trial-and-error without a clear strategy,
      unexplained leaps, approach that barely worked).
      Also: any trajectory that produced no final answer at all (gate 1).

  2 = Methodical: produced a final answer with a sound overall approach,
      BUT had at least one of the following imperfections:
        • An error requiring recovery (traceback/exception that caused a re-run/fix), OR
        • Redundancy / inefficiency (e.g., reloading the same CSV in every step,
          re-running the full analysis only to "verify" after a clean result,
          repeating code blocks without fixing an error)

  3 = Exemplary: ALL of the following must be true —
        • Produced a final answer in @var[value] format
        • No errors requiring recovery (code ran cleanly)
        • No redundancy or wasted steps
        • Inspected the data before computing
        • Sound analytical approach
      If even one criterion is missing, score 2, not 3.

Note: runtime warnings (numpy overflow etc.) are NOT errors requiring recovery."""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_run_data(judge_path: Path) -> list:
    """
    Join one judge output file with its original results to get full task content.
    Returns a list of task dicts with both agent outputs and judge scores.
    """
    judge_data = json.loads(judge_path.read_text())
    results_path = Path(judge_data["results_file"])
    results_data = json.loads(results_path.read_text())
    agent_model = judge_data["agent_model"]

    results_by_id = {r["task_id"]: r for r in results_data["results"]}
    judge_by_id = {
        t["task_id"]: t for t in judge_data["tasks"]
        if t["output_quality"]["score"] is not None
    }

    tasks = []
    for task_id, result in results_by_id.items():
        if task_id not in judge_by_id:
            continue
        j = judge_by_id[task_id]
        tasks.append({
            "task_id":            task_id,
            "model":              agent_model,
            "level":              result["level"],
            "question":           result["question"],
            "constraints":        result.get("constraints", ""),
            "format":             result.get("format", ""),
            "ground_truth":       result["ground_truth"],
            "agent_answer":       result["agent_answer"],
            "trajectory_file":    result["trajectory_file"],
            "file_name":          result.get("file_name", ""),
            "passed_verifiable":  j["passed_verifiable"],
            "judge_oq_score":     j["output_quality"]["score"],
            "judge_oq_rationale": j["output_quality"]["rationale"],
            "judge_tq_score":     j["trajectory_quality"]["score"],
            "judge_tq_rationale": j["trajectory_quality"]["rationale"],
        })
    return tasks


def _load_trajectory_for_tq(task: dict) -> tuple:
    """
    Load and format trajectory, stripping the final answer block for outcome-blinding.
    Returns (traj_str, leakage_note).
    """
    traj_path = Path(task["trajectory_file"])
    traj_data = json.loads(traj_path.read_text())
    full_traj = format_trajectory(traj_data["trajectory"])

    # Strip everything from [FINAL ANSWER] onward
    idx = full_traj.find("\n[FINAL ANSWER]")
    if idx != -1:
        traj_str = full_traj[:idx].rstrip()
        note = (
            "NOTE: the final answer block has been removed for blinding. "
            "Intermediate RESULT blocks may still show computed values — "
            "some outcome-leakage is unavoidable in code-execution trajectories."
        )
    else:
        traj_str = full_traj
        note = (
            "NOTE: no explicit final-answer block found. "
            "Intermediate RESULT blocks may reveal computed values."
        )
    return traj_str, note


# ---------------------------------------------------------------------------
# Stratified sample selection
# ---------------------------------------------------------------------------

def _select_sample(all_tasks: list, n: int, seed: int = 42) -> list:
    """
    Stratified sample across model × pass/fail.

    Target: ~n/2 per model, ~1 failed item per model (reflecting ~20-25% fail rate).
    Falls back to random draw if any stratum is too small to meet the target.

    Stratification note: all tasks are currently "hard" difficulty; difficulty
    stratification is not possible until the holdout (which includes easy/medium).
    """
    rng = random.Random(seed)

    def model_tag(t: dict) -> str:
        return "sonnet" if "sonnet" in t["model"].lower() else "haiku"

    groups: dict = {}
    for t in all_tasks:
        key = (model_tag(t), t["passed_verifiable"])
        groups.setdefault(key, []).append(t)

    per_model = n // 2
    n_fail_each = max(1, round(per_model * 0.25))
    n_pass_each = per_model - n_fail_each

    sample = []
    for m in ["sonnet", "haiku"]:
        failed = groups.get((m, False), [])
        passed = groups.get((m, True), [])
        sample.extend(rng.sample(failed, min(n_fail_each, len(failed))))
        sample.extend(rng.sample(passed, min(n_pass_each, len(passed))))

    # Pad if we came up short of n
    if len(sample) < n:
        used = {(t["task_id"], t["model"]) for t in sample}
        pool = [t for t in all_tasks if (t["task_id"], t["model"]) not in used]
        rng.shuffle(pool)
        sample.extend(pool[:n - len(sample)])

    return sample[:n]


# ---------------------------------------------------------------------------
# Opaque annotation IDs
# ---------------------------------------------------------------------------

def _make_ids(prefix: str, count: int, rng: random.Random) -> list:
    """Generate opaque annotation IDs like OQ-A7K3. No I/O/0/1 to avoid confusion."""
    chars = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    ids, seen = [], set()
    while len(ids) < count:
        suffix = "".join(rng.choice(chars) for _ in range(4))
        candidate = f"{prefix}-{suffix}"
        if candidate not in seen:
            ids.append(candidate)
            seen.add(candidate)
    return ids


# ---------------------------------------------------------------------------
# Annotation item formatting
# ---------------------------------------------------------------------------

def _format_oq_item(task: dict, ann_id: str, agent_label: str) -> str:
    """One output-quality annotation item. Shows verdict; does NOT show model identity."""
    verdict = "PASSED" if task["passed_verifiable"] else "FAILED"
    gt_parts = task["ground_truth"]
    gt_str = "  ".join(f"@{var}[{val}]" for var, val in gt_parts)
    fn = task.get("file_name", "")
    csv_link = (f"[{fn}](data/dabench/tables/{fn.replace(' ', '%20')})"
                if fn else "")

    return f"""\
---

## {ann_id}

**Dataset**: {csv_link}

**Question**: {task["question"]}

**Constraints**: {task["constraints"]}

**Expected format**: {task["format"]}

**Ground truth** (use only to verify correctness — do not anchor your score to the judge's opinion): {gt_str}

**Verifiable-eval verdict**: {verdict}

**{agent_label}'s final answer**:
{task["agent_answer"]}

**Your score (0-3)**:
**Your rationale (1-2 sentences)**:

"""


def _format_tq_item(task: dict, ann_id: str, agent_label: str,
                    traj_str: str, note: str) -> str:
    """
    One trajectory-quality annotation item.
    Does NOT show verdict, ground truth, model identity, or the final answer block.
    """
    fn = task.get("file_name", "")
    csv_link = (f"[{fn}](data/dabench/tables/{fn.replace(' ', '%20')})"
                if fn else "")

    return f"""\
---

## {ann_id}

**Dataset**: {csv_link}

**Question**: {task["question"]}

**Constraints**: {task["constraints"]}

**{agent_label}'s step-by-step trajectory**:
_{note}_

{traj_str}

**Your score (0-3)**:
**Your rationale (1-2 sentences)**:

"""


# ---------------------------------------------------------------------------
# Phase 1: generate
# ---------------------------------------------------------------------------

def generate_blind_annotation_file(
    judge_paths: list,
    n: int = 6,
    seed: int = 42,
    out_dir: Path = None,
    explicit_tasks: list = None,
) -> tuple:
    """
    Generate blind annotation file + key file.

    Returns (annotation_path, key_path).

    explicit_tasks: optional list of (task_id, model_substring) pairs.
      When provided, bypasses stratified sampling and selects exactly those tasks.
      model_substring can be 'sonnet' or 'haiku' (matched case-insensitively).
    """
    if out_dir is None:
        out_dir = Path("results")

    # Pool tasks from all judge runs
    all_tasks = []
    for p in judge_paths:
        all_tasks.extend(_load_run_data(Path(p)))

    if explicit_tasks:
        index = {(t["task_id"], t["model"]): t for t in all_tasks}
        sample = []
        for task_id, model_sub in explicit_tasks:
            match = next(
                (t for t in all_tasks
                 if t["task_id"] == task_id and model_sub.lower() in t["model"].lower()),
                None,
            )
            if match is None:
                raise ValueError(f"No task found for task_id={task_id} model=*{model_sub}*")
            sample.append(match)
    else:
        sample = _select_sample(all_tasks, n, seed=seed)

    # Assign neutral model labels (agent identity is withheld)
    models_seen = list(dict.fromkeys(t["model"] for t in sample))
    model_labels = {m: f"Agent {chr(65 + i)}" for i, m in enumerate(models_seen)}

    # Load trajectories
    for task in sample:
        traj_str, note = _load_trajectory_for_tq(task)
        task["_traj_str"] = traj_str
        task["_traj_note"] = note

    # Shuffle items independently for OQ and TQ sections
    rng = random.Random(seed + 1)
    oq_order = list(range(len(sample)))
    tq_order = list(range(len(sample)))
    rng.shuffle(oq_order)
    # Shuffle TQ differently so position doesn't correlate with OQ
    rng.shuffle(tq_order)

    oq_ids = _make_ids("OQ", len(sample), rng)
    tq_ids = _make_ids("TQ", len(sample), rng)

    # Key file: opaque ID → (task_id, model).  NO judge scores here.
    key = {}
    for i, task in enumerate(sample):
        key[oq_ids[i]] = {"task_id": task["task_id"], "model": task["model"], "dim": "oq"}
        key[tq_ids[i]] = {"task_id": task["task_id"], "model": task["model"], "dim": "tq"}

    # --- Build annotation file ---
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    oq_header = f"""\
# OUTPUT QUALITY — Blind Annotation

Score each item on the **output quality** of the agent's final answer.
You can see the task, the ground truth, and whether verifiable eval passed.
Score based on answer quality only — do NOT consider how the agent got there.

Score BEFORE reading any rationales or judge outputs.

**Rubric (0–3)**:
{_OQ_RUBRIC}

"""

    tq_header = f"""\
# TRAJECTORY QUALITY — Blind Annotation

Score each item on the **quality of the agent's problem-solving process**.
You can see the task and the step-by-step trajectory, but NOT whether the
final answer was correct. Score the process only — not the outcome.

Score BEFORE reading any rationales or judge outputs.

**Rubric (0–3)**:
{_TQ_RUBRIC}

"""

    oq_items = []
    for pos in oq_order:
        task = sample[pos]
        label = model_labels[task["model"]]
        oq_items.append(_format_oq_item(task, oq_ids[pos], label))

    tq_items = []
    for pos in tq_order:
        task = sample[pos]
        label = model_labels[task["model"]]
        tq_items.append(_format_tq_item(
            task, tq_ids[pos], label,
            task["_traj_str"], task["_traj_note"],
        ))

    annotation_content = (
        oq_header + "".join(oq_items)
        + "\n\n"
        + tq_header + "".join(tq_items)
    )

    ann_path = out_dir / f"annotation_blind_{ts}.md"
    key_path = out_dir / f"annotation_key_{ts}.json"
    ann_path.write_text(annotation_content)
    key_path.write_text(json.dumps(key, indent=2))

    # Print sampling summary (no judge scores — just metadata)
    print(f"\nSample ({n} tasks):")
    print(f"  {'ID':<8}  {'model':<22}  {'level':<6}  {'pass/fail'}")
    for i, task in enumerate(sample):
        verdict = "passed" if task["passed_verifiable"] else "FAILED"
        print(f"  OQ-{i+1:<4}  {task['model']:<22}  {task['level']:<6}  {verdict}")

    return ann_path, key_path


# ---------------------------------------------------------------------------
# Phase 2: parse + join + compute agreement
# ---------------------------------------------------------------------------

def _parse_annotations(annotation_path: Path) -> dict:
    """
    Parse a completed annotation file.
    Returns dict: ann_id → {"score": int, "rationale": str}.

    Splits on ## OQ-/TQ- headers (not on --- which can appear inside agent answers).
    Skips items where no score was filled in.
    """
    text = annotation_path.read_text()
    result = {}

    # Split just before each ## OQ-/TQ- header so agent-answer --- don't break sections
    sections = re.split(r"\n(?=## (?:OQ|TQ)-)", text)

    for section in sections:
        id_match = re.match(r"## ((OQ|TQ)-[A-Z0-9]+)", section.lstrip())
        if not id_match:
            continue
        ann_id = id_match.group(1)

        score_match = re.search(r"Your score[^:]*:\s*(\d)", section)
        if not score_match:
            continue
        score = max(0, min(3, int(score_match.group(1))))

        rationale_match = re.search(r"Your rationale[^:]*:\s*(.+)", section)
        rationale = rationale_match.group(1).strip() if rationale_match else ""

        result[ann_id] = {"score": score, "rationale": rationale}

    return result


def _weighted_kappa_linear(y_human: list, y_judge: list, scale_max: int = 3) -> float:
    """Weighted Cohen's kappa with linear weights on [0, scale_max]."""
    from sklearn.metrics import cohen_kappa_score
    labels = list(range(scale_max + 1))
    return round(float(cohen_kappa_score(y_human, y_judge,
                                         weights="linear", labels=labels)), 4)


def _krippendorff_alpha_ordinal(y1: list, y2: list) -> float:
    """
    Krippendorff's alpha with ordinal distance metric for two raters.

    Ordinal distance between values c and k:
        d(c, k)^2 = ( sum_{g=min(c,k)}^{max(c,k)} n_g  -  (n_c + n_k) / 2 )^2
    where n_g is the marginal count of value g in the pooled observations.

    Observed:  D_o = (1/n) * sum_u d(y1_u, y2_u)^2
    Expected:  D_e = (1 / (N*(N-1))) * sum_{c,k} d(c,k)^2 * n_c * n_k   (N = 2n)
    alpha = 1 - D_o / D_e
    """
    n = len(y1)
    if n == 0:
        return float("nan")

    pooled = list(y1) + list(y2)
    N = len(pooled)  # = 2n
    marginal = Counter(pooled)
    values = sorted(set(pooled))

    def d_sq(c: int, k: int) -> float:
        if c == k:
            return 0.0
        lo, hi = min(c, k), max(c, k)
        s = sum(marginal[g] for g in range(lo, hi + 1))
        return (s - (marginal[lo] + marginal[hi]) / 2.0) ** 2

    D_o = sum(d_sq(a, b) for a, b in zip(y1, y2)) / n
    D_e = sum(
        d_sq(c, k) * marginal[c] * marginal[k]
        for c in values for k in values
    ) / (N * (N - 1))

    if D_e == 0:
        return 1.0 if D_o == 0 else 0.0
    return round(1.0 - D_o / D_e, 4)


def _confusion_matrix(y_human: list, y_judge: list, scale_max: int = 3) -> list:
    """Confusion matrix as list-of-lists: rows=human, cols=judge."""
    size = scale_max + 1
    mat = [[0] * size for _ in range(size)]
    for h, j in zip(y_human, y_judge):
        mat[h][j] += 1
    return mat


def _analyze_dimension(records: list) -> dict:
    """Compute all agreement metrics for one dimension."""
    if not records:
        return {}

    y_h = [r["human_score"] for r in records]
    y_j = [r["judge_score"] for r in records]

    exact = sum(1 for h, j in zip(y_h, y_j) if h == j)
    within1 = sum(1 for h, j in zip(y_h, y_j) if abs(h - j) <= 1)
    large = [r for r in records if abs(r["human_score"] - r["judge_score"]) >= 2]

    return {
        "n":                   len(records),
        "linear_kappa":        _weighted_kappa_linear(y_h, y_j),
        "ordinal_alpha":       _krippendorff_alpha_ordinal(y_h, y_j),
        "exact_agreement_pct": round(exact / len(records) * 100, 1),
        "within1_pct":         round(within1 / len(records) * 100, 1),
        "confusion_matrix":    _confusion_matrix(y_h, y_j),
        "score_dist_human":    dict(Counter(y_h)),
        "score_dist_judge":    dict(Counter(y_j)),
        "large_disagreements": large,
    }


def compute_agreement(
    annotation_path: Path,
    key_path: Path,
    judge_paths: list,
) -> dict:
    """
    Join completed annotations with judge scores and compute agreement metrics.
    Returns a result dict with per-dimension stats and joined record lists.
    """
    ann = _parse_annotations(Path(annotation_path))
    key = json.loads(Path(key_path).read_text())

    # Index judge scores by (task_id, model)
    judge_index = {}
    for jp in judge_paths:
        jd = json.loads(Path(jp).read_text())
        agent_model = jd["agent_model"]
        for t in jd["tasks"]:
            judge_index[(t["task_id"], agent_model)] = t

    oq_records, tq_records, missing = [], [], []

    for ann_id, entry in key.items():
        task_id = entry["task_id"]
        model   = entry["model"]
        dim     = entry["dim"]

        if ann_id not in ann:
            missing.append(ann_id)
            continue

        judge_task = judge_index.get((task_id, model))
        if not judge_task:
            print(f"Warning: no judge data for task {task_id} / {model}")
            continue

        record = {
            "ann_id":      ann_id,
            "task_id":     task_id,
            "model":       model,
            "human_score": ann[ann_id]["score"],
            "human_note":  ann[ann_id]["rationale"],
        }

        if dim == "oq":
            record.update({
                "judge_score": judge_task["output_quality"]["score"],
                "judge_note":  judge_task["output_quality"]["rationale"],
                "passed":      judge_task["passed_verifiable"],
            })
            oq_records.append(record)
        else:
            record.update({
                "judge_score": judge_task["trajectory_quality"]["score"],
                "judge_note":  judge_task["trajectory_quality"]["rationale"],
                "passed":      judge_task["passed_verifiable"],
            })
            tq_records.append(record)

    if missing:
        print(f"Warning: {len(missing)} annotation IDs not scored yet: {missing}")

    return {
        "annotation_file":      str(annotation_path),
        "key_file":             str(key_path),
        "n_annotated":          len(ann),
        "output_quality":       _analyze_dimension(oq_records),
        "trajectory_quality":   _analyze_dimension(tq_records),
        "oq_records":           oq_records,
        "tq_records":           tq_records,
    }


def print_agreement_report(result: dict) -> None:
    print(f"\nAnnotation file : {result['annotation_file']}")
    print(f"Items scored    : {result['n_annotated']}")
    print("=" * 64)

    for dim_key, label in [
        ("output_quality",    "OUTPUT QUALITY"),
        ("trajectory_quality","TRAJECTORY QUALITY"),
    ]:
        stats = result.get(dim_key, {})
        if not stats:
            print(f"\n{label}: no data")
            continue

        print(f"\n{label}  (n={stats['n']})")
        print(f"  Linear kappa (weighted)     : {stats['linear_kappa']}")
        print(f"  Krippendorff alpha (ordinal): {stats['ordinal_alpha']}")
        print(f"  Exact agreement             : {stats['exact_agreement_pct']}%")
        print(f"  Within-1 agreement          : {stats['within1_pct']}%")

        print(f"\n  Score distribution (0–3):")
        print(f"    {'Score':<6}  {'Human':>6}  {'Judge':>6}")
        for v in range(4):
            h = stats["score_dist_human"].get(v, 0)
            j = stats["score_dist_judge"].get(v, 0)
            print(f"    {v:<6}  {h:>6}  {j:>6}")

        print(f"\n  Confusion matrix (rows=Human, cols=Judge):")
        header = "       " + "  ".join(f"J={v}" for v in range(4))
        print(f"  {header}")
        for i, row in enumerate(stats["confusion_matrix"]):
            print(f"    H={i}  " + "   ".join(f"{x:3d}" for x in row))

        if stats["large_disagreements"]:
            print(f"\n  Disagreements >= 2  ({len(stats['large_disagreements'])} items):")
            for r in stats["large_disagreements"]:
                print(f"    {r['ann_id']}  task={r['task_id']}  model={r['model']}")
                print(f"      Human={r['human_score']}  Judge={r['judge_score']}")
                print(f"      Human note : {r['human_note']}")
                print(f"      Judge note : {r['judge_note']}")
        else:
            print(f"\n  No disagreements >= 2")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Human validation workflow for the LLM judge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- generate ---
    gen = sub.add_parser("generate", help="Create blind annotation + key files")
    gen.add_argument(
        "judge_outputs", nargs="+",
        help="One or more judge output JSON files (e.g. haiku + sonnet)",
    )
    gen.add_argument(
        "--n", type=int, default=6,
        help="Sample size (default 6 for pilot; expand to 15-20 for full validation)",
    )
    gen.add_argument("--seed", type=int, default=42)
    gen.add_argument("--out-dir", default="results", metavar="DIR")
    gen.add_argument(
        "--pick", nargs="+", metavar="TASK_ID:MODEL",
        help=(
            "Explicit task list (bypasses random sampling). "
            "Format: TASK_ID:MODEL where MODEL is 'sonnet' or 'haiku'. "
            "Example: --pick 297:haiku 685:sonnet 574:sonnet"
        ),
    )

    # --- join ---
    join_p = sub.add_parser(
        "join", help="Compute agreement from completed annotations",
    )
    join_p.add_argument("annotation", help="Completed annotation .md file")
    join_p.add_argument("key",        help="Key .json file from the generate step")
    join_p.add_argument(
        "judge_outputs", nargs="+",
        help="Same judge output JSON files used in generate",
    )
    join_p.add_argument(
        "--output", metavar="PATH",
        help="Save agreement JSON (default: results/agreement_<TIMESTAMP>.json)",
    )

    args = parser.parse_args()

    if args.command == "generate":
        explicit = None
        if args.pick:
            explicit = []
            for item in args.pick:
                parts = item.split(":")
                if len(parts) != 2:
                    parser.error(f"--pick items must be TASK_ID:MODEL, got: {item!r}")
                explicit.append((int(parts[0]), parts[1]))

        ann_path, key_path = generate_blind_annotation_file(
            judge_paths=args.judge_outputs,
            n=args.n,
            seed=args.seed,
            out_dir=Path(args.out_dir),
            explicit_tasks=explicit,
        )
        print(f"\nFill in this file (open in any text editor):")
        print(f"  {ann_path}")
        print(f"\nDo NOT open the key file until after scoring:")
        print(f"  {key_path}")
        print(f"\nThen run:  python eval/human_validation.py join {ann_path} {key_path} <judge_files...>")

    elif args.command == "join":
        result = compute_agreement(
            annotation_path=Path(args.annotation),
            key_path=Path(args.key),
            judge_paths=args.judge_outputs,
        )
        print_agreement_report(result)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path(args.output) if args.output else Path("results") / f"agreement_{ts}.json"
        # Serialize — skip non-JSON-serializable entries (none expected here)
        out.write_text(json.dumps(result, indent=2, default=str))
        print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
