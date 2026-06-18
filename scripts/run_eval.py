"""
Run the DS agent on a set of DABench tasks and save results + trajectories.

Usage:
    python scripts/run_eval.py --split dev      # run the 20 dev tasks (use freely)
    python scripts/run_eval.py --split holdout  # FINAL ONLY — do not run until eval is complete
    python scripts/run_eval.py --ids 24 32 70   # specific tasks only

Output:
    results/<split>_<timestamp>.json              — per-task summary (answers, cost, tokens)
    results/trajectories/<split>_<timestamp>/
        task_<id>.json                            — full message history per task
"""
import json
import argparse
import time
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Task subset definitions
#
# DEV (20):     Use freely — debug, tune, analyze failure modes.
#               Original 10 dev tasks + 10 formerly-eval tasks (now contaminated
#               because we ran them and tuned the skill file based on results).
#
# HOLDOUT (60): Never run until the eval framework is complete and we are ready
#               to report final numbers. 20 easy / 20 medium / 20 hard,
#               spread across 31 CSV files. Treat as sealed.
# ---------------------------------------------------------------------------
DEV_IDS = [
    # original dev (10)
    0, 9, 18, 5, 11, 62, 7, 23, 28, 39,
    # formerly eval — now contaminated, moved to dev (10)
    24, 32, 55, 27, 66, 69, 70, 77, 118, 124,
]

HOLDOUT_IDS = [
    # Stratified random sample (seed=42): 20 easy / 20 medium / 20 hard
    # Selected from remaining 237 tasks after excluding DEV_IDS
    # easy (20)
    19, 26, 33, 72, 73, 114, 123, 174, 278, 320,
    349, 350, 354, 409, 465, 517, 666, 729, 737, 755,
    # medium (20)
    6, 132, 136, 140, 219, 244, 250, 277, 298, 408,
    447, 513, 528, 543, 588, 684, 716, 721, 739, 740,
    # hard (20)
    137, 177, 210, 224, 249, 310, 378, 423, 431, 521,
    523, 530, 590, 647, 674, 722, 723, 724, 734, 736,
]

# Pricing for claude-sonnet-4-6 (per million tokens, June 2026)
PRICE_INPUT_PER_M  = 3.00
PRICE_OUTPUT_PER_M = 15.00

DATA_DIR    = Path("data/dabench")
RESULTS_DIR = Path("results")


def serialize_message(msg: dict) -> dict:
    """
    Convert a message dict to a JSON-serializable form.

    The agent loop stores Anthropic SDK objects (TextBlock, ToolUseBlock, etc.)
    directly in the messages list — they're needed for API calls but can't be
    written to JSON as-is. This converts each block to a plain dict.
    """
    role    = msg["role"]
    content = msg["content"]

    # Assistant messages contain SDK objects; user messages with tool_results
    # are already plain dicts. We handle both.
    if isinstance(content, list):
        serialized_content = []
        for block in content:
            if isinstance(block, dict):
                # Already a plain dict (e.g. tool_result blocks we constructed)
                serialized_content.append(block)
            else:
                # SDK object — convert to dict via its model_dump() method
                serialized_content.append(block.model_dump())
        return {"role": role, "content": serialized_content}
    else:
        # String content (rare, but safe to handle)
        return {"role": role, "content": content}


def load_dabench():
    """Load all questions and labels into dicts keyed by task id."""
    questions = {
        q["id"]: q
        for q in [json.loads(l) for l in (DATA_DIR / "questions.jsonl").read_text().splitlines()]
    }
    labels = {
        l["id"]: l
        for l in [json.loads(l) for l in (DATA_DIR / "labels.jsonl").read_text().splitlines()]
    }
    return questions, labels


def run_tasks(task_ids: list, questions: dict, labels: dict, split: str) -> dict:
    """
    Run the agent on each task. For each task, save:
      - A summary record in the main results JSON (answer, cost, tokens, iterations)
      - A full trajectory JSON in results/trajectories/<split>_<timestamp>/task_<id>.json

    Correctness scoring is intentionally NOT done here — that's the evaluator's job.
    This script only produces raw agent outputs.
    """
    from agent.ds_agent import run_agent

    RESULTS_DIR.mkdir(exist_ok=True)

    # Create a timestamped folder for this run's trajectories
    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    traj_dir       = RESULTS_DIR / "trajectories" / f"{split}_{timestamp}"
    traj_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_input_tokens  = 0
    total_output_tokens = 0

    for i, task_id in enumerate(task_ids):
        q        = questions[task_id]
        csv_path = DATA_DIR / "tables" / q["file_name"]

        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(task_ids)}] Task {task_id} — {q['level'].upper()}")
        print(f"File:     {q['file_name']}")
        print(f"Question: {q['question'][:120]}...")
        print(f"{'='*60}")

        # Pass the full task spec: question + constraints + expected answer format.
        # Constraints tell the agent HOW to compute (library, rounding, etc.).
        # Format tells it WHAT SHAPE the answer should be — critical for verifiable eval.
        task_prompt = (
            q["question"]
            + "\n\nConstraints: " + q["constraints"]
            + "\n\nAnswer format: " + q["format"]
        )

        start   = time.time()
        result  = run_agent(task=task_prompt, csv_path=str(csv_path))
        elapsed = round(time.time() - start, 1)

        task_cost = (
            result["input_tokens"]  / 1_000_000 * PRICE_INPUT_PER_M +
            result["output_tokens"] / 1_000_000 * PRICE_OUTPUT_PER_M
        )
        total_input_tokens  += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        # --- Save trajectory ---
        # The trajectory is the full message history: every user message, every
        # assistant response (including tool_use blocks), every tool result.
        # We'll use this in Week 2 for trajectory evaluation (step count, error recovery).
        serialized_trajectory = [serialize_message(m) for m in result["trajectory"]]
        traj_path = traj_dir / f"task_{task_id}.json"
        traj_path.write_text(json.dumps({
            "task_id":    task_id,
            "question":   q["question"],
            "trajectory": serialized_trajectory,
        }, indent=2))

        # --- Summary record (no trajectory — keep the main file compact) ---
        record = {
            "task_id":        task_id,
            "level":          q["level"],
            "file_name":      q["file_name"],
            "concepts":       q["concepts"],
            "question":       q["question"],
            "constraints":    q["constraints"],
            "format":         q["format"],
            "ground_truth":   labels[task_id]["common_answers"],
            "agent_answer":   result["answer"],
            "iterations":     result["iterations"],
            "input_tokens":   result["input_tokens"],
            "output_tokens":  result["output_tokens"],
            "cost_usd":       round(task_cost, 5),
            "elapsed_sec":    elapsed,
            "trajectory_file": str(traj_path),  # pointer to the full trajectory
        }
        results.append(record)

        print(f"\n✓ Done in {elapsed}s | {result['iterations']} iterations | "
              f"{result['input_tokens']:,}+{result['output_tokens']:,} tokens | ${task_cost:.4f}")
        print(f"Answer preview: {result['answer'][:200]}...")

    total_cost = (
        total_input_tokens  / 1_000_000 * PRICE_INPUT_PER_M +
        total_output_tokens / 1_000_000 * PRICE_OUTPUT_PER_M
    )

    return {
        "timestamp":           timestamp,
        "split":               split,
        "n_tasks":             len(results),
        "total_input_tokens":  total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_cost_usd":      round(total_cost, 5),
        "trajectory_dir":      str(traj_dir),
        "results":             results,
    }


def main():
    parser = argparse.ArgumentParser(description="Run the DS agent on DABench tasks.")
    parser.add_argument("--split", choices=["dev", "holdout"], default="dev",
                        help="'dev' = 20 tasks, use freely. 'holdout' = 60 tasks, FINAL RUN ONLY.")
    parser.add_argument("--ids", nargs="+", type=int,
                        help="Run only these specific task IDs (overrides --split)")
    args = parser.parse_args()

    if args.split == "holdout" and not args.ids:
        confirm = input("WARNING: You are about to run the holdout set. This should only happen "
                        "once, when the eval framework is complete. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    task_ids = args.ids if args.ids else (DEV_IDS if args.split == "dev" else HOLDOUT_IDS)

    print(f"Running {len(task_ids)} tasks from '{args.split}' split")
    print(f"Task IDs: {task_ids}")

    questions, labels = load_dabench()
    summary = run_tasks(task_ids, questions, labels, split=args.split)

    out_path = RESULTS_DIR / f"{args.split}_{summary['timestamp']}.json"
    out_path.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"Run complete.")
    print(f"  Tasks:         {summary['n_tasks']}")
    print(f"  Total tokens:  {summary['total_input_tokens']:,} in + {summary['total_output_tokens']:,} out")
    print(f"  Total cost:    ${summary['total_cost_usd']:.4f}")
    print(f"  Results:       {out_path}")
    print(f"  Trajectories:  {summary['trajectory_dir']}/")


if __name__ == "__main__":
    main()
