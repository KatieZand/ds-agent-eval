"""
LLM judge for DABench agent runs.

Two independently scored dimensions per task, each with its own API call and
its own information set — by design:

  output_quality    (Call 1) — Is the final answer correct and well-reasoned?
    Sees: question, constraints, format, agent answer, PASSED/FAILED verdict.
    Does NOT see: raw ground truth values, the trajectory.
    Rationale: the verdict from verifiable eval is the clean correctness signal;
    we don't want the judge re-deriving correctness and possibly disagreeing.

  trajectory_quality (Call 2) — Did the agent follow a sensible process?
    Sees: question, constraints, full trajectory, completion status.
    Does NOT see: ground truth, verdict, or whether the answer was correct.
    Rationale: outcome-blind but completion-aware scoring. Correctness is hidden
    (a sound process reaching a wrong answer still scores well — Tasks 7, 297).
    Completion is surfaced because "never produced a final answer" is a process
    fact, not a correctness fact — it would be wrong to reward a trajectory that
    went nowhere as "Exemplary" just because its steps looked methodical.
    Completion = emitted @var[value] format in the final answer, derived from the
    trajectory text itself; ground truth and pass/fail are still NOT shown.

Keeping them separate is what makes the scores meaningful and comparable
across tasks and models. Scores are 0–3 per dimension. Judge model: Gemini
(different family from the Claude agents being evaluated).

The output record always preserves passed_verifiable and completed_trajectory
for correlation analysis (neither is shown to the trajectory judge Call 2).

Usage:
    # Judge a single run (traj dirs optional if trajectory_file paths are valid):
    PYTHONPATH=. python eval/llm_judge.py results/<run>.json

    # Sonnet spans two traj dirs (dev + hard_dev); pass both:
    PYTHONPATH=. python eval/llm_judge.py results/hard_dev_all_sonnet_combined.json \\
        results/trajectories/dev_20260618_153514/ \\
        results/trajectories/hard_dev_20260622_135738/

    # Haiku (single traj dir, inferred from trajectory_file paths in results):
    PYTHONPATH=. python eval/llm_judge.py results/hard_dev_all_haiku_20260622_152459.json

    # Extract human-scoring sample from an existing judge output:
    PYTHONPATH=. python eval/llm_judge.py results/judge_<run>.json --sample 12
"""
import json
import os
import re
import sys
import time
import random
import argparse
from pathlib import Path

from google import genai
from google.genai import types as genai_types


# ---------------------------------------------------------------------------
# Trajectory formatting
# ---------------------------------------------------------------------------

def _text_from_blocks(blocks: list) -> str:
    return " ".join(b.get("text", "") for b in blocks if b.get("type") == "text").strip()


def _code_from_blocks(blocks: list) -> str:
    snippets = [b.get("input", {}).get("code", "")
                for b in blocks if b.get("type") == "tool_use"]
    return "\n".join(snippets).strip()


def _result_from_blocks(blocks: list, max_chars: int = 800) -> str:
    parts = [b.get("content", "") for b in blocks if b.get("type") == "tool_result"]
    text = "\n".join(str(p) for p in parts)
    if len(text) > max_chars:
        text = text[:max_chars] + "\n... [truncated]"
    return text.strip()


def detect_completion(traj_str: str) -> bool:
    """
    Return True if the trajectory produced a final answer in @var[value] format.

    Looks for a [FINAL ANSWER] block in the formatted trajectory that contains
    at least one @var[...] tag. Derived entirely from the trajectory text — no
    ground truth or pass/fail verdict involved.

    This is the completion gate for trajectory scoring: a trajectory that never
    emitted @var[] is treated as "did not finish" regardless of process quality.
    """
    idx = traj_str.find("[FINAL ANSWER]")
    if idx == -1:
        return False
    final_section = traj_str[idx:]
    return bool(re.search(r"@\w+\[", final_section))


def format_trajectory(trajectory: list) -> str:
    """
    Format a trajectory (list of {role, content} steps) as readable text for the judge.

    Skips the first user message (the task prompt — shown separately in the prompt).
    Groups each code execution with its result as a numbered step.
    """
    lines = []
    step = 0
    i = 1  # skip step 0 (the task prompt string)

    while i < len(trajectory):
        msg = trajectory[i]
        role = msg["role"]
        content = msg["content"]

        if role == "assistant":
            blocks = content if isinstance(content, list) else []
            reasoning = _text_from_blocks(blocks)
            code = _code_from_blocks(blocks)

            result_text = ""
            if i + 1 < len(trajectory) and trajectory[i + 1]["role"] == "user":
                next_content = trajectory[i + 1]["content"]
                if isinstance(next_content, list):
                    result_text = _result_from_blocks(next_content)
                    if result_text:
                        i += 1  # consume the paired tool-result message

            if code:
                step += 1
                lines.append(f"\n--- STEP {step} ---")
                if reasoning:
                    lines.append(f"[REASONING] {reasoning}")
                lines.append(f"[CODE]\n{code}")
                if result_text:
                    lines.append(f"[RESULT]\n{result_text}")
            elif reasoning:
                lines.append(f"\n[FINAL ANSWER]\n{reasoning}")

        i += 1

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt construction — two separate prompts with different information sets
# ---------------------------------------------------------------------------

_OUTPUT_QUALITY_RUBRIC = """
OUTPUT QUALITY (score 0–3):
  0 = Wrong approach, answer is missing, or doesn't address the question
  1 = Right approach but wrong answer (e.g., failed to validate a preprocessing
      choice), OR correct answer but purely mechanical — no validation, no
      interpretation of the result
  2 = Correct answer (within rounding tolerance) + at least some validation or
      interpretation. Correctness is required to score 2 or higher.
  3 = Correct + validates assumptions + handles edge cases + interprets the result
""".strip()

_TRAJECTORY_QUALITY_RUBRIC = """
TRAJECTORY QUALITY (score 0–3):

──────────────────────────────────────────────────────────────
GATE 1 — COMPLETION (check first):
  Did the agent produce a final answer in @var[value] format?
  NO → score is capped at 1, regardless of process quality.
  YES → apply scores 1–3 below based on process quality.
──────────────────────────────────────────────────────────────

  0 = Disorganized: no coherent strategy; thrashing; circular reasoning;
      errors with no recovery.

  1 = Functional: produced a final answer, but via a notably flawed,
      lucky, or circuitous path (e.g., trial-and-error without a clear
      strategy, unexplained leaps, approach that barely worked).
      Also: any trajectory that produced no final answer at all (gate 1).

  2 = Methodical: produced a final answer with a sound overall approach,
      BUT had at least one of the following imperfections:
        • An error requiring recovery (see definition below), OR
        • Redundancy / inefficiency (see examples below)
      "Sound but not flawless" = 2.

  3 = Exemplary: ALL of the following must be true —
        • Produced a final answer in @var[value] format
        • No errors requiring recovery (code ran cleanly first time)
        • No redundancy or wasted steps
        • Inspected the data (shape, types, missing values) before computing
        • Sound analytical approach that demonstrably drove the result
      If even one criterion is missing, score 2, not 3.

──────────────────────────────────────────────────────────────
DEFINITIONS (apply consistently):

"Error requiring recovery" = the agent produced code that raised an
  exception or failed (e.g., NameError, ModuleNotFoundError, KeyError,
  AttributeError, a traceback) and had to be re-run or fixed to continue.
  A step that generates a recoverable error caps the trajectory at 2.
  Note: runtime warnings (e.g., numpy overflow warnings that don't stop
  execution) are NOT errors requiring recovery.

"Redundancy / inefficiency" examples (any one of these caps at 2):
  • Unnecessarily re-loading the same CSV in every step when it could
    be done once
  • Re-running the full analysis pipeline in a later step only to
    "verify" results when no error occurred in the previous step
  • Repeating substantial blocks of code without fixing an error
  • Checking library availability when there was no import failure
──────────────────────────────────────────────────────────────
""".strip()


def build_output_quality_prompt(question: str, constraints: str, fmt: str,
                                agent_answer: str, passed: bool) -> str:
    """
    Prompt for the output-quality judge.

    Information provided: task description, agent answer, pass/fail verdict.
    Information withheld: raw ground-truth values (verdict is the correctness
    signal), full trajectory (this judge scores the answer, not the process).
    """
    verdict = "PASSED — the verifiable eval confirmed the answer is correct" \
              if passed else \
              "FAILED — the verifiable eval found the answer incorrect"

    return f"""You are evaluating the final answer of a data science AI agent.

## TASK
Question: {question}
Constraints: {constraints}
Expected answer format: {fmt}

## AGENT'S FINAL ANSWER
{agent_answer}

## CORRECTNESS VERDICT
{verdict}

## SCORING RUBRIC
{_OUTPUT_QUALITY_RUBRIC}

## INSTRUCTIONS
Return a JSON object:
{{"score": <integer 0-3>, "rationale": "<one or two sentences>"}}

Rules:
- Score the final answer quality only. Do not consider how the agent got there.
- A FAILED verdict caps the score at 1, even if the approach was reasonable.
- Do not let answer length bias your score — concise and correct beats verbose and wrong.
""".strip()


def build_trajectory_quality_prompt(question: str, constraints: str,
                                    traj_str: str, completed: bool) -> str:
    """
    Prompt for the trajectory-quality judge.

    Information provided: task description, full trajectory, completion status.
    Information withheld: ground truth, correctness verdict, agent's final answer.

    Correctness-blind but completion-aware: the judge does not know whether the
    answer was right or wrong, but does know whether a @var[value] answer was
    produced at all. Completion is a process fact (did the agent finish?), not
    a correctness fact (was it right?).
    """
    completion_status = (
        "The agent DID produce a final answer in the required @var[value] format."
        if completed else
        "The agent did NOT produce a final answer in the required @var[value] format "
        "before terminating (e.g. hit iteration limit or stopped without an answer)."
    )
    return f"""You are evaluating the problem-solving process of a data science AI agent.

## TASK
Question: {question}
Constraints: {constraints}

## COMPLETION STATUS (derived from the trajectory — not from a correctness check)
{completion_status}

## AGENT'S WORK (full step-by-step trajectory)
{traj_str}

## SCORING RUBRIC
{_TRAJECTORY_QUALITY_RUBRIC}

## INSTRUCTIONS
Return a JSON object:
{{"score": <integer 0-3>, "rationale": "<one or two sentences>"}}

Rules:
- Score the PROCESS only. Do not consider whether the final answer was correct.
- Apply Gate 1 first: if no final answer was produced, cap at 1 before evaluating anything else.
- Score 3 requires ALL four criteria: no recoverable errors + no redundancy + data inspection
  + sound approach. If even one is missing, score 2.
- "Error requiring recovery" means a traceback or exception that caused a re-run/fix.
  Runtime warnings that do not stop execution (e.g., numpy overflow) are NOT errors.
- "Redundancy" includes: reloading the same CSV in every step, re-running the full
  analysis only to verify after a clean result, repeating code blocks without fixing an error.
- A wrong-but-completed trajectory with a clean process is still eligible for 3 — you do
  not know if the answer is correct, so do not let outcome suspicion lower your score.
- Do not let step count alone determine your score — 2 clean steps can be exemplary; 5
  deliberate steps with verification can also be, if each step adds something new.
""".strip()


# ---------------------------------------------------------------------------
# Gemini API call
# ---------------------------------------------------------------------------

def call_judge(prompt: str, model_name: str, api_key: str,
               max_retries: int = 5) -> dict:
    """
    Call Gemini with a judge prompt. Returns the parsed JSON dict.

    On 429 RESOURCE_EXHAUSTED: uses the API's retryDelay on the first failure,
    then doubles the wait each subsequent attempt (exponential backoff), capped
    at 300 seconds. max_retries=5 handles both short RPM spikes and longer RPD
    window resets.
    """
    import re
    client   = genai.Client(api_key=api_key)
    last_wait = 0
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
            return json.loads(response.text)
        except Exception as e:
            err_str    = str(e)
            retryable  = ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or
                          "503" in err_str or "UNAVAILABLE" in err_str)
            if retryable and attempt < max_retries - 1:
                match = re.search(r"retryDelay.*?(\d+)s", err_str)
                # First failure: use the API's own retryDelay + buffer.
                # Subsequent failures (or 503s without a retryDelay): exponential backoff.
                if match and attempt == 0:
                    wait = int(match.group(1)) + 5
                else:
                    wait = min((last_wait or 30) * 2, 300)
                last_wait = wait
                label = "rate limit" if "429" in err_str else "server unavailable"
                print(f" [{label}, retry {attempt+1}/{max_retries-1} in {wait}s]",
                      end="", flush=True)
                time.sleep(wait)
                continue
            raise


def _validate_score(result: dict) -> dict:
    """Clamp a single-dimension judge result {score, rationale} to score 0–3."""
    score = result.get("score")
    if score is None or not isinstance(score, int) or score < 0 or score > 3:
        try:
            result["score"] = max(0, min(3, int(score)))
        except (TypeError, ValueError):
            result["score"] = 0
    if "rationale" not in result:
        result["rationale"] = ""
    return result


# ---------------------------------------------------------------------------
# Single-task judge — two separate calls
# ---------------------------------------------------------------------------

def judge_task(task: dict, traj_dirs: list, model_name: str, api_key: str,
               call_delay: float = 15.0) -> dict:
    """
    Judge a single task with two separate API calls.

    Call 1 (output quality): sees task + agent answer + PASSED/FAILED verdict.
                             Does NOT see the trajectory or raw ground truth.
    Call 2 (trajectory quality): sees task + full trajectory.
                                 Does NOT see ground truth, verdict, or agent answer.

    call_delay: seconds to sleep after Call 1 before Call 2. Controls the
    per-call rate. 15s default keeps the free-tier 5 RPM limit (including
    the call_delay applied between tasks by judge_run).

    Both results are combined into one record. passed_verifiable is preserved
    in the output for correlation analysis but is never shown to Call 2.
    """
    task_id = task["task_id"]
    passed  = task.get("_passed", False)

    # Resolve trajectory file
    traj_path = Path(task.get("trajectory_file", ""))
    if not traj_path.exists():
        for d in traj_dirs:
            candidate = d / f"task_{task_id}.json"
            if candidate.exists():
                traj_path = candidate
                break
    traj_data  = json.loads(traj_path.read_text())
    traj_str   = format_trajectory(traj_data["trajectory"])
    completed  = detect_completion(traj_str)

    # Call 1 — output quality (answer-focused, outcome-aware)
    oq_prompt = build_output_quality_prompt(
        question=task["question"],
        constraints=task.get("constraints", ""),
        fmt=task.get("format", ""),
        agent_answer=task["agent_answer"],
        passed=passed,
    )
    oq = _validate_score(call_judge(oq_prompt, model_name, api_key))

    if call_delay > 0:
        time.sleep(call_delay)

    # Call 2 — trajectory quality (process-focused, outcome-blind, completion-aware)
    tq_prompt = build_trajectory_quality_prompt(
        question=task["question"],
        constraints=task.get("constraints", ""),
        traj_str=traj_str,
        completed=completed,
    )
    tq = _validate_score(call_judge(tq_prompt, model_name, api_key))

    return {
        "task_id":              task_id,
        "level":                task["level"],
        "passed_verifiable":    passed,     # kept for analysis; not shown to Call 2
        "completed_trajectory": completed,  # kept for analysis; not the same as passed
        "output_quality":       oq,
        "trajectory_quality":   tq,
    }


# ---------------------------------------------------------------------------
# Full-run judge
# ---------------------------------------------------------------------------

def judge_run(results_path: Path, traj_dirs: list, model_name: str,
              api_key: str, call_delay: float = 15.0) -> dict:
    """
    Run the judge on all tasks in a results file.

    Each task makes two API calls (output quality + trajectory quality).
    call_delay: seconds to sleep after *every* API call — both between the two
    calls within a task and between consecutive tasks. This is the primary rate
    control: 15s default keeps the Gemini free-tier 5 RPM limit with headroom.
    Set to 0 or lower for paid-tier accounts with higher quotas.
    """
    data = json.loads(results_path.read_text())

    from eval.metrics import score_results as verifiable_score
    scored   = verifiable_score(results_path)
    pass_map = {t["task_id"]: t["passed"] for t in scored["tasks"]}

    agent_model = data.get("model", "unknown")
    n_tasks     = len(data["results"])
    est_mins    = round(n_tasks * 2 * (call_delay + 3) / 60, 1)  # rough: 3s API latency
    print(f"\nJudging {n_tasks} tasks from {results_path.name}")
    print(f"  Agent model:  {agent_model}")
    print(f"  Judge model:  {model_name}  (2 calls per task)")
    print(f"  Call delay:   {call_delay}s between every call  (~{est_mins} min estimated)")
    print()

    tasks_judged = []
    for i, task in enumerate(data["results"]):
        task_id         = task["task_id"]
        task["_passed"] = pass_map.get(task_id, False)

        print(f"  [{i+1:2d}/{n_tasks}] Task {task_id} ({task['level']})...",
              end="", flush=True)

        try:
            result = judge_task(task, traj_dirs, model_name, api_key,
                                call_delay=call_delay)
            tasks_judged.append(result)
            oq        = result["output_quality"]["score"]
            tq        = result["trajectory_quality"]["score"]
            verdict   = "✓" if task["_passed"] else "✗"
            done_flag = "done" if result.get("completed_trajectory") else "NO-ANS"
            print(f" {verdict}  OQ={oq}  TQ={tq}  [{done_flag}]")
        except Exception as e:
            print(f" ERROR: {e}")
            tasks_judged.append({
                "task_id":            task_id,
                "level":              task["level"],
                "passed_verifiable":  task.get("_passed", False),
                "output_quality":     {"score": None, "rationale": f"judge error: {e}"},
                "trajectory_quality": {"score": None, "rationale": f"judge error: {e}"},
            })

        # Sleep between tasks (same call_delay — keeps inter-task gap consistent)
        if i < n_tasks - 1 and call_delay > 0:
            time.sleep(call_delay)

    return _build_summary(results_path, model_name, agent_model, tasks_judged)


def _build_summary(results_path: Path, judge_model: str, agent_model: str,
                   tasks: list) -> dict:
    valid = [t for t in tasks if t["output_quality"]["score"] is not None]

    def mean_score(subset, dim):
        scores = [t[dim]["score"] for t in subset if t[dim]["score"] is not None]
        return round(sum(scores) / len(scores), 2) if scores else None

    def aggregate(subset):
        if not subset:
            return {}
        return {
            "n":                      len(subset),
            "mean_output_quality":    mean_score(subset, "output_quality"),
            "mean_trajectory_quality": mean_score(subset, "trajectory_quality"),
        }

    levels = ["easy", "medium", "hard"]
    return {
        "results_file": str(results_path),
        "agent_model":  agent_model,
        "judge_model":  judge_model,
        "judge_design": "two_calls_per_task_v3_error_redundancy_gate",
        "overall":      aggregate(valid),
        "by_level":     {lvl: aggregate([t for t in valid if t["level"] == lvl])
                         for lvl in levels},
        "by_pass_fail": {
            "passed": aggregate([t for t in valid if t["passed_verifiable"]]),
            "failed": aggregate([t for t in valid if not t["passed_verifiable"]]),
        },
        "tasks": tasks,
    }


# ---------------------------------------------------------------------------
# Human scoring extraction
# ---------------------------------------------------------------------------

def extract_human_scoring_sample(judge_output: dict, n: int = 12,
                                 results_path: Path = None) -> list:
    """
    Return n tasks for human scoring, stratified by pass/fail.

    Each entry includes question, agent answer, judge scores, and blank
    human_* fields. Intended for Cohen's kappa / Krippendorff's alpha validation.
    """
    tasks  = judge_output["tasks"]
    passed = [t for t in tasks if t["passed_verifiable"]]
    failed = [t for t in tasks if not t["passed_verifiable"]]

    random.seed(42)
    n_failed = min(len(failed), n // 2)
    n_passed = min(len(passed), n - n_failed)
    sample   = random.sample(failed, n_failed) + random.sample(passed, n_passed)
    random.shuffle(sample)

    task_details = {}
    if results_path and Path(results_path).exists():
        raw          = json.loads(Path(results_path).read_text())
        task_details = {r["task_id"]: r for r in raw["results"]}

    return [
        {
            "task_id":                  t["task_id"],
            "level":                    t["level"],
            "passed_verifiable":        t["passed_verifiable"],
            "question":                 task_details.get(t["task_id"], {}).get("question", ""),
            "constraints":              task_details.get(t["task_id"], {}).get("constraints", ""),
            "ground_truth":             task_details.get(t["task_id"], {}).get("ground_truth", []),
            "agent_answer":             task_details.get(t["task_id"], {}).get("agent_answer", ""),
            "judge_output_quality":     t["output_quality"],
            "judge_trajectory_quality": t["trajectory_quality"],
            "human_output_quality":     None,   # fill in during validation
            "human_trajectory_quality": None,   # fill in during validation
        }
        for t in sample
    ]


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(summary: dict):
    print(f"\nResults: {summary['results_file']}")
    print(f"Agent:   {summary['agent_model']}")
    print(f"Judge:   {summary['judge_model']}  ({summary.get('judge_design', '')})")
    print("=" * 60)

    o = summary["overall"]
    print(f"\nOVERALL ({o['n']} tasks)")
    print(f"  Mean output quality:     {o['mean_output_quality']}")
    print(f"  Mean trajectory quality: {o['mean_trajectory_quality']}")

    if any(v for v in summary["by_level"].values()):
        print("\nBy difficulty:")
        for lvl, stats in summary["by_level"].items():
            if stats:
                print(f"  {lvl:<8} OQ={stats['mean_output_quality']}  "
                      f"TQ={stats['mean_trajectory_quality']}  (n={stats['n']})")

    pf = summary["by_pass_fail"]
    print("\nPassed vs failed (for TQ correlation check):")
    if pf.get("passed"):
        p = pf["passed"]
        print(f"  Passed ({p['n']}):  OQ={p['mean_output_quality']}  TQ={p['mean_trajectory_quality']}")
    if pf.get("failed"):
        f = pf["failed"]
        print(f"  Failed ({f['n']}):  OQ={f['mean_output_quality']}  TQ={f['mean_trajectory_quality']}")

    print("\nPer-task:")
    for t in sorted(summary["tasks"], key=lambda x: x["task_id"]):
        oq      = t["output_quality"]["score"]
        tq      = t["trajectory_quality"]["score"]
        verdict = "✓" if t["passed_verifiable"] else "✗"
        print(f"  {verdict} Task {t['task_id']:3d} ({t['level']:<6})  OQ={oq}  TQ={tq}")
        print(f"      OQ: {t['output_quality']['rationale']}")
        print(f"      TQ: {t['trajectory_quality']['rationale']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="LLM judge for DABench agent runs")
    parser.add_argument("results",
                        help="Run results JSON (or judge output JSON when using --sample)")
    parser.add_argument("traj_dirs", nargs="*", metavar="traj_dir",
                        help="Trajectory directories to search (optional if trajectory_file "
                             "paths in the results JSON are still valid)")
    parser.add_argument("--model", default="gemini-flash-latest",
                        help="Gemini model name (default: gemini-flash-latest)")
    parser.add_argument("--call-delay", dest="call_delay", type=float, default=15.0,
                        help="Seconds to sleep after every API call (default: 15). "
                             "Controls RPM. Free tier (5 RPM): use 15. "
                             "Paid tier with high quotas: use 1 or 0.")
    parser.add_argument("--output",
                        help="Output JSON path (default: results/judge_<run>.json)")
    parser.add_argument("--api-key", dest="api_key",
                        help="Gemini API key (default: $GEMINI_API_KEY)")
    parser.add_argument("--sample", type=int, default=0, metavar="N",
                        help="Extract N-task human-scoring sample from an existing "
                             "judge output file and exit (pass judge output as results arg)")
    args = parser.parse_args()

    results_path = Path(args.results)
    traj_dirs    = [Path(d) for d in args.traj_dirs]

    # Human-scoring sample mode — reads from an existing judge output
    if args.sample:
        judge_output = json.loads(results_path.read_text())
        orig_results = Path(judge_output.get("results_file", ""))
        sample = extract_human_scoring_sample(
            judge_output, n=args.sample,
            results_path=orig_results if orig_results.exists() else None,
        )
        out = results_path.parent / f"human_sample_{results_path.stem}.json"
        out.write_text(json.dumps(sample, indent=2))
        print(f"Human scoring sample ({len(sample)} tasks) → {out}")
        return

    api_key = args.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: set GEMINI_API_KEY or pass --api-key")
        sys.exit(1)

    out_path = Path(args.output) if args.output else (
        results_path.parent / f"judge_{results_path.stem}.json"
    )

    summary = judge_run(results_path, traj_dirs, args.model, api_key,
                        call_delay=args.call_delay)
    print_report(summary)

    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
