# DS Agent Eval — Four-Component Evaluation Framework

This project evaluates a Claude-powered data science agent on InfiAgent-DABench, a benchmark of real-world tabular data analysis tasks. The headline finding is that 40–50% of hard-task failures trace to benchmark underspecification or ambiguity rather than agent errors — raising pointed questions about what "hard" evaluation actually measures.

## The agent and the benchmark

**The agent** is a tool-use loop built on the Claude API. It receives a natural-language data analysis question, runs Python code against a CSV file in an isolated subprocess, and iterates until it produces a `@var[value]` formatted answer. The model is a CLI parameter (`--model`), making it straightforward to swap in different Claude tiers for comparison.

**The benchmark** is [InfiAgent-DABench](https://github.com/InfiAgent/InfiAgent-DABench) — 257 data analysis tasks across easy, medium, and hard difficulty levels, each with a CSV file, a question, a format specification, and a ground-truth answer. We evaluated on a **hard-only test set of 40 tasks**, held out from all development work.

## Key findings

- **Haiku costs more than Sonnet despite cheaper per-token rates.** Haiku uses more steps (median 3 vs. 2) and more tokens per step, erasing its rate advantage: $3.28 vs. $2.19 total for the same 40 tasks.

- **40–50% of hard-task failures are underspecification, not agent errors.** Classifying every failure by mechanism reveals that 4/10 Sonnet failures and 4/8 Haiku failures stem from task ambiguity or underspecification — not from the agent making a mistake.

- **TQ(failed) ≈ TQ(passed), validating the outcome-blind trajectory judge.** Sonnet's failed tasks scored higher on trajectory quality than its passed tasks (2.30 vs. 2.13); Haiku's were essentially equal (2.00 vs. 2.03). The agent's process on many "failed" tasks was clean — the task was underspecified, not the agent at fault.

---

> **Task 424 — where the metric and the eval both fail:**
> This task used a degenerate dataset (BitConnect price history where every entry fell below the threshold, making every label "Low"). Haiku committed to a meaningless answer and passed verifiable eval. Sonnet correctly diagnosed the degeneracy and reported it — but hit the iteration limit without producing a `@var[]`-formatted answer, so it failed. The trajectory judge also could not rescue Sonnet: the v3 rubric caps TQ at 1 when no `@var[]` is produced (completion gate), so Sonnet scored TQ=1/OQ=0 while Haiku scored TQ=2/OQ=3. This is one concrete example of how the metric and scaffolding together can reward the wrong behavior — and it exposes a gap: there is no graceful-termination affordance for an agent that correctly identifies a degenerate task.

---

- **OQ is comparable across models within judge noise.** Overall mean OQ is 2.42 (Sonnet) vs. 2.55 (Haiku). The difference is small and conditioned on the pass/fail composition; treating it as a signal would over-read a noisy, judge-mediated gap.

- **Human–judge agreement is moderate for OQ and lower for TQ.** Cohen's κ = 0.59 on output quality and 0.33 on trajectory quality (n=12, blind annotation). The judge is generous on the OQ 2→3 boundary; the human draws a sharper line. All TQ disagreements are within 1 point (within-1 = 100%), indicating the rubric produces consistent coarse-grained judgments even where it doesn't produce exact agreement.

## Evaluation framework

Four complementary evaluation components:

| Component | What it measures | Key design choice |
|---|---|---|
| Verifiable eval | Exact-match pass rate (`@var[value]` parsing, all-or-nothing) | All-or-nothing is stricter and more honest than partial credit |
| Trajectory eval | Step count distribution, error rate, failure location | Saved separately from results; each step is one agent API call |
| Failure taxonomy | Mechanism × source attribution for every failure | Three axes: agent-induced / benchmark-induced / eval-induced |
| LLM judge (v3) | Output quality + trajectory quality (0–3 each) | Two separate calls per task |

### Judge design

Two API calls per task — deliberately:

**Output quality (Call 1, outcome-aware):** sees the question, constraints, agent answer, and PASSED/FAILED verdict. Does not see the raw ground-truth values or the trajectory. The verdict is the clean correctness signal; letting the judge re-derive correctness would risk disagreeing with the verifiable eval.

**Trajectory quality (Call 2, outcome-blind):** sees the question, full trajectory, and whether the agent produced a `@var[]` answer — a process fact derived from the trajectory, not from the verdict. Does not see the verdict, ground truth, or agent answer. Outcome-blindness prevents hindsight bias: a clean process on a benchmark-error task should score well regardless of whether the task was passed.

**Rubric evolution:** v1 produced a ceiling effect (almost all TQ=3, not discriminative). v2 added a completion gate (no `@var[]` → TQ capped at 1). v3 added an error/redundancy gate (any traceback needing recovery, or redundant CSV reload every step → TQ capped at 2), producing meaningful score spread. Judge model: `gemini-3.1-flash-lite` (different model family from the Claude agents, reducing same-family self-preference risk); temperature=0 for reproducibility.

### Human validation

12-task blind annotation batch. Annotators scored OQ and TQ independently before seeing judge scores or model identities.

| Dimension | Cohen's κ | Krippendorff α | Exact agreement | Within-1 |
|---|---|---|---|---|
| Output quality | 0.59 | 0.68 | 58.3% | 91.7% |
| Trajectory quality | 0.33 | 0.39 | 75.0% | 100.0% |

The judge scores OQ 2→3 generously; the human draws a sharper line on answer quality vs. validation. No TQ disagreement exceeded 1 point. Both patterns are detectable and explainable — not random noise.

## Evaluation as measurement

This project treats the evaluation as a **measurement instrument**, not a one-shot scoring task. That framing — from a background in measurement science — drove most of the methodological choices, and it's what separates "we ran an LLM judge" from a result you can actually trust.

It shows up first in **guarding the reference standard**. Midway through development I realized the original evaluation set had been run and inspected during iteration — contaminating it. Rather than proceed, I demoted it to a dev set, drew a fresh blind holdout (selected by metadata only, contents uninspected), and added a run-guard against accidentally scoring the sealed set. A contaminated instrument gives precise-looking readings that mean nothing; catching this mattered more than the convenience of proceeding.

It shows up next in **calibrating the judge before trusting it**. An LLM judge is an instrument with unknown bias until validated, so I scored a blind human sample and measured agreement (weighted κ, ordinal α). The result wasn't "the judge is right" — it was a *characterization*: the judge is generous at the output-quality 2→3 boundary and strict on the trajectory redundancy gate, with 100% within-1 agreement. Knowing the *direction and size* of an instrument's bias is what makes its readings usable; treating an uncalibrated judge as ground truth is the error the whole exercise avoids.

Finally, it shapes **when to re-validate**. The judge was validated on dev and applied to the same-distribution holdout without re-validation — deliberately. The instrument is deterministic (temperature=0, fixed prompt, fixed model version) and the operating conditions (task difficulty, evaluated models, judge version) were unchanged, so the dev validation transfers. In production, the same instrument would need re-validation against fresh human labels — on any change to the judge model version, the evaluated models, or the input distribution, and on a regular schedule as a backstop, since an LLM judge's calibration can shift silently when conditions change.

The through-line: an evaluation is only as trustworthy as the instrument producing it, and instruments must be guarded from contamination, calibrated against a reference, and re-validated when conditions change.

## Results on the test set (40 hard tasks)

### Overall metrics

| Metric | Sonnet 4.6 | Haiku 4.5 |
|---|---|---|
| Pass rate (verifiable) | 30/40 (75%) | 32/40 (80%) |
| Total cost | $2.19 | $3.28 |
| Median steps | 2 | 3 |
| Code error rate | 0% | 5% |
| Mean OQ (judge, 0–3) | 2.42 | 2.55 |
| Mean TQ (judge, 0–3) | 2.17 | 2.02 |
| TQ on passed tasks | 2.13 | 2.03 |
| TQ on failed tasks | 2.30 | 2.00 |

### Failure taxonomy

| Mechanism | Source | Sonnet failures | Haiku failures |
|---|---|---|---|
| task_ambiguity | benchmark-induced | 4 | 4 |
| constraint_violation | agent-induced | 3 | 1 |
| numeric_mismatch | agent-induced | 1 | 2 |
| iteration_limit | scaffolding-induced | 1 | 0 |
| rounding_artifact | eval-induced | 1 | 0 |
| output_format_failure | agent-induced | 0 | 1 |
| **Total failures** | | **10** | **8** |

`task_ambiguity` captures 4 tasks where both models failed with the same wrong value. All 4 involve ML model metrics or multi-output analysis where the question leaves implementation details underspecified (random seed, split method, output ordering) — both models made reasonable choices that don't match the benchmark's specific expected values. 40–50% of hard-task failures fall here.

## Repo structure

```
agent/
  ds_agent.py          # Tool-use loop: generate → execute → feed back
  code_runner.py       # Isolated subprocess execution with 30s timeout
  skills/
    data_analysis.md   # Skill file loaded as system prompt

eval/
  metrics.py           # Verifiable eval: @var[value] parsing, all-or-nothing scoring
  trajectory.py        # Step counts, error types, failure location
  taxonomy.py          # Failure taxonomy: mechanism × source classification
  llm_judge.py         # LLM judge: OQ + TQ (0–3), two separate API calls per task
  human_validation.py  # Blind annotation generator + kappa/alpha agreement

scripts/
  run_eval.py          # Agent runner: --model, --split, --yes flags
  download_dabench.py  # Fetches DABench from HuggingFace
  verify_setup.py      # Sanity check: confirms API key and Claude connection

notebooks/
  data_explorer.ipynb  # Dataset overview + dev/holdout comparison
  eval_analysis.ipynb  # Results analysis and visualization

data/
  test_sales.csv       # Tiny smoke-test CSV (committed)
  dabench/             # InfiAgent-DABench (gitignored — see scripts/download_dabench.py)
```

## Reproducing

```bash
# Clone and set up
git clone <this-repo>
cd ds-agent-eval
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# API keys
cp .env.example .env
# Edit .env and add ANTHROPIC_API_KEY and GEMINI_API_KEY

# Download benchmark data
python scripts/download_dabench.py

# Verify setup
PYTHONPATH=. python scripts/verify_setup.py

# Run agent on hard-final split (40 tasks)
PYTHONPATH=. python scripts/run_eval.py --split hard_final --model claude-sonnet-4-6 --yes
PYTHONPATH=. python scripts/run_eval.py --split hard_final --model claude-haiku-4-5 --yes

# Score results
PYTHONPATH=. python eval/metrics.py results/<run>.json
PYTHONPATH=. python eval/trajectory.py results/trajectories/<dir>/ results/<run>.json
PYTHONPATH=. python eval/taxonomy.py results/metrics_<s>.json results/metrics_<h>.json \
    results/trajectories/<s_dir>/ results/trajectories/<h_dir>/
PYTHONPATH=. python eval/llm_judge.py results/<run>.json \
    --model gemini-3.1-flash-lite --call-delay 15

# Run tests (45 total)
PYTHONPATH=. python -m pytest eval/test_metrics.py eval/test_trajectory.py -v
```

## Limitations

- **Skill file tuned on Sonnet failures.** The agent's system prompt was revised after observing Sonnet's failure modes on the dev set. The hard-final test set is clean (never seen during development), but the comparison is not fully zero-shot for Sonnet.
- **Bounded trajectory length.** The agent stops at 10 iterations. Tasks requiring more steps are penalized by the completion gate; task 424 illustrates where this creates a perverse outcome.
- **Sample size.** The test set is 40 hard tasks per model. At N=40, 3–4 swing tasks can move the pass-rate ranking; point estimates are reported with that caveat.
- **LLM judge noise.** At temperature=0, the judge is reproducible but not perfectly calibrated. Human-judge agreement is moderate (κ=0.59/0.33); OQ and TQ scores are coarse signals, not fine-grained measurements. Validation was run on dev (n=12) and transferred to holdout without re-validation — see *Scope and future work*.
- **One benchmark.** DABench covers tabular data analysis in Python; findings don't generalize beyond this domain without re-running — see *Scope and future work*.

## Scope and future work

Several extensions were considered and deliberately deferred to ship a complete, validated result rather than an endless one. Each was a scope decision with a reason, not an oversight:

**Larger holdout / more models.** The primary test set is 40 hard tasks (N=20 per model), where 3–4 swing tasks can move the pass-rate ranking. A larger holdout would tighten confidence intervals, and a third capability tier (or a non-Claude agent) would strengthen the capability-comparison claim. Deferred because the current N is sufficient to characterize *failure modes and process differences* — the project's actual thesis — even if not to make fine-grained ranking claims.

**Graceful-termination affordance.** Task 424 exposed that the agent has no sanctioned way to declare a task infeasible — it must either produce an answer or hit the iteration limit. Giving the agent an explicit "I cannot complete this, because X" action would let the completion gate distinguish principled non-completion from flailing, and would let the eval *credit* correct refusal. This is the single most valuable next step, but it requires changing the agent and re-running, so it was scoped out of this iteration.

**Sharper redundancy attribution.** The trajectory rubric's efficiency criterion penalizes redundant steps (e.g. repeated CSV reloads), but doesn't fully distinguish redundancy the subprocess-isolation scaffold *forces* from redundancy that is genuinely avoidable. A cleaner version would separate scaffold-necessitated re-establishment from true waste.

**Holdout judge re-validation.** Judge validation was performed on dev (n=12) and transferred to the same-distribution holdout rather than repeated on it (see *Evaluation as measurement*). A larger validation set, and a fresh holdout validation, would tighten the agreement estimates — particularly for trajectory quality, where n=12 leaves the κ estimate noisy.

**Separating "underspecified" from "wrong ground truth."** The `task_ambiguity` category currently groups underspecified tasks with genuinely wrong ground truth. Only one case (dev task 297) was independently verified as wrong GT; the holdout `task_ambiguity` failures are all underspecification. Reporting these as separate sub-counts would make the "≈40–50% of failures are not the agent's fault" claim even more precise.

**Broader benchmark coverage.** DABench covers Python tabular data analysis. The framework (verifiable + trajectory + taxonomy + judge) is domain-general, but the specific findings don't transfer to other modalities without re-running.

The consistent principle: validate the core rigorously, ship it, and document the extensions honestly — rather than expanding scope until nothing ships.

---

*Implementation was AI-assisted (Claude Code). Evaluation design, methodology, rubric development, and analysis are my own.*
