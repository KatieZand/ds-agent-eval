import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic

from agent.code_runner import run_python_code

load_dotenv(Path(__file__).parent.parent / ".env")

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------
# This tells Claude what tools exist and how to call them.
# Claude reads the description and input_schema to know when and how to use it.
# The schema follows JSON Schema — "required" means Claude must include that field.

TOOLS = [
    {
        "name": "run_python_code",
        "description": (
            "Execute Python code and return its output. "
            "Use this to load the CSV, compute statistics, filter rows, etc. "
            "Always print results explicitly — return values are not captured."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Valid Python code to execute.",
                }
            },
            "required": ["code"],
        },
    }
]

# ---------------------------------------------------------------------------
# System prompt — loaded from the agent skill file
# ---------------------------------------------------------------------------
# Keeping instructions in a markdown file (rather than a hardcoded string)
# makes them easier to read, edit, and iterate on without touching agent logic.

SYSTEM_PROMPT = (Path(__file__).parent / "skills" / "data_analysis.md").read_text()

# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def run_agent(task: str, csv_path: str, model: str = "claude-sonnet-4-6") -> dict:
    """
    Run the data science agent on a task.

    Args:
        task: Natural language description of what to compute.
        csv_path: Path to the CSV file to analyze.

    Returns:
        A dict with the agent's final answer and the full message history
        (trajectory), which we'll need later for evaluation.
    """
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    # The messages list is the agent's entire "memory".
    # We grow it on every iteration and pass the whole thing each API call.
    messages = [
        {
            "role": "user",
            "content": f"Task: {task}\n\nCSV file path: {csv_path}",
        }
    ]

    # --- AGENT LOOP ---
    # Each iteration: call API → handle tool calls → feed results back → repeat.
    # Loop exits when Claude sets stop_reason = "end_turn" (final text answer).

    max_iterations = 10  # Safety limit — prevents runaway loops
    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Accumulate token usage across all iterations.
        # Each API call returns tokens for that call only, so we sum them up.
        total_input_tokens  += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

        # Append Claude's full response to the history.
        # We store response.content (a list of blocks), not just the text,
        # because it may contain tool_use blocks that we need to reference below.
        messages.append({"role": "assistant", "content": response.content})

        # ---------------------------------------------------------------
        # MEMORY HOOK POINT
        # For our bounded tasks (one CSV, one question) the conversation
        # never grows long enough to need this. But for longer tasks, this
        # is where you'd manage context before the next API call.
        #
        # OPTION 1 — Clipping (sliding window, simplest):
        #   Keep the original task + the last N messages, drop the middle.
        #
        #   if len(messages) > 12:
        #       messages = messages[:1] + messages[-8:]
        #
        # OPTION 2 — Summarization (more faithful, more expensive):
        #   Compress older turns into a single summary message using a
        #   cheap model, then replace them.
        #
        #   if len(messages) > 20:
        #       older = messages[1:-4]  # skip original task + keep last 4 turns
        #       summary = client.messages.create(
        #           model="claude-haiku-4-5",
        #           max_tokens=512,
        #           messages=[{
        #               "role": "user",
        #               "content": f"Summarize these agent steps concisely:\n{str(older)}"
        #           }]
        #       ).content[0].text
        #       messages = messages[:1] + [
        #           {"role": "user", "content": f"[Earlier steps summary: {summary}]"}
        #       ] + messages[-4:]
        # ---------------------------------------------------------------

        # Claude is done — extract and return the final text answer
        if response.stop_reason == "end_turn":
            final_answer = next(
                block.text for block in response.content if hasattr(block, "text")
            )
            return {
                "answer": final_answer,
                "trajectory": messages,
                "iterations": iteration + 1,
                "input_tokens": total_input_tokens,
                "output_tokens": total_output_tokens,
            }

        # Claude wants to call tools — find all tool_use blocks and execute them
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"[iteration {iteration + 1}] Running code...")
                output = run_python_code(block.input["code"])
                print(f"  → {output[:5000]}")

                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,  # Must match the block's id so Claude knows which call this answers
                        "content": output,
                    }
                )

        # Feed all tool results back as a single user turn, then loop again
        messages.append({"role": "user", "content": tool_results})

    return {
        "answer": "[agent hit max iterations without finishing]",
        "trajectory": messages,
        "iterations": max_iterations,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
    }
