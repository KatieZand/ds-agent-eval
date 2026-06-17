import subprocess
import sys
import tempfile
import os


def run_python_code(code: str, timeout: int = 30) -> str:
    """
    Execute a string of Python code in an isolated subprocess and return its output.

    Why subprocess instead of exec()?
    exec() runs code inside our own process — a crash or infinite loop in the
    agent's code would take down our whole program. subprocess spawns a fresh
    Python process, so the agent's code is fully isolated.

    Args:
        code: Python source code to run (written by the agent).
        timeout: Max seconds to wait before killing the process.

    Returns:
        The combined stdout/stderr output as a string.
    """
    # Write the code to a temporary file — subprocess needs a file path, not a string
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],  # sys.executable = the same Python that runs this file
            capture_output=True,          # capture stdout and stderr instead of printing them
            text=True,                    # decode bytes to string automatically
            timeout=timeout,
        )
        # Combine stdout and stderr so the agent sees both normal output and error messages
        output = result.stdout
        if result.stderr:
            output += "\n[stderr]\n" + result.stderr
        return output.strip() if output.strip() else "(no output)"

    except subprocess.TimeoutExpired:
        return f"[error] Code timed out after {timeout} seconds."
    except Exception as e:
        return f"[error] {e}"
    finally:
        # Always delete the temp file, even if an exception occurred
        os.unlink(tmp_path)
