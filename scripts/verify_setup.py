"""
Quick sanity check: confirms the API key loads correctly and Claude responds.
Run this once after adding your key to .env.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
import anthropic


def main():
    # Load .env from the project root (one level up from this script)
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or api_key == "your-key-here":
        print("ERROR: No API key found. Edit .env and replace 'your-key-here' with your actual key.")
        return

    print(f"API key loaded (starts with: {api_key[:12]}...)")

    # Make a minimal call to Claude to confirm everything works end-to-end
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=64,
        messages=[{"role": "user", "content": "Reply with exactly: 'Setup confirmed.'"}],
    )

    response_text = message.content[0].text
    print(f"Claude says: {response_text}")
    print("Setup complete.")


if __name__ == "__main__":
    main()
