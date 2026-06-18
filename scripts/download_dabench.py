"""
Download the InfiAgent-DABench dataset into data/dabench/.

Run once before using the benchmark:
    python scripts/download_dabench.py

What gets downloaded:
    data/dabench/questions.jsonl   — 257 tasks with format specs
    data/dabench/labels.jsonl      — ground truth answers
    data/dabench/tables/           — 68 CSV files used by the tasks

The data/ folder is gitignored — this script is how anyone who clones
the repo reproduces the data locally.
"""
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download

REPO_ID = "infiagent/DABench"
OUT_DIR = Path("data/dabench")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading questions...")
    src = hf_hub_download(repo_id=REPO_ID, filename="da-dev-questions.jsonl", repo_type="dataset")
    shutil.copy(src, OUT_DIR / "questions.jsonl")

    print("Downloading labels...")
    src = hf_hub_download(repo_id=REPO_ID, filename="da-dev-labels.jsonl", repo_type="dataset")
    shutil.copy(src, OUT_DIR / "labels.jsonl")

    print("Downloading CSV tables (this may take a moment)...")
    snapshot_dir = Path(src).parent
    tables_src = snapshot_dir / "da-dev-tables"
    tables_dst = OUT_DIR / "tables"
    if tables_dst.exists():
        shutil.rmtree(tables_dst)
    shutil.copytree(tables_src, tables_dst)

    # Quick sanity check
    questions = [json.loads(l) for l in (OUT_DIR / "questions.jsonl").read_text().splitlines()]
    labels = [json.loads(l) for l in (OUT_DIR / "labels.jsonl").read_text().splitlines()]
    tables = list(tables_dst.glob("*.csv"))

    print(f"\nDone.")
    print(f"  {len(questions)} questions")
    print(f"  {len(labels)} labels")
    print(f"  {len(tables)} CSV files")
    print(f"\nFirst task preview:")
    q = questions[0]
    print(f"  id:       {q['id']}")
    print(f"  level:    {q['level']}")
    print(f"  file:     {q['file_name']}")
    print(f"  question: {q['question']}")
    print(f"  format:   {q['format']}")


if __name__ == "__main__":
    main()
