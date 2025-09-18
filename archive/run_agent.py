import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from sample_dataset import load_dataset
from agent import run_multi_agent


def write_interaction_log(output_path: Path, question: str, state: dict):
    lines = []
    lines.append(f"QUESTION:\n{question}\n")
    lines.append("=" * 80 + "\n")
    for inter in state.get("interactions", []):
        lines.append(f"Attempt: {inter.get('attempt')}\n")
        lines.append(f"Query: {inter.get('query')}\n")
        lines.append("Results:\n")
        for r in inter.get("results_summary", []):
            lines.append(f"  [{r['index']}] {r['title']} | {r['url']}\n")
        lines.append("\n--- Prompt ---\n")
        lines.append(inter.get("prompt", "") + "\n\n")
        lines.append("--- Raw Reasoning Output ---\n")
        lines.append(inter.get("raw_reason", "") + "\n\n")
        lines.append("--- Answer ---\n")
        lines.append(inter.get("answer", "") + "\n\n")
        lines.append(f"FOLLOW_UP_QUERY: {inter.get('follow_up_query')}\n")
        lines.append("-" * 80 + "\n")
    content = "".join(lines)
    output_path.write_text(content)


def main():
    load_dotenv()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset('fullwiki', 'train')
    # Sample 5 random questions
    sampled = df.sample(n=20, random_state=42).reset_index(drop=True)

    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        question = str(row.get('question', '')).strip()
        if not question:
            continue
        state = run_multi_agent(question, max_attempts=2)
        log_path = out_dir / f"q_a{idx}.txt"
        write_interaction_log(log_path, question, state)


if __name__ == "__main__":
    main()


