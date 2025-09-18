import os
from pathlib import Path
from dotenv import load_dotenv
from sample_dataset import load_dataset
from agent import run_multi_agent


def write_interaction_log(output_path: Path, question: str, ground_truth_answer: str, state: dict):
    lines = []
    lines.append(f"QUESTION:\n{question}\n\n")
    lines.append(f"GROUND TRUTH ANSWER:\n{ground_truth_answer}\n")
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
        lines.append("--- Model Answer (this attempt) ---\n")
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
    sampled = df.sample(n=5, random_state=123).reset_index(drop=True)

    for idx, (_, row) in enumerate(sampled.iterrows(), start=1):
        question = str(row.get('question', '')).strip()
        ground_truth_answer = str(row.get('answer', '')).strip()
        if not question:
            continue
        # Prepare per-question search log path and clear previous contents
        search_log_path = out_dir / f"q_a{idx}_search.txt"
        search_log_path.write_text("")
        # Use a stable thread id per question to persist memory across runs
        sample_id = str(row.get('id', f'train-{idx}'))
        thread_id = f"hotpot-{sample_id}"
        state = run_multi_agent(question, max_attempts=2, search_log_path=str(search_log_path), thread_id=thread_id)
        log_path = out_dir / f"q_a{idx}.txt"
        write_interaction_log(log_path, question, ground_truth_answer, state)


if __name__ == "__main__":
    main()


