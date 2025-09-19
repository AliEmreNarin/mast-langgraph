# What it does
# ------------
# - Reads one or more multi-agent trace files (plain text).
# - Loads taxonomy *definitions* and optional *examples*.
# - Prompts an LLM to label each failure mode (yes/no), say if the task was completed,
#   and produce a 1–2 sentence summary.
# - Enforces a strict JSON output schema (with a fallback parser).
# - Aggregates results (counts & percentages) and saves CSV/JSON outputs.

from __future__ import annotations

import argparse
import os
import pickle
import re
import json
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv 
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


# ----------------------------- Configuration to mirror notebook -----------------------------

FAILURE_MODES: List[str] = [
    "1.1", "1.2", "1.3", "1.4", "1.5",
    "2.1", "2.2", "2.3", "2.4", "2.5", "2.6",
    "3.1", "3.2", "3.3",
]

BUDGET_LIMIT = 1_048_570


# ----------------------------- IO helpers -----------------------------

def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_text_file(path: Optional[Path]) -> str:
    if not path:
        return ""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path.read_text(encoding="utf-8", errors="ignore")


# ----------------------------- LLM call (notebook-style) -----------------------------

def make_client(api_key: Optional[str], base_url: Optional[str]) -> OpenAI:
    kwargs: Dict[str, str] = {}
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def chat_completion_request_openai(client: OpenAI, model: str, prompt: str, temperature: float) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    if resp.choices:
        return resp.choices[0].message.content or ""
    return ""


def build_notebook_prompt(trace: str, definitions: str, examples: str) -> str:
    return (
        "Below I will provide a multiagent system trace. provide me an analysis of the failure modes and inefficiencies as I will say below. \n"
        "In the traces, analyze the system behaviour."
        "There are several failure modes in multiagent systems I identified. I will provide them below. Tell me if you encounter any of them, as a binary yes or no. \n"
        "Also, give me a one sentence (be brief) summary of the problems with the inefficiencies or failure modes in the trace. Only mark a failure mode if you can provide an example of it in the trace, and specify that in your summary at the end"
        "Also tell me whether the task is successfully completed or not, as a binary yes or no."
        "At the very end, I provide you with the definitions of the failure modes and inefficiencies. After the definitions, I will provide you with examples of the failure modes and inefficiencies for you to understand them better."
        "Tell me if you encounter any of them between the @@ symbols as I will say below, as a binary yes or no."
        "Here are the things you should answer. Start after the @@ sign and end before the next @@ sign (do not include the @@ symbols in your answer):"
        "*** begin of things you should answer *** @@"
        "A. Freeform text summary of the problems with the inefficiencies or failure modes in the trace: <summary>"
        "B. Whether the task is successfully completed or not: <yes or no>"
        "C. Whether you encounter any of the failure modes or inefficiencies:"
        "1.1 Disobey Task Specification: <yes or no>"
        "1.2 Disobey Role Specification: <yes or no>"
        "1.3 Step Repetition: <yes or no>"
        "1.4 Loss of Conversation History: <yes or no>"
        "1.5 Unaware of Termination Conditions: <yes or no>"
        "2.1 Conversation Reset: <yes or no>"
        "2.2 Fail to Ask for Clarification: <yes or no>"
        "2.3 Task Derailment: <yes or no>"
        "2.4 Information Withholding: <yes or no>"
        "2.5 Ignored Other Agent's Input: <yes or no>"
        "2.6 Action-Reasoning Mismatch: <yes or no>"
        "3.1 Premature Termination: <yes or no>"
        "3.2 No or Incorrect Verification: <yes or no>"
        "3.3 Weak Verification: <yes or no>"
        "@@*** end of your answer ***"
        "An example answer is: \n"
        "A. The task is not completed due to disobeying role specification as agents went rogue and started to chat with each other instead of completing the task. Agents derailed and verifier is not strong enough to detect it.\n"
        "B. no \n"
        "C. \n"
        "1.1 no \n"
        "1.2 no \n"
        "1.3 no \n"
        "1.4 no \n"
        "1.5 no \n"
        "1.6 yes \n"
        "2.1 no \n"
        "2.2 no \n"
        "2.3 yes \n"
        "2.4 no \n"
        "2.5 no \n"
        "2.6 yes \n"
        "2.7 no \n"
        "3.1 no \n"
        "3.2 yes \n"
        "3.3 no \n"
        "Here is the trace: \n"
        f"{trace}"
        "Also, here are the explanations (definitions) of the failure modes and inefficiencies: \n"
        f"{definitions} \n"
        "Here are some examples of the failure modes and inefficiencies: \n"
        # f"{examples}"
    )


def openai_evaluator(client: OpenAI, model: str, temperature: float, trace: str, definitions: str, examples: str) -> tuple[str, str]:
    prompt = build_notebook_prompt(trace, definitions, examples)
    response = chat_completion_request_openai(client, model, prompt, temperature)
    return prompt, response


# ----------------------------- Parsing (mirror notebook) -----------------------------

def parse_responses(responses: List[str]) -> Dict[str, List[int]]:
    def parse_single_response(response: str) -> Dict[str, int]:
        parsed: Dict[str, int] = {
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
            '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
            '3.1': 0, '3.2': 0, '3.3': 0
        }
        try:
            cleaned_response = response.strip()
            if cleaned_response.startswith('@@'):
                cleaned_response = cleaned_response[2:]
            if cleaned_response.endswith('@@'):
                cleaned_response = cleaned_response[:-2]

            for mode in list(parsed.keys()):
                patterns = [
                    rf"C\..*?{mode}.*?(yes|no)",
                    rf"C{mode}\s+(yes|no)",
                    rf"{mode}\s*[:]\s*(yes|no)",
                    rf"{mode}\s+(yes|no)",
                    rf"{mode}\s*\n\s*(yes|no)",
                    rf"C\.{mode}\s*\n\s*(yes|no)",
                ]

                value_set = False
                for pattern in patterns:
                    matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if matches:
                        parsed[mode] = 1 if matches[0].lower() == 'yes' else 0
                        value_set = True
                        break

                if not value_set:
                    general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                    match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                    if match:
                        parsed[mode] = 1 if match.group(1).lower() == 'yes' else 0
        except Exception as e:
            print(f"Error parsing single response: {e}")
        return parsed

    failure_modes: Dict[str, List[int]] = {
        '1.1': [], '1.2': [], '1.3': [], '1.4': [], '1.5': [],
        '2.1': [], '2.2': [], '2.3': [], '2.4': [], '2.5': [], '2.6': [],
        '3.1': [], '3.2': [], '3.3': []
    }

    for i, response in enumerate(responses):
        parsed_single = parse_single_response(response)
        for mode, val in parsed_single.items():
            failure_modes[mode].append(val)

    max_length = max((len(values) for values in failure_modes.values()), default=0)
    for mode in failure_modes:
        if len(failure_modes[mode]) < max_length:
            failure_modes[mode].extend([0] * (max_length - len(failure_modes[mode])))

    return failure_modes

def parse_single_response_for_log(response: str) -> Dict[str, int]:
    """Light wrapper to reuse the single-response parsing from parse_responses."""
    # Re-implement minimal logic to avoid nested function access
    result: Dict[str, int] = {
        '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0, '1.5': 0,
        '2.1': 0, '2.2': 0, '2.3': 0, '2.4': 0, '2.5': 0, '2.6': 0,
        '3.1': 0, '3.2': 0, '3.3': 0
    }
    try:
        cleaned_response = response.strip()
        if cleaned_response.startswith('@@'):
            cleaned_response = cleaned_response[2:]
        if cleaned_response.endswith('@@'):
            cleaned_response = cleaned_response[:-2]
        for mode in list(result.keys()):
            patterns = [
                rf"C\..*?{mode}.*?(yes|no)",
                rf"C{mode}\s+(yes|no)",
                rf"{mode}\s*[:]\s*(yes|no)",
                rf"{mode}\s+(yes|no)",
                rf"{mode}\s*\n\s*(yes|no)",
                rf"C\.{mode}\s*\n\s*(yes|no)",
            ]
            matched = False
            for pattern in patterns:
                matches = re.findall(pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if matches:
                    result[mode] = 1 if matches[0].lower() == 'yes' else 0
                    matched = True
                    break
            if not matched:
                general_pattern = rf"(?:C\.)?{mode}.*?(yes|no)"
                match = re.search(general_pattern, cleaned_response, re.IGNORECASE | re.DOTALL)
                if match:
                    result[mode] = 1 if match.group(1).lower() == 'yes' else 0
    except Exception as e:
        print(f"Error parsing response for log: {e}")
    return result


def print_mode_stats(failure_mode_results: Dict[str, List[int]]) -> None:
    for mode, values in failure_mode_results.items():
        total_yes = sum(values)
        total = len(values) if values else 0
        pct = (total_yes / total * 100.0) if total else 0.0
        print(f"{mode}: {values[:5]} (total yes: {total_yes}/{total}, {round(pct, 2)}%)")


def compute_source_averages(failure_mode_results: Dict[str, List[int]], source_sizes: List[int]) -> Dict[str, List[float]]:
    source_indices: List[int] = [0]
    for size in source_sizes[:-1]:
        source_indices.append(source_indices[-1] + size)

    averages: Dict[str, List[float]] = {}
    for mode, values in failure_mode_results.items():
        source_averages: List[float] = []
        for i in range(len(source_sizes)):
            start_idx = source_indices[i]
            if i == len(source_sizes) - 1:
                end_idx = len(values)
            else:
                end_idx = start_idx + source_sizes[i]
            source_values = values[start_idx:end_idx]
            avg = (sum(source_values) / len(source_values)) if source_values else 0.0
            source_averages.append(avg)
        averages[mode] = source_averages
    return averages


# ----------------------------- CLI -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run notebook-equivalent LLM judge on traces (free-text parsing)")
    p.add_argument("--traces", nargs="*", help="One or more trace .txt files")
    p.add_argument("--all-traces", action="store_true", help="Run on all q_a*.txt files in output/ directory")
    p.add_argument("--definitions", required=True, help="Path to definitions.txt")
    p.add_argument("--examples", default="", help="Path to examples.txt (optional)")
    p.add_argument("--outdir", default="saved_results", help="Directory to save pickled results")
    p.add_argument("--log-file", default="", help="Path to JSONL log file for prompts/responses")
    p.add_argument("--model", default="gpt-5", help="Model name (default mirrors notebook)")
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"), help="API key or set OPENAI_API_KEY")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Optional base URL for OpenAI-compatible servers")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (default mirrors notebook)")
    p.add_argument("--source-sizes", default="", help="Comma-separated sizes to compute per-source averages, e.g. 30,30,30")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    ensure_outdir(outdir)
    
    # Determine trace files to process
    if args.all_traces:
        # Find all q_a*.txt files in output/ directory
        output_dir = Path("output")
        if not output_dir.exists():
            print("Error: output/ directory not found. Run the agent first to generate traces.")
            return
        trace_files = sorted([f for f in output_dir.glob("q_a*.txt") if not f.name.endswith("_search.txt")], 
                            key=lambda x: int(x.stem.replace('q_a', '')))
        if not trace_files:
            print("Error: No q_a*.txt files found in output/ directory.")
            return
        print(f"Found {len(trace_files)} trace files: {[f.name for f in trace_files]}")
    elif args.traces:
        trace_files = [Path(t) for t in args.traces]
    else:
        print("Error: Must specify either --traces or --all-traces")
        return
    
    # If a specific log file is provided, use it; otherwise we will write one log per trace
    global_log_file_path = Path(args.log_file) if args.log_file else None

    client = make_client(api_key=args.api_key, base_url=args.base_url)

    definitions = read_text_file(Path(args.definitions))
    examples = read_text_file(Path(args.examples)) if args.examples else ""

    full_trace_list: List[str] = []
    for trace_path in trace_files:
        trace_text = read_text_file(trace_path)
        if len(trace_text + examples) > BUDGET_LIMIT:
            trace_text = trace_text[: max(0, BUDGET_LIMIT - len(examples))]
        full_trace_list.append(trace_text)

    openai_results: List[str] = []
    checkpoint_path = outdir / 'gpt5_results_checkpoint.pkl'

    if global_log_file_path:
        print(f"Logging prompts/responses to: {global_log_file_path}")
    else:
        print(f"Logging prompts/responses per trace to: {outdir}/judge_log_{'<trace_stem>'}.jsonl")

    for i, trace_text in enumerate(full_trace_list):
        try:
            prompt, evaluation = openai_evaluator(client, args.model, args.temperature, trace_text, definitions, examples)
            openai_results.append(evaluation)

            # Write log entry for this trace
            try:
                parsed = parse_single_response_for_log(evaluation)
                log_entry = {
                    "index": i,
                    "trace_path": str(trace_files[i]) if i < len(trace_files) else "",
                    "prompt": prompt,
                    "response": evaluation,
                    "parsed": parsed,
                }
                # Determine file path: use provided --log-file, else name by trace stem
                if global_log_file_path:
                    per_trace_log_path = global_log_file_path
                else:
                    trace_stem = trace_files[i].stem if i < len(trace_files) else f"trace_{i}"
                    per_trace_log_path = outdir / f"judge_log_{trace_stem}.jsonl"
                with open(per_trace_log_path, 'a', encoding='utf-8') as lf:
                    lf.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

                # Also save a per-trace JSON with parsed results
                per_trace_path = outdir / f"judge_{trace_stem}.json"
                with open(per_trace_path, 'w', encoding='utf-8') as pf:
                    json.dump({
                        "trace_path": str(trace_files[i]) if i < len(trace_files) else "",
                        "failure_mode_results": parsed,
                        "raw_response": evaluation,
                    }, pf, ensure_ascii=False, indent=2)
            except Exception as le:
                print(f"Failed to write log entry or per-trace file for {i+1}: {le}")

            with open(checkpoint_path, 'wb') as f:
                pickle.dump(openai_results, f)

            if (i + 1) % 10 == 0:
                with open(outdir / f'o1_results_backup_{i+1}.pkl', 'wb') as f:
                    pickle.dump(openai_results, f)

            print(f"Completed and saved evaluation {i+1}/{len(full_trace_list)}")
        except Exception as e:
            print(f"Error on evaluation {i+1}: {str(e)}")
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(openai_results, f)

    o1_results = openai_results

    failure_mode_results = parse_responses(o1_results)
    print_mode_stats(failure_mode_results)

    if args.source_sizes:
        try:
            sizes = [int(x.strip()) for x in args.source_sizes.split(',') if x.strip()]
            averages = compute_source_averages(failure_mode_results, sizes)
            for mode, scores in averages.items():
                pct_scores = [round(score * 100, 2) for score in scores]
                print(f"{mode}: {pct_scores}%")
            sample_key = next(iter(averages.keys()), None)
            if sample_key:
                print("\nSample of average_scores_by_source dictionary:")
                print(f"{sample_key}: {averages[sample_key]}")
        except Exception as e:
            print(f"Failed to compute source averages: {e}")

    # Save a summary JSON of final results
    try:
        first_stem = trace_files[0].stem if trace_files else 'trace'
        summary_path = outdir / f'judge_summary_{first_stem}.json'
        with open(summary_path, 'w', encoding='utf-8') as sf:
            json.dump({
                "failure_mode_results": failure_mode_results,
            }, sf, ensure_ascii=False, indent=2)
        print(f"Saved summary to: {summary_path}")
    except Exception as se:
        print(f"Failed to save summary: {se}")



if __name__ == "__main__":
    main()

#
# Example usage:
# Run on all q_a*.txt files in output/:
# python run_judge.py --all-traces --definitions MAST/taxonomy_definitions_examples/definitions.txt --outdir mast_agent/saved_results
#
# Run on specific files:
# python run_judge.py --traces output/q_a1.txt output/q_a2.txt --definitions MAST/taxonomy_definitions_examples/definitions.txt --outdir mast_agent/saved_results