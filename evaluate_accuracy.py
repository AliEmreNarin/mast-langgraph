#!/usr/bin/env python3
"""
Evaluate agent accuracy by comparing ground truth answers with agent responses
across all q_a*.txt files in the output directory.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import anthropic
from dotenv import load_dotenv

load_dotenv()

def extract_question_and_answers(trace_file: Path) -> Tuple[str, str, str]:
    """
    Extract question, ground truth, and agent's final answer from a trace file.
    Returns: (question, ground_truth, agent_answer)
    """
    content = trace_file.read_text()
    
    # Extract question
    question_match = re.search(r'QUESTION:\n(.*?)\n\nGROUND TRUTH ANSWER:', content, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    
    # Extract ground truth
    gt_match = re.search(r'GROUND TRUTH ANSWER:\n(.*?)\n=+', content, re.DOTALL)
    ground_truth = gt_match.group(1).strip() if gt_match else ""
    
    # Extract agent's final answer (last attempt's answer)
    # Look for the last "--- Model Answer (this attempt) ---" section
    model_answers = re.findall(r'--- Model Answer \(this attempt\) ---\n(.*?)(?=\n---|$)', content, re.DOTALL)
    agent_answer = model_answers[-1].strip() if model_answers else ""
    
    # If no model answer, try to get from Raw Reasoning Output
    if not agent_answer:
        raw_answers = re.findall(r'--- Raw Reasoning Output ---\n(.*?)(?=\n---|$)', content, re.DOTALL)
        if raw_answers:
            # Extract just the ANSWER: part
            raw_text = raw_answers[-1]
            answer_match = re.search(r'ANSWER:\s*(.*?)(?:\nFOLLOW_UP_QUERY:|$)', raw_text, re.DOTALL)
            agent_answer = answer_match.group(1).strip() if answer_match else raw_text.strip()
    
    return question, ground_truth, agent_answer

def evaluate_with_claude(question: str, ground_truth: str, agent_answer: str, client: anthropic.Anthropic) -> bool:
    """
    Use Claude to evaluate if the agent's answer is correct compared to ground truth.
    Returns True if correct, False if incorrect.
    """
    prompt = f"""You are evaluating whether an AI agent's answer is correct compared to the ground truth.

Question: {question}

Ground Truth Answer: {ground_truth}

Agent's Answer: {agent_answer}

Please determine if the agent's answer is essentially correct compared to the ground truth. Consider:
- Semantic equivalence (same meaning, different wording)
- Partial credit for answers that contain the correct information
- Minor variations in formatting or phrasing

Respond with only "CORRECT" or "INCORRECT" followed by a brief explanation.
"""
    
    try:
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        result = response.content[0].text.strip()
        return result.upper().startswith("CORRECT")
    except Exception as e:
        print(f"Error evaluating with Claude: {e}")
        return False

def main():
    # Initialize Claude client
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        return
    
    client = anthropic.Anthropic(api_key=api_key)
    
    # Find all trace files
    output_dir = Path("output")
    if not output_dir.exists():
        print("Error: output/ directory not found")
        return
    
    trace_files = sorted([f for f in output_dir.glob("q_a*.txt") if not f.name.endswith("_search.txt")], 
                        key=lambda x: int(x.stem.replace('q_a', '')))
    
    if not trace_files:
        print("No trace files found")
        return
    
    print(f"Found {len(trace_files)} trace files to evaluate...")
    
    correct_count = 0
    total_count = len(trace_files)
    results = []
    
    for i, trace_file in enumerate(trace_files, 1):
        print(f"Evaluating {trace_file.name} ({i}/{total_count})...")
        
        try:
            question, ground_truth, agent_answer = extract_question_and_answers(trace_file)
            
            if not question or not ground_truth or not agent_answer:
                print(f"  Warning: Missing data in {trace_file.name}")
                results.append((trace_file.name, False, "Missing data", ground_truth, agent_answer))
                continue
            
            is_correct = evaluate_with_claude(question, ground_truth, agent_answer, client)
            
            if is_correct:
                correct_count += 1
                print(f"  ✓ CORRECT")
            else:
                print(f"  ✗ INCORRECT")
            
            results.append((trace_file.name, is_correct, question[:50] + "...", ground_truth, agent_answer[:100] + "..."))
            
        except Exception as e:
            print(f"  Error processing {trace_file.name}: {e}")
            results.append((trace_file.name, False, f"Error: {e}", "", ""))
    
    # Print summary
    print("\n" + "="*80)
    print(f"EVALUATION SUMMARY")
    print("="*80)
    print(f"Total Questions: {total_count}")
    print(f"Correct Answers: {correct_count}")
    print(f"Incorrect Answers: {total_count - correct_count}")
    print(f"Accuracy: {correct_count/total_count:.1%}")
    
    # Print detailed results
    print("\nDETAILED RESULTS:")
    print("-"*80)
    for filename, is_correct, question_snippet, gt, agent_ans in results:
        status = "✓" if is_correct else "✗"
        print(f"{status} {filename}")
        if not is_correct and gt and agent_ans:
            print(f"    GT: {gt}")
            print(f"    Agent: {agent_ans}")
        print()

if __name__ == "__main__":
    main()
