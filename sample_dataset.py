#!/usr/bin/env python3
"""
Script to sample and display random question-answer pairs from the HotpotQA dataset.
"""

import pandas as pd
import random
import json
from pathlib import Path

def load_dataset(config='fullwiki', split='train'):
    """Load the HotpotQA dataset from parquet files."""
    base_path = Path('hotpot_qa') / config
    
    if split == 'validation':
        file_path = base_path / 'validation-00000-of-00001.parquet'
    elif split == 'train':
        # Load both training files and concatenate
        train_files = list(base_path.glob('train-*.parquet'))
        dfs = [pd.read_parquet(f) for f in train_files]
        return pd.concat(dfs, ignore_index=True)
    elif split == 'test' and config == 'fullwiki':
        file_path = base_path / 'test-00000-of-00001.parquet'
    else:
        raise ValueError(f"Invalid split '{split}' for config '{config}'")
    
    return pd.read_parquet(file_path)

def format_context(context):
    """Format the context for better readability."""
    formatted = []
    for title, sentences in zip(context['title'], context['sentences']):
        formatted.append(f"\n--- {title} ---")
        for i, sentence in enumerate(sentences):
            formatted.append(f"[{i}] {sentence}")
    return '\n'.join(formatted)

def format_supporting_facts(supporting_facts):
    """Format supporting facts for better readability."""
    facts = []
    for title, sent_id in zip(supporting_facts['title'], supporting_facts['sent_id']):
        facts.append(f"  - {title}, sentence {sent_id}")
    return '\n' + '\n'.join(facts)

def display_sample(sample, index):
    """Display a single sample in a formatted way."""
    print(f"\n{'='*80}")
    print(f"SAMPLE {index + 1}")
    print(f"{'='*80}")
    print(f"ID: {sample['id']}")
    print(f"Type: {sample['type']} | Level: {sample['level']}")
    print(f"\nQUESTION:")
    print(f"  {sample['question']}")
    print(f"\nANSWER:")
    print(f"  {sample['answer']}")
    print(f"\nSUPPORTING FACTS:{format_supporting_facts(sample['supporting_facts'])}")
    print(f"\nCONTEXT PARAGRAPHS:{format_context(sample['context'])}")

def main():
    """Main function to sample and display random pairs."""
    print("HotpotQA Dataset Random Sampler")
    print("Loading dataset...")
    
    # Load the validation set from distractor config
    df = load_dataset('fullwiki', 'train')
    
    print(f"Dataset loaded: {len(df)} examples")
    print(f"Question types: {df['type'].value_counts().to_dict()}")
    print(f"Difficulty levels: {df['level'].value_counts().to_dict()}")
    
    # Sample 5 random examples
    random_samples = df.sample(n=5, random_state=42).reset_index(drop=True)
    
    print(f"\nDisplaying 5 random question-answer pairs:")
    
    for i, (_, sample) in enumerate(random_samples.iterrows()):
        display_sample(sample, i)
    
    print(f"\n{'='*80}")
    print("End of samples")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
