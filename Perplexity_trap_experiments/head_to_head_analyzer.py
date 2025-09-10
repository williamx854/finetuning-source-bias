import json
import numpy as np
import os
import argparse

from config import Config

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Performs a head-to-head analysis of perplexity vs. relevance scores.")
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., scifact).")
    parser.add_argument("--retriever", type=str, required=True, help="Name of the retriever (e.g., vanilla_contriever).")
    
    cli_args = parser.parse_args()
    config = Config()
    
    retriever_name = cli_args.retriever
    dataset = cli_args.dataset
    root_dir = config.root_dir.rstrip('/')

    print(f"--- Head-to-Head Analysis for: {retriever_name} on {dataset} ---")

    # The input for this script is the output of the previous analyze_results.py script
    analysis_input_path = os.path.join(root_dir, 'data', 'cocktail', dataset, f'analysis_{retriever_name}.jsonl')

    print(f"Reading merged analysis file from: {analysis_input_path}")

    # --- Initialize Counters ---
    # Scenario 1: The LLM text is "better" (lower perplexity)
    llm_lower_ppl_total = 0
    llm_lower_ppl_and_higher_score = 0
    
    # Scenario 2: The Human text is "better" (lower perplexity)
    human_lower_ppl_total = 0
    human_lower_ppl_and_higher_score = 0

    try:
        with open(analysis_input_path, 'r', encoding='utf8') as f_in:
            for line in f_in:
                data = json.loads(line)
                
                # Compare perplexities for this pair
                if data['ppl_llm'] < data['ppl_human']:
                    llm_lower_ppl_total += 1
                    # Check if the retriever also gave the LLM doc a higher score
                    if data['llm'] > data['human']:
                        llm_lower_ppl_and_higher_score += 1
                
                elif data['ppl_human'] < data['ppl_llm']:
                    human_lower_ppl_total += 1
                    # Check if the retriever also gave the Human doc a higher score
                    if data['human'] > data['llm']:
                        human_lower_ppl_and_higher_score += 1
                        
    except FileNotFoundError:
        print(f"ERROR: Input file not found at {analysis_input_path}")
        print("Please run analyze_results.py for this configuration first.")
        exit()

    # --- Calculate and Print Results ---
    print("\n--- Analysis Results 📊 ---")
    
    # Calculate percentage for Scenario 1
    if llm_lower_ppl_total > 0:
        trap_percentage = (llm_lower_ppl_and_higher_score / llm_lower_ppl_total) * 100
        print(f"Scenario 1: LLM text has lower perplexity.")
        print(f"  - Total cases: {llm_lower_ppl_total}")
        print(f"  - In {trap_percentage:.2f}% of these cases, the retriever gave the LLM text a HIGHER score (the 'Perplexity Trap').")
    else:
        print("Scenario 1: No cases found where LLM text had lower perplexity.")

    print("-" * 20)

    # Calculate percentage for Scenario 2
    if human_lower_ppl_total > 0:
        logical_percentage = (human_lower_ppl_and_higher_score / human_lower_ppl_total) * 100
        print(f"Scenario 2: Human text has lower perplexity.")
        print(f"  - Total cases: {human_lower_ppl_total}")
        print(f"  - In {logical_percentage:.2f}% of these cases, the retriever gave the Human text a HIGHER score ('Logical Behavior').")
    else:
        print("Scenario 2: No cases found where Human text had lower perplexity.")
    
    print("\nJob finished.")
