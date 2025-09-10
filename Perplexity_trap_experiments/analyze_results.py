import json
import numpy as np
import os
import argparse

from config import Config

RETRIEVER_TO_PPL_NICKNAME = {
    "vanilla_contriever": "vanilla_contriever",
    "contriever": "contriever-msmarco",
    "e5-base-unsupervised": "e5-base-unsupervised",
    "e5-base": "e5-base",
    "e5-ft-msmarco": "e5-ft-msmarco"

}



def load_average_perplexities(dataset, retriever_name, root_dir):
    """
    Loads the retriever-specific perplexity file, calculates the average perplexity
    for each document, and returns a dictionary mapping doc_id -> avg_perplexity.
    """
    ppl_nickname = RETRIEVER_TO_PPL_NICKNAME.get(retriever_name, retriever_name)
    
    ppl_path = os.path.join(root_dir, 'data', 'cocktail', dataset, 'perplexity', f'{ppl_nickname}.jsonl')

    print(f"Loading perplexities from: {ppl_path}")
    
    did_to_avg_ppl = {}
    with open(ppl_path, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            doc_id = data['_id']
            if data['ppl']: 
                avg_ppl = sum(data['ppl']) / len(data['ppl'])
                did_to_avg_ppl[doc_id] = avg_ppl
    return did_to_avg_ppl

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., scifact).")
    parser.add_argument("--retriever", type=str, required=True, help="Name of the retriever (e.g., vanilla_contriever).")
    
    cli_args = parser.parse_args()
    config = Config() 
    
    retriever_name = cli_args.retriever
    dataset = cli_args.dataset
    root_dir = config.root_dir

    print(f"--- Analyzing Results for: {retriever_name} on {dataset} ---")

    # 1. Load the calculated average perplexities for each document
    did_to_perplexity = load_average_perplexities(dataset, retriever_name, root_dir)

    # 2. Define input and output files
    relevance_input_path = f"{root_dir}/data/cocktail/{dataset}/pos_rel_{retriever_name}.jsonl"
    analysis_output_path = f"{root_dir}/data/cocktail/{dataset}/analysis_{retriever_name}.jsonl"

    print(f"Reading relevance scores from: {relevance_input_path}")
    print(f"Saving merged analysis file to: {analysis_output_path}")

    # 3. Initialize lists to store data for overall average calculation
    human_scores, llm_scores = [], []
    human_perplexities, llm_perplexities = [], []

    # 4. Loop through relevance file, merge with perplexity, and save
    with open(relevance_input_path, 'r', encoding='utf8') as f_in, \
         open(analysis_output_path, 'w', encoding='utf8') as f_out:
        
        for line in f_in:
            data = json.loads(line)
            human_doc_id = data['doc_id']
            llm_doc_id = '-' + human_doc_id

            # Look up the perplexity for both human and LLM docs
            ppl_human = did_to_perplexity.get(human_doc_id)
            ppl_llm = did_to_perplexity.get(llm_doc_id)

            if ppl_human is not None and ppl_llm is not None:
                # Add perplexity to the data object
                data['ppl_human'] = ppl_human
                data['ppl_llm'] = ppl_llm
                
                # Write the new enriched data to the output file
                f_out.write(json.dumps(data) + '\n')

                # Add to lists for final average calculation
                human_scores.append(data['human'])
                llm_scores.append(data['llm'])
                human_perplexities.append(ppl_human)
                llm_perplexities.append(ppl_llm)

    # 5. Calculate and print the final summary
    print("\n--- Overall Averages ---")
    if human_scores:
        print(f"Avg Human Score:       {np.mean(human_scores):.4f}")
        print(f"Avg LLM Score:         {np.mean(llm_scores):.4f}")
        print(f"Avg Human Perplexity:  {np.mean(human_perplexities):.4f}")
        print(f"Avg LLM Perplexity:    {np.mean(llm_perplexities):.4f}")
    else:
        print("No matching data found to calculate averages.")
    print("------------------------\n")
