from transformers import AutoModel, AutoTokenizer
from transformers import BertForMaskedLM, RobertaForMaskedLM, DistilBertForMaskedLM
import torch
from tqdm import tqdm
import json
import os
import argparse

# --- Global variables will be initialized in main ---
tokenizer = None
backbone = None
loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction="none")
device = None

def load_backbone(base_model_type, model_path):
    # This function remains the same as our last version
    global tokenizer, backbone, device
    # ... (function content omitted for brevity) ...
    print(f"Loading base model scaffold for type: {base_model_type}")
    if base_model_type == 'bert':
        scaffold_path = "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(scaffold_path)
        backbone = BertForMaskedLM.from_pretrained(scaffold_path).to(device)
    elif base_model_type == 'roberta':
        scaffold_path = "roberta-base"
        tokenizer = AutoTokenizer.from_pretrained(scaffold_path)
        backbone = RobertaForMaskedLM.from_pretrained(scaffold_path).to(device)
    else:
        raise ValueError(f"Unsupported base_model_type: {base_model_type}")

    print(f"Loading fine-tuned encoder weights from: {model_path}")
    model_encoder = AutoModel.from_pretrained(model_path).to(device)
    
    for name in model_encoder.state_dict():
        param_name_in_scaffold = f"{base_model_type}.{name}"
        if param_name_in_scaffold in backbone.state_dict():
            backbone.state_dict()[param_name_in_scaffold].copy_(model_encoder.state_dict()[name])
        else:
            if name.startswith("0.auto_model."):
                 new_name = name.replace("0.auto_model.", "")
                 param_name_in_scaffold = f"{base_model_type}.{new_name}"
                 if param_name_in_scaffold in backbone.state_dict():
                     backbone.state_dict()[param_name_in_scaffold].copy_(model_encoder.state_dict()[name])
    print(f"Successfully loaded weights into {base_model_type} backbone.")


def calc_ppl(text, batch_size): # <-- MODIFIED: Added batch_size as an argument
    encoded_input = tokenizer([text], return_tensors='pt', 
                              truncation=True, max_length=512, 
                              add_special_tokens=False)
    
    true_input_ids = encoded_input['input_ids'].squeeze(dim=0).to(device)

    input_ids = encoded_input['input_ids']
    if input_ids.shape[1] == 0: # Handle empty text
        return []
        
    input_ids = input_ids.repeat(input_ids.shape[1], 1)
    for i in range(input_ids.shape[0]):
        input_ids[i, i] = tokenizer.mask_token_id
    
    if 'token_type_ids' in encoded_input.keys():
        token_type_ids = encoded_input['token_type_ids']
        token_type_ids = token_type_ids.repeat(token_type_ids.shape[1], 1)
    else:
        token_type_ids = None
    
    attention_mask = encoded_input['attention_mask']
    attention_mask = attention_mask.repeat(attention_mask.shape[1], 1)

    loss = []
    # MODIFIED: Use the batch_size argument passed to the function
    for i in range(0, input_ids.shape[0], batch_size): 
        encoded_input_batch = {'input_ids': input_ids[i:i+batch_size].to(device), 
                               'attention_mask': attention_mask[i:i+batch_size].to(device)}
        if token_type_ids is not None:
            encoded_input_batch['token_type_ids'] = token_type_ids[i:i+batch_size].to(device)

        with torch.no_grad():
            model_output = backbone(**encoded_input_batch)[0]
            pred_token_ids = [model_output[j][i+j] for j in range(model_output.shape[0])]
            pred_token_ids = torch.stack(pred_token_ids).to(device)
            # MODIFIED: Use the batch_size argument passed to the function
            loss += loss_fct(pred_token_ids, true_input_ids[i:i+batch_size]).cpu().tolist()
    loss = [round(l, 6) for l in loss]
    return loss


# --- MAIN EXECUTION BLOCK ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset (e.g., scifact).")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to your main project data folder.")
    parser.add_argument("--model_path", type=str, required=True, help="Direct path to the retriever checkpoint.")
    parser.add_argument("--base_model_type", type=str, required=True, choices=['bert', 'roberta'], help="Base architecture of the retriever (e.g., bert).")
    parser.add_argument("--model_nickname", type=str, required=True, help="A short name for the model to use in the output filename.")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID to use.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for perplexity calculation.")

    args = parser.parse_args()
    device = f"cuda:{args.gpu}"

    load_backbone(args.base_model_type, args.model_path)

    corpus_merge_path = f"{args.root_dir}/data/cocktail/{args.dataset}/corpus/merge.jsonl"
    output_dir = f"{args.root_dir}/data/cocktail/{args.dataset}/perplexity/"
    output_file = f"{output_dir}/{args.model_nickname}.jsonl"
    os.makedirs(output_dir, exist_ok=True)
    
    corpus = []
    print(f"Loading merged corpus from: {corpus_merge_path}")
    with open(corpus_merge_path, 'r', encoding='utf8') as f:
        for line in f:
            data = json.loads(line)
            corpus.append((data['_id'], data['text']))
    
    print(f"Calculating perplexity for {len(corpus)} documents...")
    with open(output_file, 'w', encoding='utf8') as f_out:
        for doc_id, text in tqdm(corpus):
            # --- MODIFIED: Pass batch_size to the function ---
            loss = calc_ppl(text, args.batch_size)
            f_out.write(json.dumps({"_id": doc_id, "ppl": loss}) + '\n')
            
    print(f"Finished! Results saved to {output_file}")
