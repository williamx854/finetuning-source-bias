# Anonymous Dense Retriever Bias Experiments

This repository contains the **custom scripts and configurations** used in our ECIR 2026 submission on **source bias in dense retrievers**.

We **do not** bundle full third‑party code. Instead, please clone the upstream repositories listed below and copy our provided files into them.

---

## 0. Minimal workflow (TL;DR)

1. **Clone** the target upstream repo (Source‑Bias / SyCL / Perplexity‑Trap).
2. **Copy** our files into the indicated paths (overwrite if names match).
3. **Run** the example commands below.

---

## 1. Source‑Bias evaluation (human vs LLM)

Clone the official repository:

```bash
git clone https://github.com/KID-22/Source-Bias
```

Copy our custom script into the evaluation folder:

```bash
# from THIS repo's root
cp source_bias_custom/evaluate_custom.py Source-Bias/evaluate/
```

### Example: Evaluate **E5** on **SciFact**

```bash
# Test on human-written corpus
python evaluate/evaluate_custom.py \
    --test_dataset scifact \
    --target llama-2-7b-chat \
    --candidate_lm llama-2-7b-chat \
    --model_path intfloat/e5-base-unsupervised \
    --score_func cos_sim

# Test on LLM-generated corpus
python evaluate/evaluate_custom.py \
    --test_dataset scifact \
    --target llama-2-7b-chat \
    --candidate_lm llama-2-7b-chat \
    --model_path intfloat/e5-base-unsupervised \
    --score_func cos_sim
```

> The script reports standard retrieval metrics used in the paper.

---

## 2. Fine‑tuning & training (SyCL)

Clone the official repository:

```bash
git clone https://github.com/BatsResearch/sycl
```

Copy our data config(s) into SyCL:

```bash
# from THIS repo's root
cp sycl_configs/nq320k_30k_binary.json sycl/data_configs/
cp sycl_configs/real_binary.json        sycl/data_configs/
```

### Example A: Fine‑tune **E5** on **NQ320K** (30k pairs, InfoNCE)

```bash
# run from the sycl repo root
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='nq320k' \
    --mqrel_conf="./data_configs/nq320k_30k_binary.json" \
    --model_name_or_path="intfloat/e5-base-unsupervised" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='yes' \
    --loss="infonce" \
    --trust_remote_code='true' \
    --group_size='3' \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/ft_nq320k_30k_binary_e5_unsupervised_3epochs" \
    --report_to='none' \
    --save_strategy='epoch' \
    --per_device_train_batch_size='16' \
    --learning_rate='1e-5' \
    --num_train_epochs='3' \
    --logging_steps='10' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers='2' \
    --save_only_model='true'
```

### Example B: Fine‑tune **E5** on **MS MARCO** (1 epoch, InfoNCE)

```bash
# run from the sycl repo root
deepspeed --include localhost:0,1,2,3 train.py \
    --deepspeed="deepspeed_conf.json" \
    --dataset_name='msmarco' \
    --mqrel_conf="data_configs/real_binary.json" \
    --model_name_or_path="intfloat/e5-base-unsupervised" \
    --encoder_class='default' \
    --pooling='mean' \
    --normalize='yes' \
    --loss="infonce" \
    --trust_remote_code='true' \
    --group_size='3' \
    --query_max_len='256' \
    --passage_max_len='256' \
    --output_dir="./model_output/e5_ft_real_binary_msmarco" \
    --report_to='wandb' \
    --save_strategy='epoch' \
    --per_device_train_batch_size='16' \
    --learning_rate='1e-5' \
    --num_train_epochs='1' \
    --logging_steps='1' \
    --gradient_accumulation_steps='4' \
    --warmup_ratio='0.05' \
    --eval_strategy='no' \
    --dataloader_num_workers='2' \
    --save_only_model='true'
```

---

## 3. Perplexity experiments (Perplexity‑Trap)

Clone the official repository:

```bash
git clone https://github.com/WhyDwelledOnAi/Perplexity-Trap
```

Copy our modified experiment folder into the cloned repo and overwrite if prompted:

```bash
# from THIS repo's root
cp -r Perplexity_trap_experiments/ Perplexity-Trap/
```

Run experiments following the original repo’s commands **using the modified files**.

---

## Notes

* This repository is anonymized for review.
* After acceptance, we will release non‑anonymous materials with proper attribution and any additional integration.
