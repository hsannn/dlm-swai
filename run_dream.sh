#!/bin/bash

# score table
python build_scores.py \
    --dataset real_tox_4000 \
    --tokenizer Dream-org/Dream-v0-Instruct-7B \
    --output_dir ./lookup_scores/dream7b/real_tox_4000

# Steering + Generation
# Models: GSAI-ML/LLaDA-8B-Instruct, Dream-org/Dream-v0-Instruct-7B
# WikiPol
python dlm_logit_steering.py \
    --dataset wikipol \
    --target I \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment rewriting \
    --steering_strength 0.5 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314

python dlm_logit_steering.py \
    --dataset wikipol \
    --target P \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment rewriting \
    --steering_strength 0.5 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314

python dlm_logit_steering.py \
    --dataset wikipol \
    --target N \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment rewriting \
    --steering_strength 0.5 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314

# RealTox
python dlm_logit_steering.py \
    --dataset real_tox_4000 \
    --target T \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment rewriting \
    --steering_strength 0.5 \
    --block_length 64 \
    --max_new_tokens 64 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 400 \
    --seed 20260313

python dlm_logit_steering.py \
    --dataset real_tox_4000 \
    --target NT \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment rewriting \
    --steering_strength 0.5 \
    --block_length 64 \
    --max_new_tokens 64 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 400 \
    --seed 20260313

# OSE
python dlm_logit_steering.py \
    --dataset ose \
    --target E \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment open_ended \
    --steering_strength 0.5 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

python dlm_logit_steering.py \
    --dataset ose \
    --target I \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment open_ended \
    --steering_strength 0.5 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

python dlm_logit_steering.py \
    --dataset ose \
    --target A \
    --model_name Dream-org/Dream-v0-Instruct-7B \
    --model_tag dream7b \
    --experiment open_ended \
    --steering_strength 0.5 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

# [Step 3] Evaluation
python eval_prepare.py \
    --all \
    --model_tag dream7b

# <GPT Judge>
export OPENAI_API_KEY="YOUR_KEY"

python eval_gptjudge.py \
  --input eval_data/dream7b/eval_ose.json \
  --prompt sys_prompts/ose_sys_prompt.txt