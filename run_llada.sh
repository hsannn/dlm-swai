#!/bin/bash

# Step 1] pre-compute score table
python build_scores.py \
    --dataset real_tox_4000 \
    --tokenizer GSAI-ML/LLaDA-8B-Instruct \
    --output_dir ./lookup_scores/llada8b/real_tox_4000

# [Step 2] Steering + Generation
# Models: GSAI-ML/LLaDA-8B-Instruct, Dream-org/Dream-v0-Instruct-7B
# WikiPol steering
python dlm_logit_steering.py \
    --dataset wikipol \
    --target I \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment rewriting \
    --steering_strength 0.7 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314

python dlm_logit_steering.py \
    --dataset wikipol \
    --target P \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment rewriting \
    --steering_strength 0.7 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314

python dlm_logit_steering.py \
    --dataset wikipol \
    --target N \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment rewriting \
    --steering_strength 0.7 \
    --block_length 32 \
    --max_new_tokens 32 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 300 \
    --seed 20260314


# OSE
python dlm_logit_steering.py \
    --dataset ose \
    --target E \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment open_ended \
    --steering_strength 0.7 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

python dlm_logit_steering.py \
    --dataset ose \
    --target I \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment open_ended \
    --steering_strength 0.7 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

python dlm_logit_steering.py \
    --dataset ose \
    --target A \
    --model_name GSAI-ML/LLaDA-8B-Instruct \
    --model_tag llada8b \
    --experiment open_ended \
    --steering_strength 0.7 \
    --block_length 128 \
    --max_new_tokens 128 \
    --temperature 1.0 \
    --batch_size 2 \
    --num_samples 189

# [Step 3] Evaluation
python eval_prepare.py \
    --all \
    --model_tag llada8b

# <GPT Judge>
export OPENAI_API_KEY="YOUR_OPENAI_KEY"

python eval_gptjudge.py \
  --input eval_data/llada8b/eval_ose_open_ended.json \
  --prompt sys_prompts/ose_sys_prompt.txt

# <Perspective API toxicity>
python eval_perspective.py \
    --api_key YOUR_KEY