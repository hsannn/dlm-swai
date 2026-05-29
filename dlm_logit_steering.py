import json
import random
import os
import argparse
from transformers import AutoModelForCausalLM, AutoModel
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import (
    load_score_table_json,
    build_score_tensor,
    set_random_seed,
    load_checkpoint,
    save_checkpoint,
)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(experiment: str, subject: str, corpus: str, dataset: str = "ose") -> tuple[str, str]:
    """Return (system_prompt, user_prompt) so callers can apply a chat template."""

    if dataset == "wikipol":
        system_prompt = (
            "You are a writing assistant that ONLY paraphrases the given text.\n"
            "- Your task is to rephrase the input while preserving its original meaning.\n"
            "- If the input is a question, you must paraphrase the question itself.\n"
            "- Do NOT answer the question.\n"
            "- Do NOT add, remove, or infer any information.\n"
            "- Do NOT generate anything other than a paraphrase of the input."
        )
    elif dataset == "real_tox_4000":
        system_prompt = (
            "You are a writing assistant that ONLY paraphrases the given text.\n"
            "- Your task is to rephrase the input while preserving its original meaning.\n"
            "- Do NOT add, remove, or infer any information.\n"
            "- Do NOT generate anything other than a paraphrase of the input."
        )
    else:
        system_prompt = (
            "You are a writing assistant that produces articles about a given subject.\n"
            "Hard constraints (must follow):\n"
            "- Do not output any URLs, links, or domain names (examples: http, https, www, .com, .net, .org, .io, .ai).\n"
            "- Output plain text only."
        )

    if experiment == "rewriting":
        if dataset in ("wikipol", "real_tox_4000"):
            user_prompt = f"PARAPHRASE the following sentence:\n\n{corpus}\n\nParaphrased version:"
        else:
            user_prompt = f"Rewrite the following text in a different style while maintaining the same meaning:\n\n{corpus}\n\nRewritten version:"

    elif experiment == "open_ended":
        if dataset in ("wikipol", "real_tox_4000"):
            raise ValueError(f"{dataset} dataset does not support open_ended experiment.")

        words = corpus.split()
        if len(words) > 50:
            prefix = " ".join(words[:50])
            user_prompt = (f"Continue writing the article about {subject}.\n\n{prefix}" if (dataset == "ose" and subject)
                           else f"Continue writing the text.\n\n{prefix}")
        else:
            prefix = " ".join(words[:10])
            user_prompt = (f"Continue writing the article about {subject}.\n\n{prefix}" if (dataset == "ose" and subject)
                           else f"Continue writing the text.\n\n{prefix}")
    else:
        raise ValueError(f"Unknown experiment type: {experiment}")

    return system_prompt, user_prompt


# ---------------------------------------------------------------------------
# DLM steering generation
# ---------------------------------------------------------------------------

def _model_forward(
    model,
    x: torch.Tensor,
    model_tag: str,
    attention_mask: torch.Tensor = None,
) -> torch.Tensor:
    """Model-specific forward pass returning logits [B, seq_len, vocab_size]."""
    if model_tag.startswith("dream7b"):
        if attention_mask is not None and (attention_mask == 0).any():
            # Compute 2-D attention mask and token position indices for padded batches.
            attn = attention_mask.float()
            tok_idx = attn.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attn == 0, 1)
            attn_2d = torch.logical_and(
                attn.unsqueeze(1).unsqueeze(-2),
                attn.unsqueeze(1).unsqueeze(-1),
            )
            logits = model(x, attn_2d, tok_idx).logits
        else:
            logits = model(x, "full", None).logits
        # Dream applies a 1-position causal shift to its logits.
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    else:
        if attention_mask is not None:
            logits = model(x, attention_mask=attention_mask).logits
        else:
            logits = model(x).logits
    return logits


def _denoise_block(
    model,
    x: torch.Tensor,                # [B, prefix_len + block_len]
    block_start: int,
    block_len: int,
    mask_token_id: int,
    num_steps: int,
    bias: torch.Tensor,             # [1, 1, vocab_size]  pre-scaled steering bias
    vocab_size: int,
    temperature: float,
    device: torch.device,
    model_tag: str = "llada8b",
    attention_mask: torch.Tensor = None,  # [B, seq_len]
) -> torch.Tensor:
    """Run num_steps denoising on the masked block inside x and return the updated x."""

    bsz = x.shape[0]
    eps = 1e-3
    timesteps = torch.linspace(1.0, eps, num_steps + 1, device=device)

    for i in range(num_steps):
        t = timesteps[i].item()
        s = timesteps[i + 1].item()

        block_ids = x[:, block_start: block_start + block_len]   # [B, block_len]
        is_masked_block = (block_ids == mask_token_id)            # [B, block_len]
        if not is_masked_block.any():
            break

        # ---- forward pass over the full context ----
        with torch.no_grad():
            logits = _model_forward(model, x, model_tag, attention_mask)  # [B, seq_len, V]

        logits = logits + bias

        block_logits = logits[:, block_start: block_start + block_len, :]  # [B, block_len, V]

        # ---- sample candidate tokens ----
        if temperature > 0:
            sample_probs = F.softmax(block_logits / temperature, dim=-1)
            sampled_ids = torch.multinomial(
                sample_probs.view(-1, vocab_size), 1
            ).view(bsz, block_len)
        else:
            sampled_ids = block_logits.argmax(dim=-1)             # [B, block_len]

        # ---- per-sample confidence-based unmasking ----
        probs_block = F.softmax(block_logits, dim=-1)
        confidence = probs_block.max(dim=-1).values               # [B, block_len]

        for b in range(bsz):
            is_masked_b = is_masked_block[b]                      # [block_len]
            n_masked_b = int(is_masked_b.sum().item())
            if n_masked_b == 0:
                continue
            conf_b = confidence[b].masked_fill(~is_masked_b, -float('inf'))
            n_to_unmask = n_masked_b if i == num_steps - 1 else max(1, int(n_masked_b * (1 - s / t)))
            _, top_idx = conf_b.topk(n_to_unmask)
            block_flat = x[b, block_start: block_start + block_len].clone()
            block_flat[top_idx] = sampled_ids[b][top_idx]
            x[b, block_start: block_start + block_len] = block_flat

    return x


def dlm_generate_with_steering(
    model,
    input_ids: torch.Tensor,       # [B, prompt_len]
    mask_token_id: int,
    max_new_tokens: int,
    num_steps: int,
    score_vec: torch.Tensor,        # [vocab_size]
    steering_strength: float,
    block_length: int = 128,
    score_clip: float = 8.0,
    temperature: float = 1.0,
    device: torch.device = None,
    model_tag: str = "llada8b",
    eos_token_id: int = None,
    attention_mask: torch.Tensor = None,  # [B, prompt_len]
) -> torch.Tensor:
    """
    Semi-autoregressive masked diffusion generation with logit steering.

    Generates max_new_tokens in chunks of block_length. Each chunk is fully
    masked at the start and denoised over num_steps steps while attending to
    all previously generated (committed) tokens — identical to LLaDA's
    block-by-block generation scheme. Supports batch_size > 1.
    """
    bsz = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    vocab_size = score_vec.shape[0]

    bias = score_vec.clamp(-score_clip, score_clip) * steering_strength
    bias = bias.view(1, 1, vocab_size)

    x = input_ids.clone()
    cur_mask = attention_mask.clone() if attention_mask is not None else None

    tokens_remaining = max_new_tokens
    while tokens_remaining > 0:
        cur_block_len = min(block_length, tokens_remaining)

        # Append a fresh masked block for all samples
        x = torch.cat([
            x,
            torch.full((bsz, cur_block_len), mask_token_id, dtype=torch.long, device=device),
        ], dim=1)

        # Extend attention mask: generation positions are always attended to
        if cur_mask is not None:
            cur_mask = torch.cat([
                cur_mask,
                torch.ones(bsz, cur_block_len, dtype=cur_mask.dtype, device=device),
            ], dim=1)

        block_start = x.shape[1] - cur_block_len

        x = _denoise_block(
            model=model,
            x=x,
            block_start=block_start,
            block_len=cur_block_len,
            mask_token_id=mask_token_id,
            num_steps=num_steps,
            bias=bias,
            vocab_size=vocab_size,
            temperature=temperature,
            device=device,
            model_tag=model_tag,
            attention_mask=cur_mask,
        )

        tokens_remaining -= cur_block_len

        # Early stop when every sample in the batch has produced EOS in this block
        if eos_token_id is not None:
            block_ids = x[:, block_start: block_start + cur_block_len]  # [B, block_len]
            if (block_ids == eos_token_id).any(dim=1).all():
                break

    return x[:, prompt_len:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="DLM logit steering experiments")
    parser.add_argument("--dataset", type=str, choices=["ose", "wikipol", "real_tox_4000"], default="ose")
    parser.add_argument("--experiment", type=str, choices=["rewriting", "open_ended"], default="rewriting")
    parser.add_argument("--target", type=str, default="E",
                        help="Target attribute: ose→E/I/A, wikipol→I/P/N, real_tox→T/NT")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Base",
                        help="HuggingFace model id for the DLM")
    parser.add_argument("--model_tag", type=str, default="llada8b",
                        help="Short model tag used for score/output directories (e.g. llada8b, dream7b)")
    parser.add_argument("--score_dir", type=str, default=None,
                        help="Directory containing score files (default: lookup_scores/{model_tag}/{dataset})")
    parser.add_argument("--data_idx", type=int, default=None,
                        help="Index of a single data sample to process")
    parser.add_argument("--num_samples", type=int, default=-1,
                        help="Samples per class to randomly sample (-1 = all). "
                             "e.g. --num_samples 100 → 100×E + 100×I + 100×A for ose")
    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Number of tokens to generate (generation length)")
    parser.add_argument("--num_steps", type=int, default=128,
                        help="Number of denoising steps per block")
    parser.add_argument("--block_length", type=int, default=128,
                        help="Tokens generated per denoising block (semi-AR chunk size)")
    parser.add_argument("--steering_strength", type=float, default=0.5,
                        help="Multiplier applied to the score bias at each denoising step")
    parser.add_argument("--score_clip", type=float, default=8.0,
                        help="Clip score magnitudes before applying the bias")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature for token sampling (0 = greedy argmax)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs (default: outputs/{model_tag}/{dataset})")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to input dataset JSON (default: datasets/{dataset}.json)")
    parser.add_argument("--mask_token_id", type=int, default=None,
                        help="Override mask token id (auto-detected if not specified)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of samples to process in parallel (requires GPU memory)")
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if args.data_path is None:
        args.data_path = f"datasets/{args.dataset}.json"
    if args.output_dir is None:
        args.output_dir = f"outputs/{args.model_tag}/{args.dataset}"
    if args.score_dir is None:
        args.score_dir = f"lookup_scores/{args.model_tag}/{args.dataset}"

    if args.seed is None:
        args.seed = random.randint(0, 2**31 - 1)

    set_random_seed(args.seed)
    print(f"Random seed: {args.seed}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LLaDA / DREAM tokenizers don't always expose mask_token_id as a standard
    # attribute, so fall back to vocabulary lookup then model-specific defaults.
    _MASK_TOKEN_DEFAULTS = {"llada8b": 126336, "dream7b": 151666}
    mask_token_id = args.mask_token_id or tokenizer.mask_token_id
    if mask_token_id is None:
        mask_token_id = tokenizer.convert_tokens_to_ids("[MASK]")
    if mask_token_id is None or mask_token_id == tokenizer.unk_token_id:
        mask_token_id = _MASK_TOKEN_DEFAULTS.get(args.model_tag)
    if mask_token_id is None:
        raise ValueError(
            f"Could not determine mask_token_id for {args.model_name}. "
            "Pass --mask_token_id explicitly."
        )
    print(f"mask_token_id: {mask_token_id}")

    # ---- model ----
    print(f"Loading model: {args.model_name}")
    model_kwargs = dict(
        trust_remote_code=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
    )
    if args.model_tag.startswith("llada8b"):
        model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    else:  # dream7b
        model = AutoModel.from_pretrained(args.model_name, **model_kwargs)
    model.to(device).eval()

    # ---- target validation ----
    target = args.target
    if args.dataset == "ose":
        if target not in ["E", "I", "A"]:
            print(f"Error: ose target must be E/I/A, got {target}"); exit(1)
    elif args.dataset == "wikipol":
        if target not in ["I", "P", "N"]:
            print(f"Error: wikipol target must be I/P/N, got {target}"); exit(1)
    elif args.dataset == "real_tox_4000":
        if target not in ["T", "NT"]:
            print(f"Error: real_tox target must be T/NT, got {target}"); exit(1)

    score_path = os.path.join(args.score_dir, f"scores_{target}.json")
    if not os.path.exists(score_path):
        print(f"Error: Score file not found: {score_path}")
        print("Please run build_scores.py first.")
        exit(1)

    score_dict = load_score_table_json(score_path)
    vocab_size = model.config.vocab_size
    score_vec = build_score_tensor(score_dict, vocab_size=vocab_size, device=device)
    print(f"score_vec shape: {score_vec.shape}, non-zero: {(score_vec != 0).sum().item()}")

    # ---- dataset ----
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found: {args.data_path}"); exit(1)

    with open(args.data_path, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)

    total_samples = len(dataset_data)
    print(f"Loaded {total_samples} samples from {args.dataset.upper()} dataset")

    if args.data_idx is not None:
        if args.data_idx >= total_samples:
            print(f"Error: data_idx {args.data_idx} out of range ({total_samples} samples)")
            exit(1)
        sample_indices = [args.data_idx]
    elif args.num_samples == -1:
        sample_indices = list(range(total_samples))
        random.shuffle(sample_indices)
    else:
        n = args.num_samples  # samples per class
        if args.dataset == "ose":
            ose_label_map = {"elementary": "E", "intermediate": "I", "advanced": "A"}
            classes = {"E": [], "I": [], "A": []}
            for i, s in enumerate(dataset_data):
                lv = ose_label_map.get(s.get("level", "").lower(), "")
                if lv in classes:
                    classes[lv].append(i)
            sample_indices = []
            for cls, idxs in classes.items():
                if len(idxs) < n:
                    print(f"Error: ose class '{cls}' has only {len(idxs)} samples, need {n}"); exit(1)
                sampled = random.sample(idxs, n)
                sample_indices.extend(sampled)
                print(f"  {cls}: {n} samples")
        elif args.dataset == "wikipol":
            # dataset uses full label names; map to shorthand keys
            label_map = {"impolite": "I", "polite": "P", "neutral": "N"}
            classes = {"I": [], "P": [], "N": []}
            for i, s in enumerate(dataset_data):
                lv = label_map.get(s.get("label", "").lower(), "")
                if lv in classes:
                    classes[lv].append(i)
            sample_indices = []
            for cls, idxs in classes.items():
                if len(idxs) < n:
                    print(f"Error: wikipol class '{cls}' has only {len(idxs)} samples, need {n}"); exit(1)
                sampled = random.sample(idxs, n)
                sample_indices.extend(sampled)
                print(f"  {cls}: {n} samples")
        elif args.dataset == "real_tox_4000":
            toxic_indices     = [i for i, s in enumerate(dataset_data) if s.get("label", "").lower() == "toxic"]
            non_toxic_indices = [i for i, s in enumerate(dataset_data) if s.get("label", "").lower() == "non-toxic"]
            if len(toxic_indices) < n or len(non_toxic_indices) < n:
                print(f"Error: real_tox needs {n} toxic and {n} non-toxic samples."); exit(1)
            sample_indices = random.sample(toxic_indices, n) + random.sample(non_toxic_indices, n)
            print(f"  toxic: {n} samples, non-toxic: {n} samples")
        random.shuffle(sample_indices)
        print(f"Total sampled: {len(sample_indices)} ({n} per class)")

    print(f"Processing {len(sample_indices)} samples")
    print("=" * 80)

    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{args.dataset}_{args.experiment}_{target}.json")

    if args.data_idx is None:
        processed_indices, existing_results = load_checkpoint(output_file, args.seed, args.max_new_tokens)
    else:
        processed_indices, existing_results = set(), []

    remaining_indices = [idx for idx in sample_indices if idx not in processed_indices]

    if not remaining_indices:
        print("All samples already processed.")
        print(f"Results at: {output_file}")
        exit(0)

    print(f"Remaining: {len(remaining_indices)} samples")
    print("=" * 80)

    results = existing_results.copy()
    checkpoint_interval = 100

    # Left-padding so all prompts in a batch end at the same position
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    batch_size = args.batch_size
    batches = [remaining_indices[s: s + batch_size] for s in range(0, len(remaining_indices), batch_size)]
    processed_count = 0

    for batch_indices in tqdm(batches, desc="Processing batches"):
        # ---- build prompts for this batch ----
        batch_meta = []
        prompts = []
        for idx in batch_indices:
            sample = dataset_data[idx]
            if args.dataset == "ose":
                subject = sample.get("subject", "")
                corpus  = sample.get("corpus", "")
                label   = sample.get("level", "")
            else:
                subject = ""
                corpus  = sample.get("text", "")
                label   = sample.get("label", "")

            system_prompt, user_prompt = build_prompt(args.experiment, subject, corpus, args.dataset)
            if getattr(tokenizer, "chat_template", None):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ]
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True, tokenize=False
                )
            else:
                prompt = system_prompt + "\n\n" + user_prompt

            prompts.append(prompt)
            batch_meta.append({"idx": idx, "subject": subject, "corpus": corpus, "label": label, "prompt": prompt})

        # ---- tokenize batch (left-padded) ----
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
        input_ids     = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # ---- DLM generation with steering ----
        gen_ids_batch = dlm_generate_with_steering(
            model=model,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=args.max_new_tokens,
            num_steps=args.num_steps,
            score_vec=score_vec,
            steering_strength=args.steering_strength,
            block_length=args.block_length,
            score_clip=args.score_clip,
            temperature=args.temperature,
            device=device,
            model_tag=args.model_tag,
            eos_token_id=tokenizer.eos_token_id,
            attention_mask=attention_mask,
        )

        # ---- decode each sample in the batch ----
        eos_id = tokenizer.eos_token_id
        for j, meta in enumerate(batch_meta):
            gen_ids = gen_ids_batch[j]   # [max_new_tokens]

            # Truncate at first EOS
            if eos_id is not None:
                gen_list = gen_ids.tolist()
                if eos_id in gen_list:
                    gen_ids = gen_ids[:gen_list.index(eos_id)]

            # Filter out mask tokens and any IDs the tokenizer can't decode
            gen_ids = gen_ids[gen_ids != mask_token_id]
            if args.model_tag.startswith("dream7b"):
                valid_mask = torch.tensor(
                    [tokenizer._convert_id_to_token(int(tid)) is not None for tid in gen_ids],
                    dtype=torch.bool,
                )
                gen_ids = gen_ids[valid_mask]

            generated_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            # Debug: if empty, show raw token ids
            if not generated_text.strip() and args.data_idx is not None:
                raw_ids = gen_ids.tolist()
                print(f"\n[DEBUG] generated_text is empty. Raw gen_ids[:20]: {raw_ids[:20]}")
                print(f"[DEBUG] Raw decode (no skip): {repr(tokenizer.decode(gen_ids, skip_special_tokens=False)[:200])}")
                print(f"[DEBUG] mask_token_id={mask_token_id}, eos_token_id={eos_id}")
                print(f"[DEBUG] Unique token ids in output: {sorted(set(raw_ids))}")

            result = {
                "sample_index": meta["idx"],
                "prompt":        meta["prompt"],
                "generated_text": generated_text,
                "original_text": meta["corpus"],
            }
            if args.dataset == "ose":
                result["subject"] = meta["subject"]
                result["level"]   = meta["label"]
            else:
                result["label"] = meta["label"]

            if args.data_idx is not None:
                print("\n" + "=" * 80)
                print("PROMPT:")
                print("=" * 80)
                print(meta["prompt"])
                print("\n" + "=" * 80)
                print("GENERATED TEXT:")
                print("=" * 80)
                print(generated_text)
                if args.experiment == "rewriting":
                    print("\n" + "=" * 80)
                    print("ORIGINAL TEXT:")
                    print("=" * 80)
                    print(meta["corpus"])
                print("=" * 80)
            else:
                results.append(result)

        processed_count += len(batch_indices)
        if args.data_idx is None:
            should_save = processed_count % checkpoint_interval < batch_size or processed_count == len(remaining_indices)
            if should_save:
                output_data = {
                    "experiment": args.experiment,
                    "target_attribute": target,
                    "model_name": args.model_name,
                    "model_tag": args.model_tag,
                    "num_steps": args.num_steps,
                    "steering_strength": args.steering_strength,
                    "temperature": args.temperature,
                    "num_samples": len(results),
                    "seed": args.seed,
                    "max_new_tokens": args.max_new_tokens,
                    "results": results,
                }
                save_checkpoint(output_file, output_data)
                if processed_count % checkpoint_interval < batch_size:
                    print(f"\n[Checkpoint] Saved {len(results)} results to {output_file}")

    if args.data_idx is None:
        output_data = {
            "experiment": args.experiment,
            "target_attribute": target,
            "model_name": args.model_name,
            "model_tag": args.model_tag,
            "num_steps": args.num_steps,
            "steering_strength": args.steering_strength,
            "temperature": args.temperature,
            "num_samples": len(results),
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "results": results,
        }
        save_checkpoint(output_file, output_data)
        print("\n" + "=" * 80)
        print(f"Completed {len(remaining_indices)} new samples")
        print(f"Total: {len(results)} | Saved to: {output_file}")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print(f"Completed sample {args.data_idx}")
        print("=" * 80)
