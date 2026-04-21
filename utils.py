import json
import math
import os
import random
import numpy as np
import torch
from transformers import LogitsProcessor


def load_score_table_json(path: str) -> dict[int, float]:
    """Load score table from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {int(k): float(v) for k, v in obj.items()}


def build_score_tensor(score_dict: dict[int, float], vocab_size: int, device: torch.device) -> torch.Tensor:
    """
    score_vec[token_id] = score
    """
    score_vec = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    for tid, sc in score_dict.items():
        if 0 <= tid < vocab_size:
            score_vec[tid] = float(sc)
    return score_vec


def quantile_threshold(values: torch.Tensor, q: float) -> float:
    """
    Returns quantile threshold as a scalar float.
    If values is empty, returns 0.0.
    """
    if values.numel() == 0:
        return 0.0
    values_cpu = values.detach().cpu()
    q = float(max(0.0, min(1.0, q)))
    result = torch.quantile(values_cpu, q)
    if isinstance(result, torch.Tensor):
        return float(result.item())
    return float(result)


class AttributeSteeringProcessor(LogitsProcessor):
    """
    candidate(top-k) 내부에서:
      - token score a[v] 계산
      - theta = quantile(a, q=lambda_max)  (lambda↑ -> 더 strict)
      - preferred P = {a>=theta}
      - favored F = preferred 먼저 채우고 부족하면 score 높은 순으로 채워서 m개 고정
      - favored에 +delta bias
    """
    def __init__(
        self,
        score_vec: torch.Tensor,   # [V]
        K: int = 100,
        rho: float = 0.5,
        delta: float = 1.5,
        lambda_max: float = 0.7,
        score_clip: float | None = 8.0,
    ):
        self.score_vec = score_vec
        self.K = K
        self.rho = rho
        self.delta = delta
        self.lambda_max = lambda_max
        self.score_clip = score_clip

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        scores: [batch, V] (pre-softmax logits)
        """
        bsz, V = scores.shape
        K = min(self.K, V)
        m = max(1, int(math.ceil(self.rho * K)))
        
        score_vec_size = self.score_vec.shape[0]
        actual_vocab_size = min(V, score_vec_size)

        for b in range(bsz):
            z = scores[b]  # [V]

            # candidate set C = topK(z)
            cand_ids = torch.topk(z, k=K, dim=-1).indices  # [K]
            cand_ids = torch.clamp(cand_ids, 0, score_vec_size - 1)

            # candidate score a[v]
            a = self.score_vec[cand_ids]  # [K]
            if self.score_clip is not None:
                a = torch.clamp(a, -self.score_clip, self.score_clip)

            theta = quantile_threshold(a, q=self.lambda_max)
            preferred = a >= theta  # [K] bool

            # pick favored set
            order = torch.argsort(a, descending=True)
            sorted_ids = cand_ids[order]
            preferred_sorted = preferred[order]

            pref_ids = sorted_ids[preferred_sorted]
            rest_ids = sorted_ids[~preferred_sorted]

            if pref_ids.numel() >= m:
                favored_ids = pref_ids[:m]
            else:
                need = m - pref_ids.numel()
                if rest_ids.numel() > 0:
                    favored_ids = torch.cat([pref_ids, rest_ids[:need]], dim=0)
                else:
                    favored_ids = pref_ids

            # delta bias
            scores[b, favored_ids] = scores[b, favored_ids] + self.delta

        return scores


def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_checkpoint(output_file: str, current_seed: int, current_max_new_tokens: int):
    """
    Load existing checkpoint file and return processed indices and existing results.
    Returns: (processed_indices: set, existing_results: list)
    """
    processed_indices = set()
    existing_results = []
    
    if not os.path.exists(output_file):
        return processed_indices, existing_results
    
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            existing_data = json.load(f)
            existing_results = existing_data.get("results", [])
            processed_indices = {r["sample_index"] for r in existing_results}
            
            if existing_data.get("seed") != current_seed:
                print(f"Warning: Existing file has seed {existing_data.get('seed')}, but current seed is {current_seed}")
            if existing_data.get("max_new_tokens") != current_max_new_tokens:
                print(f"Warning: Existing file has max_new_tokens {existing_data.get('max_new_tokens')}, but current is {current_max_new_tokens}")
            
            print(f"Found existing checkpoint: {len(existing_results)} samples already processed")
            if len(processed_indices) > 10:
                print(f"Resuming from checkpoint. Processed indices: {sorted(list(processed_indices))[:10]}...")
            else:
                print(f"Resuming from checkpoint. Processed indices: {sorted(list(processed_indices))}")
    except Exception as e:
        print(f"Warning: Could not load existing checkpoint file: {e}")
        print("Starting fresh...")
        existing_results = []
        processed_indices = set()
    
    return processed_indices, existing_results


def save_checkpoint(output_file: str, output_data: dict):
    """Save checkpoint atomically using temporary file."""
    temp_file = output_file + ".tmp"
    with open(temp_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_file, output_file)



