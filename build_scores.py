from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from collections import Counter
from typing import Dict, Iterable, List, Optional, Tuple, Union

import torch
from transformers import AutoTokenizer


def iter_texts_from_txt(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t:
                yield t


def iter_texts_from_jsonl(path: str, field: str = "text") -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            t = obj.get(field, "")
            if isinstance(t, str) and t.strip():
                yield t.strip()


def iter_texts_from_json_by_level(path: str, level: str, text_field: str = "corpus", level_field: str = "level") -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for obj in data:
            if obj.get(level_field) == level:
                t = obj.get(text_field, "")
                if isinstance(t, str) and t.strip():
                    yield t.strip()


def iter_texts_from_json_by_label(path: str, label: str, text_field: str = "text", label_field: str = "label") -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        for obj in data:
            if obj.get(label_field) == label:
                t = obj.get(text_field, "")
                if isinstance(t, str) and t.strip():
                    yield t.strip()


@dataclass
class CountConfig:
    add_special_tokens: bool = False
    ignore_token_ids: Optional[List[int]] = None
    max_tokens_per_doc: Optional[int] = None


def count_tokens(texts: Iterable[str], tokenizer: AutoTokenizer, cfg: CountConfig) -> Counter:
    tokenizer,
    ignore = set(cfg.ignore_token_ids or [])
    counts = Counter()
    model_max_length = getattr(tokenizer, 'model_max_length', 1024)
    if model_max_length > 100000:
        model_max_length = 1024
    max_len = cfg.max_tokens_per_doc
    if max_len is None:
        max_len = model_max_length
    else:
        max_len = min(max_len, model_max_length)

    for t in texts:
        ids = tokenizer.encode(
            t, 
            add_special_tokens=cfg.add_special_tokens,
            max_length=max_len,
            truncation=True
        )

        for tid in ids:
            if tid in ignore:
                continue
            counts[int(tid)] += 1

    return counts


@dataclass
class LogOddsConfig:
    alpha: float = 0.01
    prior_mode: str = "pooled"
    compute_z: bool = True
    eps: float = 1e-12


def _counter_total(c: Counter) -> int:
    return sum(c.values())


def build_prior(counts_E: Counter, counts_I: Counter, counts_A: Counter, vocab_size: int, mode: str = "pooled") -> torch.Tensor:
    if mode == "uniform":
        return torch.ones(vocab_size, dtype=torch.float64)

    # pooled
    pooled = torch.zeros(vocab_size, dtype=torch.float64)
    for c in (counts_E, counts_I, counts_A):
        for tid, cnt in c.items():
            if 0 <= tid < vocab_size:
                pooled[tid] += float(cnt)
    # avoid all-zero
    pooled = torch.clamp(pooled, min=1.0)
    return pooled


def build_prior_2way(counts_pos: Counter, counts_neg: Counter, vocab_size: int, mode: str = "pooled") -> torch.Tensor:
    if mode == "uniform":
        return torch.ones(vocab_size, dtype=torch.float64)

    # pooled
    pooled = torch.zeros(vocab_size, dtype=torch.float64)
    for c in (counts_pos, counts_neg):
        for tid, cnt in c.items():
            if 0 <= tid < vocab_size:
                pooled[tid] += float(cnt)
    # avoid all-zero
    pooled = torch.clamp(pooled, min=1.0)
    return pooled


def log_odds_one_vs_rest(counts_pos: Counter, counts_neg: Counter, prior_alpha0: torch.Tensor, cfg: LogOddsConfig, vocab_size: int) -> Tuple[Dict[int, float], Optional[Dict[int, float]]]:
    alpha = float(cfg.alpha)
    alpha0 = prior_alpha0.to(torch.float64) * alpha  # [V]

    # totals
    n_pos = float(_counter_total(counts_pos))
    n_neg = float(_counter_total(counts_neg))
    a0_sum = float(alpha0.sum().item())

    score: Dict[int, float] = {}
    zscore: Dict[int, float] = {}

    token_set = set(counts_pos.keys()) | set(counts_neg.keys())

    for tid in token_set:
        if tid < 0 or tid >= vocab_size:
            continue

        y_pos = float(counts_pos.get(tid, 0))
        y_neg = float(counts_neg.get(tid, 0))
        a0 = float(alpha0[tid].item())

        # log-odds with prior
        num_pos = y_pos + a0
        den_pos = (n_pos + a0_sum) - num_pos
        num_neg = y_neg + a0
        den_neg = (n_neg + a0_sum) - num_neg

        # numeric safety
        num_pos = max(num_pos, cfg.eps); den_pos = max(den_pos, cfg.eps)
        num_neg = max(num_neg, cfg.eps); den_neg = max(den_neg, cfg.eps)

        delta = math.log(num_pos / den_pos) - math.log(num_neg / den_neg)
        score[tid] = float(delta)

        if cfg.compute_z:
            var = (1.0 / num_pos) + (1.0 / num_neg)
            z = delta / math.sqrt(max(var, cfg.eps))
            zscore[tid] = float(z)

    return score, (zscore if cfg.compute_z else None)


@dataclass
class BuildConfig:
    count_cfg: CountConfig = field(default_factory=CountConfig)
    logodds_cfg: LogOddsConfig = field(default_factory=LogOddsConfig)
    use_zscore_for_ranking: bool = True


def build_3way_one_vs_rest_scores(texts_E: Iterable[str], texts_I: Iterable[str], texts_A: Iterable[str], tokenizer: AutoTokenizer, cfg: BuildConfig) -> Dict[str, Dict[int, float]]:

    vocab_size = int(tokenizer.vocab_size)

    cE = count_tokens(texts_E, tokenizer, cfg.count_cfg)
    cI = count_tokens(texts_I, tokenizer, cfg.count_cfg)
    cA = count_tokens(texts_A, tokenizer, cfg.count_cfg)

    prior_alpha0 = build_prior(cE, cI, cA, vocab_size=vocab_size, mode=cfg.logodds_cfg.prior_mode)

    # one-vs-rest
    # E vs (I+A)
    cIA = cI + cA
    sE, zE = log_odds_one_vs_rest(cE, cIA, prior_alpha0, cfg.logodds_cfg, vocab_size)
    # I vs (E+A)
    cEA = cE + cA
    sI, zI = log_odds_one_vs_rest(cI, cEA, prior_alpha0, cfg.logodds_cfg, vocab_size)
    # A vs (E+I)
    cEI = cE + cI
    sA, zA = log_odds_one_vs_rest(cA, cEI, prior_alpha0, cfg.logodds_cfg, vocab_size)

    def pick(s: Dict[int, float], z: Optional[Dict[int, float]]) -> Dict[int, float]:
        if cfg.use_zscore_for_ranking and z is not None:
            return z
        return s

    return {
        "E": pick(sE, zE),
        "I": pick(sI, zI),
        "A": pick(sA, zA),
    }


def build_3way_one_vs_rest_scores_generic(texts_1: Iterable[str], texts_2: Iterable[str], texts_3: Iterable[str], tokenizer: AutoTokenizer, cfg: BuildConfig, label_names: Tuple[str, str, str] = ("1", "2", "3")) -> Dict[str, Dict[int, float]]:
    vocab_size = int(tokenizer.vocab_size)

    c1 = count_tokens(texts_1, tokenizer, cfg.count_cfg)
    c2 = count_tokens(texts_2, tokenizer, cfg.count_cfg)
    c3 = count_tokens(texts_3, tokenizer, cfg.count_cfg)

    prior_alpha0 = build_prior(c1, c2, c3, vocab_size=vocab_size, mode=cfg.logodds_cfg.prior_mode)

    # 1 vs (2+3)
    c23 = c2 + c3
    s1, z1 = log_odds_one_vs_rest(c1, c23, prior_alpha0, cfg.logodds_cfg, vocab_size)
    # 2 vs (1+3)
    c13 = c1 + c3
    s2, z2 = log_odds_one_vs_rest(c2, c13, prior_alpha0, cfg.logodds_cfg, vocab_size)
    # 3 vs (1+2)
    c12 = c1 + c2
    s3, z3 = log_odds_one_vs_rest(c3, c12, prior_alpha0, cfg.logodds_cfg, vocab_size)

    def pick(s: Dict[int, float], z: Optional[Dict[int, float]]) -> Dict[int, float]:
        if cfg.use_zscore_for_ranking and z is not None:
            return z
        return s

    return {
        label_names[0]: pick(s1, z1),
        label_names[1]: pick(s2, z2),
        label_names[2]: pick(s3, z3),
    }


def build_2way_one_vs_rest_scores(texts_pos: Iterable[str], texts_neg: Iterable[str], tokenizer: AutoTokenizer, cfg: BuildConfig, label_names: Tuple[str, str] = ("pos", "neg")) -> Dict[str, Dict[int, float]]:

    vocab_size = int(tokenizer.vocab_size)

    c_pos = count_tokens(texts_pos, tokenizer, cfg.count_cfg)
    c_neg = count_tokens(texts_neg, tokenizer, cfg.count_cfg)

    prior_alpha0 = build_prior_2way(c_pos, c_neg, vocab_size=vocab_size, mode=cfg.logodds_cfg.prior_mode)

    # pos vs neg
    s_pos, z_pos = log_odds_one_vs_rest(c_pos, c_neg, prior_alpha0, cfg.logodds_cfg, vocab_size)
    # neg vs pos (reverse)
    s_neg, z_neg = log_odds_one_vs_rest(c_neg, c_pos, prior_alpha0, cfg.logodds_cfg, vocab_size)

    def pick(s: Dict[int, float], z: Optional[Dict[int, float]]) -> Dict[int, float]:
        if cfg.use_zscore_for_ranking and z is not None:
            return z
        return s

    return {
        label_names[0]: pick(s_pos, z_pos),
        label_names[1]: pick(s_neg, z_neg),
    }


def make_attribute_list(score_table: Dict[int, float], *, top_n: Optional[int] = None, min_score: Optional[float] = None, exclude_token_ids: Optional[List[int]] = None) -> List[int]:
    exclude = set(exclude_token_ids or [])

    items = [(tid, sc) for tid, sc in score_table.items() if tid not in exclude]
    items.sort(key=lambda x: x[1], reverse=True)

    if min_score is not None:
        return [tid for tid, sc in items if sc >= min_score]

    if top_n is None:
        top_n = 5000
    return [tid for tid, _ in items[:top_n]]


def save_score_table_json(path: str, score_table: Dict[int, float]) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(k): float(v) for k, v in score_table.items()}, f, ensure_ascii=False)


def load_score_table_json(path: str) -> Dict[int, float]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return {int(k): float(v) for k, v in obj.items()}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build token scores for different datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["ose", "wikipol", "real_tox_4000"],
        default="ose",
        help="Dataset to process: 'ose', 'wikipol', or 'real_tox'"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="GSAI-ML/LLaDA-8B-Instruct",
        help="Tokenizer model name"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./lookup/llada8b/real_tox_4000",
        help="Output directory for score files"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="Dirichlet prior coefficient for log-odds smoothing (default: 0.01)"
    )
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, torch_dtype="auto")

    ignore_ids = []
    if tokenizer.eos_token_id is not None:
        ignore_ids.append(tokenizer.eos_token_id)

    cfg = BuildConfig(
        count_cfg=CountConfig(
            add_special_tokens=False,
            ignore_token_ids=ignore_ids,
            max_tokens_per_doc=None,
        ),
        logodds_cfg=LogOddsConfig(
            alpha=args.alpha,
            prior_mode="pooled",  # or "uniform"
            compute_z=True,
        ),
        use_zscore_for_ranking=True,  # z-score recommended for "distinctive token" ranking
    )

    if args.dataset == "ose":
        # OSE dataset: elementary/intermediate/advanced -> E/I/A
        json_path = "datasets/ose.json"
        texts_E = iter_texts_from_json_by_level(json_path, level="elementary", text_field="corpus")
        texts_I = iter_texts_from_json_by_level(json_path, level="intermediate", text_field="corpus")
        texts_A = iter_texts_from_json_by_level(json_path, level="advanced", text_field="corpus")
        
        scores = build_3way_one_vs_rest_scores(texts_E, texts_I, texts_A, tokenizer, cfg)
        
        save_score_table_json(f"{args.output_dir}/scores_E.json", scores["E"])
        save_score_table_json(f"{args.output_dir}/scores_I.json", scores["I"])
        save_score_table_json(f"{args.output_dir}/scores_A.json", scores["A"])
        
        ele_list = make_attribute_list(scores["E"], top_n=5000, exclude_token_ids=ignore_ids)
        int_list = make_attribute_list(scores["I"], top_n=5000, exclude_token_ids=ignore_ids)
        adv_list = make_attribute_list(scores["A"], top_n=5000, exclude_token_ids=ignore_ids)
        
        print("OSE dataset - E/I/A list sizes:", len(ele_list), len(int_list), len(adv_list))
        
    elif args.dataset == "wikipol":
        json_path = "datasets/wikipol.json"
        texts_impolite = iter_texts_from_json_by_label(json_path, label="impolite", text_field="text")
        texts_polite = iter_texts_from_json_by_label(json_path, label="polite", text_field="text")
        texts_neutral = iter_texts_from_json_by_label(json_path, label="neutral", text_field="text")
        
        scores = build_3way_one_vs_rest_scores_generic(
            texts_impolite,
            texts_polite,
            texts_neutral,
            tokenizer,
            cfg,
            label_names=("I", "P", "N")  # I=impolite, P=polite, N=neutral
        )
        
        save_score_table_json(f"{args.output_dir}/scores_I.json", scores["I"])
        save_score_table_json(f"{args.output_dir}/scores_P.json", scores["P"])
        save_score_table_json(f"{args.output_dir}/scores_N.json", scores["N"])
        
        impolite_list = make_attribute_list(scores["I"], top_n=5000, exclude_token_ids=ignore_ids)
        polite_list = make_attribute_list(scores["P"], top_n=5000, exclude_token_ids=ignore_ids)
        neutral_list = make_attribute_list(scores["N"], top_n=5000, exclude_token_ids=ignore_ids)
        
        print("Wikipol dataset - I/P/N list sizes:", len(impolite_list), len(polite_list), len(neutral_list))
        
    elif args.dataset == "real_tox_4000":
        json_path = "datasets/real_tox_4000.json"
        texts_toxic = iter_texts_from_json_by_label(json_path, label="toxic", text_field="text")
        texts_non_toxic = iter_texts_from_json_by_label(json_path, label="non-toxic", text_field="text")
        
        scores = build_2way_one_vs_rest_scores(
            texts_toxic,
            texts_non_toxic,
            tokenizer,
            cfg,
            label_names=("T", "NT")  # T=toxic, NT=non-toxic
        )
        
        save_score_table_json(f"{args.output_dir}/scores_T.json", scores["T"])
        save_score_table_json(f"{args.output_dir}/scores_NT.json", scores["NT"])
        
        toxic_list = make_attribute_list(scores["T"], top_n=5000, exclude_token_ids=ignore_ids)
        non_toxic_list = make_attribute_list(scores["NT"], top_n=5000, exclude_token_ids=ignore_ids)
        
        print("Real Toxicity Prompt dataset - T/NT list sizes:", len(toxic_list), len(non_toxic_list))
