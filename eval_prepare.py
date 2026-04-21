import json
import os
import argparse
from typing import Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def index_by_sample(results: List[dict]) -> Dict[int, dict]:
    return {r["sample_index"]: r for r in results}


# ---------------------------------------------------------------------------
# Per-dataset builders
# ---------------------------------------------------------------------------

def build_ose(dataset, model_tag: str, outputs_dir: str, output_path: str) -> None:
    if dataset == "ose_rewriting":
        base = os.path.join(outputs_dir, model_tag, "ose", "ose_rewriting")
    elif dataset == "ose_open_ended":
        base = os.path.join(outputs_dir, model_tag, "ose", "ose_open_ended")
    targets = ["E", "I", "A"]
    target_to_level = {"E": "elementary", "I": "intermediate", "A": "advanced"}

    data_by_target = {}
    for t in targets:
        path = f"{base}_{t}.json"
        data_by_target[t] = load_json(path)
        print(f"  Loaded {t}: {len(data_by_target[t]['results'])} samples  ({path})")

    by_idx = {t: index_by_sample(data_by_target[t]["results"]) for t in targets}
    all_indices = sorted(set(by_idx["E"]) | set(by_idx["I"]) | set(by_idx["A"]))
    print(f"  Total sample_indices: {len(all_indices)}")

    groups = []
    for idx in all_indices:
        base_target = next(t for t in targets if idx in by_idx[t])
        base_sample = by_idx[base_target][idx]
        group = {
            "sample_index": idx,
            "subject": base_sample.get("subject", ""),
            "original_text": base_sample.get("original_text", ""),
            "original_class": base_sample.get("level", ""),
            "prompt": base_sample.get("prompt", ""),
        }
        for t in targets:
            group[f"{t}_text"] = by_idx[t][idx].get("generated_text", "") if idx in by_idx[t] else ""
        groups.append(group)

    meta = {k: v for k, v in data_by_target["E"].items() if k != "results"}
    meta["target_attribute"] = "combined"
    meta["num_groups"] = len(groups)
    meta["groups"] = groups

    save_json(meta, output_path)
    print(f"  Saved {len(groups)} triplet groups → {output_path}\n")


def build_wikipol(model_tag: str, outputs_dir: str, output_path: str) -> None:
    base = os.path.join(outputs_dir, model_tag, "wikipol", "wikipol_rewriting")
    targets = ["I", "N", "P"]

    data_by_target = {}
    for t in targets:
        path = f"{base}_{t}.json"
        data_by_target[t] = load_json(path)
        print(f"  Loaded {t}: {len(data_by_target[t]['results'])} samples  ({path})")

    by_idx = {t: index_by_sample(data_by_target[t]["results"]) for t in targets}
    common = sorted(set(by_idx["I"]) & set(by_idx["N"]) & set(by_idx["P"]))
    print(f"  Intersection (I ∩ N ∩ P): {len(common)} sample_indices")

    groups = []
    for idx in common:
        base_sample = by_idx["I"][idx]
        group = {
            "sample_index": idx,
            "original_text": base_sample.get("original_text", ""),
            "original_class": base_sample.get("label", ""),
            "prompt": base_sample.get("prompt", ""),
        }
        for t in targets:
            group[f"{t}_text"] = by_idx[t][idx].get("generated_text", "")
        groups.append(group)

    meta = {k: v for k, v in data_by_target["I"].items() if k != "results"}
    meta["target_attribute"] = "combined"
    meta["num_groups"] = len(groups)
    meta["groups"] = groups

    save_json(meta, output_path)
    print(f"  Saved {len(groups)} triplet groups → {output_path}\n")


def build_real_tox(model_tag: str, outputs_dir: str, output_path: str,
                   dataset: str = "real_tox_4000") -> None:
    base = os.path.join(outputs_dir, model_tag, dataset, f"{dataset}_rewriting")
    targets = ["T", "NT"]

    data_by_target = {}
    for t in targets:
        path = f"{base}_{t}.json"
        data_by_target[t] = load_json(path)
        print(f"  Loaded {t}: {len(data_by_target[t]['results'])} samples  ({path})")

    by_idx = {t: index_by_sample(data_by_target[t]["results"]) for t in targets}
    common = sorted(set(by_idx["T"]) & set(by_idx["NT"]))
    print(f"  Intersection (T ∩ NT): {len(common)} sample_indices")

    groups = []
    for idx in common:
        base_sample = by_idx["T"][idx]
        group = {
            "sample_index": idx,
            "original_text": base_sample.get("original_text", ""),
            "original_class": base_sample.get("label", ""),
            "prompt": base_sample.get("prompt", ""),
        }
        for t in targets:
            group[f"{t}_text"] = by_idx[t][idx].get("generated_text", "")
        groups.append(group)

    meta = {k: v for k, v in data_by_target["T"].items() if k != "results"}
    meta["target_attribute"] = "combined"
    meta["num_groups"] = len(groups)
    meta["groups"] = groups

    save_json(meta, output_path)
    print(f"  Saved {len(groups)} pair groups → {output_path}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare eval_{dataset}.json for eval_gptjudge.py"
    )
    parser.add_argument("--dataset", default=None,
                        choices=["ose_rewriting","ose_open_ended", "wikipol", "real_tox_4000"],
                        help="Dataset to process (omit with --all)")
    parser.add_argument("--all", action="store_true",
                        help="Process ose, wikipol, and real_tox_4000")
    parser.add_argument("--model_tag", default="dream7b",
                        help="dream 7b or llada8b; this is just for naming the directory")
    parser.add_argument("--outputs_dir", default="outputs",
                        help="Root outputs directory (default: outputs)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: eval_data/{model_tag})")
    args = parser.parse_args()

    if not args.all and args.dataset is None:
        parser.error("Specify --dataset or --all")

    output_dir = args.output_dir or os.path.join("eval_data", args.model_tag)
    datasets = ["ose_rewriting", "ose_open_ended", "wikipol", "real_tox_4000"] if args.all else [args.dataset]

    for dataset in datasets:
        print(f"[{dataset}]")
        output_path = os.path.join(output_dir, f"eval_{dataset}.json")
        if "ose" in dataset:
            build_ose(dataset, args.model_tag, args.outputs_dir, output_path)
        elif dataset == "wikipol":
            build_wikipol(args.model_tag, args.outputs_dir, output_path)
        elif dataset == "real_tox_4000":
            build_real_tox(args.model_tag, args.outputs_dir, output_path)


if __name__ == "__main__":
    main()
