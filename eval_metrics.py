import json
import os
import argparse
import random
from collections import defaultdict
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_results(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # perspective results use perspective_evaluation
    results = data.get("results", [])
    return results


def extract_pairs(results: List[Dict]) -> Tuple[List[str], List[str], List[float]]:
    """Return (y_true, y_pred, confidences) as lists."""
    y_true, y_pred, confidences = [], [], []
    skipped = 0
    for r in results:
        ev = r.get("judge_evaluation") or r.get("perspective_evaluation")
        if ev is None:
            skipped += 1
            continue
        expected = ev.get("expected_label", "").upper().strip()
        predicted = ev.get("label", "").upper().strip()
        if not expected or not predicted:
            skipped += 1
            continue
        y_true.append(expected)
        y_pred.append(predicted)
        # confidence: judge_evaluation has 'confidence', perspective has 'toxicity_score'
        conf = ev.get("confidence")
        if conf is None:
            conf = ev.get("toxicity_score")
        confidences.append(conf)
    if skipped:
        print(f"  [warn] Skipped {skipped} samples with missing evaluation fields.")
    return y_true, y_pred, confidences


# ---------------------------------------------------------------------------
# Metrics (no sklearn dependency)
# ---------------------------------------------------------------------------

def compute_metrics(y_true: List[str], y_pred: List[str], confidences: List[float] = None) -> Dict:
    classes = sorted(set(y_true) | set(y_pred))
    n = len(y_true)

    # Per-class counts
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    for yt, yp in zip(y_true, y_pred):
        for c in classes:
            if yt == c and yp == c:
                tp[c] += 1
            elif yt != c and yp == c:
                fp[c] += 1
            elif yt == c and yp != c:
                fn[c] += 1

    # Per-class confidence
    class_confs = defaultdict(list)
    if confidences:
        for yt, conf in zip(y_true, confidences):
            if conf is not None:
                class_confs[yt].append(conf)

    per_class = {}
    for c in classes:
        prec = tp[c] / (tp[c] + fp[c]) if (tp[c] + fp[c]) > 0 else 0.0
        rec  = tp[c] / (tp[c] + fn[c]) if (tp[c] + fn[c]) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        support = tp[c] + fn[c]
        avg_conf = sum(class_confs[c]) / len(class_confs[c]) if class_confs[c] else None
        per_class[c] = {"precision": prec, "recall": rec, "f1": f1, "support": support, "avg_confidence": avg_conf}

    acc = sum(yt == yp for yt, yp in zip(y_true, y_pred)) / n if n > 0 else 0.0

    macro_prec = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_rec  = sum(v["recall"]    for v in per_class.values()) / len(classes)
    macro_f1   = sum(v["f1"]        for v in per_class.values()) / len(classes)

    # Overall avg confidence
    valid_confs = [c for c in (confidences or []) if c is not None]
    avg_conf_all = sum(valid_confs) / len(valid_confs) if valid_confs else None

    return {
        "n_samples":  n,
        "accuracy":   acc,
        "macro_precision": macro_prec,
        "macro_recall":    macro_rec,
        "macro_f1":        macro_f1,
        "avg_confidence":  avg_conf_all,
        "per_class":  per_class,
    }


def print_metrics(metrics: Dict, title: str = ""):
    if title:
        print(f"\n{'=' * 70}")
        print(f"  {title}")
    print(f"{'=' * 70}")
    print(f"  Samples   : {metrics['n_samples']}")
    print(f"  Accuracy  : {metrics['accuracy']:.4f}")
    print(f"  Macro P   : {metrics['macro_precision']:.4f}")
    print(f"  Macro R   : {metrics['macro_recall']:.4f}")
    print(f"  Macro F1  : {metrics['macro_f1']:.4f}")
    if metrics.get("avg_confidence") is not None:
        print(f"  Avg Conf  : {metrics['avg_confidence']:.4f}")
    print(f"  {'─' * 76}")
    print(f"  {'Class':<16} {'Class Acc':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Conf':>8} {'Count':>8}")
    print(f"  {'─' * 76}")
    for cls, v in sorted(metrics["per_class"].items()):
        class_acc = v["recall"]  # TP / support = fraction correct among true-class samples
        conf_str = f"{v['avg_confidence']:.4f}" if v.get("avg_confidence") is not None else "   -"
        print(f"  {cls:<16} {class_acc:>10.4f} {v['precision']:>10.4f} {v['recall']:>10.4f} {v['f1']:>10.4f} {conf_str:>8} {v['support']:>8}")
    print(f"{'=' * 70}")


def breakdown_by_target(results: List[Dict]) -> Dict[str, Tuple[List, List, List]]:
    """Group samples by target_attribute and return per-target (y_true, y_pred, confidences)."""
    groups = defaultdict(list)
    for r in results:
        ev = r.get("judge_evaluation") or r.get("perspective_evaluation")
        if ev is None:
            continue
        target = (
            ev.get("target_attribute")
            or r.get("target_attribute")
            or r.get("level")
            or "?"
        )
        groups[target.upper()].append(r)

    result = {}
    for target, samples in sorted(groups.items()):
        yt, yp, confs = extract_pairs(samples)
        result[target] = (yt, yp, confs)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compute accuracy / precision / recall / F1 for GPT judge or Perspective outputs"
    )
    parser.add_argument(
        "input",
        nargs="+",
        help="One or more judged JSON files (e.g. gpt_judge_results/dream7b/eval_ose_judged_triplet.json)"
    )
    parser.add_argument(
        "--by-target",
        action="store_true",
        default=False,
        help="Also print per-target-attribute breakdown (e.g. per steering direction E/I/A)"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        metavar="N",
        help="Randomly sample N results before computing metrics (default: use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Save metrics as JSON to this path"
    )
    args = parser.parse_args()

    all_metrics = {}

    for path in args.input:
        if not os.path.exists(path):
            print(f"Error: File not found: {path}")
            continue

        print(f"\nLoading: {path}")
        results = load_results(path)

        if args.sample is not None and args.sample < len(results):
            random.seed(args.seed)
            total = len(results)
            results = random.sample(results, args.sample)
            print(f"  Randomly sampled {args.sample} / {total} results (seed={args.seed})")

        print(f"  Total samples with judge_evaluation: ", end="")

        y_true, y_pred, confidences = extract_pairs(results)
        print(len(y_true))

        if not y_true:
            print("  No evaluated samples found. Skipping.")
            continue

        metrics = compute_metrics(y_true, y_pred, confidences)
        label = os.path.basename(path)
        print_metrics(metrics, title=label)
        all_metrics[path] = {"overall": metrics}

        if args.by_target:
            target_groups = breakdown_by_target(results)
            for target, (yt, yp, confs) in target_groups.items():
                if not yt:
                    continue
                tm = compute_metrics(yt, yp, confs)
                print_metrics(tm, title=f"{label}  [target={target}]")
                all_metrics[path][f"target_{target}"] = tm

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(all_metrics, f, ensure_ascii=False, indent=2)
        print(f"\nMetrics saved to: {args.save}")


if __name__ == "__main__":
    main()
