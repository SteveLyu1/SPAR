from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TOKEN_RE = re.compile(r"\S+")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value))
    text = text.replace("<focus>", "").replace("</focus>", "")
    text = " ".join(text.strip().split())
    if not text:
        return None
    lowered = text.casefold()
    if lowered in {"null", "none", "nan"}:
        return None
    return text


def normalize_match_text(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    return text.casefold()


def normalize_target(value: Any) -> Optional[str]:
    text = normalize_match_text(value)
    if text in {None, "implicit_target", "null"}:
        return None
    return text


def normalize_argument(value: Any) -> Optional[str]:
    text = normalize_match_text(value)
    if text in {None, "null"}:
        return None
    return text


def normalize_label(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("harmful") if "harmful" in item else item.get("hateful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    text = normalize_match_text(raw)
    if text is None:
        text = normalize_match_text(item.get("harmful_direction"))
    if text is None:
        return None
    if text in {"1", "harmful", "hateful", "hate", "true"}:
        return 1
    if text in {"0", "non-harmful", "non_harmful", "non-hate", "non_hate", "false"}:
        return 0
    return None


def tokenize_span(value: Any) -> List[str]:
    text = normalize_match_text(value)
    if text is None:
        return []
    return TOKEN_RE.findall(text)


def safe_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def macro_f1_binary(gold: Sequence[int], pred: Sequence[int]) -> float:
    labels = [0, 1]
    scores: List[float] = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(safe_f1(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0


def macro_f1_binary_from_counts(tp_by_label: Dict[int, int], fp_by_label: Dict[int, int], fn_by_label: Dict[int, int]) -> float:
    scores: List[float] = []
    for label in (0, 1):
        tp = tp_by_label.get(label, 0)
        fp = fp_by_label.get(label, 0)
        fn = fn_by_label.get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(safe_f1(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0


def count_exact_matches(
    gold_items: Sequence[Tuple[Any, ...]],
    pred_items: Sequence[Tuple[Any, ...]],
) -> int:
    gold_counter = Counter(gold_items)
    pred_counter = Counter(pred_items)
    return sum(min(pred_counter[key], gold_counter[key]) for key in pred_counter.keys() & gold_counter.keys())


def evaluate_phase0_like(input_path: Path) -> Dict[str, Any]:
    rows = [row for row in read_jsonl(input_path) if str(row.get("language") or "") == "en"]

    target_token_tp = 0
    target_token_gold_total = 0
    target_token_pred_total = 0
    argument_token_tp = 0
    argument_token_gold_total = 0
    argument_token_pred_total = 0

    targeted_tp = 0
    targeted_gold_total = 0
    targeted_pred_total = 0
    pair_tp = 0
    pair_gold_total = 0
    pair_pred_total = 0
    triple_tp = 0
    triple_gold_total = 0
    triple_pred_total = 0
    tuple_harm_gold_total = 0
    tuple_harm_pred_total = 0

    sample_gold_harmful: List[int] = []
    sample_pred_harmful: List[int] = []
    tuple_harm_tp = {0: 0, 1: 0}
    tuple_harm_fp = {0: 0, 1: 0}
    tuple_harm_fn = {0: 0, 1: 0}

    for row in rows:
        gold = [item for item in (row.get("ground_truth") or []) if isinstance(item, dict)]
        pred = [item for item in (row.get("predictions") or []) if isinstance(item, dict)]

        gold_target_counter: Counter[str] = Counter()
        pred_target_counter: Counter[str] = Counter()
        gold_arg_counter: Counter[str] = Counter()
        pred_arg_counter: Counter[str] = Counter()
        gold_targeted: List[Tuple[Optional[str], int]] = []
        pred_targeted: List[Tuple[Optional[str], int]] = []
        gold_pairs: List[Tuple[Optional[str], Optional[str]]] = []
        pred_pairs: List[Tuple[Optional[str], Optional[str]]] = []
        gold_triples: List[Tuple[Optional[str], Optional[str], Optional[int]]] = []
        pred_triples: List[Tuple[Optional[str], Optional[str], Optional[int]]] = []
        gold_pair_labels: Dict[Tuple[Optional[str], Optional[str]], int] = {}
        pred_pair_labels: Dict[Tuple[Optional[str], Optional[str]], int] = {}

        for item in gold:
            target = normalize_target(item.get("target"))
            argument = normalize_argument(item.get("argument"))
            label = normalize_label(item)

            gold_target_counter.update(tokenize_span(target))
            gold_arg_counter.update(tokenize_span(argument))
            if target is not None and label is not None:
                gold_targeted.append((target, label))
            if argument is not None:
                gold_pairs.append((target, argument))
                if label is not None:
                    gold_pair_labels[(target, argument)] = label
                if label is not None:
                    gold_triples.append((target, argument, label))

        for item in pred:
            target = normalize_target(item.get("target"))
            argument = normalize_argument(item.get("argument"))
            label = normalize_label(item)

            pred_target_counter.update(tokenize_span(target))
            pred_arg_counter.update(tokenize_span(argument))
            if target is not None and label is not None:
                pred_targeted.append((target, label))
            if argument is not None:
                pred_pairs.append((target, argument))
                if label is not None:
                    pred_pair_labels[(target, argument)] = label
                if label is not None:
                    pred_triples.append((target, argument, label))

        target_token_tp += sum(min(pred_target_counter[token], gold_target_counter[token]) for token in pred_target_counter.keys() & gold_target_counter.keys())
        target_token_gold_total += sum(gold_target_counter.values())
        target_token_pred_total += sum(pred_target_counter.values())

        argument_token_tp += sum(min(pred_arg_counter[token], gold_arg_counter[token]) for token in pred_arg_counter.keys() & gold_arg_counter.keys())
        argument_token_gold_total += sum(gold_arg_counter.values())
        argument_token_pred_total += sum(pred_arg_counter.values())

        targeted_tp += count_exact_matches(gold_targeted, pred_targeted)
        targeted_gold_total += len(gold_targeted)
        targeted_pred_total += len(pred_targeted)

        pair_tp += count_exact_matches(gold_pairs, pred_pairs)
        pair_gold_total += len(gold_pairs)
        pair_pred_total += len(pred_pairs)

        triple_tp += count_exact_matches(gold_triples, pred_triples)
        triple_gold_total += len(gold_triples)
        triple_pred_total += len(pred_triples)

        all_pair_keys = set(gold_pair_labels) | set(pred_pair_labels)
        tuple_harm_gold_total += len(gold_pair_labels)
        tuple_harm_pred_total += len(pred_pair_labels)
        for pair_key in all_pair_keys:
            gold_label = gold_pair_labels.get(pair_key)
            pred_label = pred_pair_labels.get(pair_key)
            for label in (0, 1):
                if gold_label == label and pred_label == label:
                    tuple_harm_tp[label] += 1
                elif gold_label == label and pred_label != label:
                    tuple_harm_fn[label] += 1
                elif pred_label == label and gold_label != label:
                    tuple_harm_fp[label] += 1

        sample_gold_harmful.append(1 if any(normalize_label(item) == 1 for item in gold) else 0)
        sample_pred_harmful.append(1 if any(normalize_label(item) == 1 for item in pred) else 0)

    def token_metric(gold_total: int, pred_total: int, tp: int) -> Dict[str, float]:
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        return {
            "gold_total": gold_total,
            "pred_total": pred_total,
            "true_positives": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(safe_f1(precision, recall), 6),
        }

    def exact_metric(gold_total: int, pred_total: int, tp: int) -> Dict[str, float]:
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        return {
            "gold_total": gold_total,
            "pred_total": pred_total,
            "true_positives": tp,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(safe_f1(precision, recall), 6),
        }

    sample_accuracy = (
        sum(1 for gold_label, pred_label in zip(sample_gold_harmful, sample_pred_harmful) if gold_label == pred_label) / len(sample_gold_harmful)
        if sample_gold_harmful
        else 0.0
    )

    report = {
        "input_jsonl": str(input_path.resolve()),
        "sample_count": len(rows),
        "metrics": {
            "target_span": token_metric(target_token_gold_total, target_token_pred_total, target_token_tp),
            "argument_span": token_metric(argument_token_gold_total, argument_token_pred_total, argument_token_tp),
            "targeted": exact_metric(targeted_gold_total, targeted_pred_total, targeted_tp),
            "ta_pair": exact_metric(pair_gold_total, pair_pred_total, pair_tp),
            "taf1": exact_metric(triple_gold_total, triple_pred_total, triple_tp),
            "tuple_harmful": {
                "macro_f1": round(macro_f1_binary_from_counts(tuple_harm_tp, tuple_harm_fp, tuple_harm_fn), 6),
                "true_positives_by_label": tuple_harm_tp,
                "false_positives_by_label": tuple_harm_fp,
                "false_negatives_by_label": tuple_harm_fn,
                "gold_total": tuple_harm_gold_total,
                "pred_total": tuple_harm_pred_total,
                "note": "Tuple harmfulness macro-F1 over exact (target, argument) identities.",
            },
            "sample_harmful": {
                "accuracy": round(sample_accuracy, 6),
                "macro_f1": round(macro_f1_binary(sample_gold_harmful, sample_pred_harmful), 6),
            },
        },
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate English TBO phase0_like rows with paper-aligned exact metrics.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_phase0_like(Path(args.input_jsonl).resolve())

    if args.output_json:
        write_json(Path(args.output_json).resolve(), report)
    if args.output_csv:
        rows = []
        for metric_name, metric_value in report["metrics"].items():
            if isinstance(metric_value, dict) and "f1" in metric_value:
                rows.append({"metric": metric_name, **metric_value})
        write_csv(
            Path(args.output_csv).resolve(),
            rows,
            fieldnames=["metric", "gold_total", "pred_total", "true_positives", "precision", "recall", "f1"],
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
