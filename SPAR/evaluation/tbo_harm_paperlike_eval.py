from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence


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


def normalize_string(value: Any) -> str | None:
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


def normalize_label(item: Dict[str, Any]) -> int | None:
    raw = item.get("harmful") if "harmful" in item else item.get("hateful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    text = normalize_string(raw)
    if text is None:
        text = normalize_string(item.get("harmful_direction"))
    if text is None:
        return None
    lowered = text.casefold()
    if lowered in {"1", "harmful", "hateful", "hate", "true"}:
        return 1
    if lowered in {"0", "non-harmful", "non_harmful", "non-hate", "non_hate", "false"}:
        return 0
    return None


def safe_f1(precision: float, recall: float) -> float:
    return (2 * precision * recall / (precision + recall)) if precision + recall else 0.0


def macro_f1_from_counts(tp_by_label: Dict[int, int], fp_by_label: Dict[int, int], fn_by_label: Dict[int, int]) -> float:
    scores: List[float] = []
    for label in (0, 1):
        tp = tp_by_label.get(label, 0)
        fp = fp_by_label.get(label, 0)
        fn = fn_by_label.get(label, 0)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(safe_f1(precision, recall))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_phase0_like(input_path: Path) -> Dict[str, Any]:
    rows = [
        row
        for row in read_jsonl(input_path)
        if str(row.get("language") or "") == "en" and str(row.get("split") or "") == "test"
    ]

    tp_by_label = {0: 0, 1: 0}
    fp_by_label = {0: 0, 1: 0}
    fn_by_label = {0: 0, 1: 0}
    gold_total = 0
    pred_total = 0

    for row in rows:
        gold_labels = [normalize_label(item) for item in (row.get("ground_truth") or []) if isinstance(item, dict)]
        pred_labels = [normalize_label(item) for item in (row.get("predictions") or []) if isinstance(item, dict)]

        gold_counter = Counter(label for label in gold_labels if label is not None)
        pred_counter = Counter(label for label in pred_labels if label is not None)

        gold_total += sum(gold_counter.values())
        pred_total += sum(pred_counter.values())

        for label in (0, 1):
            tp = min(gold_counter[label], pred_counter[label])
            tp_by_label[label] += tp
            if gold_counter[label] > pred_counter[label]:
                fn_by_label[label] += gold_counter[label] - pred_counter[label]
            if pred_counter[label] > gold_counter[label]:
                fp_by_label[label] += pred_counter[label] - gold_counter[label]

    return {
        "input_jsonl": str(input_path.resolve()),
        "sample_count": len(rows),
        "metric_name": "tuple_harmful_paperlike",
        "notes": [
            "Approximate paper-like harmfulness-only tuple Macro-F1.",
            "Evaluation unit is tuple label count, not exact (target, argument) identity.",
            "Within each sample, gold and predicted harmful/non-harmful tuple counts are matched as label multisets.",
            "This is intended for analysis when the original paper-level harmfulness protocol is underspecified or unavailable.",
        ],
        "metrics": {
            "tuple_harmful_paperlike": {
                "macro_f1": round(macro_f1_from_counts(tp_by_label, fp_by_label, fn_by_label), 6),
                "true_positives_by_label": tp_by_label,
                "false_positives_by_label": fp_by_label,
                "false_negatives_by_label": fn_by_label,
                "gold_total": gold_total,
                "pred_total": pred_total,
                "note": "Tuple harmfulness Macro-F1 over per-sample harmfulness label multisets.",
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate TBO harmfulness with a paper-like tuple-label-only Macro-F1 approximation."
    )
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
        metric = report["metrics"]["tuple_harmful_paperlike"]
        write_csv(
            Path(args.output_csv).resolve(),
            [
                {
                    "metric": "tuple_harmful_paperlike",
                    "macro_f1": metric["macro_f1"],
                    "gold_total": metric["gold_total"],
                    "pred_total": metric["pred_total"],
                    "tp_label_0": metric["true_positives_by_label"][0],
                    "tp_label_1": metric["true_positives_by_label"][1],
                    "fp_label_0": metric["false_positives_by_label"][0],
                    "fp_label_1": metric["false_positives_by_label"][1],
                    "fn_label_0": metric["false_negatives_by_label"][0],
                    "fn_label_1": metric["false_negatives_by_label"][1],
                }
            ],
            fieldnames=[
                "metric",
                "macro_f1",
                "gold_total",
                "pred_total",
                "tp_label_0",
                "tp_label_1",
                "fp_label_0",
                "fp_label_1",
                "fn_label_0",
                "fn_label_1",
            ],
        )

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
