from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List


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


def normalize_label(item: Dict[str, Any]) -> int | None:
    raw = item.get("harmful") if "harmful" in item else item.get("hateful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if text in {"1", "harmful", "hate", "hateful", "true"}:
        return 1
    if text in {"0", "non-harmful", "non_harmful", "non-hate", "non_hate", "false"}:
        return 0
    return None


def evaluate_phase0_like(input_jsonl: Path) -> Dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    class_values = [0, 1]
    class_f1: Dict[str, float] = {}
    tp_map: Dict[str, int] = {}
    gold_total_map: Dict[str, int] = {}
    pred_total_map: Dict[str, int] = {}

    for label_value in class_values:
        tp = 0
        gold_total = 0
        pred_total = 0
        for row in rows:
            gold_counter = Counter(
                normalize_label(item)
                for item in (row.get("ground_truth") or [])
                if isinstance(item, dict) and normalize_label(item) is not None
            )
            pred_counter = Counter(
                normalize_label(item)
                for item in (row.get("predictions") or [])
                if isinstance(item, dict) and normalize_label(item) is not None
            )
            gold_total += int(gold_counter.get(label_value, 0))
            pred_total += int(pred_counter.get(label_value, 0))
            tp += min(int(gold_counter.get(label_value, 0)), int(pred_counter.get(label_value, 0)))
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        class_f1[str(label_value)] = round(f1, 6)
        tp_map[str(label_value)] = tp
        gold_total_map[str(label_value)] = gold_total
        pred_total_map[str(label_value)] = pred_total

    macro_f1 = sum(class_f1.values()) / len(class_values)
    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "row_count": len(rows),
        "metrics": {
            "tuple_harmful_paperlike": {
                "class_f1": class_f1,
                "tp": tp_map,
                "gold_total": gold_total_map,
                "pred_total": pred_total_map,
                "macro_f1": round(macro_f1, 6),
            }
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate paper-like tuple harmful Macro-F1 on phase0_like JSONL.")
    parser.add_argument("--input-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_phase0_like(Path(args.input_jsonl).resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
