from __future__ import annotations

import argparse
import csv
import html
import json
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


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


def normalize_group(value: Any) -> Optional[str]:
    text = normalize_match_text(value)
    if text is None:
        return None
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if not parts:
        return None
    return ", ".join(sorted(parts))


def normalize_label(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("hateful") if "hateful" in item else item.get("harmful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    text = normalize_match_text(raw)
    if text is None:
        text = normalize_match_text(item.get("harmful_direction"))
    if text is None:
        return None
    if text in {"1", "hate", "hateful", "harmful", "true"}:
        return 1
    if text in {"0", "non-hate", "non_hate", "non-harmful", "non_harmful", "false"}:
        return 0
    return None


def soft_span_match(left: Any, right: Any, threshold: float = 0.5) -> bool:
    left_text = normalize_match_text(left)
    right_text = normalize_match_text(right)
    if left_text is None or right_text is None:
        return left_text is None and right_text is None
    if left_text == right_text:
        return True
    ratio = SequenceMatcher(None, left_text, right_text).ratio()
    contains = left_text in right_text or right_text in left_text
    return contains and ratio >= threshold


def metric_signature(item: Dict[str, Any], metric: str) -> Tuple[Any, ...]:
    target = normalize_target(item.get("target"))
    argument = normalize_argument(item.get("argument"))
    label = normalize_label(item)
    group = normalize_group(item.get("group"))

    if metric == "target":
        return (target,)
    if metric == "argument":
        return (argument,)
    if metric == "ta_pair":
        return (target, argument)
    if metric == "tah_tri":
        return (target, argument, label)
    if metric == "quad":
        return (target, argument, label, group)
    raise ValueError(f"Unsupported metric: {metric}")


def metric_applicable(item: Dict[str, Any], metric: str) -> bool:
    target = normalize_target(item.get("target"))
    argument = normalize_argument(item.get("argument"))
    label = normalize_label(item)
    group = normalize_group(item.get("group"))

    if metric == "target":
        return target is not None
    if metric == "argument":
        return argument is not None
    if metric == "ta_pair":
        return argument is not None
    if metric == "tah_tri":
        return argument is not None and label is not None
    if metric == "quad":
        return argument is not None and label is not None and group is not None
    raise ValueError(f"Unsupported metric: {metric}")


def units_match(pred: Dict[str, Any], gold: Dict[str, Any], metric: str, match_mode: str, soft_threshold: float = 0.5) -> bool:
    pred_target = normalize_target(pred.get("target"))
    gold_target = normalize_target(gold.get("target"))
    pred_argument = normalize_argument(pred.get("argument"))
    gold_argument = normalize_argument(gold.get("argument"))
    pred_label = normalize_label(pred)
    gold_label = normalize_label(gold)
    pred_group = normalize_group(pred.get("group"))
    gold_group = normalize_group(gold.get("group"))

    if metric == "target":
        return pred_target == gold_target if match_mode == "hard" else soft_span_match(pred_target, gold_target, threshold=soft_threshold)
    if metric == "argument":
        return pred_argument == gold_argument if match_mode == "hard" else soft_span_match(pred_argument, gold_argument, threshold=soft_threshold)
    if metric == "ta_pair":
        target_ok = pred_target == gold_target if match_mode == "hard" else soft_span_match(pred_target, gold_target, threshold=soft_threshold)
        argument_ok = pred_argument == gold_argument if match_mode == "hard" else soft_span_match(pred_argument, gold_argument, threshold=soft_threshold)
        return target_ok and argument_ok
    if metric == "tah_tri":
        if pred_label != gold_label:
            return False
        target_ok = pred_target == gold_target if match_mode == "hard" else soft_span_match(pred_target, gold_target, threshold=soft_threshold)
        argument_ok = pred_argument == gold_argument if match_mode == "hard" else soft_span_match(pred_argument, gold_argument, threshold=soft_threshold)
        return target_ok and argument_ok
    if metric == "quad":
        if pred_label != gold_label or pred_group != gold_group:
            return False
        target_ok = pred_target == gold_target if match_mode == "hard" else soft_span_match(pred_target, gold_target, threshold=soft_threshold)
        argument_ok = pred_argument == gold_argument if match_mode == "hard" else soft_span_match(pred_argument, gold_argument, threshold=soft_threshold)
        return target_ok and argument_ok
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_metric(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    match_mode: str,
    soft_threshold: float,
) -> Dict[str, Any]:
    sample_count = len(rows)
    pred_total = 0
    gold_total = 0
    true_positives = 0
    hit_samples = 0
    exact_match_samples = 0

    for row in rows:
        gold_items = [item for item in (row.get("ground_truth") or []) if isinstance(item, dict) and metric_applicable(item, metric)]
        pred_items = [item for item in (row.get("predictions") or []) if isinstance(item, dict) and metric_applicable(item, metric)]
        pred_total += len(pred_items)
        gold_total += len(gold_items)

        matched_gold = [False] * len(gold_items)
        sample_tp = 0
        for pred in pred_items:
            matched_index = None
            for idx, gold in enumerate(gold_items):
                if matched_gold[idx]:
                    continue
                if units_match(pred, gold, metric=metric, match_mode=match_mode, soft_threshold=soft_threshold):
                    matched_index = idx
                    break
            if matched_index is not None:
                matched_gold[matched_index] = True
                sample_tp += 1
                true_positives += 1

        if sample_tp > 0:
            hit_samples += 1

        pred_set = {metric_signature(item, metric) for item in pred_items}
        gold_set = {metric_signature(item, metric) for item in gold_items}
        if pred_set == gold_set:
            exact_match_samples += 1

    precision = true_positives / pred_total if pred_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = hit_samples / sample_count if sample_count else 0.0
    exact_set_accuracy = exact_match_samples / sample_count if sample_count else 0.0

    return {
        "sample_count": sample_count,
        "pred_total": pred_total,
        "gold_total": gold_total,
        "true_positives": true_positives,
        "hit_samples": hit_samples,
        "exact_match_samples": exact_match_samples,
        "accuracy": round(accuracy, 6),
        "exact_set_accuracy": round(exact_set_accuracy, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "match_mode": match_mode,
        "metric": metric,
        "soft_match_threshold": None if match_mode == "hard" else soft_threshold,
    }


def build_scopes(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    scopes: Dict[str, List[Dict[str, Any]]] = {"overall": list(rows)}
    splits = sorted({str(row.get("split") or "") for row in rows})
    for split in splits:
        if split:
            scopes[split] = [row for row in rows if str(row.get("split") or "") == split]
    return scopes


def evaluate_phase35_jsonl(input_jsonl: Path, soft_match_threshold: float = 0.5) -> Dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    scopes = build_scopes(rows)

    csv_rows: List[Dict[str, Any]] = []
    scope_payload: Dict[str, Any] = {}

    for scope_name, scope_rows in scopes.items():
        scope_languages = sorted({str(row.get("language") or "") for row in scope_rows if row.get("language") is not None})
        all_zh = bool(scope_rows) and all(str(row.get("language") or "") == "zh" for row in scope_rows)
        metric_names = ["target", "argument", "ta_pair", "tah_tri"]
        if all_zh:
            metric_names.append("quad")

        metrics: Dict[str, Any] = {}
        for metric in metric_names:
            hard = evaluate_metric(scope_rows, metric=metric, match_mode="hard", soft_threshold=soft_match_threshold)
            soft = evaluate_metric(scope_rows, metric=metric, match_mode="soft", soft_threshold=soft_match_threshold)
            metrics[metric] = {
                "applicable": True,
                "hard": hard,
                "soft": soft,
            }
            csv_rows.append({"scope": scope_name, **hard})
            csv_rows.append({"scope": scope_name, **soft})

        if not all_zh:
            metrics["quad"] = {
                "applicable": False,
                "note": "Quad is only applicable to all-ZH scopes.",
            }

        scope_payload[scope_name] = {
            "sample_count": len(scope_rows),
            "language_distribution": {
                language: sum(1 for row in scope_rows if str(row.get("language") or "") == language)
                for language in scope_languages
            },
            "split_distribution": {
                split: sum(1 for row in scope_rows if str(row.get("split") or "") == split)
                for split in sorted({str(row.get("split") or "") for row in scope_rows})
                if split
            },
            "metrics": metrics,
        }

    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "input_row_count": len(rows),
        "row_count": len(rows),
        "excluded_missing_ground_truth": 0,
        "scopes": scope_payload,
        "csv_rows": csv_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase0_like rows with STATE-style sample metrics.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate_phase35_jsonl(Path(args.input_jsonl).resolve())

    if args.output_json:
        payload = dict(report)
        payload.pop("csv_rows", None)
        write_json(Path(args.output_json).resolve(), payload)
    if args.output_csv:
        write_csv(
            Path(args.output_csv).resolve(),
            report["csv_rows"],
            fieldnames=[
                "scope",
                "metric",
                "match_mode",
                "sample_count",
                "pred_total",
                "gold_total",
                "true_positives",
                "hit_samples",
                "exact_match_samples",
                "accuracy",
                "exact_set_accuracy",
                "precision",
                "recall",
                "f1",
                "soft_match_threshold",
            ],
        )

    printable = dict(report)
    printable.pop("csv_rows", None)
    print(json.dumps(printable, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
