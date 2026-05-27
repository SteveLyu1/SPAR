from __future__ import annotations

import argparse
import csv
import html
import json
import re
from collections import Counter
from difflib import SequenceMatcher
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
    if text in {"implicit_target"}:
        return None
    return text


def normalize_argument(value: Any) -> Optional[str]:
    return normalize_match_text(value)


def normalize_label(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("harmful") if "harmful" in item else item.get("hateful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    text = normalize_match_text(raw)
    if text is None:
        return None
    if text in {"1", "harmful", "hate", "hateful", "true"}:
        return 1
    if text in {"0", "non-harmful", "non_harmful", "non-hate", "non_hate", "false"}:
        return 0
    return None


def tokenize(text: str) -> List[str]:
    return [match.group(0) for match in TOKEN_RE.finditer(text)]


def exact_token_match_positions(text_tokens: Sequence[str], span_tokens: Sequence[str]) -> List[Tuple[int, int]]:
    if not span_tokens or len(span_tokens) > len(text_tokens):
        return []
    results: List[Tuple[int, int]] = []
    n = len(span_tokens)
    for i in range(0, len(text_tokens) - n + 1):
        if list(text_tokens[i : i + n]) == list(span_tokens):
            results.append((i, i + n))
    return results


def best_fuzzy_token_window(text_tokens: Sequence[str], span_tokens: Sequence[str]) -> Optional[Tuple[int, int, float]]:
    if not text_tokens or not span_tokens:
        return None
    target_len = len(span_tokens)
    min_len = max(1, target_len - 2)
    max_len = min(len(text_tokens), target_len + 2)
    best: Optional[Tuple[int, int, float]] = None
    span_joined = " ".join(span_tokens)
    for window_len in range(min_len, max_len + 1):
        for i in range(0, len(text_tokens) - window_len + 1):
            candidate_tokens = text_tokens[i : i + window_len]
            candidate_joined = " ".join(candidate_tokens)
            ratio = SequenceMatcher(None, span_joined, candidate_joined).ratio()
            if best is None or ratio > best[2]:
                best = (i, i + window_len, ratio)
    return best


def locate_span_token_indices(source_text: Any, span_text: Any) -> List[int]:
    source = normalize_string(source_text)
    span = normalize_string(span_text)
    if source is None or span is None:
        return []
    source_tokens = tokenize(source.casefold())
    span_tokens = tokenize(span.casefold())
    if not span_tokens:
        return []

    exact = exact_token_match_positions(source_tokens, span_tokens)
    if exact:
        start, end = exact[0]
        return list(range(start, end))

    fuzzy = best_fuzzy_token_window(source_tokens, span_tokens)
    if fuzzy is None or fuzzy[2] < 0.6:
        return []
    start, end, _ = fuzzy
    return list(range(start, end))


def span_token_f1(rows: Sequence[Dict[str, Any]], field: str) -> Dict[str, Any]:
    gold_total = 0
    pred_total = 0
    true_positives = 0

    for row in rows:
        text = row.get("text")
        gold_items = row.get("ground_truth") or []
        pred_items = row.get("predictions") or []

        gold_tokens: set[int] = set()
        pred_tokens: set[int] = set()

        for item in gold_items:
            if not isinstance(item, dict):
                continue
            value = item.get(field)
            token_ids = locate_span_token_indices(text, value)
            gold_tokens.update(token_ids)

        for item in pred_items:
            if not isinstance(item, dict):
                continue
            value = item.get(field)
            token_ids = locate_span_token_indices(text, value)
            pred_tokens.update(token_ids)

        gold_total += len(gold_tokens)
        pred_total += len(pred_tokens)
        true_positives += len(gold_tokens & pred_tokens)

    precision = true_positives / pred_total if pred_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "gold_total": gold_total,
        "pred_total": pred_total,
        "true_positives": true_positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def build_units(row: Dict[str, Any], metric: str) -> Tuple[List[Tuple[Any, ...]], List[Tuple[Any, ...]]]:
    gold_units: List[Tuple[Any, ...]] = []
    pred_units: List[Tuple[Any, ...]] = []

    def convert(item: Dict[str, Any]) -> Optional[Tuple[Any, ...]]:
        target = normalize_target(item.get("target"))
        argument = normalize_argument(item.get("argument"))
        label = normalize_label(item)
        if metric == "targeted":
            if label is None:
                return None
            return (target, label)
        if metric == "ta_pair":
            if argument is None:
                return None
            return (target, argument)
        if metric == "taf1":
            if argument is None or label is None:
                return None
            return (target, argument, label)
        raise ValueError(f"Unsupported metric: {metric}")

    for item in row.get("ground_truth") or []:
        if isinstance(item, dict):
            unit = convert(item)
            if unit is not None:
                gold_units.append(unit)

    for item in row.get("predictions") or []:
        if isinstance(item, dict):
            unit = convert(item)
            if unit is not None:
                pred_units.append(unit)

    return gold_units, pred_units


def exact_unit_f1(rows: Sequence[Dict[str, Any]], metric: str) -> Dict[str, Any]:
    gold_total = 0
    pred_total = 0
    true_positives = 0

    for row in rows:
        gold_units, pred_units = build_units(row, metric=metric)
        gold_counter = Counter(gold_units)
        pred_counter = Counter(pred_units)
        gold_total += sum(gold_counter.values())
        pred_total += sum(pred_counter.values())
        true_positives += sum(min(gold_counter[k], pred_counter[k]) for k in set(gold_counter) | set(pred_counter))

    precision = true_positives / pred_total if pred_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "gold_total": gold_total,
        "pred_total": pred_total,
        "true_positives": true_positives,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def tuple_harmful_debug(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    class_values = [0, 1]
    class_f1: Dict[str, float] = {}

    for label_value in class_values:
        gold_total = 0
        pred_total = 0
        tp = 0
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

    macro_f1 = sum(class_f1.values()) / len(class_values)
    return {
        "class_f1": class_f1,
        "macro_f1": round(macro_f1, 6),
    }


def sample_harmful_metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    gold_labels: List[int] = []
    pred_labels: List[int] = []
    for row in rows:
        gold_harmful = any(normalize_label(item) == 1 for item in (row.get("ground_truth") or []) if isinstance(item, dict))
        pred_harmful = any(normalize_label(item) == 1 for item in (row.get("predictions") or []) if isinstance(item, dict))
        gold_labels.append(1 if gold_harmful else 0)
        pred_labels.append(1 if pred_harmful else 0)

    accuracy = sum(1 for g, p in zip(gold_labels, pred_labels) if g == p) / len(gold_labels) if gold_labels else 0.0
    class_f1: Dict[str, float] = {}
    for label_value in [0, 1]:
        tp = sum(1 for g, p in zip(gold_labels, pred_labels) if g == label_value and p == label_value)
        pred_total = sum(1 for p in pred_labels if p == label_value)
        gold_total = sum(1 for g in gold_labels if g == label_value)
        precision = tp / pred_total if pred_total else 0.0
        recall = tp / gold_total if gold_total else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        class_f1[str(label_value)] = round(f1, 6)
    macro_f1 = sum(class_f1.values()) / 2.0
    return {
        "accuracy": round(accuracy, 6),
        "macro_f1": round(macro_f1, 6),
        "class_f1": class_f1,
    }


def evaluate_phase0_like(input_jsonl: Path) -> Dict[str, Any]:
    rows = read_jsonl(input_jsonl)
    metrics = {
        "target_span": span_token_f1(rows, field="target"),
        "argument_span": span_token_f1(rows, field="argument"),
        "targeted": exact_unit_f1(rows, metric="targeted"),
        "ta_pair": exact_unit_f1(rows, metric="ta_pair"),
        "taf1": exact_unit_f1(rows, metric="taf1"),
        "tuple_harmful": tuple_harmful_debug(rows),
        "sample_harmful": sample_harmful_metrics(rows),
    }
    return {
        "input_jsonl": str(input_jsonl.resolve()),
        "row_count": len(rows),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TBO-style English metrics on phase0_like JSONL.")
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
        for metric_name, payload in report["metrics"].items():
            if isinstance(payload, dict) and "f1" in payload:
                rows.append({"metric": metric_name, **payload})
        if rows:
            write_csv(Path(args.output_csv).resolve(), rows, fieldnames=list(rows[0].keys()))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
