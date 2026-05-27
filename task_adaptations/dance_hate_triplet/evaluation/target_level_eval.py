from __future__ import annotations

import argparse
import csv
import html
import json
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


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


@dataclass(frozen=True)
class Unit:
    target: Optional[str]
    argument: Optional[str]
    label: Optional[int]
    group: Optional[str]


def build_unit(item: Dict[str, Any]) -> Unit:
    return Unit(
        target=normalize_target(item.get("target")),
        argument=normalize_argument(item.get("argument")),
        label=normalize_label(item),
        group=normalize_group(item.get("group")),
    )


def metric_signature(unit: Unit, metric: str) -> Tuple[Any, ...]:
    if metric == "target":
        return (unit.target,)
    if metric == "argument":
        return (unit.argument,)
    if metric == "ta_pair":
        return (unit.target, unit.argument)
    if metric == "tah_tri":
        return (unit.target, unit.argument, unit.label)
    if metric == "quad":
        return (unit.target, unit.argument, unit.label, unit.group)
    raise ValueError(f"Unsupported metric: {metric}")


def metric_applicable(unit: Unit, metric: str) -> bool:
    if metric == "target":
        return unit.target is not None
    if metric == "argument":
        return unit.argument is not None
    if metric == "ta_pair":
        return unit.argument is not None
    if metric == "tah_tri":
        return unit.argument is not None and unit.label is not None
    if metric == "quad":
        return unit.argument is not None and unit.label is not None and unit.group is not None
    raise ValueError(f"Unsupported metric: {metric}")


def units_match(pred: Unit, gold: Unit, metric: str, match_mode: str, soft_threshold: float = 0.5) -> bool:
    if metric == "target":
        if match_mode == "hard":
            return pred.target == gold.target
        return soft_span_match(pred.target, gold.target, threshold=soft_threshold)

    if metric == "argument":
        if match_mode == "hard":
            return pred.argument == gold.argument
        return soft_span_match(pred.argument, gold.argument, threshold=soft_threshold)

    if metric == "ta_pair":
        target_ok = pred.target == gold.target if match_mode == "hard" else soft_span_match(pred.target, gold.target, threshold=soft_threshold)
        arg_ok = pred.argument == gold.argument if match_mode == "hard" else soft_span_match(pred.argument, gold.argument, threshold=soft_threshold)
        return target_ok and arg_ok

    if metric == "tah_tri":
        if pred.label != gold.label:
            return False
        target_ok = pred.target == gold.target if match_mode == "hard" else soft_span_match(pred.target, gold.target, threshold=soft_threshold)
        arg_ok = pred.argument == gold.argument if match_mode == "hard" else soft_span_match(pred.argument, gold.argument, threshold=soft_threshold)
        return target_ok and arg_ok

    if metric == "quad":
        if pred.label != gold.label or pred.group != gold.group:
            return False
        target_ok = pred.target == gold.target if match_mode == "hard" else soft_span_match(pred.target, gold.target, threshold=soft_threshold)
        arg_ok = pred.argument == gold.argument if match_mode == "hard" else soft_span_match(pred.argument, gold.argument, threshold=soft_threshold)
        return target_ok and arg_ok

    raise ValueError(f"Unsupported metric: {metric}")


def parse_reference_jsonls(raw: Optional[str]) -> List[Path]:
    if not raw:
        return []
    return [Path(item).resolve() for item in raw.split(",") if item.strip()]


def load_phase0_like_rows(
    input_path: Path,
    input_format: str,
    reference_jsonls: Sequence[Path],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if input_format != "phase0_like":
        raise ValueError(f"Unsupported input_format: {input_format}")

    rows = read_jsonl(input_path)
    summary = {
        "input_row_count": len(rows),
        "phase0_like_row_count": len(rows),
        "missing_reference_samples": 0,
    }
    if reference_jsonls:
        summary["reference_jsonls"] = [str(path) for path in reference_jsonls]
    return rows, summary


def dedupe_units(units: Sequence[Unit], metric: str) -> List[Unit]:
    seen = set()
    out: List[Unit] = []
    for unit in units:
        signature = metric_signature(unit, metric)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(unit)
    return out


def evaluate_metric_standard(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    match_mode: str,
    soft_threshold: float,
) -> Dict[str, Any]:
    gold_total = 0
    pred_total = 0
    true_positives = 0

    for row in rows:
        gold_units = [build_unit(item) for item in (row.get("ground_truth") or []) if isinstance(item, dict)]
        pred_units = [build_unit(item) for item in (row.get("predictions") or []) if isinstance(item, dict)]

        gold_units = [unit for unit in gold_units if metric_applicable(unit, metric)]
        pred_units = [unit for unit in pred_units if metric_applicable(unit, metric)]

        gold_total += len(gold_units)
        pred_total += len(pred_units)

        matched_gold = [False] * len(gold_units)
        for pred in pred_units:
            matched_index = None
            for idx, gold in enumerate(gold_units):
                if matched_gold[idx]:
                    continue
                if units_match(pred, gold, metric=metric, match_mode=match_mode, soft_threshold=soft_threshold):
                    matched_index = idx
                    break
            if matched_index is not None:
                matched_gold[matched_index] = True
                true_positives += 1

    precision = true_positives / pred_total if pred_total else 0.0
    recall = true_positives / gold_total if gold_total else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "gold_unit_count": gold_total,
        "pred_unit_count": pred_total,
        "true_positives": true_positives,
        "gold_hit_rate": round(recall, 6),
        "accuracy": round(recall, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "match_mode": match_mode,
        "metric": metric,
        "scoring_mode": "standard",
        "soft_match_threshold": None if match_mode == "hard" else soft_threshold,
    }


def evaluate_metric_standard_macro(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    match_mode: str,
    soft_threshold: float,
) -> Dict[str, Any]:
    sample_precision_sum = 0.0
    sample_recall_sum = 0.0
    sample_f1_sum = 0.0
    sample_count = 0

    gold_total = 0
    pred_total = 0
    true_positives = 0

    for row in rows:
        gold_units = [build_unit(item) for item in (row.get("ground_truth") or []) if isinstance(item, dict)]
        pred_units = [build_unit(item) for item in (row.get("predictions") or []) if isinstance(item, dict)]

        gold_units = [unit for unit in gold_units if metric_applicable(unit, metric)]
        pred_units = [unit for unit in pred_units if metric_applicable(unit, metric)]

        if not gold_units and not pred_units:
            continue

        gold_total += len(gold_units)
        pred_total += len(pred_units)

        matched_gold = [False] * len(gold_units)
        row_tp = 0
        for pred in pred_units:
            matched_index = None
            for idx, gold in enumerate(gold_units):
                if matched_gold[idx]:
                    continue
                if units_match(pred, gold, metric=metric, match_mode=match_mode, soft_threshold=soft_threshold):
                    matched_index = idx
                    break
            if matched_index is not None:
                matched_gold[matched_index] = True
                row_tp += 1

        true_positives += row_tp
        row_precision = row_tp / len(pred_units) if pred_units else 0.0
        row_recall = row_tp / len(gold_units) if gold_units else 0.0
        row_f1 = (2 * row_precision * row_recall / (row_precision + row_recall)) if (row_precision + row_recall) else 0.0

        sample_precision_sum += row_precision
        sample_recall_sum += row_recall
        sample_f1_sum += row_f1
        sample_count += 1

    macro_precision = sample_precision_sum / sample_count if sample_count else 0.0
    macro_recall = sample_recall_sum / sample_count if sample_count else 0.0
    macro_f1 = sample_f1_sum / sample_count if sample_count else 0.0

    return {
        "gold_unit_count": gold_total,
        "pred_unit_count": pred_total,
        "true_positives": true_positives,
        "sample_count": sample_count,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "match_mode": match_mode,
        "metric": metric,
        "scoring_mode": "standard_macro",
        "soft_match_threshold": None if match_mode == "hard" else soft_threshold,
    }


def evaluate_metric_gold_anchor(
    rows: Sequence[Dict[str, Any]],
    metric: str,
    match_mode: str,
    soft_threshold: float,
) -> Dict[str, Any]:
    gold_total = 0
    pred_total = 0
    true_positives = 0

    for row in rows:
        gold_units = [build_unit(item) for item in (row.get("ground_truth") or []) if isinstance(item, dict)]
        pred_units = [build_unit(item) for item in (row.get("predictions") or []) if isinstance(item, dict)]

        gold_units = dedupe_units([unit for unit in gold_units if metric_applicable(unit, metric)], metric=metric)
        pred_units = dedupe_units([unit for unit in pred_units if metric_applicable(unit, metric)], metric=metric)

        gold_total += len(gold_units)
        pred_total += len(pred_units)

        for gold in gold_units:
            if any(units_match(pred, gold, metric=metric, match_mode=match_mode, soft_threshold=soft_threshold) for pred in pred_units):
                true_positives += 1

    gold_hit_rate = true_positives / gold_total if gold_total else 0.0
    return {
        "gold_unit_count": gold_total,
        "pred_unit_count": pred_total,
        "true_positives": true_positives,
        "gold_hit_rate": round(gold_hit_rate, 6),
        "accuracy": round(gold_hit_rate, 6),
        "precision": round(gold_hit_rate, 6),
        "recall": round(gold_hit_rate, 6),
        "f1": round(gold_hit_rate, 6),
        "match_mode": match_mode,
        "metric": metric,
        "scoring_mode": "gold_anchor",
        "soft_match_threshold": None if match_mode == "hard" else soft_threshold,
    }


def build_scope_rows(all_rows: Sequence[Dict[str, Any]], requested_splits: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    scopes: Dict[str, List[Dict[str, Any]]] = {}
    rows = [row for row in all_rows if isinstance(row, dict)]
    scopes["overall"] = rows

    requested = [split for split in requested_splits if split]
    present_splits = sorted({str(row.get("split") or "") for row in rows})
    for split in requested:
        scopes[split] = [row for row in rows if str(row.get("split") or "") == split]
    for split in present_splits:
        if split and split not in scopes:
            scopes[split] = [row for row in rows if str(row.get("split") or "") == split]
    return scopes


def evaluate_input(
    input_path: Path,
    input_format: str = "phase0_like",
    reference_jsonls: Sequence[Path] = (),
    requested_splits: Sequence[str] = ("test",),
    scoring_mode: str = "standard",
    soft_match_threshold: float = 0.5,
) -> Dict[str, Any]:
    rows, build_summary = load_phase0_like_rows(
        input_path=input_path,
        input_format=input_format,
        reference_jsonls=reference_jsonls,
    )
    scopes = build_scope_rows(rows, requested_splits=requested_splits)

    csv_rows: List[Dict[str, Any]] = []
    scope_payload: Dict[str, Any] = {}
    evaluator = evaluate_metric_standard if scoring_mode == "standard" else evaluate_metric_gold_anchor

    for scope_name, scope_rows in scopes.items():
        metrics: Dict[str, Any] = {}
        scope_languages = sorted({str(row.get("language") or "") for row in scope_rows if row.get("language") is not None})
        all_zh = bool(scope_rows) and all(str(row.get("language") or "") == "zh" for row in scope_rows)

        metric_names = ["target", "argument", "ta_pair", "tah_tri"]
        if all_zh:
            metric_names.append("quad")

        for metric in metric_names:
            metric_result = {
                "applicable": True,
                "hard": evaluator(scope_rows, metric=metric, match_mode="hard", soft_threshold=soft_match_threshold),
                "soft": evaluator(scope_rows, metric=metric, match_mode="soft", soft_threshold=soft_match_threshold),
                "hard_macro": evaluate_metric_standard_macro(scope_rows, metric=metric, match_mode="hard", soft_threshold=soft_match_threshold),
                "soft_macro": evaluate_metric_standard_macro(scope_rows, metric=metric, match_mode="soft", soft_threshold=soft_match_threshold),
            }
            metrics[metric] = metric_result

            for mode_key in ("hard", "soft"):
                csv_rows.append(
                    {
                        "scope": scope_name,
                        "metric": metric,
                        "scoring_mode": scoring_mode,
                        "match_mode": mode_key,
                        "sample_count": len(scope_rows),
                        **metric_result[mode_key],
                    }
                )

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

    notes = [
        "Evaluation unit is target/tuple, not sample-level hit accuracy.",
        "String matching is case-insensitive.",
        "ZH group combinations are normalized into sorted sets, so 'LGBTQ, Sexism, Racism' equals 'Sexism, LGBTQ, Racism'.",
        "Reported accuracy is target-level gold-hit coverage; under this unit it equals recall.",
    ]
    if scoring_mode == "gold_anchor":
        notes.extend(
            [
                "Gold-anchor scoring uses metric-projected gold units as the denominator. If any prediction matches a gold unit, that gold counts as hit; extra unmatched predictions do not reduce the score.",
                "Under gold-anchor scoring, precision/recall/f1 fields are populated with the same gold-hit-rate value for easier comparison, but this is not a standard precision-penalized F1.",
            ]
        )

    return {
        "input_jsonl": str(input_path.resolve()),
        "input_format": input_format,
        "reference_jsonls": [str(path.resolve()) for path in reference_jsonls],
        "build_summary": build_summary,
        "matching_rules": {
            "target_level_unit": True,
            "case_insensitive": True,
            "group_order_insensitive": True,
            "soft_match_threshold": soft_match_threshold,
            "scoring_mode": scoring_mode,
        },
        "notes": notes,
        "scopes": scope_payload,
        "csv_rows": csv_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate phase0_like rows with target-level hard/soft metrics.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--input-format", default="phase0_like")
    parser.add_argument("--reference-jsonls", default=None)
    parser.add_argument("--requested-splits", default="test")
    parser.add_argument("--scoring-mode", choices=["standard", "gold_anchor"], default="standard")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested_splits = [item.strip() for item in args.requested_splits.split(",") if item.strip()]
    report = evaluate_input(
        input_path=Path(args.input_jsonl).resolve(),
        input_format=args.input_format,
        reference_jsonls=parse_reference_jsonls(args.reference_jsonls),
        requested_splits=requested_splits or ["test"],
        scoring_mode=args.scoring_mode,
    )

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
                "scoring_mode",
                "match_mode",
                "sample_count",
                "gold_unit_count",
                "pred_unit_count",
                "true_positives",
                "gold_hit_rate",
                "accuracy",
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
