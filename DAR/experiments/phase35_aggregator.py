from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
VALID_HARMFUL_DIRECTION = {"harmful", "non-harmful", "unknown"}


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    input_jsonl: Path
    reference_jsonls: List[Path]
    output_root: Path
    run_root: Path


@dataclass
class GroupResult:
    decision_id: str
    group_id: str
    sample_id: str
    local_group_id: int
    source: str
    split: str
    language: str
    sentence_ids: List[int]
    group_text: str
    final_tuple: Dict[str, Any]
    final_tuples: List[Dict[str, Any]]
    audit_confidence: float
    veto_flag: bool


class EventLogger:
    def __init__(self, run_id: str, log_path: Path, level: str = "INFO") -> None:
        self.run_id = run_id
        self.log_path = log_path
        self.level = level if level in LOG_LEVEL_ORDER else "INFO"
        ensure_dir(log_path.parent)
        self._fh = log_path.open("w", encoding="utf-8")

    def event(self, stage: str, event: str, level: str = "INFO", **payload: Any) -> None:
        safe_level = level if level in LOG_LEVEL_ORDER else "INFO"
        if LOG_LEVEL_ORDER[safe_level] < LOG_LEVEL_ORDER[self.level]:
            return
        record = {
            "ts": utc_now(),
            "run_id": self.run_id,
            "level": safe_level,
            "stage": stage,
            "event": event,
            "payload": payload,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_sha1(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "none", "nan"}:
        return None
    return text


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def safe_sentence_ids(value: Any) -> List[int]:
    out: List[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and item >= 0:
                out.append(item)
            elif isinstance(item, str) and item.isdigit():
                out.append(int(item))
    return sorted(set(out))


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def detect_language(text: Optional[str]) -> str:
    if text and CJK_RE.search(text):
        return "zh"
    return "en"


def normalize_harmful_direction(direction: Any, harmful: Any) -> str:
    normalized = normalize_string(direction)
    if normalized is not None:
        lowered = normalized.lower()
        if lowered in VALID_HARMFUL_DIRECTION:
            return lowered
    if isinstance(harmful, bool):
        return "harmful" if harmful else "non-harmful"
    if isinstance(harmful, (int, float)):
        return "harmful" if int(harmful) == 1 else "non-harmful"
    harmful_text = normalize_string(harmful)
    if harmful_text is not None:
        lowered = harmful_text.lower()
        if lowered in {"1", "harmful", "hate", "hateful", "true"}:
            return "harmful"
        if lowered in {"0", "non-harmful", "non_harmful", "false"}:
            return "non-harmful"
    return "unknown"


def harmful_direction_to_binary(direction: str) -> Optional[int]:
    if direction == "harmful":
        return 1
    if direction == "non-harmful":
        return 0
    return None


def normalize_tuple_fields(tuple_obj: Dict[str, Any], language: str) -> Dict[str, Any]:
    target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    argument = normalize_string(tuple_obj.get("argument"))
    group_label = normalize_string(tuple_obj.get("group"))
    if language == "en":
        group_label = group_label or None
    else:
        group_label = group_label or "unknown"

    harmful_direction = normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful"))
    harmful = tuple_obj.get("harmful")
    if harmful is None:
        harmful = harmful_direction_to_binary(harmful_direction)
    elif not isinstance(harmful, (int, float)):
        harmful = harmful_direction_to_binary(harmful_direction)
    else:
        harmful = int(harmful)

    return {
        "target": target,
        "argument": argument,
        "group": group_label,
        "harmful": harmful,
        "harmful_direction": harmful_direction,
    }


def normalize_sig_value(value: Any, language: str) -> str:
    text = normalize_string(value)
    if text is None:
        return "__missing__"
    if language == "en":
        return text.lower()
    return text


def tuple_signature(tuple_obj: Dict[str, Any], language: str) -> Tuple[str, str, str, str]:
    return (
        normalize_sig_value(tuple_obj.get("target"), language),
        normalize_sig_value(tuple_obj.get("argument"), language),
        normalize_sig_value(tuple_obj.get("group"), language),
        normalize_sig_value(tuple_obj.get("harmful_direction"), "en"),
    )


def extract_row_final_tuples(row: Dict[str, Any], language: str) -> List[Dict[str, Any]]:
    tuples: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str, str, str]] = set()

    raw_list = row.get("final_tuples") if isinstance(row.get("final_tuples"), list) else None
    candidates = raw_list if raw_list else [row.get("final_tuple") if isinstance(row.get("final_tuple"), dict) else {}]
    for item in candidates:
        if not isinstance(item, dict):
            continue
        normalized = normalize_tuple_fields(item, language=language)
        signature = tuple_signature(normalized, language=language)
        if signature in seen:
            continue
        seen.add(signature)
        tuples.append(normalized)

    if tuples:
        return tuples

    fallback = normalize_tuple_fields(row.get("final_tuple") if isinstance(row.get("final_tuple"), dict) else {}, language=language)
    return [fallback]


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


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_path(workspace_root: Path, config_dir: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if raw.startswith("./") or raw.startswith("../"):
        return (config_dir / p).resolve()
    return (workspace_root / p).resolve()


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAR-Hate Phase 3.5 Aggregator")
    default_config = str(Path(__file__).resolve().parents[1] / "configs" / "phase35_template.json")
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--input-jsonl", default=None, help="Override phase3 final tuple JSONL")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N samples")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--dry-run", action="store_true", help="Skip candidate JSONL writes")
    parser.add_argument("--strict", action="store_true", help="Fail when quality gate fails")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config. Use JSON config or install pyyaml.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    override_map = [
        (args.input_jsonl, ["paths", "input_jsonl"]),
        (args.output_root, ["paths", "output_root"]),
        (args.log_level, ["logging", "level"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)
    if args.splits:
        set_nested(cfg, ["runtime", "default_splits"], [x.strip() for x in args.splits.split(",") if x.strip()])
    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg["paths"]

    input_jsonl = resolve_path(workspace_root, config_dir, str(cfg_paths["input_jsonl"]))
    reference_jsonls = parse_reference_jsonls(cfg=cfg, workspace_root=workspace_root, config_dir=config_dir)
    output_root = resolve_path(workspace_root, config_dir, str(cfg_paths["output_root"]))

    run_id = args.run_id or f"phase35_aggregator_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_id

    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        input_jsonl=input_jsonl,
        reference_jsonls=reference_jsonls,
        output_root=output_root,
        run_root=run_root,
    )


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    splits = runtime_cfg.get("default_splits", ["train", "dev", "test"])
    if not isinstance(splits, list):
        return ["train", "dev", "test"]
    cleaned = [str(x).strip() for x in splits if str(x).strip()]
    return cleaned or ["train", "dev", "test"]


def parse_reference_jsonls(cfg: Dict[str, Any], workspace_root: Path, config_dir: Path) -> List[Path]:
    cfg_paths = cfg.get("paths", {}) if isinstance(cfg.get("paths"), dict) else {}
    raw = cfg_paths.get("reference_jsonl")

    items: List[str] = []
    if isinstance(raw, str) and raw.strip():
        items = [raw.strip()]
    elif isinstance(raw, list):
        items = [str(item).strip() for item in raw if str(item).strip()]

    if not items:
        items = [
            "DAR/outputs/phase0/cleaned_zh_train.jsonl",
            "DAR/outputs/phase0/cleaned_zh_test.jsonl",
            "DAR/outputs/phase0/cleaned_en_train.jsonl",
            "DAR/outputs/phase0/cleaned_en_test.jsonl",
        ]

    resolved: List[Path] = []
    seen: Set[Path] = set()
    for item in items:
        path = resolve_path(workspace_root, config_dir, item)
        if path in seen:
            continue
        resolved.append(path)
        seen.add(path)
    return resolved


def parse_group_result(row: Dict[str, Any]) -> GroupResult:
    language = str(row.get("language") or detect_language(normalize_string(row.get("group_text"))))
    final_tuples = extract_row_final_tuples(row, language=language)
    final_tuple = final_tuples[0] if final_tuples else normalize_tuple_fields({}, language=language)
    try:
        local_group_id = int(row.get("local_group_id") or 0)
    except (TypeError, ValueError):
        local_group_id = 0

    return GroupResult(
        decision_id=str(row.get("decision_id") or ""),
        group_id=str(row.get("group_id") or make_sha1("group", row.get("sample_id"), row.get("group_text"))[:16]),
        sample_id=str(row.get("sample_id") or make_sha1("sample", row.get("group_text"))[:16]),
        local_group_id=max(0, local_group_id),
        source=str(row.get("source") or "unknown"),
        split=str(row.get("split") or "train"),
        language=language,
        sentence_ids=safe_sentence_ids(row.get("sentence_ids")),
        group_text=normalize_string(row.get("group_text")) or "",
        final_tuple=final_tuple,
        final_tuples=final_tuples,
        audit_confidence=round(clamp(safe_float(row.get("audit_confidence"), default=0.5)), 4),
        veto_flag=bool(row.get("veto_flag")),
    )


def copy_json_compatible(value: Any) -> Any:
    return copy.deepcopy(value)


def normalize_meta_info(value: Any) -> Dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, val in value.items():
        if val is None:
            continue
        out[str(key)] = copy_json_compatible(val)
    return out


def load_reference_sample_index(reference_paths: Sequence[Path], logger: Optional[EventLogger] = None) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    loaded_paths = 0
    missing_paths = 0
    duplicate_ids = 0
    loaded_rows = 0

    for path in reference_paths:
        if not path.exists():
            missing_paths += 1
            if logger is not None:
                logger.event("reference", "reference_missing", level="WARNING", path=str(path))
            continue

        loaded_paths += 1
        rows = read_jsonl(path)
        loaded_rows += len(rows)
        for row in rows:
            sample_id = normalize_string(row.get("id")) or normalize_string(row.get("sample_id"))
            if sample_id is None:
                continue
            if sample_id in index:
                duplicate_ids += 1
                continue
            index[sample_id] = {
                "id": sample_id,
                "original_id": row.get("original_id"),
                "source": row.get("source"),
                "language": row.get("language"),
                "split": row.get("split"),
                "text": row.get("text"),
                "meta_info": normalize_meta_info(row.get("meta_info")),
                "ground_truth": copy_json_compatible(row.get("ground_truth")) if isinstance(row.get("ground_truth"), list) else [],
            }

    if logger is not None:
        logger.event(
            "reference",
            "reference_loaded",
            paths_requested=len(reference_paths),
            paths_loaded=loaded_paths,
            paths_missing=missing_paths,
            rows_loaded=loaded_rows,
            duplicate_ids_skipped=duplicate_ids,
            unique_samples=len(index),
        )

    return index


def normalize_phase0_target(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text == "IMPLICIT_TARGET":
        return None
    return text


def normalize_prediction_label(tuple_obj: Dict[str, Any]) -> Optional[int]:
    harmful = tuple_obj.get("harmful")
    if isinstance(harmful, bool):
        return int(harmful)
    if isinstance(harmful, (int, float)):
        return int(harmful)
    return harmful_direction_to_binary(
        normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful"))
    )


def fallback_original_id(sample_id: str, source: str) -> Optional[str]:
    if source == "STATE-ToxiCN" and sample_id.startswith("zh_"):
        _, _, suffix = sample_id.partition("_")
        return suffix or None
    return None


def convert_tuple_to_phase0_like_prediction(tuple_obj: Dict[str, Any], language: str) -> Dict[str, Any]:
    target = normalize_phase0_target(tuple_obj.get("target"))
    argument = normalize_string(tuple_obj.get("argument"))
    label = normalize_prediction_label(tuple_obj)

    if language == "zh":
        group_label = normalize_string(tuple_obj.get("group"))
        if label == 0 and group_label in {None, "unknown"}:
            group_label = "non-hate"
        return {
            "target": target,
            "argument": argument,
            "group": group_label or "unknown",
            "hateful": label,
        }

    return {
        "target": target,
        "argument": argument,
        "harmful": label,
    }


def build_phase0_like_row(
    aggregated_row: Dict[str, Any],
    reference_row: Optional[Dict[str, Any]],
    include_reference_ground_truth: bool,
) -> Dict[str, Any]:
    sample_id = str(aggregated_row.get("sample_id") or "")
    language = str((reference_row or {}).get("language") or aggregated_row.get("language") or "en")
    source = str((reference_row or {}).get("source") or aggregated_row.get("source") or "unknown")

    predictions = [
        convert_tuple_to_phase0_like_prediction(tuple_obj=item, language=language)
        for item in (aggregated_row.get("restored_multi_target_tuples") or [])
        if isinstance(item, dict)
    ]

    row = {
        "id": str((reference_row or {}).get("id") or sample_id),
        "original_id": (reference_row or {}).get("original_id") or fallback_original_id(sample_id, source),
        "source": source,
        "language": language,
        "split": str((reference_row or {}).get("split") or aggregated_row.get("split") or "train"),
        "text": (reference_row or {}).get("text") or str(aggregated_row.get("restored_text") or ""),
        "meta_info": normalize_meta_info((reference_row or {}).get("meta_info")),
        "predictions": predictions,
    }

    if include_reference_ground_truth and isinstance((reference_row or {}).get("ground_truth"), list):
        row["ground_truth"] = copy_json_compatible((reference_row or {}).get("ground_truth"))

    return row


def aggregate_sample(
    sample_id: str,
    rows: Sequence[GroupResult],
    dedupe_exact_tuple: bool,
    keep_group_provenance: bool,
) -> Dict[str, Any]:
    sorted_rows = sorted(
        rows,
        key=lambda r: (
            (min(r.sentence_ids) if r.sentence_ids else 10**9),
            r.local_group_id if r.local_group_id > 0 else 10**9,
            r.group_id,
        ),
    )

    source = sorted_rows[0].source if sorted_rows else "unknown"
    split = sorted_rows[0].split if sorted_rows else "train"
    language = sorted_rows[0].language if sorted_rows else "unknown"

    tuple_map: Dict[Tuple[str, str, str, str], Dict[str, Any]] = {}
    tuple_list: List[Dict[str, Any]] = []

    for group in sorted_rows:
        for merged_tuple in group.final_tuples:
            merged_tuple = dict(merged_tuple)
            signature = tuple_signature(merged_tuple, language=language)

            if dedupe_exact_tuple:
                existing = tuple_map.get(signature)
                if existing is not None:
                    provenance = existing.setdefault("provenance", {})
                    provenance.setdefault("group_ids", []).append(group.group_id)
                    provenance.setdefault("decision_ids", []).append(group.decision_id)
                    provenance["sentence_ids"] = sorted(
                        set(provenance.get("sentence_ids", [])) | set(group.sentence_ids)
                    )
                    provenance["max_audit_confidence"] = round(
                        max(safe_float(provenance.get("max_audit_confidence"), default=0.0), group.audit_confidence),
                        4,
                    )
                    provenance["veto_count"] = int(provenance.get("veto_count", 0)) + (1 if group.veto_flag else 0)
                    continue

            item = {
                "target": merged_tuple.get("target"),
                "argument": merged_tuple.get("argument"),
                "group": merged_tuple.get("group"),
                "harmful": merged_tuple.get("harmful"),
                "harmful_direction": merged_tuple.get("harmful_direction"),
            }
            if keep_group_provenance:
                item["provenance"] = {
                    "group_ids": [group.group_id],
                    "decision_ids": [group.decision_id],
                    "sentence_ids": list(group.sentence_ids),
                    "max_audit_confidence": group.audit_confidence,
                    "veto_count": 1 if group.veto_flag else 0,
                }

            tuple_list.append(item)
            tuple_map[signature] = item

    group_records = [
        {
            "group_id": g.group_id,
            "decision_id": g.decision_id,
            "local_group_id": g.local_group_id,
            "sentence_ids": g.sentence_ids,
            "group_text": g.group_text,
            "audit_confidence": g.audit_confidence,
            "veto_flag": g.veto_flag,
            "final_tuple": g.final_tuple,
            "final_tuples": g.final_tuples,
        }
        for g in sorted_rows
    ]

    text_segments = [g.group_text for g in sorted_rows if g.group_text]
    restored_text = "\n".join(dict.fromkeys(text_segments))

    return {
        "sample_id": sample_id,
        "source": source,
        "split": split,
        "language": language,
        "restored_text": restored_text,
        "restored_multi_target_tuples": tuple_list,
        "groups": group_records,
        "phase35_aggregator_trace": {
            "group_count": len(sorted_rows),
            "restored_tuple_count": len(tuple_list),
            "dedupe_exact_tuple": bool(dedupe_exact_tuple),
            "keep_group_provenance": bool(keep_group_provenance),
        },
    }


def quality_gates(rows: Sequence[Dict[str, Any]], require_non_empty_tuples: bool) -> Dict[str, Any]:
    duplicate_sample_id = 0
    invalid_split = 0
    empty_tuple_samples = 0

    seen: Set[str] = set()
    for row in rows:
        sample_id = str(row.get("sample_id") or "")
        if sample_id in seen:
            duplicate_sample_id += 1
        seen.add(sample_id)

        if str(row.get("split")) not in {"train", "dev", "test"}:
            invalid_split += 1

        tuples = row.get("restored_multi_target_tuples")
        if not isinstance(tuples, list) or not tuples:
            empty_tuple_samples += 1

    passed = duplicate_sample_id == 0 and invalid_split == 0
    if require_non_empty_tuples:
        passed = passed and empty_tuple_samples == 0

    return {
        "total_samples": len(rows),
        "duplicate_sample_id": duplicate_sample_id,
        "invalid_split": invalid_split,
        "empty_tuple_samples": empty_tuple_samples,
        "require_non_empty_tuples": bool(require_non_empty_tuples),
        "passed": bool(passed),
    }


def build_summary(
    rows: Sequence[Dict[str, Any]],
    input_group_rows: int,
    phase0_reference_hits: int,
    phase0_reference_misses: int,
) -> Dict[str, Any]:
    split_dist = dict(sorted(Counter(str(r.get("split")) for r in rows).items()))
    lang_dist = dict(sorted(Counter(str(r.get("language")) for r in rows).items()))
    source_dist = dict(sorted(Counter(str(r.get("source")) for r in rows).items()))

    tuple_counts = [len(r.get("restored_multi_target_tuples") or []) for r in rows]
    avg_tuple_count = round(sum(tuple_counts) / len(tuple_counts), 4) if tuple_counts else 0.0

    return {
        "sample_count": len(rows),
        "input_group_rows": int(input_group_rows),
        "split_distribution": split_dist,
        "language_distribution": lang_dist,
        "source_distribution": source_dist,
        "avg_tuple_count_per_sample": avg_tuple_count,
        "max_tuple_count_per_sample": max(tuple_counts) if tuple_counts else 0,
        "phase0_reference_hits": int(phase0_reference_hits),
        "phase0_reference_misses": int(phase0_reference_misses),
    }


def bucket_rows_by_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("split") or "train")].append(row)
    return buckets


def bucket_rows_by_language_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        language = str(row.get("language") or "unknown")
        split = str(row.get("split") or "train")
        buckets[f"{language}_{split}"].append(row)
    return buckets


def write_candidate_outputs(run_root: Path, rows: Sequence[Dict[str, Any]], phase0_like_rows: Sequence[Dict[str, Any]]) -> None:
    write_jsonl(run_root / "candidates" / "aggregated_samples.jsonl", rows)
    write_jsonl(run_root / "candidates" / "restored_multi_target.jsonl", rows)

    by_split = bucket_rows_by_split(rows)
    for split in ["train", "dev", "test"]:
        write_jsonl(run_root / "candidates" / "by_split" / f"{split}.jsonl", by_split.get(split, []))

    write_jsonl(run_root / "candidates" / "phase0_like.jsonl", phase0_like_rows)

    phase0_like_by_split = bucket_rows_by_split(phase0_like_rows)
    for split in ["train", "dev", "test"]:
        write_jsonl(
            run_root / "candidates" / "phase0_like_by_split" / f"{split}.jsonl",
            phase0_like_by_split.get(split, []),
        )

    phase0_like_by_language_split = bucket_rows_by_language_split(phase0_like_rows)
    for language in ["en", "zh"]:
        for split in ["train", "dev", "test"]:
            key = f"{language}_{split}"
            write_jsonl(
                run_root / "candidates" / "phase0_like_by_lang_split" / f"{key}.jsonl",
                phase0_like_by_language_split.get(key, []),
            )


def run_phase35(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cli_args = args or parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, cli_args)
    paths = build_paths(args=cli_args, cfg=cfg, workspace_root=workspace_root)
    run_id = paths.run_root.name

    logger = EventLogger(
        run_id=run_id,
        log_path=paths.run_root / "logs" / "events.jsonl",
        level=str(cfg.get("logging", {}).get("level", "INFO")),
    )

    try:
        if not paths.input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL not found: {paths.input_jsonl}")

        logger.event(
            "bootstrap",
            "run_started",
            config_path=str(paths.config_path),
            input_jsonl=str(paths.input_jsonl),
            output_root=str(paths.run_root),
        )

        phase35_cfg = cfg.get("phase35", {}) if isinstance(cfg.get("phase35"), dict) else {}
        dedupe_exact_tuple = parse_bool(phase35_cfg.get("dedupe_exact_tuple"), default=False)
        keep_group_provenance = parse_bool(phase35_cfg.get("keep_group_provenance"), default=True)
        export_phase0_like = parse_bool(phase35_cfg.get("export_phase0_like"), default=True)
        include_reference_ground_truth = parse_bool(phase35_cfg.get("include_reference_ground_truth"), default=True)

        all_rows = read_jsonl(paths.input_jsonl)
        logger.event("load", "input_loaded", rows=len(all_rows))

        parsed = [parse_group_result(row) for row in all_rows]
        allowed_splits = set(parse_allowed_splits(cfg))
        filtered = [r for r in parsed if r.split in allowed_splits]

        max_samples = cli_args.max_samples
        if max_samples is None:
            runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
            max_samples = runtime_cfg.get("max_samples")

        grouped: Dict[str, List[GroupResult]] = defaultdict(list)
        for row in filtered:
            grouped[row.sample_id].append(row)

        sample_ids = sorted(grouped.keys())
        if isinstance(max_samples, int) and max_samples > 0:
            sample_ids = sample_ids[:max_samples]

        aggregated_rows: List[Dict[str, Any]] = []
        progress_interval = int(max(1, (cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}).get("progress_interval", 200)))
        for idx, sample_id in enumerate(sample_ids):
            item = aggregate_sample(
                sample_id=sample_id,
                rows=grouped[sample_id],
                dedupe_exact_tuple=dedupe_exact_tuple,
                keep_group_provenance=keep_group_provenance,
            )
            aggregated_rows.append(item)
            if idx == 0 or (idx + 1) % progress_interval == 0:
                logger.event("aggregate", "progress", index=idx, sample_id=sample_id)

        reference_index = load_reference_sample_index(paths.reference_jsonls, logger=logger)
        phase0_like_rows: List[Dict[str, Any]] = []
        phase0_reference_hits = 0
        phase0_reference_misses = 0
        if export_phase0_like:
            for row in aggregated_rows:
                sample_id = str(row.get("sample_id") or "")
                reference_row = reference_index.get(sample_id)
                if reference_row is None:
                    phase0_reference_misses += 1
                else:
                    phase0_reference_hits += 1
                phase0_like_rows.append(
                    build_phase0_like_row(
                        aggregated_row=row,
                        reference_row=reference_row,
                        include_reference_ground_truth=include_reference_ground_truth,
                    )
                )
            logger.event(
                "reference",
                "phase0_like_rows_built",
                rows=len(phase0_like_rows),
                hits=phase0_reference_hits,
                misses=phase0_reference_misses,
                include_reference_ground_truth=bool(include_reference_ground_truth),
            )
        else:
            logger.event("reference", "phase0_like_export_disabled")

        quality_cfg = cfg.get("quality", {}) if isinstance(cfg.get("quality"), dict) else {}
        quality = quality_gates(
            rows=aggregated_rows,
            require_non_empty_tuples=parse_bool(quality_cfg.get("require_non_empty_tuples"), default=True),
        )
        write_json(paths.run_root / "reports" / "quality_gates.json", quality)
        logger.event("quality", "quality_report_written", **quality)

        strict_mode = parse_bool(quality_cfg.get("strict"), default=False)
        if strict_mode and not quality.get("passed", False):
            raise RuntimeError(f"Phase3.5 quality gate failed under strict mode: {quality}")

        summary = build_summary(
            rows=aggregated_rows,
            input_group_rows=len(filtered),
            phase0_reference_hits=phase0_reference_hits,
            phase0_reference_misses=phase0_reference_misses,
        )
        write_json(paths.run_root / "reports" / "summary.json", summary)
        logger.event("report", "summary_written", **summary)

        if not cli_args.dry_run:
            write_candidate_outputs(paths.run_root, aggregated_rows, phase0_like_rows)
            write_candidate_outputs(paths.output_root / "latest", aggregated_rows, phase0_like_rows)
            logger.event(
                "export",
                "candidates_written",
                total=len(aggregated_rows),
                phase0_like_total=len(phase0_like_rows),
            )
        else:
            logger.event("export", "dry_run_skip_large_writes")

        run_summary = {
            "run_id": run_id,
            "created_at": utc_now(),
            "config_path": str(paths.config_path),
            "input_jsonl": str(paths.input_jsonl),
            "reference_jsonl": [str(path) for path in paths.reference_jsonls],
            "output_root": str(paths.run_root),
            "quality": quality,
            "summary": summary,
        }
        write_json(paths.run_root / "run_summary.json", run_summary)
        logger.event("bootstrap", "run_finished", status="ok", samples=len(aggregated_rows))
        return run_summary
    except Exception as exc:
        logger.event("bootstrap", "run_failed", level="ERROR", error=str(exc))
        raise
    finally:
        logger.close()


def main() -> None:
    run_phase35()


if __name__ == "__main__":
    main()
