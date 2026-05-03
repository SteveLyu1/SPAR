from __future__ import annotations

import argparse
import copy
import html
import json
import time
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
DEFAULT_REFERENCE_JSONLS = [
    "DAR/outputs/phase0/cleaned_zh_train.jsonl",
    "DAR/outputs/phase0/cleaned_zh_test.jsonl",
    "DAR/outputs/phase0/cleaned_en_train.jsonl",
    "DAR/outputs/phase0/cleaned_en_test.jsonl",
]
ROLE_CHOICES = ("all_roles_concat", "victim_only", "moderator_only", "bystander_only", "moderator_anchor")


class EventLogger:
    def __init__(self, run_id: str, log_path: Path, level: str = "INFO", append: bool = False) -> None:
        self.run_id = run_id
        self.log_path = log_path
        self.level = level if level in LOG_LEVEL_ORDER else "INFO"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        self._fh = log_path.open(mode, encoding="utf-8")

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


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


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


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
    if not text:
        return None
    if text.lower() in {"null", "none", "nan"}:
        return None
    return text


def normalize_match_text(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    return " ".join(text.split()).casefold()


def normalize_target(value: Any) -> Optional[str]:
    text = normalize_match_text(value)
    if text == "implicit_target":
        return None
    return text


def normalize_group(value: Any, language: str) -> Optional[str]:
    if language != "zh":
        return None
    text = normalize_match_text(value)
    return text


def normalize_label(item: Dict[str, Any]) -> Optional[int]:
    raw = item.get("hateful") if "hateful" in item else item.get("harmful")
    if isinstance(raw, bool):
        return int(raw)
    if isinstance(raw, (int, float)):
        return int(raw)
    text = normalize_string(raw)
    if text is None:
        return None
    lowered = text.casefold()
    if lowered in {"1", "harmful", "hateful", "hate", "true"}:
        return 1
    if lowered in {"0", "non-harmful", "non_harmful", "non-hate", "non_hate", "false"}:
        return 0
    return None


def canonical_signature(item: Dict[str, Any], language: str) -> Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]:
    return (
        normalize_target(item.get("target")),
        normalize_match_text(item.get("argument")),
        normalize_label(item),
        normalize_group(item.get("group"), language=language),
    )


def dedupe_predictions(predictions: Sequence[Dict[str, Any]], language: str) -> List[Dict[str, Any]]:
    seen: set[Tuple[Optional[str], Optional[str], Optional[int], Optional[str]]] = set()
    out: List[Dict[str, Any]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        sig = canonical_signature(item, language=language)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(item)
    return out


def similarity(a: Any, b: Any) -> float:
    ta = normalize_match_text(a)
    tb = normalize_match_text(b)
    if ta is None and tb is None:
        return 1.0
    if ta is None or tb is None:
        return 0.0
    return SequenceMatcher(None, ta, tb).ratio()


def sameish_tuple(a: Dict[str, Any], b: Dict[str, Any], language: str) -> bool:
    if normalize_label(a) != normalize_label(b):
        return False
    if normalize_group(a.get("group"), language=language) != normalize_group(b.get("group"), language=language):
        return False
    if similarity(a.get("target"), b.get("target")) < 0.85:
        return False
    if similarity(a.get("argument"), b.get("argument")) < 0.75:
        return False
    return True


def select_more_precise_pred(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    arg_a = normalize_string(a.get("argument")) or ""
    arg_b = normalize_string(b.get("argument")) or ""
    if arg_a and arg_b:
        return copy.deepcopy(a if len(arg_a) <= len(arg_b) else b)
    if arg_a:
        return copy.deepcopy(a)
    return copy.deepcopy(b)


def normalize_prediction_for_output(pred: Dict[str, Any], language: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "target": normalize_string(pred.get("target")) or "IMPLICIT_TARGET",
        "argument": normalize_string(pred.get("argument")),
    }
    if language == "zh":
        out["group"] = normalize_string(pred.get("group")) or "unknown"
        if "hateful" in pred:
            out["hateful"] = normalize_label(pred)
        else:
            out["hateful"] = normalize_label(pred)
    else:
        out["harmful"] = normalize_label(pred)
    return out


def build_victim_bystander_consensus(
    victims: Sequence[Dict[str, Any]],
    bystanders: Sequence[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    consensus: List[Dict[str, Any]] = []
    for victim in victims:
        for bystander in bystanders:
            if not sameish_tuple(victim, bystander, language=language):
                continue
            consensus.append(select_more_precise_pred(victim, bystander))
            break
    return dedupe_predictions(consensus, language=language)


def choose_group_policy_predictions(group_row: Dict[str, Any], policy: str, language: str) -> List[Dict[str, Any]]:
    raw = group_row.get("phase2_predictions") or {}
    if not isinstance(raw, dict):
        return []

    victim = [normalize_prediction_for_output(item, language) for item in (raw.get("victim") or []) if isinstance(item, dict)]
    moderator = [normalize_prediction_for_output(item, language) for item in (raw.get("moderator") or []) if isinstance(item, dict)]
    bystander = [normalize_prediction_for_output(item, language) for item in (raw.get("bystander") or raw.get("cultural_bystander") or []) if isinstance(item, dict)]

    victim = dedupe_predictions(victim, language=language)
    moderator = dedupe_predictions(moderator, language=language)
    bystander = dedupe_predictions(bystander, language=language)

    if policy == "victim_only":
        return victim
    if policy == "moderator_only":
        return moderator
    if policy == "bystander_only":
        return bystander
    if policy == "all_roles_concat":
        return dedupe_predictions([*victim, *moderator, *bystander], language=language)
    if policy != "moderator_anchor":
        raise ValueError(f"Unsupported policy: {policy}")

    if moderator:
        return moderator

    consensus = build_victim_bystander_consensus(victim, bystander, language=language)
    if consensus:
        return consensus
    if victim:
        return victim
    if bystander:
        return bystander
    return []


def load_reference_index(reference_jsonls: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for path in reference_jsonls:
        if not path.exists():
            continue
        for row in read_jsonl(path):
            sample_id = normalize_string(row.get("id"))
            if sample_id is None or sample_id in index:
                continue
            index[sample_id] = row
    return index


def bucket_rows_by_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("split") or "unknown")].append(row)
    return buckets


def bucket_rows_by_language_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = f"{str(row.get('language') or 'unknown')}_{str(row.get('split') or 'unknown')}"
        buckets[key].append(row)
    return buckets


def resolve_reference_paths(workspace_root: Path, raw: Optional[str]) -> List[Path]:
    if raw:
        return [Path(item).resolve() for item in raw.split(",") if item.strip()]
    return [(workspace_root / item).resolve() for item in DEFAULT_REFERENCE_JSONLS]


def parse_csv_filter(raw: Optional[str]) -> Optional[set[str]]:
    if not raw:
        return None
    items = {item.strip() for item in raw.split(",") if item.strip()}
    return items or None


def compute_eta_iso(start_ts: float, completed: int, total: int) -> Optional[str]:
    if completed <= 0 or total <= completed:
        return None
    elapsed = max(0.001, time.time() - start_ts)
    rate = completed / elapsed
    if rate <= 0:
        return None
    remaining = (total - completed) / rate
    eta_ts = time.time() + remaining
    return datetime.fromtimestamp(eta_ts, tz=timezone.utc).isoformat()


def build_phase0_like_rows(
    group_rows: Sequence[Dict[str, Any]],
    reference_index: Dict[str, Dict[str, Any]],
    policy: str,
    logger: EventLogger,
    checkpoint_path: Path,
    partial_output_path: Path,
    progress_interval: int,
    checkpoint_interval: int,
    resume_from_checkpoint: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    missing_reference = 0
    for row in group_rows:
        sample_id = normalize_string(row.get("sample_id"))
        if sample_id is None:
            continue
        if sample_id not in grouped:
            grouped[sample_id] = []
            order.append(sample_id)
        grouped[sample_id].append(row)

    processed_rows: List[Dict[str, Any]] = []
    processed_ids: set[str] = set()
    start_ts = time.time()
    resumed = False
    if resume_from_checkpoint and checkpoint_path.exists() and partial_output_path.exists():
        try:
            checkpoint = read_json(checkpoint_path)
            partial_rows = read_jsonl(partial_output_path)
            processed_rows = partial_rows
            processed_ids = {str(row.get("id") or "") for row in partial_rows if str(row.get("id") or "")}
            if int(checkpoint.get("completed_samples") or 0) == len(processed_rows):
                resumed = True
                logger.event(
                    "checkpoint",
                    "resume_loaded",
                    checkpoint_path=str(checkpoint_path),
                    completed_samples=len(processed_rows),
                )
        except Exception as exc:
            processed_rows = []
            processed_ids = set()
            logger.event(
                "checkpoint",
                "resume_ignored",
                level="WARNING",
                error=str(exc),
            )

    if not resumed:
        logger.event("checkpoint", "resume_fresh_start")

    total_samples = len(order)

    def save_checkpoint(reason: str) -> None:
        eta_iso = compute_eta_iso(start_ts, len(processed_rows), total_samples)
        write_jsonl(partial_output_path, processed_rows)
        write_json(
            checkpoint_path,
            {
                "updated_at": utc_now(),
                "reason": reason,
                "policy": policy,
                "completed_samples": len(processed_rows),
                "total_samples": total_samples,
                "latest_eta_iso": eta_iso,
            },
        )
        logger.event(
            "checkpoint",
            "checkpoint_written",
            reason=reason,
            completed_samples=len(processed_rows),
            total_samples=total_samples,
            latest_eta_iso=eta_iso,
        )

    next_checkpoint = max(checkpoint_interval, ((len(processed_rows) // checkpoint_interval) + 1) * checkpoint_interval)
    for idx, sample_id in enumerate(order, start=1):
        if sample_id in processed_ids:
            continue
        sample_rows = grouped[sample_id]
        reference_row = reference_index.get(sample_id)
        if reference_row is None:
            missing_reference += 1
            continue

        language = str(reference_row.get("language") or sample_rows[0].get("language") or "en")
        split = str(reference_row.get("split") or sample_rows[0].get("split") or "")
        source = str(reference_row.get("source") or sample_rows[0].get("source") or "")

        predictions: List[Dict[str, Any]] = []
        for sample_row in sample_rows:
            predictions.extend(choose_group_policy_predictions(sample_row, policy=policy, language=language))

        processed_rows.append(
            {
                "id": sample_id,
                "original_id": reference_row.get("original_id"),
                "source": source,
                "language": language,
                "split": split,
                "text": reference_row.get("text"),
                "meta_info": reference_row.get("meta_info") or {},
                "predictions": dedupe_predictions(predictions, language=language),
                "ground_truth": reference_row.get("ground_truth") or [],
            }
        )
        processed_ids.add(sample_id)

        completed = len(processed_rows)
        if completed == 1 or completed % progress_interval == 0:
            logger.event(
                "export",
                "progress",
                sample_index=completed,
                total_samples=total_samples,
                sample_id=sample_id,
                eta_iso=compute_eta_iso(start_ts, completed, total_samples),
            )
        if completed >= next_checkpoint:
            save_checkpoint(reason=f"interval_{completed}")
            next_checkpoint = ((completed // checkpoint_interval) + 1) * checkpoint_interval

    save_checkpoint(reason="completed")
    summary = {
        "input_group_row_count": len(group_rows),
        "grouped_sample_count": len(grouped),
        "phase0_like_row_count": len(processed_rows),
        "missing_reference_samples": missing_reference,
    }
    return processed_rows, summary


def write_views(run_root: Path, rows: Sequence[Dict[str, Any]]) -> None:
    write_jsonl(run_root / "candidates" / "phase0_like.jsonl", rows)
    by_split = bucket_rows_by_split(rows)
    for split, split_rows in by_split.items():
        write_jsonl(run_root / "candidates" / "phase0_like_by_split" / f"{split}.jsonl", split_rows)
    by_lang_split = bucket_rows_by_language_split(rows)
    for key, scoped_rows in by_lang_split.items():
        write_jsonl(run_root / "candidates" / "phase0_like_by_lang_split" / f"{key}.jsonl", scoped_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export policy-based phase0_like rows from Phase2 group predictions.")
    parser.add_argument("--input-jsonl", default="DAR/outputs/phase2/latest/candidates/group_predictions.jsonl")
    parser.add_argument("--output-root", default="DAR/outputs/phase2_policy")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--policy", required=True, choices=list(ROLE_CHOICES))
    parser.add_argument("--reference-jsonls", default=None, help="Comma-separated cleaned phase0 JSONLs.")
    parser.add_argument("--languages", default=None, help="Comma-separated languages, e.g. zh,en")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,test")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit by grouped sample count after filtering.")
    parser.add_argument("--progress-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=250)
    parser.add_argument("--no-resume-from-checkpoint", action="store_true")
    parser.add_argument("--write-latest", action="store_true")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    input_path = (workspace_root / args.input_jsonl).resolve() if not Path(args.input_jsonl).is_absolute() else Path(args.input_jsonl)
    output_root = (workspace_root / args.output_root).resolve() if not Path(args.output_root).is_absolute() else Path(args.output_root)
    run_id = args.run_id or f"phase2_policy_{args.policy}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_id
    ensure_dir(run_root)

    logger = EventLogger(run_id=run_id, log_path=run_root / "logs" / "events.jsonl", level=args.log_level)
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input JSONL not found: {input_path}")

        reference_paths = resolve_reference_paths(workspace_root, args.reference_jsonls)
        reference_index = load_reference_index(reference_paths)
        allowed_languages = parse_csv_filter(args.languages)
        allowed_splits = parse_csv_filter(args.splits)

        group_rows = read_jsonl(input_path)
        filtered_rows = []
        seen_samples: set[str] = set()
        for row in group_rows:
            language = str(row.get("language") or "en")
            split = str(row.get("split") or "train")
            sample_id = normalize_string(row.get("sample_id"))
            if allowed_languages and language not in allowed_languages:
                continue
            if allowed_splits and split not in allowed_splits:
                continue
            if args.max_samples and sample_id and sample_id not in seen_samples and len(seen_samples) >= args.max_samples:
                continue
            if sample_id:
                seen_samples.add(sample_id)
            filtered_rows.append(row)

        logger.event(
            "bootstrap",
            "run_started",
            input_jsonl=str(input_path),
            policy=args.policy,
            filtered_group_rows=len(filtered_rows),
            filtered_sample_count=len(seen_samples),
            allowed_languages=sorted(allowed_languages) if allowed_languages else [],
            allowed_splits=sorted(allowed_splits) if allowed_splits else [],
            max_samples=args.max_samples,
            cpu_only=True,
            gpu_oom_risk=False,
        )

        rows, summary = build_phase0_like_rows(
            group_rows=filtered_rows,
            reference_index=reference_index,
            policy=args.policy,
            logger=logger,
            checkpoint_path=run_root / "reports" / "checkpoint_progress.json",
            partial_output_path=run_root / "candidates" / "phase0_like.partial.jsonl",
            progress_interval=max(1, args.progress_interval),
            checkpoint_interval=max(1, args.checkpoint_interval),
            resume_from_checkpoint=not args.no_resume_from_checkpoint,
        )

        write_views(run_root, rows)
        if args.write_latest:
            latest_root = output_root / "latest"
            write_views(latest_root, rows)
            write_json(latest_root / "reports" / "summary.json", summary)

        summary_payload = {
            "run_id": run_id,
            "policy": args.policy,
            "input_jsonl": str(input_path),
            "reference_jsonls": [str(path) for path in reference_paths],
            "filters": {
                "languages": sorted(allowed_languages) if allowed_languages else [],
                "splits": sorted(allowed_splits) if allowed_splits else [],
                "max_samples": args.max_samples,
            },
            "summary": summary,
            "cpu_only": True,
            "gpu_oom_risk": False,
            "completed_at": utc_now(),
        }
        write_json(run_root / "reports" / "summary.json", summary_payload)
        write_json(run_root / "run_summary.json", summary_payload)
        logger.event("export", "completed", **summary_payload["summary"])
    finally:
        logger.close()


if __name__ == "__main__":
    main()
