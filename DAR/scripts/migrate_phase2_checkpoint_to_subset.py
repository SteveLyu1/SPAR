from __future__ import annotations

import argparse
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter a saved phase2 checkpoint to the minimal train/test subset while preserving recoverable progress."
    )
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--target-checkpoint", required=True)
    parser.add_argument("--mintrain-subset-root", required=True)
    parser.add_argument("--mintest-subset-root", required=True)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, Any]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return obj


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_id_set(subset_root: Path, language: str, split: str) -> Set[str]:
    sample_ids_path = subset_root / f"{language}_{split}.sample_ids.txt"
    if sample_ids_path.exists():
        return {
            line.strip()
            for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    jsonl_path = subset_root / f"cleaned_{language}_{split}.jsonl"
    rows = read_jsonl(jsonl_path)
    return {str(row.get("id") or row.get("sample_id") or "") for row in rows if row}


def build_group_id_signature(rows: List[Dict[str, Any]]) -> str:
    joined = "||".join(str(row.get("group_id") or "") for row in rows)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def row_sample_id(row: Dict[str, Any]) -> str:
    value = row.get("sample_id")
    if value is None:
        group_id = str(row.get("group_id") or "")
        if "_g" in group_id:
            return group_id.rsplit("_g", 1)[0]
        value = row.get("id")
    return str(value or "")


def keep_row(row: Dict[str, Any], keep_map: Dict[Tuple[str, str], Set[str]]) -> bool:
    language = str(row.get("language") or "")
    split = str(row.get("split") or "")
    sample_id = row_sample_id(row)
    return sample_id in keep_map.get((language, split), set())


def main() -> None:
    args = parse_args()
    source_checkpoint = Path(args.source_checkpoint).resolve()
    target_checkpoint = Path(args.target_checkpoint).resolve()
    mintrain_root = Path(args.mintrain_subset_root).resolve()
    mintest_root = Path(args.mintest_subset_root).resolve()

    checkpoint = read_json(source_checkpoint)
    source_rows = checkpoint.get("group_rows")
    if not isinstance(source_rows, list):
        raise RuntimeError(f"Checkpoint does not contain group_rows: {source_checkpoint}")

    keep_map: Dict[Tuple[str, str], Set[str]] = {
        ("zh", "train"): load_id_set(mintrain_root, "zh", "train"),
        ("en", "train"): load_id_set(mintrain_root, "en", "train"),
        ("zh", "test"): load_id_set(mintest_root, "zh", "test"),
        ("en", "test"): load_id_set(mintest_root, "en", "test"),
    }

    kept_rows: List[Dict[str, Any]] = []
    kept_indices: List[int] = []
    for idx, raw_row in enumerate(source_rows):
        if isinstance(raw_row, dict) and keep_row(raw_row, keep_map=keep_map):
            kept_rows.append(raw_row)
            kept_indices.append(idx)

    if not kept_rows:
        raise RuntimeError("Filtered checkpoint would be empty; aborting.")

    processed_by_role = checkpoint.get("processed_by_role")
    if not isinstance(processed_by_role, dict):
        processed_by_role = {}
    migrated_processed: Dict[str, int] = {}
    for role_name, raw_count in processed_by_role.items():
        try:
            old_count = max(0, int(raw_count))
        except (TypeError, ValueError):
            old_count = 0
        migrated_processed[str(role_name)] = sum(1 for idx in kept_indices if idx < old_count)

    checkpoint["group_rows"] = kept_rows
    checkpoint["group_id_signature"] = build_group_id_signature(kept_rows)
    checkpoint["processed_by_role"] = migrated_processed
    checkpoint["total_groups"] = len(kept_rows)
    role_order = checkpoint.get("role_order")
    role_count = len(role_order) if isinstance(role_order, list) and role_order else 3
    checkpoint["total_role_groups"] = len(kept_rows) * role_count
    checkpoint["updated_at"] = utc_now()
    checkpoint["completed"] = False
    checkpoint["reason"] = "migrated_to_min_subset"
    checkpoint["migration_meta"] = {
        "generated_at": utc_now(),
        "source_checkpoint": str(source_checkpoint),
        "mintrain_subset_root": str(mintrain_root),
        "mintest_subset_root": str(mintest_root),
        "source_total_groups": len(source_rows),
        "target_total_groups": len(kept_rows),
        "kept_prefix_counts_by_role": migrated_processed,
    }

    write_json(target_checkpoint, checkpoint)
    print(
        json.dumps(
            {
                "source_checkpoint": str(source_checkpoint),
                "target_checkpoint": str(target_checkpoint),
                "source_total_groups": len(source_rows),
                "target_total_groups": len(kept_rows),
                "processed_by_role": migrated_processed,
                "signature": checkpoint["group_id_signature"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
