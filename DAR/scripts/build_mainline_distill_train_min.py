from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


DEFAULT_SUBSET_NAME = "mainline_distill_train_min_v1"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_lines(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{line}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the minimal train subset actually used by the mainline phase4 distillation dataset."
    )
    parser.add_argument("--subset-name", default=DEFAULT_SUBSET_NAME)
    parser.add_argument(
        "--phase4-dataset-root",
        default="DAR/outputs/main/phase4_teacher_distill/latest/datasets/by_lang_split",
        help="Directory containing phase4 by_lang_split train JSONL files.",
    )
    parser.add_argument(
        "--phase0-root",
        default="DAR/outputs/main/phase0",
        help="Directory containing cleaned_{zh,en}_train.jsonl.",
    )
    parser.add_argument(
        "--output-root",
        default="DAR/outputs/main/phase0/subsets",
        help="Directory under which the named subset folder will be created.",
    )
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return repo_root() / raw


def row_id(row: Dict[str, Any]) -> str:
    value = row.get("sample_id")
    if value is None:
        value = row.get("id")
    return str(value or "")


def collect_unique_phase4_train_ids(path: Path) -> List[str]:
    seen: Set[str] = set()
    ordered_ids: List[str] = []
    for row in read_jsonl(path):
        sample_id = row_id(row)
        if sample_id and sample_id not in seen:
            ordered_ids.append(sample_id)
            seen.add(sample_id)
    return ordered_ids


def filter_phase0_rows(path: Path, keep_ids: Set[str]) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    return [row for row in rows if row_id(row) in keep_ids]


def main() -> None:
    args = parse_args()
    phase4_root = resolve_path(args.phase4_dataset_root)
    phase0_root = resolve_path(args.phase0_root)
    subsets_root = resolve_path(args.output_root)
    subset_root = subsets_root / args.subset_name

    manifest: Dict[str, Any] = {
        "subset_name": args.subset_name,
        "generated_at": utc_now(),
        "phase4_dataset_root": str(phase4_root),
        "phase0_root": str(phase0_root),
        "languages": {},
    }

    for language in ("zh", "en"):
        phase4_train_path = phase4_root / f"{language}_train.jsonl"
        phase0_train_path = phase0_root / f"cleaned_{language}_train.jsonl"
        if not phase4_train_path.exists():
            raise FileNotFoundError(f"Missing phase4 train source: {phase4_train_path}")
        if not phase0_train_path.exists():
            raise FileNotFoundError(f"Missing phase0 train source: {phase0_train_path}")

        phase4_rows = read_jsonl(phase4_train_path)
        unique_ids = collect_unique_phase4_train_ids(phase4_train_path)
        unique_id_set = set(unique_ids)
        filtered_rows = filter_phase0_rows(phase0_train_path, keep_ids=unique_id_set)
        filtered_ids = [row_id(row) for row in filtered_rows]
        missing_ids = [sample_id for sample_id in unique_ids if sample_id not in set(filtered_ids)]
        if missing_ids:
            raise RuntimeError(
                f"{language} minimal-train build found ids present in phase4 but missing in phase0: {missing_ids[:10]}"
            )

        write_jsonl(subset_root / f"cleaned_{language}_train.jsonl", filtered_rows)
        write_lines(subset_root / f"{language}_train.sample_ids.txt", unique_ids)

        manifest["languages"][language] = {
            "phase4_train_path": str(phase4_train_path),
            "phase0_train_path": str(phase0_train_path),
            "subset_train_path": str(subset_root / f"cleaned_{language}_train.jsonl"),
            "sample_ids_path": str(subset_root / f"{language}_train.sample_ids.txt"),
            "phase4_group_rows": len(phase4_rows),
            "unique_sample_ids": len(unique_ids),
            "subset_rows": len(filtered_rows),
        }

    write_json(subset_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
