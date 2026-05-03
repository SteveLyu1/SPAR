from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter archived phase1 outputs down to the configured minimal train/test subsets."
    )
    parser.add_argument(
        "--source-phase1-root",
        required=True,
        help="Directory containing full phase1_{zh,en}_{train,test}.jsonl files.",
    )
    parser.add_argument(
        "--target-phase1-root",
        required=True,
        help="Directory to write filtered phase1_{zh,en}_{train,test}.jsonl files.",
    )
    parser.add_argument(
        "--mintrain-subset-root",
        required=True,
        help="Directory containing cleaned_{zh,en}_train.jsonl or *_train.sample_ids.txt.",
    )
    parser.add_argument(
        "--mintest-subset-root",
        required=True,
        help="Directory containing cleaned_{zh,en}_test.jsonl or *_test.sample_ids.txt.",
    )
    return parser.parse_args()


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


def row_id(row: Dict[str, Any]) -> str:
    group_id = str(row.get("group_id") or "")
    if "_g" in group_id:
        return group_id.rsplit("_g", 1)[0]
    sample_id = row.get("sample_id")
    if sample_id is not None:
        return str(sample_id)
    sample_id = row.get("id")
    return str(sample_id or "")


def load_id_set(subset_root: Path, language: str, split: str) -> Set[str]:
    sample_ids_path = subset_root / f"{language}_{split}.sample_ids.txt"
    if sample_ids_path.exists():
        return {
            line.strip()
            for line in sample_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    jsonl_path = subset_root / f"cleaned_{language}_{split}.jsonl"
    return {str(row.get("id") or row.get("sample_id") or "") for row in read_jsonl(jsonl_path) if row}


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_phase1_root).resolve()
    target_root = Path(args.target_phase1_root).resolve()
    mintrain_root = Path(args.mintrain_subset_root).resolve()
    mintest_root = Path(args.mintest_subset_root).resolve()

    manifest: Dict[str, Any] = {
        "generated_at": utc_now(),
        "source_phase1_root": str(source_root),
        "target_phase1_root": str(target_root),
        "mintrain_subset_root": str(mintrain_root),
        "mintest_subset_root": str(mintest_root),
        "files": {},
    }

    for language, split in (("zh", "train"), ("zh", "test"), ("en", "train"), ("en", "test")):
        subset_root = mintrain_root if split == "train" else mintest_root
        keep_ids = load_id_set(subset_root, language=language, split=split)
        source_path = source_root / f"phase1_{language}_{split}.jsonl"
        target_path = target_root / f"phase1_{language}_{split}.jsonl"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing phase1 source file: {source_path}")
        source_rows = read_jsonl(source_path)
        filtered_rows = [row for row in source_rows if row_id(row) in keep_ids]
        write_jsonl(target_path, filtered_rows)
        manifest["files"][f"{language}_{split}"] = {
            "source_path": str(source_path),
            "target_path": str(target_path),
            "keep_sample_ids": len(keep_ids),
            "source_rows": len(source_rows),
            "filtered_rows": len(filtered_rows),
            "filtered_unique_sample_ids": len({row_id(row) for row in filtered_rows}),
        }

    write_json(target_root / "subset_manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
