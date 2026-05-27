from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove redundant sample_id and meta_info.platform fields from Phase0 JSONL outputs"
    )
    default_root = Path(__file__).resolve().parents[2] / "SPAR" / "outputs" / "phase0"
    parser.add_argument("--output-root", default=str(default_root), help=f"Phase0 output directory. Default: {default_root}")
    return parser.parse_args()


def sanitize_row(row: Dict[str, Any]) -> Tuple[Dict[str, Any], int, int]:
    removed_sample_id = 0
    removed_platform = 0

    if "sample_id" in row:
        row.pop("sample_id", None)
        removed_sample_id = 1

    meta_info = row.get("meta_info")
    if isinstance(meta_info, dict) and "platform" in meta_info:
        meta_info.pop("platform", None)
        removed_platform = 1

    return row, removed_sample_id, removed_platform


def sanitize_jsonl(path: Path) -> Dict[str, int]:
    temp_path = path.with_suffix(path.suffix + ".tmp")
    line_count = 0
    removed_sample_id = 0
    removed_platform = 0

    with path.open("r", encoding="utf-8") as src, temp_path.open("w", encoding="utf-8") as dst:
        for raw in src:
            if not raw.strip():
                continue
            line_count += 1
            row = json.loads(raw)
            if not isinstance(row, dict):
                raise ValueError(f"Expected JSON object per line in {path}, got {type(row)}")
            row, rm_sid, rm_platform = sanitize_row(row)
            removed_sample_id += rm_sid
            removed_platform += rm_platform
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    temp_path.replace(path)
    return {
        "lines": line_count,
        "removed_sample_id": removed_sample_id,
        "removed_platform": removed_platform,
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if not output_root.exists() or not output_root.is_dir():
        raise FileNotFoundError(f"Output directory not found: {output_root}")

    summary: Dict[str, Any] = {
        "output_root": str(output_root),
        "files": {},
        "totals": {
            "files": 0,
            "lines": 0,
            "removed_sample_id": 0,
            "removed_platform": 0,
        },
    }

    for path in sorted(output_root.glob("*.jsonl")):
        stats = sanitize_jsonl(path)
        summary["files"][path.name] = stats
        summary["totals"]["files"] += 1
        summary["totals"]["lines"] += stats["lines"]
        summary["totals"]["removed_sample_id"] += stats["removed_sample_id"]
        summary["totals"]["removed_platform"] += stats["removed_platform"]

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
