from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List


KEEP_TOP_LEVEL = {"main", "qwen3.5-9B", "qwen3.5-35B"}
PHASE35_MAINLINE_RUN = "phase35_fullrun_from_phase3_qfix4_14b_20260428"
PHASE3_MAINLINE_RUN = "phase3_full_nothink_qfix4_14b_20260428"


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


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reorganize DAR outputs into main/qwen3.5-9B/qwen3.5-35B.")
    parser.add_argument("--outputs-root", default=None, help="Override DAR/outputs root")
    parser.add_argument("--execute", action="store_true", help="Apply filesystem moves instead of dry-run only")
    return parser.parse_args()


def resolve_outputs_root(raw: str | None) -> Path:
    if raw:
        return Path(raw).resolve()
    return Path(__file__).resolve().parents[1] / "outputs"


def move_legacy_entries(outputs_root: Path, main_root: Path, execute: bool) -> List[Dict[str, str]]:
    planned: List[Dict[str, str]] = []
    for entry in sorted(outputs_root.iterdir(), key=lambda item: item.name):
        if entry.name in KEEP_TOP_LEVEL:
            continue
        target = main_root / entry.name
        planned.append({"source": str(entry), "target": str(target)})
        if not execute:
            continue
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing target: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(entry), str(target))
    return planned


def reconstruct_phase3_from_phase35(main_root: Path) -> Dict[str, Any]:
    phase3_root = main_root / "phase3"
    reconstructed_root = phase3_root / PHASE3_MAINLINE_RUN
    final_tuples_path = reconstructed_root / "candidates" / "final_tuples.jsonl"
    latest_final_tuples_path = phase3_root / "latest" / "candidates" / "final_tuples.jsonl"
    if final_tuples_path.exists():
        return {
            "status": "existing",
            "run_root": str(reconstructed_root),
            "final_tuples": str(final_tuples_path),
        }

    aggregated_path = main_root / "phase3.5" / PHASE35_MAINLINE_RUN / "candidates" / "aggregated_samples.jsonl"
    if not aggregated_path.exists():
        return {
            "status": "missing_source",
            "run_root": str(reconstructed_root),
            "source": str(aggregated_path),
        }

    rows = read_jsonl(aggregated_path)
    flattened: List[Dict[str, Any]] = []
    for sample_row in rows:
        sample_id = str(sample_row.get("sample_id") or "")
        source = str(sample_row.get("source") or "unknown")
        split = str(sample_row.get("split") or "train")
        language = str(sample_row.get("language") or "en")
        for group in sample_row.get("groups", []):
            if not isinstance(group, dict):
                continue
            final_tuple = group.get("final_tuple") if isinstance(group.get("final_tuple"), dict) else {}
            final_tuples = group.get("final_tuples") if isinstance(group.get("final_tuples"), list) else [final_tuple]
            flattened.append(
                {
                    "decision_id": str(group.get("decision_id") or ""),
                    "group_id": str(group.get("group_id") or ""),
                    "sample_id": sample_id,
                    "local_group_id": int(group.get("local_group_id") or 0),
                    "source": source,
                    "split": split,
                    "language": language,
                    "sentence_ids": group.get("sentence_ids") if isinstance(group.get("sentence_ids"), list) else [],
                    "group_text": str(group.get("group_text") or ""),
                    "audit_confidence": group.get("audit_confidence"),
                    "veto_flag": bool(group.get("veto_flag")),
                    "final_tuple": final_tuple,
                    "final_tuples": final_tuples,
                    "reconstructed_from": str(aggregated_path),
                    "reconstruction_method": "flattened_from_phase35_aggregated_samples",
                }
            )

    write_jsonl(final_tuples_path, flattened)
    write_jsonl(latest_final_tuples_path, flattened)
    write_json(
        reconstructed_root / "reconstruction_note.json",
        {
            "status": "reconstructed",
            "reason": "Original phase3 mainline directory was not present under DAR/outputs during reorganization.",
            "source": str(aggregated_path),
            "method": "Flattened group-level finals from phase3.5 aggregated_samples.jsonl.",
            "unavailable_original_files": [
                "candidates/group_decisions.jsonl",
                "candidates/rectified_roles.jsonl",
                "candidates/phase4_trace_pack.jsonl",
            ],
            "row_count": len(flattened),
        },
    )
    write_json(
        reconstructed_root / "reports" / "summary.json",
        {
            "reconstructed": True,
            "group_row_count": len(flattened),
            "source": str(aggregated_path),
        },
    )
    return {
        "status": "reconstructed",
        "run_root": str(reconstructed_root),
        "final_tuples": str(final_tuples_path),
        "row_count": len(flattened),
    }


def build_manifest(main_root: Path, phase3_status: Dict[str, Any], moved_entries: List[Dict[str, str]]) -> Dict[str, Any]:
    return {
        "outputs_root": str(main_root.parent),
        "main_root": str(main_root),
        "root_layout_target": ["main", "qwen3.5-9B", "qwen3.5-35B"],
        "moved_legacy_entries": moved_entries,
        "best_results": {
            "phase1_mainline_input": str(main_root / "phase1" / "phase1_focus"),
            "phase2_mainline_pipeline": str(main_root / "phase2" / "phase2_full_nothink_qfix2_20260427"),
            "phase2_eval_best_zh": str(main_root / "phase2_policy" / "phase2_moderator_anchor_full_20260429" / "candidates" / "phase0_like_by_lang_split" / "zh_test.jsonl"),
            "phase2_eval_best_en": str(main_root / "phase2_policy" / "phase2_victim_only_full_20260429" / "candidates" / "phase0_like_by_lang_split" / "en_test.jsonl"),
            "phase3_mainline": phase3_status,
            "phase35_mainline": str(main_root / "phase3.5" / PHASE35_MAINLINE_RUN),
            "phase4_official_eval_zh": str(main_root / "phase4_teacher_distill" / "phase4_teacher_distill_qfix6_prepare" / "training" / "full_eval" / "official_eval_zh" / "exports" / "phase0_like.jsonl"),
            "phase4_official_eval_en": str(main_root / "phase4_teacher_distill" / "phase4_teacher_distill_qfix6_prepare" / "training" / "full_eval" / "official_eval_en" / "exports" / "phase0_like.jsonl"),
        },
        "notes": [
            "All former DAR/outputs top-level phase/history directories are now nested under DAR/outputs/main.",
            "Comparison experiments should write only to DAR/outputs/qwen3.5-9B and DAR/outputs/qwen3.5-35B.",
        ],
    }


def main() -> None:
    args = parse_args()
    outputs_root = resolve_outputs_root(args.outputs_root)
    main_root = outputs_root / "main"
    if args.execute:
        main_root.mkdir(parents=True, exist_ok=True)

    moved_entries = move_legacy_entries(outputs_root=outputs_root, main_root=main_root, execute=args.execute)
    phase3_status = (
        reconstruct_phase3_from_phase35(main_root=main_root)
        if args.execute
        else {
            "status": "dry_run_pending",
            "run_root": str(main_root / "phase3" / PHASE3_MAINLINE_RUN),
            "source": str(main_root / "phase3.5" / PHASE35_MAINLINE_RUN / "candidates" / "aggregated_samples.jsonl"),
        }
    )
    manifest = build_manifest(main_root=main_root, phase3_status=phase3_status, moved_entries=moved_entries)
    if args.execute:
        write_json(main_root / "mainline_manifest.json", manifest)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
