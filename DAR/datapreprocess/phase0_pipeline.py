from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


TUPLE_KEY_RE = re.compile(r"^(T|Q)(\d+)\s+(.+)$")


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    output_root: Path
    state_train: Path
    state_test: Path
    tbo: Path


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_json_array(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}, got {type(data)}")
    return data


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


def normalize_binary_label(raw: Any, source: str) -> int:
    value = normalize_string(raw)
    if value is None:
        return 0
    token = value.lower()

    if source == "STATE-ToxiCN":
        if token == "hate":
            return 1
        if token == "non-hate":
            return 0
    if source == "TBO":
        if token == "yes":
            return 1
        if token == "no":
            return 0

    if token in {"1", "true"}:
        return 1
    if token in {"0", "false"}:
        return 0
    return 0


def normalize_group(raw: Any, typo_map: Dict[str, str], taxonomy: Sequence[str]) -> Optional[str]:
    value = normalize_string(raw)
    if value is None:
        return None
    mapped = typo_map.get(value, value)
    if mapped in taxonomy:
        return mapped
    lowered_map = {k.lower(): k for k in taxonomy}
    return lowered_map.get(mapped.lower(), mapped)


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
    parser = argparse.ArgumentParser(description="Phase0 schema normalization to cleaned zh/en JSONL files")
    default_config = str((Path(__file__).resolve().parents[1] / "configs" / "phase0_template.json"))
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    parser.add_argument("--tbo-train-size", type=int, default=None, help="Override TBO train doc size")
    parser.add_argument("--tbo-test-size", type=int, default=None, help="Override TBO test doc size")
    parser.add_argument("--max-state-train-docs", type=int, default=None, help="STATE train cap for smoke")
    parser.add_argument("--max-state-test-docs", type=int, default=None, help="STATE test cap for smoke")
    parser.add_argument("--max-tbo-docs", type=int, default=None, help="TBO cap for smoke")
    parser.add_argument("--dry-run", action="store_true", help="Run logic without writing files")
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")
    if suffix == ".json":
        return json.loads(text)
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    override_map = [
        (args.seed, ["seed"]),
        (args.tbo_train_size, ["split", "tbo_train_size"]),
        (args.tbo_test_size, ["split", "tbo_test_size"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)

    if args.max_state_train_docs is not None:
        set_nested(cfg, ["runtime", "max_state_train_docs"], args.max_state_train_docs)
    if args.max_state_test_docs is not None:
        set_nested(cfg, ["runtime", "max_state_test_docs"], args.max_state_test_docs)
    if args.max_tbo_docs is not None:
        set_nested(cfg, ["runtime", "max_tbo_docs"], args.max_tbo_docs)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg["paths"]

    output_root_raw = args.output_root if args.output_root else cfg_paths.get("output_root", "DAR/outputs/phase0")
    output_root = resolve_path(workspace_root, config_dir, output_root_raw)

    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        output_root=output_root,
        state_train=resolve_path(workspace_root, config_dir, cfg_paths["state_train"]),
        state_test=resolve_path(workspace_root, config_dir, cfg_paths["state_test"]),
        tbo=resolve_path(workspace_root, config_dir, cfg_paths["tbo"]),
    )


def collect_tuple_indices(doc: Dict[str, Any], prefix: str) -> List[int]:
    found = set()
    for key in doc.keys():
        m = TUPLE_KEY_RE.match(key)
        if not m:
            continue
        if m.group(1) != prefix:
            continue
        found.add(int(m.group(2)))
    return sorted(found)


def build_state_ground_truth(
    doc: Dict[str, Any],
    group_taxonomy: Sequence[str],
    group_typo_map: Dict[str, str],
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in collect_tuple_indices(doc, "Q"):
        hateful_raw = normalize_string(doc.get(f"Q{idx} hateful"))
        if hateful_raw is None:
            hateful_raw = normalize_string(doc.get(f"Q{idx} Harmful"))
        out.append(
            {
                "target": normalize_string(doc.get(f"Q{idx} Target")),
                "argument": normalize_string(doc.get(f"Q{idx} Argument")),
                "group": normalize_group(doc.get(f"Q{idx} Group"), typo_map=group_typo_map, taxonomy=group_taxonomy),
                "hateful": normalize_binary_label(hateful_raw, source="STATE-ToxiCN"),
            }
        )
    return out


def build_tbo_ground_truth(doc: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx in collect_tuple_indices(doc, "T"):
        out.append(
            {
                "target": normalize_string(doc.get(f"T{idx} Target")),
                "argument": normalize_string(doc.get(f"T{idx} Argument")),
                "harmful": normalize_binary_label(doc.get(f"T{idx} Harmful"), source="TBO"),
            }
        )
    return out


def build_state_record(
    doc: Dict[str, Any],
    *,
    split: str,
    group_taxonomy: Sequence[str],
    group_typo_map: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    text = normalize_string(doc.get("content"))
    raw_id = doc.get("id")
    if text is None or raw_id is None:
        return None

    original_id = str(raw_id)
    sid = f"zh_{original_id}"
    return {
        "id": sid,
        "original_id": original_id,
        "source": "STATE-ToxiCN",
        "language": "zh",
        "split": split,
        "text": text,
        "meta_info": {
            "topic": doc.get("topic"),
        },
        "ground_truth": build_state_ground_truth(
            doc,
            group_taxonomy=group_taxonomy,
            group_typo_map=group_typo_map,
        ),
    }


def build_tbo_record(doc: Dict[str, Any], *, split: str, record_id: str) -> Optional[Dict[str, Any]]:
    text = normalize_string(doc.get("text"))
    raw_id = doc.get("id")
    if text is None or raw_id is None:
        return None

    return {
        "id": record_id,
        "original_id": str(raw_id),
        "source": "TBO",
        "language": "en",
        "split": split,
        "text": text,
        "meta_info": {},
        "ground_truth": build_tbo_ground_truth(doc),
    }


def split_tbo_docs(docs: Sequence[Dict[str, Any]], train_size: int, test_size: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    all_docs = list(docs)
    total_need = train_size + test_size
    if len(all_docs) < total_need:
        raise ValueError(f"TBO size too small: have {len(all_docs)}, need {total_need}")

    rng = random.Random(seed)
    idx = list(range(len(all_docs)))
    rng.shuffle(idx)
    shuffled = [all_docs[i] for i in idx]
    return shuffled[:train_size], shuffled[train_size : train_size + test_size]


def quality_gates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    seen = set()
    duplicate_id = 0
    missing_text = 0
    invalid_split = 0
    invalid_ground_truth = 0
    tbo_group_violation = 0

    for row in rows:
        row_id = row.get("id")
        if row_id in seen:
            duplicate_id += 1
        seen.add(row_id)

        if normalize_string(row.get("text")) is None:
            missing_text += 1
        if row.get("split") not in {"train", "test"}:
            invalid_split += 1

        gt = row.get("ground_truth")
        if not isinstance(gt, list) or not gt:
            invalid_ground_truth += 1
            continue

        language = row.get("language")
        for item in gt:
            if not isinstance(item, dict):
                invalid_ground_truth += 1
                break
            if language == "zh":
                need = {"target", "argument", "group", "hateful"}
                if any(k not in item for k in need):
                    invalid_ground_truth += 1
                    break
                if item.get("hateful") not in {0, 1}:
                    invalid_ground_truth += 1
                    break
            if language == "en":
                need = {"target", "argument", "harmful"}
                if any(k not in item for k in need):
                    invalid_ground_truth += 1
                    break
                if "group" in item:
                    tbo_group_violation += 1
                    break
                if item.get("harmful") not in {0, 1}:
                    invalid_ground_truth += 1
                    break

    passed = (
        duplicate_id == 0
        and missing_text == 0
        and invalid_split == 0
        and invalid_ground_truth == 0
        and tbo_group_violation == 0
    )

    return {
        "total_rows": len(rows),
        "duplicate_id": duplicate_id,
        "missing_text": missing_text,
        "invalid_split": invalid_split,
        "invalid_ground_truth": invalid_ground_truth,
        "tbo_group_violation": tbo_group_violation,
        "passed": passed,
    }


def export_cleaned(output_root: Path, zh_train: Sequence[Dict[str, Any]], zh_test: Sequence[Dict[str, Any]], en_train: Sequence[Dict[str, Any]], en_test: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    write_jsonl(output_root / "cleaned_zh_train.jsonl", zh_train)
    write_jsonl(output_root / "cleaned_zh_test.jsonl", zh_test)
    write_jsonl(output_root / "cleaned_en_train.jsonl", en_train)
    write_jsonl(output_root / "cleaned_en_test.jsonl", en_test)

    return {
        "cleaned_zh_train": len(zh_train),
        "cleaned_zh_test": len(zh_test),
        "cleaned_en_train": len(en_train),
        "cleaned_en_test": len(en_test),
    }


def main() -> None:
    args = parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, args)
    paths = build_paths(args, cfg, workspace_root=workspace_root)

    seed = int(cfg.get("seed", 20260415))
    split_cfg = cfg.get("split", {})
    runtime_cfg = cfg.get("runtime", {})
    schema_cfg = cfg.get("schema", {})

    tbo_train_size = int(split_cfg.get("tbo_train_size", 3200))
    tbo_test_size = int(split_cfg.get("tbo_test_size", 800))

    state_train_docs = read_json_array(paths.state_train)
    state_test_docs = read_json_array(paths.state_test)
    tbo_docs = read_json_array(paths.tbo)

    if runtime_cfg.get("max_state_train_docs"):
        state_train_docs = state_train_docs[: int(runtime_cfg["max_state_train_docs"])]
    if runtime_cfg.get("max_state_test_docs"):
        state_test_docs = state_test_docs[: int(runtime_cfg["max_state_test_docs"])]
    if runtime_cfg.get("max_tbo_docs"):
        tbo_docs = tbo_docs[: int(runtime_cfg["max_tbo_docs"])]

    group_taxonomy = list(schema_cfg.get("group_taxonomy", []))
    group_typo_map = dict(schema_cfg.get("group_typo_map", {}))

    zh_train: List[Dict[str, Any]] = []
    for doc in state_train_docs:
        row = build_state_record(
            doc,
            split="train",
            group_taxonomy=group_taxonomy,
            group_typo_map=group_typo_map,
        )
        if row is not None:
            zh_train.append(row)

    zh_test: List[Dict[str, Any]] = []
    for doc in state_test_docs:
        row = build_state_record(
            doc,
            split="test",
            group_taxonomy=group_taxonomy,
            group_typo_map=group_typo_map,
        )
        if row is not None:
            zh_test.append(row)

    tbo_train_docs, tbo_test_docs = split_tbo_docs(
        tbo_docs,
        train_size=tbo_train_size,
        test_size=tbo_test_size,
        seed=seed,
    )

    en_train: List[Dict[str, Any]] = []
    seq = 1
    for doc in tbo_train_docs:
        row_id = f"en_{seq:04d}"
        seq += 1
        row = build_tbo_record(doc, split="train", record_id=row_id)
        if row is not None:
            en_train.append(row)

    en_test: List[Dict[str, Any]] = []
    for doc in tbo_test_docs:
        row_id = f"en_{seq:04d}"
        seq += 1
        row = build_tbo_record(doc, split="test", record_id=row_id)
        if row is not None:
            en_test.append(row)

    all_rows = zh_train + zh_test + en_train + en_test
    quality = quality_gates(all_rows)
    if not quality["passed"]:
        raise RuntimeError(f"Quality gate failed: {quality}")

    if not args.dry_run:
        counts = export_cleaned(paths.output_root, zh_train, zh_test, en_train, en_test)
        print(json.dumps({"quality": quality, "counts": counts}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps({"quality": quality, "dry_run": True}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
