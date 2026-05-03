from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


ROLE_ORDER = ["victim", "moderator", "cultural_bystander"]
CULTURAL_ERROR_TAG = "cultural_metaphor_sarcasm_misread"
FOCUS_TAG_RE = re.compile(r"</?focus>")


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    output_root: Path
    run_root: Path
    phase0_samples: Optional[Path]
    phase1_candidates: Optional[Path]
    phase2_candidates: Optional[Path]
    phase3_trace_pack: Optional[Path]
    phase3_group_decisions: Optional[Path]
    phase3_rectified_roles: Optional[Path]
    phase3_final_tuples: Optional[Path]
    zh_lexicon_path: Optional[Path]


DEFAULT_PROMPT_PROFILES: Dict[str, Dict[str, Any]] = {
    "zh_group_hateful": {
        "match_languages": ["zh"],
        "match_sources": ["STATE-ToxiCN"],
        "system_prompt": (
            "你是 DAR-Hate 的蒸馏学生模型。任务是针对中文样本做 group-aware 的有害言论解析。"
            "你会看到原文、focus/group 切分结果、三种角色视角、冲突诊断、修正动作，以及可选的中文 lexicon 证据。"
            "必须先依据证据完成五步思考：判断当前 focus/group 是否还需拆分、比较三种角色候选、识别冲突、应用修正、最终仲裁。"
            "在 `<think>` 中只保留当前样本真正起作用的证据，不要逐条复述提示词。"
            "`target` 和 `argument` 必须回贴 source/focus 中可见的最小原文 span，不扩写、不改写、不重复无关词。"
            "其中 `group` 对中文样本是有效语义字段，若证据支持，应输出具体 group 标签；若不构成仇恨，输出 `non-hate`。"
            "最终只输出严格 JSON，字段固定为 `target`, `argument`, `group`, `harmful`, `harmful_direction`，不得新增字段。"
        ),
        "user_instruction": (
            "请依据下面的 phase traces 执行自然语言伪推理链蒸馏：\n"
            "1. 先确认当前样本是否已经完成拆分，并锁定 focus/group 与 canonical target。\n"
            "2. 只保留三角色里会影响最终裁决的字段与短理由。\n"
            "3. 只对 high_risk_fields 做 auditor 仲裁。\n"
            "4. 在最终裁决前先做 span anchoring：target/argument 必须回贴原文最小 span。\n"
            "5. `<think>` 要短，不要复述任务规则；最后输出严格 JSON。"
        ),
        "package_label": "输入证据包",
        "schema_label": "输出 JSON 约束",
        "assistant_chain_style": "danceha_natural",
        "step_titles": [
            "步骤A：分解",
            "步骤B：并行角色假设",
            "步骤C：冲突诊断",
            "步骤D：修正动作",
            "步骤E：最终仲裁",
        ],
    },
    "en_span_harmful": {
        "match_languages": ["en"],
        "match_sources": ["TBO"],
        "system_prompt": (
            "You are the DAR-Hate distillation student for English harmful-span parsing. "
            "You will receive the source text, focus/group decomposition, three role hypotheses, conflict diagnosis, "
            "rectification actions, and optional lexicon evidence. First reason through five steps: decide whether the current focus/group "
            "still needs splitting, compare roles, diagnose conflict, apply rectification, and arbitrate the final answer. "
            "Inside `<think>`, keep only sample-specific evidence and do not restate the task instructions. "
            "`target` and `argument` must be copied back from the smallest visible source spans instead of expanded paraphrases. "
            "For English samples, `group` is usually a placeholder and should default to `unknown` unless the evidence clearly supports otherwise. "
            "Return strict JSON only with keys `target`, `argument`, `group`, `harmful`, `harmful_direction`."
        ),
        "user_instruction": (
            "Use the provided phase traces to perform natural-language pseudo-chain distillation:\n"
            "1. First confirm the current focus/group unit and canonical target.\n"
            "2. Keep only the role fields and short reasons that can change the final answer.\n"
            "3. Use the auditor stage only on high_risk_fields.\n"
            "4. Before the final decision, do span anchoring so target/argument copy the smallest source spans.\n"
            "5. Keep `<think>` short and finish with strict JSON."
        ),
        "package_label": "Input evidence package",
        "schema_label": "Required JSON schema",
        "assistant_chain_style": "danceha_natural",
        "step_titles": [
            "Step A: Decompose",
            "Step B: Parallel Role Hypotheses",
            "Step C: Conflict Diagnosis",
            "Step D: Rectification Actions",
            "Step E: Final Arbitration",
        ],
    },
}

DEFAULT_DATASET_VARIANTS: List[Dict[str, Any]] = [
    {
        "name": "legacy_full_teacher",
        "description": "Legacy full-trace teacher-supervised dataset kept for backward compatibility.",
        "filter_mode": "all",
        "supervision_target": "teacher",
    },
    {
        "name": "stage1_clean",
        "description": "High-precision curriculum stage using profile-aware exact teacher-vs-ground-truth agreement.",
        "filter_mode": "profile_clean",
        "supervision_target": "ground_truth_primary",
    },
    {
        "name": "stage2_silver",
        "description": "Expanded curriculum stage using profile-aware silver agreement with ground truth.",
        "filter_mode": "profile_silver",
        "supervision_target": "ground_truth_primary",
    },
]

PRIMARY_VARIANT_NAME = "stage2_silver"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sha1(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = html.unescape(str(value)).strip()
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


def safe_sentence_ids(value: Any) -> List[int]:
    out: List[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and item >= 0:
                out.append(item)
            elif isinstance(item, str) and item.isdigit():
                out.append(int(item))
    return sorted(set(out))


def normalize_harmful_direction(direction: Any, harmful: Any) -> str:
    text = normalize_string(direction)
    if text is not None:
        lowered = text.lower()
        if lowered in {"harmful", "yes", "true", "1"}:
            return "harmful"
        if lowered in {"non-harmful", "non_harmful", "not_harmful", "no", "false", "0"}:
            return "non-harmful"
        if lowered in {"unknown", "uncertain", "null", "none"}:
            return "unknown"

    if harmful in {1, "1", True, "true", "True"}:
        return "harmful"
    if harmful in {0, "0", False, "false", "False"}:
        return "non-harmful"
    return "unknown"


def direction_to_harmful_binary(direction: str) -> Optional[int]:
    if direction == "harmful":
        return 1
    if direction == "non-harmful":
        return 0
    return None


def normalize_final_tuple(raw_tuple: Any) -> Dict[str, Any]:
    tup = raw_tuple if isinstance(raw_tuple, dict) else {}
    direction = normalize_harmful_direction(tup.get("harmful_direction"), tup.get("harmful"))
    return {
        "target": normalize_string(tup.get("target")) or "IMPLICIT_TARGET",
        "argument": normalize_string(tup.get("argument")),
        "group": normalize_string(tup.get("group")) or "unknown",
        "harmful": direction_to_harmful_binary(direction),
        "harmful_direction": direction,
    }


def dedupe_preserve_order(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen: Set[str] = set()
    for value in values:
        key = json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else str(value)
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out


def strip_focus_tags(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    return FOCUS_TAG_RE.sub("", text)


def normalize_text_for_match(value: Any) -> str:
    text = strip_focus_tags(value)
    if text is None:
        return ""
    return "".join(ch.lower() for ch in text if not ch.isspace())


def parse_json_object(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    text = normalize_string(value)
    if text is None:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def load_jsonl_if_exists(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    return read_jsonl(path)


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


def choose_existing_path(primary: Optional[Path], fallbacks: Sequence[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if primary is not None:
        candidates.append(primary)
    candidates.extend(fallbacks)
    for cand in candidates:
        if cand.exists():
            return cand
    return primary


def sibling(path: Optional[Path], name: str) -> Optional[Path]:
    if path is None:
        return None
    return path.with_name(name)


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    runtime = cfg.get("runtime", {})
    splits = runtime.get("default_splits", ["train", "dev", "test"])
    if not isinstance(splits, list):
        return ["train", "dev", "test"]
    out = [str(x).strip() for x in splits if str(x).strip()]
    return out or ["train", "dev", "test"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase4 single-pass superprompt dataset")
    default_config = str((Path(__file__).resolve().parents[1] / "configs" / "phase4_distill_template.json"))

    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--output-root", default=None, help="Override output root")

    parser.add_argument("--phase0-samples", default=None, help="Override combined phase0 sample path")
    parser.add_argument("--phase1-candidates", default=None, help="Override phase1 all.jsonl path")
    parser.add_argument("--phase2-candidates", default=None, help="Override phase2 all_roles.jsonl path")
    parser.add_argument("--phase3-trace-pack", default=None, help="Override phase3 trace pack path")
    parser.add_argument("--phase3-group-decisions", default=None, help="Override phase3 group decisions path")
    parser.add_argument("--phase3-rectified-roles", default=None, help="Override phase3 rectified roles path")
    parser.add_argument("--phase3-final-tuples", default=None, help="Override phase3 final tuples path")

    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N samples")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--hard-case-only", action="store_true", help="Keep hard cases only")
    parser.add_argument("--hard-case-threshold", type=float, default=None, help="Override hard-case threshold")

    parser.add_argument("--strict", action="store_true", help="Fail on quality issues")
    parser.add_argument("--dry-run", action="store_true", help="Skip dataset JSONL writes")
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
            raise RuntimeError("PyYAML is required for YAML config. Use JSON or install pyyaml.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    override_map = [
        (args.output_root, ["paths", "output_root"]),
        (args.phase0_samples, ["paths", "phase0_samples"]),
        (args.phase1_candidates, ["paths", "phase1_candidates"]),
        (args.phase2_candidates, ["paths", "phase2_candidates"]),
        (args.phase3_trace_pack, ["paths", "phase3_trace_pack"]),
        (args.phase3_group_decisions, ["paths", "phase3_group_decisions"]),
        (args.phase3_rectified_roles, ["paths", "phase3_rectified_roles"]),
        (args.phase3_final_tuples, ["paths", "phase3_final_tuples"]),
        (args.max_samples, ["runtime", "max_samples"]),
        (args.hard_case_threshold, ["phase4", "hard_case_threshold"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)

    if args.splits:
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        set_nested(cfg, ["runtime", "default_splits"], splits)
    if args.hard_case_only:
        set_nested(cfg, ["phase4", "hard_case_only"], True)
    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    path_cfg = cfg.get("paths", {})

    def resolve_opt(raw: Any) -> Optional[Path]:
        if not isinstance(raw, str) or not raw.strip():
            return None
        return resolve_path(workspace_root, config_dir, raw)

    phase0_samples = resolve_opt(path_cfg.get("phase0_samples"))
    phase1_candidates = resolve_opt(path_cfg.get("phase1_candidates"))
    phase2_candidates = resolve_opt(path_cfg.get("phase2_candidates"))

    phase3_trace_pack = resolve_opt(path_cfg.get("phase3_trace_pack"))
    phase3_group_decisions = resolve_opt(path_cfg.get("phase3_group_decisions"))
    phase3_rectified_roles = resolve_opt(path_cfg.get("phase3_rectified_roles"))
    phase3_final_tuples = resolve_opt(path_cfg.get("phase3_final_tuples"))
    zh_lexicon_path = resolve_opt(path_cfg.get("zh_lexicon_path"))

    output_root = resolve_opt(path_cfg.get("output_root"))
    if output_root is None:
        output_root = (workspace_root / "DAR" / "outputs" / "phase4").resolve()

    run_id = args.run_id or f"phase4_distill_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_id

    phase3_group_decisions = choose_existing_path(
        phase3_group_decisions,
        [x for x in [sibling(phase3_group_decisions, "group_audits.jsonl")] if x is not None],
    )
    phase3_rectified_roles = choose_existing_path(
        phase3_rectified_roles,
        [x for x in [sibling(phase3_rectified_roles, "rectified_all.jsonl")] if x is not None],
    )
    phase3_final_tuples = choose_existing_path(
        phase3_final_tuples,
        [x for x in [sibling(phase3_final_tuples, "final.jsonl"), sibling(phase3_final_tuples, "final_all.jsonl")] if x is not None],
    )

    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        output_root=output_root,
        run_root=run_root,
        phase0_samples=phase0_samples,
        phase1_candidates=phase1_candidates,
        phase2_candidates=phase2_candidates,
        phase3_trace_pack=phase3_trace_pack,
        phase3_group_decisions=phase3_group_decisions,
        phase3_rectified_roles=phase3_rectified_roles,
        phase3_final_tuples=phase3_final_tuples,
        zh_lexicon_path=zh_lexicon_path,
    )


def load_prompt_profiles(cfg: Dict[str, Any], workspace_root: Path, config_path: Path) -> Dict[str, Dict[str, Any]]:
    phase4_cfg = cfg.get("phase4", {})
    raw_profiles = phase4_cfg.get("prompt_profiles", {})
    config_dir = config_path.parent

    if raw_profiles is None:
        raw_profiles = {}
    if not isinstance(raw_profiles, dict):
        raise ValueError("phase4.prompt_profiles must be a mapping when provided.")

    profile_names = dedupe_preserve_order(list(DEFAULT_PROMPT_PROFILES.keys()) + [str(name) for name in raw_profiles.keys()])
    loaded: Dict[str, Dict[str, Any]] = {}

    for name in profile_names:
        merged = dict(DEFAULT_PROMPT_PROFILES.get(name, {}))
        override = raw_profiles.get(name, {})
        if override is not None and not isinstance(override, dict):
            raise ValueError(f"phase4.prompt_profiles.{name} must be a mapping.")

        if isinstance(override, dict):
            for key, value in override.items():
                if key in {"system_prompt_path", "user_instruction_path"}:
                    continue
                merged[key] = value

            for text_key, path_key in [("system_prompt", "system_prompt_path"), ("user_instruction", "user_instruction_path")]:
                text_value = normalize_string(override.get(text_key))
                raw_path = normalize_string(override.get(path_key))
                if raw_path is not None:
                    prompt_path = resolve_path(workspace_root, config_dir, raw_path)
                    if not prompt_path.exists():
                        raise FileNotFoundError(f"Prompt file not found for {name}.{path_key}: {prompt_path}")
                    text_value = prompt_path.read_text(encoding="utf-8")
                if text_value is not None:
                    merged[text_key] = text_value

        if not normalize_string(merged.get("system_prompt")):
            raise ValueError(f"Prompt profile {name} is missing system_prompt content.")
        if not normalize_string(merged.get("user_instruction")):
            raise ValueError(f"Prompt profile {name} is missing user_instruction content.")

        loaded[name] = merged

    return loaded


def load_dataset_variants(phase4_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw_variants = phase4_cfg.get("dataset_variants")
    if raw_variants is None:
        return [dict(item) for item in DEFAULT_DATASET_VARIANTS]
    if not isinstance(raw_variants, list):
        raise ValueError("phase4.dataset_variants must be a list when provided.")

    variants: List[Dict[str, Any]] = []
    for item in raw_variants:
        if not isinstance(item, dict):
            continue
        name = normalize_string(item.get("name"))
        if name is None:
            continue
        variant = dict(item)
        variant["name"] = name
        variant.setdefault("description", "")
        variant.setdefault("filter_mode", "all")
        variant.setdefault("supervision_target", "teacher")
        variants.append(variant)

    return variants or [dict(item) for item in DEFAULT_DATASET_VARIANTS]


def validate_protocol_contract(
    cfg: Dict[str, Any],
    protocol_cfg: Dict[str, Any],
    dataset_variants: Sequence[Dict[str, Any]],
    allowed_splits: Sequence[str],
    workspace_root: Path,
    config_path: Path,
) -> None:
    if not protocol_cfg:
        return

    training_supervision = normalize_string(protocol_cfg.get("training_supervision"))
    approved_eval_protocols = protocol_cfg.get("approved_eval_protocols")
    if approved_eval_protocols is not None:
        if not isinstance(approved_eval_protocols, list):
            raise ValueError("protocol.approved_eval_protocols must be a list when provided.")
        normalized_protocols = [normalize_string(item) for item in approved_eval_protocols]
        if not any(item for item in normalized_protocols):
            raise ValueError("protocol.approved_eval_protocols must contain at least one non-empty protocol name.")

    plan_path_raw = normalize_string(protocol_cfg.get("plan_path"))
    if plan_path_raw is not None:
        plan_path = resolve_path(workspace_root, config_path.parent, plan_path_raw)
        if not plan_path.exists():
            raise FileNotFoundError(f"Protocol plan_path does not exist: {plan_path}")

    if training_supervision != "teacher_only":
        return

    phase4_cfg = cfg.get("phase4", {}) if isinstance(cfg.get("phase4"), dict) else {}
    if parse_bool(phase4_cfg.get("include_ground_truth_in_user_prompt"), default=False):
        raise ValueError("Teacher-only distillation cannot include ground truth inside the user prompt.")

    normalized_splits = {str(split).strip().lower() for split in allowed_splits if str(split).strip()}
    if "test" in normalized_splits:
        raise ValueError("Teacher-only distillation protocol forbids including the test split in the training dataset build.")

    non_teacher_variants = [
        normalize_string(variant.get("name")) or "unknown"
        for variant in dataset_variants
        if (normalize_string(variant.get("supervision_target")) or "teacher") != "teacher"
    ]
    if non_teacher_variants:
        raise ValueError(
            "Teacher-only distillation protocol forbids non-teacher supervision variants: "
            + ", ".join(non_teacher_variants)
        )


def load_zh_lexicon_terms(path: Optional[Path]) -> List[Dict[str, str]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return []

    rows: List[Dict[str, str]] = []
    for item in payload.get("terms", []):
        if not isinstance(item, dict):
            continue
        term = normalize_string(item.get("term"))
        if term is None:
            continue
        rows.append(
            {
                "term": term,
                "category": normalize_string(item.get("category")) or "unknown",
                "definition": normalize_string(item.get("definition")) or "",
            }
        )
    rows.sort(key=lambda x: (-len(x["term"]), x["term"]))
    return rows


def load_phase0_like_selector_indices(
    phase4_cfg: Dict[str, Any],
    workspace_root: Path,
    config_path: Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    selector_cfg = phase4_cfg.get("quality_selectors", {})
    if not isinstance(selector_cfg, dict):
        return {}

    by_language = selector_cfg.get("by_language_phase0_like", {})
    if not isinstance(by_language, dict):
        return {}

    config_dir = config_path.parent
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for raw_lang, raw_path in by_language.items():
        lang = normalize_string(raw_lang)
        path_str = normalize_string(raw_path)
        if lang is None or path_str is None:
            continue
        resolved = resolve_path(workspace_root, config_dir, path_str)
        if not resolved.exists():
            raise FileNotFoundError(f"Quality selector phase0_like path not found for language={lang}: {resolved}")
        rows = read_jsonl(resolved)
        out[lang.lower()] = {
            normalize_string(row.get("id")) or normalize_string(row.get("sample_id")) or "": row
            for row in rows
            if isinstance(row, dict)
        }
    return out


def build_phase1_decompose_index(phase1_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in phase1_rows:
        group_id = normalize_string(row.get("group_id"))
        if group_id is None:
            continue
        payload = row.get("phase4_decompose") if isinstance(row.get("phase4_decompose"), dict) else None
        if payload is None:
            payload = {
                "task": "decompose",
                "sentence_ids": safe_sentence_ids(row.get("sentence_ids")),
                "group_text": normalize_string(row.get("group_text")) or "",
                "canonical_target": normalize_string(row.get("canonical_target")) or "IMPLICIT_TARGET",
                "escalation_flag": bool(row.get("escalation_flag")),
            }
        out[group_id] = payload
    return out


def build_phase2_role_index(phase2_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in phase2_rows:
        rcid = normalize_string(row.get("role_candidate_id"))
        if rcid is None:
            continue
        out[rcid] = row
    return out


def build_phase0_sample_index(phase0_rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for row in phase0_rows:
        sample_id = normalize_string(row.get("id")) or normalize_string(row.get("sample_id"))
        if sample_id is None:
            continue
        out[sample_id] = row
    return out


def normalize_ground_truth_list(sample_row: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(sample_row, dict):
        return []

    ground_truth = sample_row.get("ground_truth")
    if not isinstance(ground_truth, list):
        return []

    rows: List[Dict[str, Any]] = []
    for item in ground_truth:
        if not isinstance(item, dict):
            continue
        harmful = item.get("hateful") if "hateful" in item else item.get("harmful")
        rows.append(
            {
                "target": normalize_string(item.get("target")) or "IMPLICIT_TARGET",
                "argument": normalize_string(item.get("argument")),
                "group": normalize_string(item.get("group")) or "unknown",
                "harmful": direction_to_harmful_binary(normalize_harmful_direction(item.get("harmful_direction"), harmful)),
                "harmful_direction": normalize_harmful_direction(item.get("harmful_direction"), harmful),
            }
        )
    return rows


def normalize_candidate_tuple(candidate: Dict[str, Any]) -> Dict[str, Any]:
    tuple_obj = candidate.get("tuple") if isinstance(candidate.get("tuple"), dict) else candidate
    return normalize_final_tuple(tuple_obj)


def choose_best_candidate(candidates: Any) -> Dict[str, Any]:
    if not isinstance(candidates, list):
        return {}

    best_item: Dict[str, Any] = {}
    best_key: Tuple[int, float, int, int] = (-1, -1, -1, -1)
    for index, item in enumerate(candidates):
        if not isinstance(item, dict):
            continue
        valid = 1 if parse_bool(item.get("valid_for_fusion"), default=True) else 0
        quality_score = clamp(safe_float(item.get("quality_score"), default=0.0), 0.0, 1.0)
        issues = item.get("issues") if isinstance(item.get("issues"), list) else []
        key = (valid, quality_score, -len(issues), -index)
        if key > best_key:
            best_item = item
            best_key = key
    return best_item


def normalize_role_hypothesis_item(role_entry: Dict[str, Any], group_id: str, sample_id: str) -> Dict[str, Any]:
    role_name = str(role_entry.get("role") or role_entry.get("task") or "unknown").strip().lower()

    if isinstance(role_entry.get("tuple"), dict):
        tuple_obj = normalize_final_tuple(role_entry.get("tuple"))
        return {
            "task": role_name,
            "role": role_name,
            "role_candidate_id": normalize_string(role_entry.get("role_candidate_id")) or "",
            "group_id": group_id,
            "sample_id": sample_id,
            "tuple": tuple_obj,
            "rationale": normalize_string(role_entry.get("rationale")) or "",
            "uncertainty_flag": bool(role_entry.get("uncertainty_flag")),
            "escalation_flag": bool(role_entry.get("escalation_flag")),
            "role_status": normalize_string(role_entry.get("role_status")) or "original",
            "candidate_count": int(role_entry.get("candidate_count") or 1),
            "quality_score": clamp(safe_float(role_entry.get("quality_score"), default=0.0), 0.0, 1.0),
            "issues": list(role_entry.get("issues") or []),
            "lexicon_hit": bool(role_entry.get("lexicon_hit")),
        }

    candidates = role_entry.get("candidates") if isinstance(role_entry.get("candidates"), list) else []
    best = choose_best_candidate(candidates)
    tuple_obj = normalize_candidate_tuple(best)
    issues = best.get("issues") if isinstance(best.get("issues"), list) else []
    quality_score = clamp(safe_float(best.get("quality_score"), default=0.0), 0.0, 1.0)
    valid_for_fusion = parse_bool(best.get("valid_for_fusion"), default=True)
    role_status = normalize_string(role_entry.get("role_status")) or "original"
    uncertainty_flag = bool(issues) or (not valid_for_fusion) or quality_score < 0.67

    return {
        "task": role_name,
        "role": role_name,
        "role_candidate_id": normalize_string(best.get("candidate_id")) or "",
        "group_id": group_id,
        "sample_id": sample_id,
        "tuple": tuple_obj,
        "rationale": normalize_string(best.get("rationale")) or "",
        "uncertainty_flag": uncertainty_flag,
        "escalation_flag": role_status == "regenerated" or uncertainty_flag,
        "role_status": role_status,
        "candidate_count": int(role_entry.get("candidate_count") or len(candidates) or 1),
        "quality_score": quality_score,
        "issues": issues,
        "valid_for_fusion": valid_for_fusion,
        "lexicon_hit": bool(best.get("lexicon_hit") or role_entry.get("lexicon_hit")),
        "raw_candidates": candidates,
    }


def build_field_conflict_stats(role_items: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    fields: Dict[str, Dict[str, Any]] = {}
    for field in ["target", "argument", "group", "harmful_direction"]:
        votes: Dict[str, int] = defaultdict(int)
        for item in role_items:
            tuple_obj = item.get("tuple") if isinstance(item.get("tuple"), dict) else {}
            raw_value = tuple_obj.get(field)
            if field == "argument":
                value = normalize_string(raw_value) or "__null__"
            else:
                value = normalize_string(raw_value) or "unknown"
            votes[value] += 1
        if votes:
            fields[field] = {
                "unique_values": len(votes),
                "votes": dict(sorted(votes.items(), key=lambda kv: (-kv[1], kv[0]))),
            }
    return fields


def derive_high_risk_fields(field_stats: Dict[str, Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for field_name, stats in field_stats.items():
        if int(stats.get("unique_values") or 0) > 1:
            out.append(field_name)
    return sorted(out)


def derive_error_tags_from_dynamic(
    lane: str,
    triage: Dict[str, Any],
    signals: Dict[str, Any],
    role_items: Sequence[Dict[str, Any]],
    correction_trace: Sequence[Dict[str, Any]],
) -> List[str]:
    tags: List[str] = []
    if lane == "conflict":
        tags.append("cross_role_conflict")
    if lane == "defective":
        tags.append("defective_role")
    if isinstance(triage.get("missing_roles"), list) and triage.get("missing_roles"):
        tags.append("missing_roles")
    if isinstance(triage.get("regenerated_roles"), list) and triage.get("regenerated_roles"):
        tags.append("role_regenerated")
    if parse_bool(signals.get("quote_caution"), default=False):
        tags.append("quote_caution")
    if parse_bool(signals.get("lexicon_hit"), default=False):
        tags.append("lexicon_hit")

    for role_item in role_items:
        for issue in role_item.get("issues") if isinstance(role_item.get("issues"), list) else []:
            issue_text = normalize_string(issue)
            if issue_text is not None:
                tags.append(issue_text)

    for item in correction_trace:
        if not isinstance(item, dict):
            continue
        action = normalize_string(item.get("action"))
        if action is not None:
            tags.append(f"auditor_{action}")

    return dedupe_preserve_order(tags)


def score_conflict_for_lane(
    lane: str,
    high_risk_fields: Sequence[str],
    regenerated_roles: Sequence[str],
    cluster_count: int,
) -> float:
    lane_base = {
        "consensus": 0.12,
        "conflict": 0.72,
        "defective": 0.84,
        "unknown": 0.55,
    }
    score = lane_base.get(lane, 0.55)
    score += min(0.16, 0.04 * len(high_risk_fields))
    score += min(0.12, 0.04 * len(regenerated_roles))
    score += min(0.12, 0.03 * max(cluster_count - 1, 0))
    return round(clamp(score, 0.0, 1.0), 4)


def normalize_step_a(trace_row: Dict[str, Any]) -> Dict[str, Any]:
    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else None
    if step_a is None:
        step_a = trace_row.get("step_a_decomposition_evidence") if isinstance(trace_row.get("step_a_decomposition_evidence"), dict) else {}

    return {
        "task": "decompose",
        "sentence_ids": safe_sentence_ids(step_a.get("sentence_ids") or trace_row.get("sentence_ids")),
        "group_text": normalize_string(step_a.get("group_text")) or normalize_string(trace_row.get("group_text")) or "",
        "focus_text": normalize_string(step_a.get("focus_text")) or normalize_string(trace_row.get("focus_text")),
        "canonical_target": normalize_string(step_a.get("canonical_target")) or normalize_string(trace_row.get("canonical_target")) or "IMPLICIT_TARGET",
        "escalation_flag": bool(step_a.get("escalation_flag") or trace_row.get("escalation_flag")),
    }


def normalize_step_b(trace_row: Dict[str, Any], group_id: str, sample_id: str) -> List[Dict[str, Any]]:
    step_b = trace_row.get("step_b_role_hypotheses") if isinstance(trace_row.get("step_b_role_hypotheses"), list) else []
    normalized: List[Dict[str, Any]] = []
    for item in step_b:
        if not isinstance(item, dict):
            continue
        normalized.append(normalize_role_hypothesis_item(item, group_id=group_id, sample_id=sample_id))
    normalized.sort(key=lambda x: (ROLE_ORDER.index(x["role"]) if x["role"] in ROLE_ORDER else 99, x["role_candidate_id"]))
    return normalized


def normalize_step_c(trace_row: Dict[str, Any], role_items: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    if "overall_conflict_score" in step_c:
        return {
            "overall_conflict_score": round(clamp(safe_float(step_c.get("overall_conflict_score"), default=0.0), 0.0, 1.0), 4),
            "overall_conflict_level": str(step_c.get("overall_conflict_level") or "none"),
            "high_risk_fields": list(step_c.get("high_risk_fields") or []),
            "fields": step_c.get("fields") if isinstance(step_c.get("fields"), dict) else {},
            "error_tags": list(step_c.get("error_tags") or []),
            "field_findings": list(step_c.get("field_findings") or []),
            "summary": normalize_string(step_c.get("summary")) or "",
            "triage": step_c.get("triage") if isinstance(step_c.get("triage"), dict) else {},
            "signals": step_c.get("signals") if isinstance(step_c.get("signals"), dict) else {},
            "applied_beacons": step_c.get("applied_beacons") if isinstance(step_c.get("applied_beacons"), list) else [],
            "clusters": step_c.get("clusters") if isinstance(step_c.get("clusters"), list) else [],
        }

    triage = step_c.get("triage") if isinstance(step_c.get("triage"), dict) else {}
    signals = step_c.get("signals") if isinstance(step_c.get("signals"), dict) else {}
    clusters = step_c.get("clusters") if isinstance(step_c.get("clusters"), list) else []
    lane = str(triage.get("lane") or "unknown")
    field_stats = build_field_conflict_stats(role_items)
    high_risk_fields = derive_high_risk_fields(field_stats)
    correction_trace = (
        trace_row.get("step_e_final_arbitration", {}) if isinstance(trace_row.get("step_e_final_arbitration"), dict) else {}
    ).get("correction_trace")
    correction_trace = correction_trace if isinstance(correction_trace, list) else []
    error_tags = derive_error_tags_from_dynamic(lane, triage, signals, role_items, correction_trace)
    summary = normalize_string(((trace_row.get("step_e_final_arbitration") or {}).get("fused_rationale"))) or (
        f"lane={lane}, high_risk_fields={high_risk_fields}, applied_beacons={list(step_c.get('applied_beacons') or [])}"
    )

    return {
        "overall_conflict_score": score_conflict_for_lane(
            lane=lane,
            high_risk_fields=high_risk_fields,
            regenerated_roles=list(triage.get("regenerated_roles") or []),
            cluster_count=len(clusters),
        ),
        "overall_conflict_level": lane,
        "high_risk_fields": high_risk_fields,
        "fields": field_stats,
        "error_tags": error_tags,
        "field_findings": [
            f"{field}: {stats['votes']}" for field, stats in field_stats.items() if int(stats.get("unique_values") or 0) > 1
        ],
        "summary": summary,
        "triage": triage,
        "signals": signals,
        "applied_beacons": step_c.get("applied_beacons") if isinstance(step_c.get("applied_beacons"), list) else [],
        "clusters": clusters,
    }


def normalize_step_e(trace_row: Dict[str, Any], step_c: Dict[str, Any]) -> Dict[str, Any]:
    step_e = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else None
    if step_e is not None:
        return {
            "final_tuple": normalize_final_tuple(step_e.get("final_tuple")),
            "audit_confidence": round(clamp(safe_float(step_e.get("audit_confidence"), default=0.0), 0.0, 1.0), 4),
            "veto_recommend": bool(step_e.get("veto_recommend")),
            "veto_flag": bool(step_e.get("veto_flag")),
            "veto_reason": normalize_string(step_e.get("veto_reason")) or "",
            "consensus_trace": step_e.get("consensus_trace") if isinstance(step_e.get("consensus_trace"), dict) else {},
            "fused_rationale": normalize_string(step_e.get("fused_rationale")) or "",
            "correction_trace": step_e.get("correction_trace") if isinstance(step_e.get("correction_trace"), list) else [],
        }

    step_e_alt = trace_row.get("step_e_final_arbitration") if isinstance(trace_row.get("step_e_final_arbitration"), dict) else {}
    triage = step_c.get("triage") if isinstance(step_c.get("triage"), dict) else {}
    correction_trace = step_e_alt.get("correction_trace") if isinstance(step_e_alt.get("correction_trace"), list) else []
    downweighted_roles = [
        str(item.get("role") or "")
        for item in correction_trace
        if isinstance(item, dict) and normalize_string(item.get("action")) in {"downweighted", "reject_hypothesis"}
    ]
    return {
        "final_tuple": normalize_final_tuple(step_e_alt.get("final_tuple")),
        "audit_confidence": round(clamp(safe_float(step_e_alt.get("audit_confidence"), default=0.0), 0.0, 1.0), 4),
        "veto_recommend": False,
        "veto_flag": False,
        "veto_reason": "",
        "consensus_trace": {
            "lane": str(triage.get("lane") or "unknown"),
            "selected_cluster_id": normalize_string(step_e_alt.get("selected_cluster_id")),
            "regenerated_roles": list(triage.get("regenerated_roles") or []),
            "downweighted_roles": dedupe_preserve_order([role for role in downweighted_roles if role]),
            "rejected_roles": [],
            "consensus_aligned_roles": list(((step_c.get("signals") or {}).get("consensus_aligned_roles")) or []),
        },
        "fused_rationale": normalize_string(step_e_alt.get("fused_rationale")) or "",
        "correction_trace": correction_trace,
    }


def normalize_step_d(trace_row: Dict[str, Any], role_items: Sequence[Dict[str, Any]], step_e: Dict[str, Any]) -> List[Dict[str, Any]]:
    step_d = trace_row.get("step_d_rectification_actions")
    if isinstance(step_d, list):
        out: List[Dict[str, Any]] = []
        for item in step_d:
            if not isinstance(item, dict):
                continue
            out.append(
                {
                    "role": str(item.get("role") or "unknown"),
                    "role_candidate_id": str(item.get("role_candidate_id") or ""),
                    "decision_action": str(item.get("decision_action") or "keep_candidate"),
                    "decision_reason": normalize_string(item.get("decision_reason")) or "",
                    "is_revised": bool(item.get("is_revised")),
                    "rejected_by_engine": bool(item.get("rejected_by_engine")),
                    "selected_for_arbitration": bool(item.get("selected_for_arbitration")),
                    "original_predicted_tuple": normalize_final_tuple(item.get("original_predicted_tuple")),
                    "rectified_tuple": normalize_final_tuple(item.get("rectified_tuple")),
                }
            )
        return out

    correction_trace = step_e.get("correction_trace") if isinstance(step_e.get("correction_trace"), list) else []
    correction_by_role: Dict[str, Dict[str, Any]] = {}
    for item in correction_trace:
        if not isinstance(item, dict):
            continue
        role_name = normalize_string(item.get("role"))
        if role_name is None:
            continue
        correction_by_role[role_name.lower()] = item

    regen_trace = step_d.get("regeneration_trace") if isinstance(step_d, dict) and isinstance(step_d.get("regeneration_trace"), list) else []
    regen_roles = {
        str(item.get("role") or "").strip().lower()
        for item in regen_trace
        if isinstance(item, dict) and normalize_string(item.get("role")) is not None
    }

    out = []
    for role_item in role_items:
        role_name = str(role_item.get("role") or "unknown").lower()
        correction = correction_by_role.get(role_name, {})
        action = normalize_string(correction.get("action"))
        if action is None:
            action = "regenerated_candidate" if role_name in regen_roles or role_item.get("role_status") == "regenerated" else "keep_candidate"
        reason = normalize_string(correction.get("reason")) or ""
        tuple_obj = role_item.get("tuple") if isinstance(role_item.get("tuple"), dict) else {}
        out.append(
            {
                "role": role_name,
                "role_candidate_id": str(role_item.get("role_candidate_id") or ""),
                "decision_action": action,
                "decision_reason": reason,
                "is_revised": action != "keep_candidate" or role_name in regen_roles or role_item.get("role_status") == "regenerated",
                "rejected_by_engine": action == "reject_hypothesis",
                "selected_for_arbitration": True,
                "original_predicted_tuple": normalize_final_tuple(tuple_obj),
                "rectified_tuple": normalize_final_tuple(tuple_obj),
            }
        )
    return out


def normalize_trace_row(raw_trace: Dict[str, Any]) -> Dict[str, Any]:
    sample_id = normalize_string(raw_trace.get("sample_id")) or ""
    group_id = normalize_string(raw_trace.get("group_id")) or ""

    step_a = normalize_step_a(raw_trace)
    step_b = normalize_step_b(raw_trace, group_id=group_id, sample_id=sample_id)
    step_c = normalize_step_c(raw_trace, role_items=step_b)
    step_e = normalize_step_e(raw_trace, step_c=step_c)
    step_d = normalize_step_d(raw_trace, role_items=step_b, step_e=step_e)

    normalized = dict(raw_trace)
    normalized["sample_id"] = sample_id
    normalized["group_id"] = group_id
    normalized["decision_id"] = normalize_string(raw_trace.get("decision_id")) or ""
    normalized["source"] = str(raw_trace.get("source") or "unknown")
    normalized["split"] = str(raw_trace.get("split") or "train")
    normalized["language"] = str(raw_trace.get("language") or "unknown")
    normalized["step_a_decompose"] = step_a
    normalized["step_b_role_hypotheses"] = step_b
    normalized["step_c_conflict_diagnosis"] = step_c
    normalized["step_d_rectification_actions"] = step_d
    normalized["step_e_final_decision"] = step_e
    return normalized


def score_ground_truth_match(trace_row: Dict[str, Any], ground_truth_tuple: Dict[str, Any]) -> float:
    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {}
    step_e = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else {}
    final_tuple = step_e.get("final_tuple") if isinstance(step_e.get("final_tuple"), dict) else {}

    group_text = normalize_text_for_match(step_a.get("group_text"))
    focus_text = normalize_text_for_match(step_a.get("focus_text"))
    canonical_target = normalize_text_for_match(step_a.get("canonical_target"))
    gt_argument = normalize_text_for_match(ground_truth_tuple.get("argument"))
    gt_target = normalize_text_for_match(ground_truth_tuple.get("target"))

    score = 0.0
    if gt_argument and (gt_argument in group_text or gt_argument in focus_text):
        score += 3.0
    if gt_target and (gt_target == canonical_target or gt_target in group_text or gt_target in focus_text):
        score += 2.0
    if normalize_harmful_direction(ground_truth_tuple.get("harmful_direction"), ground_truth_tuple.get("harmful")) == normalize_harmful_direction(final_tuple.get("harmful_direction"), final_tuple.get("harmful")):
        score += 1.0
    if normalize_text_for_match(final_tuple.get("argument")) and normalize_text_for_match(final_tuple.get("argument")) == gt_argument:
        score += 2.0
    if normalize_text_for_match(final_tuple.get("target")) and normalize_text_for_match(final_tuple.get("target")) == gt_target:
        score += 1.0
    return score


def attach_sample_context(
    trace_row: Dict[str, Any],
    sample_index: Dict[str, Dict[str, Any]],
    min_score: float,
) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    sample_row = sample_index.get(str(trace_row.get("sample_id") or ""))
    if not isinstance(sample_row, dict):
        return None, [], None

    normalized_ground_truth = normalize_ground_truth_list(sample_row)
    ranked = [
        (score_ground_truth_match(trace_row, item), item)
        for item in normalized_ground_truth
    ]
    ranked.sort(key=lambda x: (-x[0], normalize_string(x[1].get("argument")) or ""))
    matched = [item for score, item in ranked if score >= min_score]
    primary = matched[0] if matched else None
    return sample_row, matched, primary


def get_row_decision_id(row: Dict[str, Any]) -> str:
    return (
        normalize_string(row.get("decision_id"))
        or normalize_string(row.get("audit_id"))
        or ""
    )


def build_decision_payload(decision: Dict[str, Any]) -> Dict[str, Any]:
    payload = decision.get("decision_payload") if isinstance(decision.get("decision_payload"), dict) else None
    if payload is not None:
        return payload

    critique = decision.get("critique") if isinstance(decision.get("critique"), dict) else {}
    actions = decision.get("actions") if isinstance(decision.get("actions"), list) else []
    return {
        "summary": normalize_string(critique.get("summary")) or "",
        "error_tags": list(critique.get("error_tags") or []),
        "field_findings": list(critique.get("field_findings") or []),
        "actions": actions,
        "veto_recommend": bool(decision.get("veto_recommend")),
    }


def select_tuple_for_role(row: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(row.get("rectified_tuple"), dict):
        return normalize_final_tuple(row.get("rectified_tuple"))
    if isinstance(row.get("original_predicted_tuple"), dict):
        return normalize_final_tuple(row.get("original_predicted_tuple"))
    if isinstance(row.get("predicted_tuple"), dict):
        return normalize_final_tuple(row.get("predicted_tuple"))
    return normalize_final_tuple({})


def majority_value(values: Sequence[str]) -> Optional[str]:
    if not values:
        return None
    counter: Dict[str, int] = defaultdict(int)
    for value in values:
        counter[value] += 1
    ranked = sorted(counter.keys(), key=lambda x: (-counter[x], x))
    return ranked[0]


def build_majority_final_tuple(group_roles: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    target_votes: List[str] = []
    argument_votes: List[str] = []
    group_votes: List[str] = []
    harmful_votes: List[str] = []

    for row in group_roles:
        tup = select_tuple_for_role(row)

        target_votes.append(normalize_string(tup.get("target")) or "IMPLICIT_TARGET")
        argument_votes.append(normalize_string(tup.get("argument")) or "")
        group_votes.append(normalize_string(tup.get("group")) or "unknown")
        harmful_votes.append(normalize_harmful_direction(tup.get("harmful_direction"), tup.get("harmful")))

    target = majority_value(target_votes) or "IMPLICIT_TARGET"
    argument = majority_value(argument_votes) or ""
    group_label = majority_value(group_votes) or "unknown"
    harmful_direction = majority_value(harmful_votes) or "unknown"

    return {
        "target": target,
        "argument": argument if argument else None,
        "group": group_label,
        "harmful": direction_to_harmful_binary(harmful_direction),
        "harmful_direction": harmful_direction,
    }


def bucket_rows_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        out[str(row.get(key))].append(row)
    return out


def synthesize_trace_pack_rows(
    group_decisions: Sequence[Dict[str, Any]],
    rectified_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
    phase1_index: Dict[str, Dict[str, Any]],
    phase2_index: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rectified_by_decision: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rectified_rows:
        key = get_row_decision_id(row)
        if key:
            rectified_by_decision[key].append(row)

    final_by_decision: Dict[str, Dict[str, Any]] = {}
    for row in final_rows:
        key = get_row_decision_id(row)
        if key:
            final_by_decision[key] = row

    rows: List[Dict[str, Any]] = []
    for decision in group_decisions:
        decision_id = get_row_decision_id(decision)
        if not decision_id:
            decision_id = make_sha1(decision.get("group_id"), decision.get("sample_id"))[:20]
        group_id = str(decision.get("group_id") or "")
        group_roles = sorted(
            rectified_by_decision.get(decision_id, []),
            key=lambda x: (str(x.get("role") or ""), str(x.get("role_candidate_id") or "")),
        )
        final_row = final_by_decision.get(decision_id)

        phase4_decompose: Optional[Dict[str, Any]] = None
        for role_row in group_roles:
            input_package = role_row.get("input_package") if isinstance(role_row.get("input_package"), dict) else {}
            candidate = input_package.get("phase4_decompose")
            if isinstance(candidate, dict):
                phase4_decompose = candidate
                break
        if phase4_decompose is None:
            phase4_decompose = phase1_index.get(group_id)
        if phase4_decompose is None:
            phase4_decompose = {
                "task": "decompose",
                "sentence_ids": safe_sentence_ids(decision.get("sentence_ids")),
                "group_text": normalize_string(decision.get("group_text")) or "",
                "canonical_target": "IMPLICIT_TARGET",
                "escalation_flag": False,
            }

        role_hypotheses: List[Dict[str, Any]] = []
        rectification_actions: List[Dict[str, Any]] = []

        for role_row in group_roles:
            rcid = normalize_string(role_row.get("role_candidate_id"))
            original_tuple = role_row.get("original_predicted_tuple") if isinstance(role_row.get("original_predicted_tuple"), dict) else {}
            role_payload = role_row.get("phase4_role_hypothesis") if isinstance(role_row.get("phase4_role_hypothesis"), dict) else None
            if role_payload is None and rcid is not None:
                phase2_row = phase2_index.get(rcid, {})
                phase2_payload = phase2_row.get("phase4_role_hypothesis") if isinstance(phase2_row.get("phase4_role_hypothesis"), dict) else None
                role_payload = phase2_payload
            if role_payload is None:
                role_payload = {
                    "task": str(role_row.get("role") or "unknown"),
                    "role": str(role_row.get("role") or "unknown"),
                    "role_candidate_id": str(role_row.get("role_candidate_id") or ""),
                    "group_id": str(role_row.get("group_id") or ""),
                    "sample_id": str(role_row.get("sample_id") or ""),
                    "tuple": original_tuple,
                    "rationale": normalize_string(role_row.get("rationale")) or "",
                    "uncertainty_flag": bool(role_row.get("uncertainty_flag")),
                    "escalation_flag": bool(role_row.get("escalation_flag")),
                }
            role_hypotheses.append(role_payload)

            rectification_actions.append(
                {
                    "role": str(role_row.get("role") or "unknown"),
                    "role_candidate_id": str(role_row.get("role_candidate_id") or ""),
                    "decision_action": str(role_row.get("decision_action") or role_row.get("audit_action") or "keep_candidate"),
                    "decision_reason": normalize_string(role_row.get("decision_reason") or role_row.get("audit_reason")) or "",
                    "is_revised": bool(role_row.get("is_revised")),
                    "rejected_by_engine": bool(role_row.get("rejected_by_engine") or role_row.get("rejected_by_auditor")),
                    "selected_for_arbitration": bool(role_row.get("selected_for_arbitration") or role_row.get("selected_for_phase4")),
                    "original_predicted_tuple": original_tuple,
                    "rectified_tuple": role_row.get("rectified_tuple") if isinstance(role_row.get("rectified_tuple"), dict) else original_tuple,
                }
            )

        decision_payload = build_decision_payload(decision)
        conflict_matrix = decision.get("conflict_matrix") if isinstance(decision.get("conflict_matrix"), dict) else {}

        final_tuple = {}
        veto_flag = False
        veto_reason = ""
        consensus_trace = decision.get("consensus_trace") if isinstance(decision.get("consensus_trace"), dict) else {}
        if isinstance(final_row, dict):
            final_tuple = final_row.get("final_tuple") if isinstance(final_row.get("final_tuple"), dict) else {}
            veto_flag = bool(final_row.get("veto_flag"))
            veto_reason = normalize_string(final_row.get("veto_reason")) or ""
            if not consensus_trace:
                consensus_trace = final_row.get("consensus_trace") if isinstance(final_row.get("consensus_trace"), dict) else {}
        if not final_tuple:
            final_tuple = decision.get("final_tuple") if isinstance(decision.get("final_tuple"), dict) else {}
        if not final_tuple:
            final_tuple = build_majority_final_tuple(group_roles)

        rows.append(
            {
                "decision_id": decision_id,
                "group_id": group_id,
                "sample_id": str(decision.get("sample_id") or ""),
                "source": str(decision.get("source") or "unknown"),
                "split": str(decision.get("split") or "train"),
                "language": str(decision.get("language") or "unknown"),
                "step_a_decompose": phase4_decompose,
                "step_b_role_hypotheses": role_hypotheses,
                "step_c_conflict_diagnosis": {
                    "overall_conflict_score": round(clamp(safe_float(conflict_matrix.get("overall_conflict_score"), default=0.0), 0.0, 1.0), 4),
                    "overall_conflict_level": str(conflict_matrix.get("overall_conflict_level") or "none"),
                    "high_risk_fields": list(conflict_matrix.get("high_risk_fields") or []),
                    "fields": conflict_matrix.get("fields") if isinstance(conflict_matrix.get("fields"), dict) else {},
                    "error_tags": list(decision_payload.get("error_tags") or []),
                    "field_findings": list(decision_payload.get("field_findings") or []),
                    "summary": normalize_string(decision_payload.get("summary")) or "",
                },
                "step_d_rectification_actions": rectification_actions,
                "step_e_final_decision": {
                    "final_tuple": normalize_final_tuple(final_tuple),
                    "veto_recommend": bool(decision_payload.get("veto_recommend")),
                    "veto_flag": veto_flag,
                    "veto_reason": veto_reason,
                    "consensus_trace": consensus_trace,
                },
            }
        )

    return rows


def extract_role_hypothesis(role_entry: Dict[str, Any], role_name: str) -> Optional[Dict[str, Any]]:
    for item in role_entry.get("step_b_role_hypotheses", []):
        if not isinstance(item, dict):
            continue
        if str(item.get("role") or "").lower() == role_name:
            return item
    return None


def get_task_label(
    trace_row: Dict[str, Any],
    task_name: str,
    final_tuple_override: Optional[Dict[str, Any]] = None,
) -> str:
    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {}
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    step_d = trace_row.get("step_d_rectification_actions") if isinstance(trace_row.get("step_d_rectification_actions"), list) else []
    step_e = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else {}

    if task_name == "decompose":
        return (
            f"group_id={trace_row.get('group_id')}, canonical_target={step_a.get('canonical_target')}, "
            f"sentence_ids={step_a.get('sentence_ids')}, escalation={step_a.get('escalation_flag')}"
        )

    if task_name in set(ROLE_ORDER):
        role_item = extract_role_hypothesis(trace_row, task_name)
        if role_item is None:
            return f"{task_name}: missing"
        tuple_obj = role_item.get("tuple") if isinstance(role_item.get("tuple"), dict) else {}
        return (
            f"{task_name}: target={tuple_obj.get('target')}, group={tuple_obj.get('group')}, "
            f"harmful_direction={tuple_obj.get('harmful_direction')}"
        )

    if task_name == "conflict":
        return (
            f"overall_conflict_score={step_c.get('overall_conflict_score')}, "
            f"high_risk_fields={step_c.get('high_risk_fields')}, error_tags={step_c.get('error_tags')}"
        )

    if task_name == "rectification":
        action_bits: List[str] = []
        for action in step_d:
            if not isinstance(action, dict):
                continue
            action_bits.append(f"{action.get('role')}->{action.get('decision_action')}")
        return "; ".join(action_bits) if action_bits else "no_rectification_actions"

    if task_name == "final":
        if isinstance(final_tuple_override, dict):
            final_tuple = normalize_final_tuple(final_tuple_override)
            veto_flag = False
        else:
            final_tuple = normalize_final_tuple(step_e.get("final_tuple"))
            veto_flag = bool(step_e.get("veto_flag"))
        return f"final_tuple={final_tuple}, veto_flag={veto_flag}"

    return "unknown_task"


def get_step_titles(prompt_profile: Dict[str, Any]) -> List[str]:
    raw_titles = prompt_profile.get("step_titles")
    if isinstance(raw_titles, list):
        titles = [str(item) for item in raw_titles if str(item).strip()]
        if len(titles) >= 5:
            return titles[:5]
    return list(DEFAULT_PROMPT_PROFILES["en_span_harmful"]["step_titles"])


def resolve_assistant_chain_style(prompt_profile: Dict[str, Any]) -> str:
    style = normalize_string(prompt_profile.get("assistant_chain_style"))
    if style:
        return style
    return "compact_ae"


def normalize_reasoning_snippet(value: Any, max_chars: int = 220) -> str:
    text = normalize_string(value) or ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def is_trace_zh(prompt_profile: Dict[str, Any], trace_row: Dict[str, Any]) -> bool:
    task_profile = normalize_string(trace_row.get("task_profile")) or ""
    return is_profile_zh(task_profile, prompt_profile)


def get_role_display_name(role: str, use_zh: bool) -> str:
    mapping_zh = {
        "victim": "victim",
        "moderator": "moderator",
        "cultural_bystander": "cultural_bystander",
    }
    mapping_en = {
        "victim": "victim",
        "moderator": "moderator",
        "cultural_bystander": "cultural_bystander",
    }
    mapping = mapping_zh if use_zh else mapping_en
    return mapping.get(role, role)


def normalize_lexicon_hits(lexicon_evidence: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    for item in lexicon_evidence or []:
        if not isinstance(item, dict):
            continue
        term = normalize_string(item.get("term"))
        if term is None:
            continue
        hits.append(
            {
                "term": term,
                "category": normalize_string(item.get("category")) or "unknown",
                "definition": normalize_reasoning_snippet(item.get("definition"), max_chars=120),
                "matched_fields": list(item.get("matched_fields") or []),
            }
        )
    return hits


def format_tuple_summary(tuple_obj: Dict[str, Any], use_zh: bool) -> str:
    target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    argument = normalize_string(tuple_obj.get("argument")) or "null"
    group = normalize_string(tuple_obj.get("group")) or "unknown"
    harmful_direction = normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful"))
    harmful = tuple_obj.get("harmful")
    if use_zh:
        return (
            f"target={target}，argument={argument}，group={group}，"
            f"harmful={harmful}，harmful_direction={harmful_direction}"
        )
    return (
        f"target={target}, argument={argument}, group={group}, "
        f"harmful={harmful}, harmful_direction={harmful_direction}"
    )


def format_compact_tuple_anchor(tuple_obj: Dict[str, Any], use_zh: bool) -> str:
    target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    argument = normalize_string(tuple_obj.get("argument")) or "null"
    group = normalize_string(tuple_obj.get("group")) or "unknown"
    harmful_direction = normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful"))
    if use_zh:
        return f"target={target}；argument={argument}；group={group}；dir={harmful_direction}"
    return f"target={target}; argument={argument}; group={group}; dir={harmful_direction}"


def format_role_flags(role_item: Dict[str, Any], use_zh: bool) -> str:
    flags: List[str] = []
    if bool(role_item.get("uncertainty_flag")):
        flags.append("uncertain")
    if bool(role_item.get("escalation_flag")):
        flags.append("escalate")
    if bool(role_item.get("lexicon_hit")):
        flags.append("lexicon")
    if flags:
        joined = "/".join(flags)
        return f"flags={joined}"
    return ""


def format_span_anchor(value: Any, use_zh: bool) -> str:
    text = normalize_string(value)
    if text is None:
        return "null"
    quote = "“{}”" if use_zh else "\"{}\""
    return quote.format(text)


def format_list_for_chain(values: Sequence[Any], empty_text: str) -> str:
    items = [normalize_string(value) for value in values]
    items = [item for item in items if item is not None]
    return ", ".join(items) if items else empty_text


def build_think_chain(
    trace_row: Dict[str, Any],
    prompt_profile: Dict[str, Any],
    final_tuple_override: Optional[Dict[str, Any]] = None,
) -> str:
    step_titles = get_step_titles(prompt_profile)
    lines: List[str] = []
    lines.append(f"[{step_titles[0]}]")
    lines.append(get_task_label(trace_row, "decompose"))

    lines.append(f"[{step_titles[1]}]")
    for role in ROLE_ORDER:
        lines.append(get_task_label(trace_row, role))

    lines.append(f"[{step_titles[2]}]")
    lines.append(get_task_label(trace_row, "conflict"))

    lines.append(f"[{step_titles[3]}]")
    lines.append(get_task_label(trace_row, "rectification"))

    lines.append(f"[{step_titles[4]}]")
    lines.append(get_task_label(trace_row, "final", final_tuple_override=final_tuple_override))
    return "\n".join(lines)


def build_natural_think_chain(
    trace_row: Dict[str, Any],
    prompt_profile: Dict[str, Any],
    final_tuple_override: Optional[Dict[str, Any]] = None,
    lexicon_evidence: Optional[Sequence[Dict[str, Any]]] = None,
    chain_style: str = "danceha_natural",
) -> str:
    use_zh = is_trace_zh(prompt_profile, trace_row)
    softspan_mode = chain_style == "danceha_compact_softspan"
    cuepreserve_mode = chain_style == "danceha_natural_cuepreserve"
    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {}
    step_b = trace_row.get("step_b_role_hypotheses") if isinstance(trace_row.get("step_b_role_hypotheses"), list) else []
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    step_d = trace_row.get("step_d_rectification_actions") if isinstance(trace_row.get("step_d_rectification_actions"), list) else []
    step_e = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else {}
    final_tuple = normalize_final_tuple(final_tuple_override if isinstance(final_tuple_override, dict) else step_e.get("final_tuple"))
    lexicon_hits = normalize_lexicon_hits(lexicon_evidence)

    group_id = normalize_string(trace_row.get("group_id")) or ""
    focus_text = normalize_reasoning_snippet(step_a.get("focus_text"), max_chars=120)
    group_text = normalize_reasoning_snippet(step_a.get("group_text"), max_chars=120)
    canonical_target = normalize_string(step_a.get("canonical_target")) or "IMPLICIT_TARGET"
    escalation = bool(step_a.get("escalation_flag"))
    focus_anchor = strip_focus_tags(focus_text) or strip_focus_tags(group_text) or canonical_target
    final_target_anchor = format_span_anchor(final_tuple.get("target"), use_zh=use_zh)
    final_argument_anchor = format_span_anchor(final_tuple.get("argument"), use_zh=use_zh)

    lines: List[str] = []
    if softspan_mode:
        if use_zh:
            lines.append(
                f"step1：当前 focus/group 已固定，canonical_target={format_span_anchor(canonical_target, use_zh=True)}"
                + ("，存在 escalation 信号。" if escalation else "，本轮不再继续拆分。")
            )
        else:
            lines.append(
                f"step1: The current focus/group unit is fixed, with canonical_target={format_span_anchor(canonical_target, use_zh=False)}"
                + (", and escalation=True." if escalation else ", so I do not split it further.")
            )
    else:
        if use_zh:
            lines.append(
                f"step1：当前只处理这个 focus/group 单元：focus={format_span_anchor(focus_anchor, use_zh=True)}，"
                f"canonical_target={format_span_anchor(canonical_target, use_zh=True)}"
                + ("，存在 escalation 信号。" if escalation else "，这一轮不再继续拆分。")
            )
        else:
            lines.append(
                f"step1: I only handle this focus/group unit: focus={format_span_anchor(focus_anchor, use_zh=False)}, "
                f"canonical_target={format_span_anchor(canonical_target, use_zh=False)}"
                + (", with escalation=True." if escalation else ", so I do not split it further.")
            )

    role_lookup = {
        str(item.get("role") or "").lower(): item
        for item in step_b
        if isinstance(item, dict) and normalize_string(item.get("role")) is not None
    }
    role_sentences: List[str] = []
    for role in ROLE_ORDER:
        role_item = role_lookup.get(role)
        if role_item is None:
            if use_zh:
                role_sentences.append(f"{get_role_display_name(role, use_zh)}=missing")
            else:
                role_sentences.append(f"{get_role_display_name(role, use_zh)}=missing")
            continue
        tuple_obj = role_item.get("tuple") if isinstance(role_item.get("tuple"), dict) else {}
        rationale = normalize_reasoning_snippet(role_item.get("rationale"), max_chars=40 if softspan_mode else 56)
        tuple_summary = format_compact_tuple_anchor(tuple_obj, use_zh=use_zh)
        flag_summary = format_role_flags(role_item, use_zh=use_zh)
        if use_zh:
            flag_part = f"；{flag_summary}" if flag_summary else ""
            role_sentences.append(
                f"{get_role_display_name(role, use_zh)}[{tuple_summary}{flag_part}；why={rationale or '无'}]"
            )
        else:
            flag_part = f"; {flag_summary}" if flag_summary else ""
            role_sentences.append(
                f"{get_role_display_name(role, use_zh)}[{tuple_summary}{flag_part}; why={rationale or 'none'}]"
            )

    if use_zh:
        lines.append("step2：三角色候选只保留会影响最终裁决的字段：")
    else:
        lines.append("step2: I keep only the role evidence that can change the final tuple:")
    lines.append(" | ".join(role_sentences))

    high_risk_fields = list(step_c.get("high_risk_fields") or [])
    error_tags = list(step_c.get("error_tags") or [])
    overall_conflict_score = round(clamp(safe_float(step_c.get("overall_conflict_score"), default=0.0), 0.0, 1.0), 4)
    if use_zh:
        conflict_line = (
            f"step3：auditor 只重点看 [{format_list_for_chain(high_risk_fields, '无')}]；"
            f"conflict_score={overall_conflict_score}"
        )
        if error_tags:
            conflict_line += f"；error_tags=[{format_list_for_chain(error_tags, '无')}]"
        conflict_line += "。"
        lines.append(conflict_line)
    else:
        conflict_line = (
            f"step3: As auditor, I only focus on [{format_list_for_chain(high_risk_fields, 'none')}]; "
            f"conflict_score={overall_conflict_score}"
        )
        if error_tags:
            conflict_line += f"; error_tags=[{format_list_for_chain(error_tags, 'none')}]"
        conflict_line += "."
        lines.append(conflict_line)

    action_bits: List[str] = []
    for action in step_d:
        if not isinstance(action, dict):
            continue
        role_name = normalize_string(action.get("role")) or "unknown"
        decision_action = normalize_string(action.get("decision_action")) or "keep_candidate"
        decision_reason = normalize_reasoning_snippet(action.get("decision_reason"), max_chars=120)
        if use_zh:
            text = f"{role_name}->{decision_action}"
            if decision_reason:
                text += f"（{decision_reason}）"
        else:
            text = f"{role_name}->{decision_action}"
            if decision_reason:
                text += f" ({decision_reason})"
        action_bits.append(text)
    if use_zh:
        action_sentence = "；".join(action_bits) if action_bits else "没有额外修正动作"
        if softspan_mode:
            lines.append(
                "step4：结合 rectification actions，我保留更贴近原文、范围更窄的 target/argument span，"
                f"不沿用更长的解释性片段；动作是：{action_sentence}。"
            )
        elif cuepreserve_mode:
            lines.append(
                f"step4：先做 span anchoring：target 回贴 {final_target_anchor}，argument 回贴 {final_argument_anchor}；"
                f"其中 target 只保留被指向对象，argument 只保留攻击性属性/动作/评价，不把 target 机械重复进 argument；"
                f"再应用动作 {action_sentence}。"
            )
        else:
            lines.append(
                f"step4：先做 span anchoring：target 回贴 {final_target_anchor}，argument 回贴 {final_argument_anchor}；"
                f"再应用动作 {action_sentence}。"
            )
        if lexicon_hits:
            lex_bits = [
                f"{item['term']}({item['category']})"
                for item in lexicon_hits
            ]
            lines.append("lexicon 只作辅助参考：" + "；".join(lex_bits) + "。")
    else:
        action_sentence = "; ".join(action_bits) if action_bits else "no extra rectification actions"
        if softspan_mode:
            lines.append(
                "step4: After rectification, I keep the narrower source-grounded target/argument spans "
                f"instead of longer explanatory phrases; actions: {action_sentence}."
            )
        elif cuepreserve_mode:
            lines.append(
                f"step4: I first anchor spans back to the source: target={final_target_anchor}, argument={final_argument_anchor}; "
                "target keeps only the attacked entity/group span, while argument keeps the offensive cue, attribute, action, or predicate, "
                f"so I do not mechanically duplicate target inside argument; then I apply {action_sentence}."
            )
        else:
            lines.append(
                f"step4: I first anchor spans back to the source: target={final_target_anchor}, argument={final_argument_anchor}; "
                f"then I apply {action_sentence}."
            )
        if lexicon_hits:
            lex_bits = [
                f"{item['term']}({item['category']})"
                for item in lexicon_hits
            ]
            lines.append("lexicon is auxiliary only: " + "; ".join(lex_bits) + ".")

    final_tuple_summary = format_tuple_summary(final_tuple, use_zh=use_zh)
    if use_zh:
        lines.append(f"step5：最终裁决：{final_tuple_summary}。")
    else:
        lines.append(f"step5: Final arbitration: {final_tuple_summary}.")
    return "\n".join(lines)


def compact_role_payload(role_payload: Dict[str, Any]) -> Dict[str, Any]:
    tuple_obj = role_payload.get("tuple") if isinstance(role_payload.get("tuple"), dict) else {}
    return {
        "role": str(role_payload.get("role") or "unknown"),
        "role_status": str(role_payload.get("role_status") or "original"),
        "candidate_count": int(role_payload.get("candidate_count") or 1),
        "tuple": {
            "target": normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET",
            "argument": normalize_string(tuple_obj.get("argument")),
            "group": normalize_string(tuple_obj.get("group")) or "unknown",
            "harmful": tuple_obj.get("harmful"),
            "harmful_direction": normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful")),
        },
        "quality_score": round(clamp(safe_float(role_payload.get("quality_score"), default=0.0), 0.0, 1.0), 4),
        "issues": list(role_payload.get("issues") or []),
        "uncertainty_flag": bool(role_payload.get("uncertainty_flag")),
        "escalation_flag": bool(role_payload.get("escalation_flag")),
        "lexicon_hit": bool(role_payload.get("lexicon_hit")),
        "rationale": normalize_string(role_payload.get("rationale")) or "",
    }


def compact_rectification_payload(action: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "role": str(action.get("role") or "unknown"),
        "decision_action": str(action.get("decision_action") or "keep_candidate"),
        "decision_reason": normalize_string(action.get("decision_reason")) or "",
        "is_revised": bool(action.get("is_revised")),
        "original_predicted_tuple": action.get("original_predicted_tuple") if isinstance(action.get("original_predicted_tuple"), dict) else {},
        "rectified_tuple": action.get("rectified_tuple") if isinstance(action.get("rectified_tuple"), dict) else {},
    }


def is_profile_zh(task_profile: str, prompt_profile: Dict[str, Any]) -> bool:
    languages = prompt_profile.get("match_languages")
    if isinstance(languages, list):
        lowered = {str(item).strip().lower() for item in languages if str(item).strip()}
        if "zh" in lowered:
            return True
    return task_profile.lower().startswith("zh")


def resolve_task_profile(trace_row: Dict[str, Any], prompt_profiles: Dict[str, Dict[str, Any]]) -> str:
    if not prompt_profiles:
        raise ValueError("Prompt profiles cannot be empty.")

    language = (normalize_string(trace_row.get("language")) or "").lower()
    source = (normalize_string(trace_row.get("source")) or "").lower()

    for name, profile in prompt_profiles.items():
        match_languages = profile.get("match_languages")
        match_sources = profile.get("match_sources")
        language_ok = True
        source_ok = True

        if isinstance(match_languages, list) and match_languages:
            language_ok = language in {str(item).strip().lower() for item in match_languages if str(item).strip()}
        if isinstance(match_sources, list) and match_sources:
            source_ok = source in {str(item).strip().lower() for item in match_sources if str(item).strip()}

        if language_ok and source_ok:
            return name

    if language == "zh":
        for name, profile in prompt_profiles.items():
            if is_profile_zh(name, profile):
                return name
    if language == "en":
        for name, profile in prompt_profiles.items():
            if not is_profile_zh(name, profile):
                return name

    return next(iter(prompt_profiles.keys()))


def extract_zh_lexicon_evidence(
    trace_row: Dict[str, Any],
    sample_row: Optional[Dict[str, Any]],
    task_profile: str,
    prompt_profile: Dict[str, Any],
    lexicon_terms: Sequence[Dict[str, str]],
    max_hits: int,
) -> List[Dict[str, Any]]:
    if max_hits <= 0 or not lexicon_terms or not is_profile_zh(task_profile, prompt_profile):
        return []

    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {}
    step_b = trace_row.get("step_b_role_hypotheses") if isinstance(trace_row.get("step_b_role_hypotheses"), list) else []
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    signals = step_c.get("signals") if isinstance(step_c.get("signals"), dict) else {}
    lexicon_signal = parse_bool(signals.get("lexicon_hit"), default=False) or any(
        bool(item.get("lexicon_hit")) for item in step_b if isinstance(item, dict)
    )

    field_texts = {
        "source_text": strip_focus_tags((sample_row or {}).get("text")),
        "focus_text": strip_focus_tags(step_a.get("focus_text")),
        "group_text": strip_focus_tags(step_a.get("group_text")),
        "canonical_target": strip_focus_tags(step_a.get("canonical_target")),
    }

    hits: List[Dict[str, Any]] = []
    for lex_item in lexicon_terms:
        term = lex_item.get("term", "")
        if not term:
            continue
        matched_fields = [field for field, text in field_texts.items() if text and term in text]
        if not matched_fields:
            continue
        hits.append(
            {
                "term": term,
                "category": lex_item.get("category", "unknown"),
                "definition": lex_item.get("definition", ""),
                "matched_fields": matched_fields,
                "lexicon_signal": lexicon_signal,
            }
        )
        if len(hits) >= max_hits:
            break

    return hits


def tuples_equal_on_field(left: Dict[str, Any], right: Dict[str, Any], field: str) -> bool:
    return normalize_text_for_match(left.get(field)) == normalize_text_for_match(right.get(field))


def compute_profile_alignment(
    task_profile: str,
    prompt_profile: Dict[str, Any],
    teacher_final_tuple: Dict[str, Any],
    primary_ground_truth: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(primary_ground_truth, dict):
        return {
            "has_primary_ground_truth": False,
            "same_label": False,
            "same_target": False,
            "same_argument": False,
            "same_group": False,
            "clean": False,
            "silver": False,
        }

    teacher_tuple = normalize_final_tuple(teacher_final_tuple)
    gt_tuple = normalize_final_tuple(primary_ground_truth)

    same_label = normalize_harmful_direction(teacher_tuple.get("harmful_direction"), teacher_tuple.get("harmful")) == normalize_harmful_direction(
        gt_tuple.get("harmful_direction"),
        gt_tuple.get("harmful"),
    )
    same_target = tuples_equal_on_field(teacher_tuple, gt_tuple, "target")
    same_argument = tuples_equal_on_field(teacher_tuple, gt_tuple, "argument")
    same_group = tuples_equal_on_field(teacher_tuple, gt_tuple, "group")

    if is_profile_zh(task_profile, prompt_profile):
        clean = same_label and same_target and same_argument and same_group
        silver = same_label and same_group and (same_target or same_argument)
    else:
        clean = same_label and same_target and same_argument
        silver = same_label and (same_target or same_argument)

    return {
        "has_primary_ground_truth": True,
        "same_label": same_label,
        "same_target": same_target,
        "same_argument": same_argument,
        "same_group": same_group,
        "clean": clean,
        "silver": silver,
    }


def normalize_phase0_like_prediction_list(selector_row: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not isinstance(selector_row, dict):
        return []
    predictions = selector_row.get("predictions")
    if not isinstance(predictions, list):
        return []

    rows: List[Dict[str, Any]] = []
    for item in predictions:
        if not isinstance(item, dict):
            continue
        harmful = item.get("hateful") if "hateful" in item else item.get("harmful")
        rows.append(
            {
                "target": normalize_string(item.get("target")) or "IMPLICIT_TARGET",
                "argument": normalize_string(item.get("argument")),
                "group": normalize_string(item.get("group")) or "unknown",
                "harmful": direction_to_harmful_binary(normalize_harmful_direction(item.get("harmful_direction"), harmful)),
                "harmful_direction": normalize_harmful_direction(item.get("harmful_direction"), harmful),
            }
        )
    return rows


def compute_external_selector_alignment(
    task_profile: str,
    prompt_profile: Dict[str, Any],
    selector_row: Optional[Dict[str, Any]],
    primary_ground_truth: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if not isinstance(primary_ground_truth, dict):
        return {
            "has_primary_ground_truth": False,
            "selector_row_found": isinstance(selector_row, dict),
            "prediction_count": len(normalize_phase0_like_prediction_list(selector_row)),
            "clean": False,
            "silver": False,
        }

    normalized_predictions = normalize_phase0_like_prediction_list(selector_row)
    if not normalized_predictions:
        return {
            "has_primary_ground_truth": True,
            "selector_row_found": isinstance(selector_row, dict),
            "prediction_count": 0,
            "clean": False,
            "silver": False,
        }

    clean = False
    silver = False
    for pred in normalized_predictions:
        alignment = compute_profile_alignment(
            task_profile=task_profile,
            prompt_profile=prompt_profile,
            teacher_final_tuple=pred,
            primary_ground_truth=primary_ground_truth,
        )
        clean = clean or bool(alignment.get("clean"))
        silver = silver or bool(alignment.get("silver"))
        if clean and silver:
            break

    return {
        "has_primary_ground_truth": True,
        "selector_row_found": isinstance(selector_row, dict),
        "prediction_count": len(normalized_predictions),
        "clean": clean,
        "silver": silver,
    }


def collect_grounding_texts(trace_row: Optional[Dict[str, Any]], sample_row: Optional[Dict[str, Any]]) -> List[str]:
    step_a = trace_row.get("step_a_decompose") if isinstance((trace_row or {}).get("step_a_decompose"), dict) else {}
    values = [
        step_a.get("group_text"),
        step_a.get("focus_text"),
        step_a.get("canonical_target"),
        (sample_row or {}).get("text"),
    ]
    grounded: List[str] = []
    for value in values:
        normalized = normalize_text_for_match(value)
        if normalized:
            grounded.append(normalized)
    return grounded


def tuple_value_grounded_in_text(value: Any, texts: Sequence[str], allow_empty: bool = False) -> bool:
    normalized = normalize_text_for_match(value)
    if not normalized:
        return bool(allow_empty)
    if normalized == normalize_text_for_match("IMPLICIT_TARGET"):
        return True
    return any(normalized in text for text in texts if text)


def find_selected_cluster(trace_row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(trace_row, dict):
        return None
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    clusters = step_c.get("clusters") if isinstance(step_c.get("clusters"), list) else []
    if not clusters:
        return None
    step_e = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else {}
    consensus_trace = step_e.get("consensus_trace") if isinstance(step_e.get("consensus_trace"), dict) else {}
    selected_cluster_id = normalize_string(consensus_trace.get("selected_cluster_id"))
    if selected_cluster_id is None:
        return clusters[0] if len(clusters) == 1 and isinstance(clusters[0], dict) else None
    for cluster in clusters:
        if isinstance(cluster, dict) and normalize_string(cluster.get("cluster_id")) == selected_cluster_id:
            return cluster
    return None


def compute_teacher_trace_quality(
    trace_row: Optional[Dict[str, Any]],
    sample_row: Optional[Dict[str, Any]],
    teacher_final_tuple: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_teacher = normalize_final_tuple(teacher_final_tuple)
    texts = collect_grounding_texts(trace_row, sample_row)
    selected_cluster = find_selected_cluster(trace_row)
    support_roles = (
        selected_cluster.get("support_roles")
        if isinstance(selected_cluster, dict) and isinstance(selected_cluster.get("support_roles"), list)
        else []
    )
    step_e = trace_row.get("step_e_final_decision") if isinstance((trace_row or {}).get("step_e_final_decision"), dict) else {}
    audit_confidence = round(clamp(safe_float(step_e.get("audit_confidence"), default=0.0), 0.0, 1.0), 4)
    target_grounded = tuple_value_grounded_in_text(normalized_teacher.get("target"), texts, allow_empty=False)
    argument_grounded = tuple_value_grounded_in_text(normalized_teacher.get("argument"), texts, allow_empty=True)
    return {
        "audit_confidence": audit_confidence,
        "selected_cluster_id": normalize_string((selected_cluster or {}).get("cluster_id")) or "",
        "support_role_count": len([role for role in support_roles if normalize_string(role)]),
        "support_roles": [str(role) for role in support_roles if normalize_string(role)],
        "target_grounded": bool(target_grounded),
        "argument_grounded": bool(argument_grounded),
        "tuple_grounded": bool(target_grounded and argument_grounded),
    }


def resolve_prompt_override_text(
    variant_cfg: Dict[str, Any],
    scalar_key: str,
    mapping_key: str,
    task_profile: Optional[str],
) -> Optional[str]:
    mapping = variant_cfg.get(mapping_key)
    if task_profile is not None and isinstance(mapping, dict):
        mapped_value = normalize_string(mapping.get(task_profile))
        if mapped_value is not None:
            return mapped_value
    return normalize_string(variant_cfg.get(scalar_key))


def apply_variant_prompt_overrides(
    prompt_profile: Dict[str, Any],
    variant_cfg: Dict[str, Any],
    task_profile: Optional[str] = None,
) -> Dict[str, Any]:
    profile = dict(prompt_profile)
    assistant_chain_style = normalize_string(variant_cfg.get("assistant_chain_style"))
    if assistant_chain_style is not None:
        profile["assistant_chain_style"] = assistant_chain_style

    system_prompt_append = resolve_prompt_override_text(
        variant_cfg=variant_cfg,
        scalar_key="system_prompt_append",
        mapping_key="system_prompt_append_by_task_profile",
        task_profile=task_profile,
    )
    if system_prompt_append:
        base = normalize_string(profile.get("system_prompt")) or ""
        profile["system_prompt"] = (base + "\n" + system_prompt_append).strip()

    user_instruction_append = resolve_prompt_override_text(
        variant_cfg=variant_cfg,
        scalar_key="user_instruction_append",
        mapping_key="user_instruction_append_by_task_profile",
        task_profile=task_profile,
    )
    if user_instruction_append:
        base = normalize_string(profile.get("user_instruction")) or ""
        profile["user_instruction"] = (base + "\n" + user_instruction_append).strip()
    return profile


def build_user_prompt(
    trace_row: Dict[str, Any],
    prompt_profile: Dict[str, Any],
    task_profile: str,
    sample_row: Optional[Dict[str, Any]] = None,
    matched_ground_truth: Optional[Sequence[Dict[str, Any]]] = None,
    primary_ground_truth: Optional[Dict[str, Any]] = None,
    lexicon_evidence: Optional[Sequence[Dict[str, Any]]] = None,
    include_source_text: bool = True,
    include_ground_truth: bool = False,
) -> str:
    role_payloads = trace_row.get("step_b_role_hypotheses") if isinstance(trace_row.get("step_b_role_hypotheses"), list) else []
    action_payloads = trace_row.get("step_d_rectification_actions") if isinstance(trace_row.get("step_d_rectification_actions"), list) else []

    compact_input: Dict[str, Any] = {
        "sample_id": str(trace_row.get("sample_id") or ""),
        "group_id": str(trace_row.get("group_id") or ""),
        "source": str(trace_row.get("source") or "unknown"),
        "split": str(trace_row.get("split") or "train"),
        "language": str(trace_row.get("language") or "unknown"),
        "task_profile": task_profile,
        "decompose": trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {},
        "role_hypotheses": [compact_role_payload(x) for x in role_payloads if isinstance(x, dict)],
        "conflict_diagnosis": trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {},
        "rectification_actions": [compact_rectification_payload(x) for x in action_payloads if isinstance(x, dict)],
        "lexicon_evidence": [dict(item) for item in (lexicon_evidence or []) if isinstance(item, dict)],
    }

    if include_source_text and isinstance(sample_row, dict):
        compact_input["source_text"] = normalize_string(sample_row.get("text")) or ""

    if include_ground_truth:
        gt_rows = [x for x in (matched_ground_truth or []) if isinstance(x, dict)]
        compact_input["matched_ground_truth"] = gt_rows
        if isinstance(primary_ground_truth, dict):
            compact_input["primary_ground_truth"] = primary_ground_truth

    schema = {
        "target": "string",
        "argument": "string_or_null",
        "group": "string_or_unknown",
        "harmful": "1_or_0_or_null",
        "harmful_direction": "harmful|non-harmful|unknown",
    }

    instruction = normalize_string(prompt_profile.get("user_instruction")) or ""
    package_label = normalize_string(prompt_profile.get("package_label")) or "Input evidence package"
    schema_label = normalize_string(prompt_profile.get("schema_label")) or "Required JSON schema"

    return (
        instruction
        + "\n\n"
        + package_label
        + ":\n"
        + json.dumps(compact_input, ensure_ascii=False, indent=2)
        + "\n\n"
        + schema_label
        + ":\n"
        + json.dumps(schema, ensure_ascii=False, indent=2)
    )


def build_assistant_content(
    trace_row: Dict[str, Any],
    prompt_profile: Dict[str, Any],
    include_think_tags: bool,
    final_tuple_override: Optional[Dict[str, Any]] = None,
    lexicon_evidence: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    chain_style = resolve_assistant_chain_style(prompt_profile)
    if chain_style == "danceha_natural":
        think_chain = build_natural_think_chain(
            trace_row,
            prompt_profile=prompt_profile,
            final_tuple_override=final_tuple_override,
            lexicon_evidence=lexicon_evidence,
            chain_style=chain_style,
        )
    elif chain_style == "danceha_natural_cuepreserve":
        think_chain = build_natural_think_chain(
            trace_row,
            prompt_profile=prompt_profile,
            final_tuple_override=final_tuple_override,
            lexicon_evidence=lexicon_evidence,
            chain_style=chain_style,
        )
    elif chain_style == "danceha_compact_softspan":
        think_chain = build_natural_think_chain(
            trace_row,
            prompt_profile=prompt_profile,
            final_tuple_override=final_tuple_override,
            lexicon_evidence=lexicon_evidence,
            chain_style=chain_style,
        )
    else:
        think_chain = build_think_chain(trace_row, prompt_profile=prompt_profile, final_tuple_override=final_tuple_override)
    final_block = trace_row.get("step_e_final_decision") if isinstance(trace_row.get("step_e_final_decision"), dict) else {}
    final_tuple = normalize_final_tuple(final_tuple_override if isinstance(final_tuple_override, dict) else final_block.get("final_tuple"))
    final_json = json.dumps(final_tuple, ensure_ascii=False)

    if include_think_tags:
        return "<think>\n" + think_chain + "\n</think>\n" + final_json
    return final_json


def extract_error_tags(trace_row: Dict[str, Any]) -> List[str]:
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}
    tags = step_c.get("error_tags") if isinstance(step_c.get("error_tags"), list) else []
    return [str(x) for x in tags]


def is_hard_case(trace_row: Dict[str, Any], threshold: float) -> bool:
    step_a = trace_row.get("step_a_decompose") if isinstance(trace_row.get("step_a_decompose"), dict) else {}
    step_b = trace_row.get("step_b_role_hypotheses") if isinstance(trace_row.get("step_b_role_hypotheses"), list) else []
    step_c = trace_row.get("step_c_conflict_diagnosis") if isinstance(trace_row.get("step_c_conflict_diagnosis"), dict) else {}

    if bool(step_a.get("escalation_flag")):
        return True

    for role_payload in step_b:
        if not isinstance(role_payload, dict):
            continue
        if bool(role_payload.get("uncertainty_flag")) or bool(role_payload.get("escalation_flag")):
            return True

    if safe_float(step_c.get("overall_conflict_score"), default=0.0) >= threshold:
        return True

    high_risk_fields = step_c.get("high_risk_fields") if isinstance(step_c.get("high_risk_fields"), list) else []
    if high_risk_fields:
        return True

    tags = set(extract_error_tags(trace_row))
    if CULTURAL_ERROR_TAG in tags:
        return True

    return False


def build_base_dataset_rows(
    trace_rows: Sequence[Dict[str, Any]],
    sample_index: Dict[str, Dict[str, Any]],
    prompt_profiles: Dict[str, Dict[str, Any]],
    quality_selector_indices: Dict[str, Dict[str, Dict[str, Any]]],
    include_lexicon_evidence: bool,
    zh_lexicon_terms: Sequence[Dict[str, str]],
    zh_lexicon_max_hits: int,
    hard_case_only: bool,
    hard_case_threshold: float,
    allowed_splits: Set[str],
    max_samples: Optional[int],
    ground_truth_match_min_score: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    rows: List[Dict[str, Any]] = []
    drop_stats = {
        "drop_invalid_split": 0,
        "drop_missing_ids": 0,
        "drop_hard_case_filter": 0,
        "drop_duplicate_decision_id": 0,
    }
    seen_decision_ids: Set[str] = set()

    for raw_trace in trace_rows:
        trace = normalize_trace_row(raw_trace)
        split = str(trace.get("split") or "train")
        if split not in allowed_splits:
            drop_stats["drop_invalid_split"] += 1
            continue

        decision_id = normalize_string(trace.get("decision_id"))
        sample_id = normalize_string(trace.get("sample_id"))
        group_id = normalize_string(trace.get("group_id"))
        if decision_id is None or sample_id is None or group_id is None:
            drop_stats["drop_missing_ids"] += 1
            continue

        if decision_id in seen_decision_ids:
            drop_stats["drop_duplicate_decision_id"] += 1
            continue
        seen_decision_ids.add(decision_id)

        hard_case_flag = is_hard_case(trace, threshold=hard_case_threshold)
        if hard_case_only and not hard_case_flag:
            drop_stats["drop_hard_case_filter"] += 1
            continue

        sample_row, matched_ground_truth, primary_ground_truth = attach_sample_context(
            trace_row=trace,
            sample_index=sample_index,
            min_score=ground_truth_match_min_score,
        )

        step_e = trace.get("step_e_final_decision") if isinstance(trace.get("step_e_final_decision"), dict) else {}
        teacher_final_tuple = normalize_final_tuple(step_e.get("final_tuple"))
        task_profile = resolve_task_profile(trace, prompt_profiles=prompt_profiles)
        prompt_profile = prompt_profiles[task_profile]
        language_key = (normalize_string(trace.get("language")) or "").lower()
        selector_row = quality_selector_indices.get(language_key, {}).get(sample_id)
        lexicon_evidence = (
            extract_zh_lexicon_evidence(
                trace_row=trace,
                sample_row=sample_row,
                task_profile=task_profile,
                prompt_profile=prompt_profile,
                lexicon_terms=zh_lexicon_terms,
                max_hits=zh_lexicon_max_hits,
            )
            if include_lexicon_evidence
            else []
        )
        alignment = compute_profile_alignment(
            task_profile=task_profile,
            prompt_profile=prompt_profile,
            teacher_final_tuple=teacher_final_tuple,
            primary_ground_truth=primary_ground_truth,
        )
        external_selector_alignment = compute_external_selector_alignment(
            task_profile=task_profile,
            prompt_profile=prompt_profile,
            selector_row=selector_row,
            primary_ground_truth=primary_ground_truth,
        )
        teacher_trace_quality = compute_teacher_trace_quality(
            trace_row=trace,
            sample_row=sample_row,
            teacher_final_tuple=teacher_final_tuple,
        )

        item = {
            "sample_id": sample_id,
            "group_id": group_id,
            "decision_id": decision_id,
            "source": str(trace.get("source") or "unknown"),
            "split": split,
            "language": str(trace.get("language") or "unknown"),
            "task_profile": task_profile,
            "hard_case": bool(hard_case_flag),
            "source_text": normalize_string((sample_row or {}).get("text")),
            "ground_truth": normalize_ground_truth_list(sample_row),
            "group_ground_truth_candidates": matched_ground_truth,
            "group_ground_truth_primary": primary_ground_truth,
            "teacher_final_tuple": teacher_final_tuple,
            "teacher_rationale": normalize_string(step_e.get("fused_rationale")),
            "lexicon_evidence": lexicon_evidence,
            "teacher_matches_primary_gt_clean": bool(alignment.get("clean")),
            "teacher_matches_primary_gt_silver": bool(alignment.get("silver")),
            "external_selector_matches_primary_gt_clean": bool(external_selector_alignment.get("clean")),
            "external_selector_matches_primary_gt_silver": bool(external_selector_alignment.get("silver")),
            "metadata": {
                "error_tags": extract_error_tags(trace),
                "overall_conflict_score": safe_float(
                    ((trace.get("step_c_conflict_diagnosis") or {}).get("overall_conflict_score")),
                    default=0.0,
                ),
                "focus_text": normalize_string(((trace.get("step_a_decompose") or {}).get("focus_text"))),
                "canonical_target": normalize_string(((trace.get("step_a_decompose") or {}).get("canonical_target"))),
                "consensus_trace": step_e.get("consensus_trace") if isinstance(step_e.get("consensus_trace"), dict) else {},
                "audit_confidence": teacher_trace_quality.get("audit_confidence"),
                "teacher_primary_gt_alignment": alignment,
                "external_selector_alignment": external_selector_alignment,
                "matched_ground_truth_count": len(matched_ground_truth),
                "lexicon_signal": bool(lexicon_evidence),
                "teacher_trace_quality": teacher_trace_quality,
            },
            "_trace_row": trace,
            "_sample_row": sample_row,
        }
        rows.append(item)

        if isinstance(max_samples, int) and max_samples > 0 and len(rows) >= max_samples:
            break

    return rows, drop_stats


def row_matches_variant(base_row: Dict[str, Any], variant_cfg: Dict[str, Any]) -> bool:
    supervision_target = normalize_string(variant_cfg.get("supervision_target")) or "teacher"
    filter_mode = normalize_string(variant_cfg.get("filter_mode")) or "all"
    alignment = (base_row.get("metadata") or {}).get("teacher_primary_gt_alignment")
    alignment = alignment if isinstance(alignment, dict) else {}
    external_selector_alignment = (base_row.get("metadata") or {}).get("external_selector_alignment")
    external_selector_alignment = external_selector_alignment if isinstance(external_selector_alignment, dict) else {}

    if supervision_target == "ground_truth_primary" and not alignment.get("has_primary_ground_truth"):
        return False

    if filter_mode == "all":
        return True
    if filter_mode == "profile_clean":
        return bool(alignment.get("clean"))
    if filter_mode == "profile_silver":
        return bool(alignment.get("silver"))
    if filter_mode == "external_selector_clean":
        return bool(external_selector_alignment.get("clean"))
    if filter_mode == "external_selector_silver":
        return bool(external_selector_alignment.get("silver"))
    if filter_mode == "teacher_trace_hq":
        quality = (base_row.get("metadata") or {}).get("teacher_trace_quality")
        quality = quality if isinstance(quality, dict) else {}
        language = (normalize_string(base_row.get("language")) or "").lower()
        conflict_score = safe_float((base_row.get("metadata") or {}).get("overall_conflict_score"), default=1.0)
        error_tags = {str(tag) for tag in ((base_row.get("metadata") or {}).get("error_tags") or [])}

        min_audit_confidence = safe_float(variant_cfg.get("teacher_min_audit_confidence"), default=0.8)
        min_support_roles = int(variant_cfg.get("teacher_min_support_roles") or 2)
        require_grounded_target = parse_bool(variant_cfg.get("teacher_require_grounded_target"), default=True)
        require_grounded_argument = parse_bool(variant_cfg.get("teacher_require_grounded_argument"), default=True)
        require_tuple_grounded = parse_bool(variant_cfg.get("teacher_require_tuple_grounded"), default=False)

        exclude_error_tags = {str(tag) for tag in (variant_cfg.get("teacher_exclude_error_tags") or [])}
        if error_tags & exclude_error_tags:
            return False

        max_conflict = safe_float(variant_cfg.get("teacher_max_conflict_score"), default=0.67)
        max_conflict_by_language = variant_cfg.get("teacher_max_conflict_by_language")
        if isinstance(max_conflict_by_language, dict):
            language_override = max_conflict_by_language.get(language)
            if language_override is not None:
                max_conflict = safe_float(language_override, default=max_conflict)

        if safe_float(quality.get("audit_confidence"), default=0.0) < min_audit_confidence:
            return False
        if int(quality.get("support_role_count") or 0) < min_support_roles:
            return False
        if require_grounded_target and not bool(quality.get("target_grounded")):
            return False
        if require_grounded_argument and not bool(quality.get("argument_grounded")):
            return False
        if require_tuple_grounded and not bool(quality.get("tuple_grounded")):
            return False
        return conflict_score < max_conflict
    if filter_mode == "selector_silver_teacher_bridge":
        if bool(external_selector_alignment.get("silver")):
            return True
        language = (normalize_string(base_row.get("language")) or "").lower()
        conflict_score = safe_float((base_row.get("metadata") or {}).get("overall_conflict_score"), default=1.0)
        error_tags = {str(tag) for tag in ((base_row.get("metadata") or {}).get("error_tags") or [])}
        require_same_label = parse_bool(variant_cfg.get("bridge_require_same_label"), default=True)
        if require_same_label and not bool(alignment.get("same_label")):
            return False

        require_any_fields = variant_cfg.get("bridge_require_any_teacher_fields")
        if isinstance(require_any_fields, list) and require_any_fields:
            if not any(bool(alignment.get(str(field))) for field in require_any_fields):
                return False

        exclude_error_tags = {str(tag) for tag in (variant_cfg.get("bridge_exclude_error_tags") or [])}
        if error_tags & exclude_error_tags:
            return False

        max_conflict = safe_float(variant_cfg.get("bridge_max_conflict_score"), default=0.67)
        max_conflict_by_language = variant_cfg.get("bridge_max_conflict_by_language")
        if isinstance(max_conflict_by_language, dict):
            language_override = max_conflict_by_language.get(language)
            if language_override is not None:
                max_conflict = safe_float(language_override, default=max_conflict)
        return conflict_score < max_conflict
    raise ValueError(f"Unsupported dataset variant filter_mode: {filter_mode}")


def render_variant_row(
    base_row: Dict[str, Any],
    variant_cfg: Dict[str, Any],
    prompt_profiles: Dict[str, Dict[str, Any]],
    include_think_tags: bool,
    include_source_text_in_user_prompt: bool,
    include_ground_truth_in_user_prompt: bool,
) -> Dict[str, Any]:
    task_profile = str(base_row.get("task_profile") or "")
    prompt_profile = apply_variant_prompt_overrides(prompt_profiles[task_profile], variant_cfg, task_profile=task_profile)
    trace_row = base_row.get("_trace_row") if isinstance(base_row.get("_trace_row"), dict) else {}
    sample_row = base_row.get("_sample_row") if isinstance(base_row.get("_sample_row"), dict) else None
    row_include_think_tags = parse_bool(variant_cfg.get("include_think_tags"), default=include_think_tags)

    supervision_target = normalize_string(variant_cfg.get("supervision_target")) or "teacher"
    if supervision_target == "ground_truth_primary":
        supervision_final_tuple = normalize_final_tuple(base_row.get("group_ground_truth_primary"))
    else:
        supervision_final_tuple = normalize_final_tuple(base_row.get("teacher_final_tuple"))

    user_prompt = build_user_prompt(
        trace_row=trace_row,
        prompt_profile=prompt_profile,
        task_profile=task_profile,
        sample_row=sample_row,
        matched_ground_truth=base_row.get("group_ground_truth_candidates"),
        primary_ground_truth=base_row.get("group_ground_truth_primary"),
        lexicon_evidence=base_row.get("lexicon_evidence"),
        include_source_text=include_source_text_in_user_prompt,
        include_ground_truth=include_ground_truth_in_user_prompt,
    )
    assistant_content = build_assistant_content(
        trace_row=trace_row,
        prompt_profile=prompt_profile,
        include_think_tags=row_include_think_tags,
        final_tuple_override=supervision_final_tuple,
        lexicon_evidence=base_row.get("lexicon_evidence"),
    )
    assistant_format = "reasoning_json" if row_include_think_tags else "json_only"

    metadata = dict(base_row.get("metadata") or {})
    metadata["dataset_variant"] = normalize_string(variant_cfg.get("name")) or "unknown"
    metadata["supervision_target"] = supervision_target
    metadata["assistant_format"] = assistant_format

    return {
        "sample_id": base_row.get("sample_id"),
        "group_id": base_row.get("group_id"),
        "decision_id": base_row.get("decision_id"),
        "source": base_row.get("source"),
        "split": base_row.get("split"),
        "language": base_row.get("language"),
        "task_profile": task_profile,
        "dataset_variant": metadata["dataset_variant"],
        "supervision_target": supervision_target,
        "hard_case": bool(base_row.get("hard_case")),
        "source_text": base_row.get("source_text"),
        "ground_truth": list(base_row.get("ground_truth") or []),
        "group_ground_truth_candidates": list(base_row.get("group_ground_truth_candidates") or []),
        "group_ground_truth_primary": base_row.get("group_ground_truth_primary"),
        "teacher_final_tuple": normalize_final_tuple(base_row.get("teacher_final_tuple")),
        "teacher_rationale": base_row.get("teacher_rationale"),
        "supervision_final_tuple": supervision_final_tuple,
        "assistant_format": assistant_format,
        "lexicon_evidence": list(base_row.get("lexicon_evidence") or []),
        "teacher_matches_primary_gt_clean": bool(base_row.get("teacher_matches_primary_gt_clean")),
        "teacher_matches_primary_gt_silver": bool(base_row.get("teacher_matches_primary_gt_silver")),
        "external_selector_matches_primary_gt_clean": bool(base_row.get("external_selector_matches_primary_gt_clean")),
        "external_selector_matches_primary_gt_silver": bool(base_row.get("external_selector_matches_primary_gt_silver")),
        "messages": [
            {"role": "system", "content": normalize_string(prompt_profile.get("system_prompt")) or ""},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_content},
        ],
        "metadata": metadata,
    }


def build_variant_datasets(
    base_rows: Sequence[Dict[str, Any]],
    dataset_variants: Sequence[Dict[str, Any]],
    prompt_profiles: Dict[str, Dict[str, Any]],
    include_think_tags: bool,
    include_source_text_in_user_prompt: bool,
    include_ground_truth_in_user_prompt: bool,
) -> Dict[str, List[Dict[str, Any]]]:
    datasets: Dict[str, List[Dict[str, Any]]] = {}
    for variant_cfg in dataset_variants:
        name = normalize_string(variant_cfg.get("name")) or "unknown_variant"
        rows: List[Dict[str, Any]] = []
        for base_row in base_rows:
            if not row_matches_variant(base_row, variant_cfg):
                continue
            rows.append(
                render_variant_row(
                    base_row=base_row,
                    variant_cfg=variant_cfg,
                    prompt_profiles=prompt_profiles,
                    include_think_tags=include_think_tags,
                    include_source_text_in_user_prompt=include_source_text_in_user_prompt,
                    include_ground_truth_in_user_prompt=include_ground_truth_in_user_prompt,
                )
            )
        datasets[name] = rows
    return datasets


def quality_gates(dataset_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    duplicate_decision_id = 0
    missing_messages = 0
    invalid_message_schema = 0
    invalid_final_tuple = 0
    missing_think_wrapper = 0
    non_strict_json_only = 0

    seen: Set[str] = set()
    for row in dataset_rows:
        decision_id = str(row.get("decision_id") or "")
        if decision_id in seen:
            duplicate_decision_id += 1
        seen.add(decision_id)

        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            missing_messages += 1
            continue

        roles = [msg.get("role") for msg in messages if isinstance(msg, dict)]
        if roles != ["system", "user", "assistant"]:
            invalid_message_schema += 1

        assistant = messages[2].get("content") if isinstance(messages[2], dict) else None
        if not isinstance(assistant, str):
            invalid_final_tuple += 1
            continue

        assistant_format = normalize_string(row.get("assistant_format")) or normalize_string((row.get("metadata") or {}).get("assistant_format")) or "reasoning_json"
        stripped_assistant = assistant.strip()
        if assistant_format == "reasoning_json":
            if not stripped_assistant.startswith("<think>") or "</think>" not in stripped_assistant:
                missing_think_wrapper += 1

            json_start = assistant.rfind("{")
            if json_start < 0:
                invalid_final_tuple += 1
                continue

            try:
                parsed = json.loads(assistant[json_start:])
            except Exception:
                invalid_final_tuple += 1
                continue
        elif assistant_format == "json_only":
            if not stripped_assistant.startswith("{") or not stripped_assistant.endswith("}"):
                non_strict_json_only += 1
                invalid_final_tuple += 1
                continue
            try:
                parsed = json.loads(stripped_assistant)
            except Exception:
                non_strict_json_only += 1
                invalid_final_tuple += 1
                continue
        else:
            json_start = assistant.rfind("{")
            if json_start < 0:
                invalid_final_tuple += 1
                continue

            try:
                parsed = json.loads(assistant[json_start:])
            except Exception:
                invalid_final_tuple += 1
                continue

        normalized = normalize_final_tuple(parsed)
        if not normalize_string(normalized.get("target")):
            invalid_final_tuple += 1

    passed = (
        duplicate_decision_id == 0
        and missing_messages == 0
        and invalid_message_schema == 0
        and invalid_final_tuple == 0
        and missing_think_wrapper == 0
        and non_strict_json_only == 0
    )

    return {
        "total_rows": len(dataset_rows),
        "duplicate_decision_id": duplicate_decision_id,
        "missing_messages": missing_messages,
        "invalid_message_schema": invalid_message_schema,
        "invalid_final_tuple": invalid_final_tuple,
        "missing_think_wrapper": missing_think_wrapper,
        "non_strict_json_only": non_strict_json_only,
        "passed": passed,
    }


def build_summary(dataset_rows: Sequence[Dict[str, Any]], drop_stats: Dict[str, int], trace_source: str) -> Dict[str, Any]:
    split_dist = dict(sorted(Counter(str(row.get("split")) for row in dataset_rows).items()))
    lang_dist = dict(sorted(Counter(str(row.get("language")) for row in dataset_rows).items()))
    source_dist = dict(sorted(Counter(str(row.get("source")) for row in dataset_rows).items()))
    profile_dist = dict(sorted(Counter(str(row.get("task_profile")) for row in dataset_rows).items()))
    variant_dist = dict(sorted(Counter(str(row.get("dataset_variant")) for row in dataset_rows).items()))
    supervision_dist = dict(sorted(Counter(str(row.get("supervision_target")) for row in dataset_rows).items()))
    assistant_format_dist = dict(sorted(Counter(str(row.get("assistant_format") or "unknown") for row in dataset_rows).items()))
    hard_case_count = sum(1 for row in dataset_rows if bool(row.get("hard_case")))
    gt_matched_count = sum(1 for row in dataset_rows if bool(row.get("group_ground_truth_candidates")))
    lexicon_evidence_count = sum(
        1 for row in dataset_rows if isinstance(row.get("lexicon_evidence"), list) and bool(row.get("lexicon_evidence"))
    )
    clean_agree_count = sum(1 for row in dataset_rows if bool(row.get("teacher_matches_primary_gt_clean")))
    silver_agree_count = sum(1 for row in dataset_rows if bool(row.get("teacher_matches_primary_gt_silver")))
    selector_clean_count = sum(1 for row in dataset_rows if bool(row.get("external_selector_matches_primary_gt_clean")))
    selector_silver_count = sum(1 for row in dataset_rows if bool(row.get("external_selector_matches_primary_gt_silver")))

    return {
        "trace_source": trace_source,
        "dataset_count": len(dataset_rows),
        "split_distribution": split_dist,
        "language_distribution": lang_dist,
        "source_distribution": source_dist,
        "task_profile_distribution": profile_dist,
        "dataset_variant_distribution": variant_dist,
        "supervision_target_distribution": supervision_dist,
        "assistant_format_distribution": assistant_format_dist,
        "hard_case_count": hard_case_count,
        "hard_case_ratio": round(hard_case_count / len(dataset_rows), 4) if dataset_rows else 0.0,
        "ground_truth_matched_count": gt_matched_count,
        "ground_truth_matched_ratio": round(gt_matched_count / len(dataset_rows), 4) if dataset_rows else 0.0,
        "lexicon_evidence_count": lexicon_evidence_count,
        "teacher_matches_primary_gt_clean_count": clean_agree_count,
        "teacher_matches_primary_gt_silver_count": silver_agree_count,
        "external_selector_matches_primary_gt_clean_count": selector_clean_count,
        "external_selector_matches_primary_gt_silver_count": selector_silver_count,
        "drop_stats": drop_stats,
    }


def write_dataset_variant_outputs(base_dir: Path, dataset_rows: Sequence[Dict[str, Any]]) -> None:
    write_jsonl(base_dir / "single_pass_superprompt.jsonl", dataset_rows)
    by_split = bucket_rows_by_key(dataset_rows, "split")
    for split in ["train", "dev", "test"]:
        write_jsonl(base_dir / "by_split" / f"{split}.jsonl", by_split.get(split, []))
    by_lang_split: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        language = normalize_string(row.get("language")) or "unknown"
        split = normalize_string(row.get("split")) or "train"
        by_lang_split[(language, split)].append(row)
    for (language, split), rows in sorted(by_lang_split.items()):
        write_jsonl(base_dir / "by_lang_split" / f"{language}_{split}.jsonl", rows)


def write_dataset_outputs(
    output_root: Path,
    variant_rows_map: Dict[str, List[Dict[str, Any]]],
    primary_variant_name: str,
) -> None:
    primary_rows = variant_rows_map.get(primary_variant_name, [])
    write_dataset_variant_outputs(output_root / "datasets", primary_rows)

    manifest = {
        "primary_variant_name": primary_variant_name,
        "variants": {},
    }
    for variant_name, rows in variant_rows_map.items():
        variant_dir = output_root / "datasets" / "variants" / variant_name
        write_dataset_variant_outputs(variant_dir, rows)
        manifest["variants"][variant_name] = {
            "dataset_path": str(variant_dir / "single_pass_superprompt.jsonl"),
            "split_counts": {split: sum(1 for row in rows if str(row.get("split")) == split) for split in ["train", "dev", "test"]},
            "language_split_counts": {
                str(language): {
                    split: sum(
                        1
                        for row in rows
                        if str(row.get("language")) == str(language) and str(row.get("split")) == split
                    )
                    for split in ["train", "dev", "test"]
                }
                for language in sorted({str(row.get("language")) for row in rows})
            },
        }

    write_json(output_root / "datasets" / "variants_manifest.json", manifest)


def load_trace_rows(paths: Paths) -> Tuple[List[Dict[str, Any]], str]:
    if paths.phase3_trace_pack is not None and paths.phase3_trace_pack.exists():
        return read_jsonl(paths.phase3_trace_pack), "phase3_trace_pack"

    group_decisions = load_jsonl_if_exists(paths.phase3_group_decisions)
    rectified_roles = load_jsonl_if_exists(paths.phase3_rectified_roles)
    final_tuples = load_jsonl_if_exists(paths.phase3_final_tuples)

    if not group_decisions:
        raise FileNotFoundError(
            "Cannot build phase4 dataset: missing phase3 trace pack and missing group decisions artifact."
        )

    phase1_rows = load_jsonl_if_exists(paths.phase1_candidates)
    phase2_rows = load_jsonl_if_exists(paths.phase2_candidates)

    phase1_index = build_phase1_decompose_index(phase1_rows)
    phase2_index = build_phase2_role_index(phase2_rows)

    trace_rows = synthesize_trace_pack_rows(
        group_decisions=group_decisions,
        rectified_rows=rectified_roles,
        final_rows=final_tuples,
        phase1_index=phase1_index,
        phase2_index=phase2_index,
    )
    return trace_rows, "phase3_joined"


def run_phase4_distill(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cli_args = args or parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, cli_args)
    paths = build_paths(args=cli_args, cfg=cfg, workspace_root=workspace_root)
    run_id = paths.run_root.name

    phase4_cfg = cfg.get("phase4", {})
    protocol_cfg = cfg.get("protocol", {}) if isinstance(cfg.get("protocol"), dict) else {}
    runtime_cfg = cfg.get("runtime", {})

    prompt_profiles = load_prompt_profiles(cfg, workspace_root=workspace_root, config_path=config_path)
    dataset_variants = load_dataset_variants(phase4_cfg)
    quality_selector_indices = load_phase0_like_selector_indices(
        phase4_cfg=phase4_cfg,
        workspace_root=workspace_root,
        config_path=config_path,
    )
    primary_variant_name = normalize_string(phase4_cfg.get("primary_variant_name")) or PRIMARY_VARIANT_NAME
    variant_names = [normalize_string(item.get("name")) or "" for item in dataset_variants]
    if primary_variant_name not in variant_names:
        primary_variant_name = variant_names[0] if variant_names else PRIMARY_VARIANT_NAME

    trace_rows, trace_source = load_trace_rows(paths)
    phase0_rows = load_jsonl_if_exists(paths.phase0_samples)
    sample_index = build_phase0_sample_index(phase0_rows)

    allowed_splits = set(parse_allowed_splits(cfg))
    validate_protocol_contract(
        cfg=cfg,
        protocol_cfg=protocol_cfg,
        dataset_variants=dataset_variants,
        allowed_splits=sorted(allowed_splits),
        workspace_root=workspace_root,
        config_path=config_path,
    )
    max_samples = runtime_cfg.get("max_samples")
    max_samples = max_samples if isinstance(max_samples, int) and max_samples > 0 else None

    hard_case_only = parse_bool(phase4_cfg.get("hard_case_only"), default=False)
    hard_case_threshold = safe_float(phase4_cfg.get("hard_case_threshold", 0.67), default=0.67)
    include_think_tags = parse_bool(phase4_cfg.get("include_think_tags"), default=True)
    ground_truth_match_min_score = safe_float(phase4_cfg.get("ground_truth_match_min_score", 3.0), default=3.0)
    include_source_text_in_user_prompt = parse_bool(phase4_cfg.get("include_source_text_in_user_prompt"), default=True)
    include_ground_truth_in_user_prompt = parse_bool(phase4_cfg.get("include_ground_truth_in_user_prompt"), default=False)
    include_lexicon_evidence = parse_bool(phase4_cfg.get("include_lexicon_evidence"), default=True)
    zh_lexicon_max_hits = int(phase4_cfg.get("zh_lexicon_max_hits") or 3)
    zh_lexicon_terms = load_zh_lexicon_terms(paths.zh_lexicon_path) if include_lexicon_evidence else []

    base_rows, drop_stats = build_base_dataset_rows(
        trace_rows=trace_rows,
        sample_index=sample_index,
        prompt_profiles=prompt_profiles,
        quality_selector_indices=quality_selector_indices,
        include_lexicon_evidence=include_lexicon_evidence,
        zh_lexicon_terms=zh_lexicon_terms,
        zh_lexicon_max_hits=zh_lexicon_max_hits,
        hard_case_only=hard_case_only,
        hard_case_threshold=hard_case_threshold,
        allowed_splits=allowed_splits,
        max_samples=max_samples,
        ground_truth_match_min_score=ground_truth_match_min_score,
    )
    variant_rows_map = build_variant_datasets(
        base_rows=base_rows,
        dataset_variants=dataset_variants,
        prompt_profiles=prompt_profiles,
        include_think_tags=include_think_tags,
        include_source_text_in_user_prompt=include_source_text_in_user_prompt,
        include_ground_truth_in_user_prompt=include_ground_truth_in_user_prompt,
    )

    quality_by_variant = {name: quality_gates(rows) for name, rows in variant_rows_map.items()}
    summary_by_variant: Dict[str, Any] = {}
    for name, rows in variant_rows_map.items():
        variant_drop_stats = {
            "base_rows_available": len(base_rows),
            "excluded_by_variant_filter": max(len(base_rows) - len(rows), 0),
        }
        if name == primary_variant_name:
            variant_drop_stats = dict(drop_stats)
            variant_drop_stats["base_rows_available"] = len(base_rows)
            variant_drop_stats["excluded_by_variant_filter"] = max(len(base_rows) - len(rows), 0)
        summary_by_variant[name] = build_summary(rows, drop_stats=variant_drop_stats, trace_source=trace_source)

    quality_report = {
        "primary_variant_name": primary_variant_name,
        "variants": quality_by_variant,
    }
    summary_report = {
        "trace_source": trace_source,
        "base_dataset_count": len(base_rows),
        "primary_variant_name": primary_variant_name,
        "variant_summaries": summary_by_variant,
        "drop_stats": drop_stats,
    }

    write_json(paths.run_root / "reports" / "quality_gates.json", quality_report)
    write_json(paths.run_root / "reports" / "summary.json", summary_report)

    strict_mode = parse_bool(cfg.get("quality", {}).get("strict", False), default=False)
    failed_variants = [name for name, quality in quality_by_variant.items() if not quality.get("passed", False)]
    if strict_mode and failed_variants:
        raise RuntimeError(f"Phase4 quality gate failed under strict mode for variants: {failed_variants}")

    if not cli_args.dry_run:
        write_dataset_outputs(paths.run_root, variant_rows_map=variant_rows_map, primary_variant_name=primary_variant_name)
        write_dataset_outputs(paths.output_root / "latest", variant_rows_map=variant_rows_map, primary_variant_name=primary_variant_name)

    run_summary = {
        "run_id": run_id,
        "created_at": utc_now(),
        "config_path": str(paths.config_path),
        "output_root": str(paths.run_root),
        "trace_source": trace_source,
        "counts": {
            "input_trace_rows": len(trace_rows),
            "phase0_rows": len(phase0_rows),
            "base_dataset_rows": len(base_rows),
            "output_dataset_rows": len(variant_rows_map.get(primary_variant_name, [])),
            "variant_counts": {name: len(rows) for name, rows in variant_rows_map.items()},
        },
        "phase4": {
            "hard_case_only": bool(hard_case_only),
            "hard_case_threshold": hard_case_threshold,
            "include_think_tags": bool(include_think_tags),
            "ground_truth_match_min_score": ground_truth_match_min_score,
            "include_source_text_in_user_prompt": bool(include_source_text_in_user_prompt),
            "include_ground_truth_in_user_prompt": bool(include_ground_truth_in_user_prompt),
            "include_lexicon_evidence": bool(include_lexicon_evidence),
            "zh_lexicon_max_hits": zh_lexicon_max_hits,
            "allowed_splits": sorted(allowed_splits),
            "max_samples": max_samples,
            "primary_variant_name": primary_variant_name,
            "prompt_profile_names": sorted(prompt_profiles.keys()),
            "dataset_variants": variant_names,
            "quality_selector_languages": sorted(quality_selector_indices.keys()),
        },
        "protocol": protocol_cfg,
        "quality": quality_report,
        "summary": summary_report,
    }
    write_json(paths.run_root / "run_summary.json", run_summary)
    return run_summary


def main() -> None:
    result = run_phase4_distill()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
