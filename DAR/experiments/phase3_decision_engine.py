from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

VALID_HARMFUL_DIRECTION = {"harmful", "non-harmful", "unknown"}
DEFAULT_EXPECTED_ROLES = ["victim", "moderator", "cultural_bystander"]

DEFAULT_ACTION_ALLOWLIST = ["reject_hypothesis", "revise_group", "revise_harmful", "keep_candidate"]
DEFAULT_ERROR_TAG_ALLOWLIST = [
    "implicit_target_miss",
    "cultural_metaphor_sarcasm_misread",
    "group_boundary_error",
    "harmful_over_under_estimation",
    "reasoning_leap",
]

CRITICAL_VETO_TAGS = {
    "cultural_metaphor_sarcasm_misread",
    "harmful_over_under_estimation",
}

DEFAULT_ASSIST_PROMPT = """You are an optional model-assist module for a deterministic hate tuple decision engine.

Input group package:
{{GROUP_PAYLOAD_JSON}}

Input role outputs:
{{ROLE_OUTPUTS_JSON}}

Field-level conflict matrix:
{{CONFLICT_MATRIX_JSON}}

Rule draft decision:
{{RULE_DECISION_JSON}}

Task:
1) Keep deterministic behavior as primary. Only override rule draft if grounded role outputs strongly support changes.
2) Return strict JSON only.
3) Use allowed actions and error tags only.
4) Avoid speculative edits; preserve grounded spans.

Allowed actions:
- reject_hypothesis
- revise_group
- revise_harmful
- keep_candidate

Allowed error tags:
- implicit_target_miss
- cultural_metaphor_sarcasm_misread
- group_boundary_error
- harmful_over_under_estimation
- reasoning_leap

Output schema:
{{OUTPUT_SCHEMA_JSON}}

Retry note:
{{RETRY_NOTE}}
"""


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    input_jsonl: Path
    output_root: Path
    run_root: Path
    model_root: Path
    assist_model_path: Path
    prompt_path: Optional[Path]


@dataclass
class GroupBundle:
    group_id: str
    sample_id: str
    source: str
    split: str
    language: str
    group_text: str
    sentence_ids: List[int]
    candidates_by_role: Dict[str, Dict[str, Any]]
    all_candidates: List[Dict[str, Any]]
    missing_roles: List[str]
    duplicate_roles: Dict[str, int]
    unknown_roles: List[str]


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
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"null", "none", "nan"}:
        return None
    return text


def dedupe_preserve_order(items: Sequence[str], case_insensitive: bool = True) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    for item in items:
        if not item:
            continue
        key = item.lower() if case_insensitive else item
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def detect_language(text: Optional[str]) -> str:
    if text and CJK_RE.search(text):
        return "zh"
    return "en"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


def get_segmenter_confidence(payload: Dict[str, Any], default: float = 0.0) -> float:
    raw = payload.get("segmenter_confidence")
    return safe_float(raw, default=default)


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
    ids: List[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and item >= 0:
                ids.append(item)
            elif isinstance(item, str) and item.isdigit():
                ids.append(int(item))
    return sorted(set(ids))


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


def resolve_existing_model_path(
    configured_model_path: Path,
    workspace_root: Path,
    config_dir: Path,
    model_root: Path,
    raw_candidates: Any,
) -> Path:
    if configured_model_path.exists():
        return configured_model_path

    candidates: List[Path] = []
    if isinstance(raw_candidates, list):
        for raw in raw_candidates:
            if not isinstance(raw, str) or not raw.strip():
                continue
            candidates.append(resolve_path(workspace_root, config_dir, raw.strip()))

    if not candidates:
        candidates = [
            model_root / "Qwen3-14B",
            model_root / "Qwen3.5-9B",
            model_root / "Qwen3-8B",
            model_root / "Qwen3.5-35B-A3B-FP8",
        ]

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return configured_model_path


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DAR-Hate Phase3 deterministic decision engine")
    default_config = str((Path(__file__).resolve().parents[1] / "configs" / "phase3_template.json"))

    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--mode", choices=["rule_only", "qwen_assist"], default=None)
    parser.add_argument("--enable-model-assist", action="store_true", help="Enable optional model-assisted overrides")
    parser.add_argument("--input-jsonl", default=None, help="Override phase2 candidate input JSONL")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--model-path", default=None, help="Override optional assist model path")
    parser.add_argument("--prompt-path", default=None, help="Override optional assist prompt path")
    parser.add_argument("--roles", default=None, help="Comma-separated expected roles")
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N groups")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Override generation max_new_tokens")
    parser.add_argument("--temperature", type=float, default=None, help="Override generation temperature")
    parser.add_argument("--top-p", type=float, default=None, help="Override generation top_p")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--max-parse-retries", type=int, default=None, help="Override parse retries")
    parser.add_argument(
        "--disable-rectification",
        action="store_true",
        help="Disable rectification actions and keep candidates as-is (ablation switch)",
    )
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
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config. Use JSON config or install pyyaml.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


def get_phase3_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Backward-compatible config lookup:
    # - preferred key: phase3
    # - legacy key: phase3_4
    if isinstance(cfg.get("phase3"), dict):
        phase3 = cfg.get("phase3")
    elif isinstance(cfg.get("phase3_4"), dict):
        phase3 = cfg.get("phase3_4")
    else:
        phase3 = {}
    out = {
        "mode": "rule_only",
        "expected_roles": list(DEFAULT_EXPECTED_ROLES),
        "confidence_threshold": 0.8,
        "veto_threshold": 0.8,
        "disagreement_high_threshold": 0.67,
        "disagreement_medium_threshold": 0.34,
        "max_parse_retries": 1,
        "enable_model_assist": False,
        "model_assist_on_high_risk_only": True,
        "model_assist_risk_threshold": 0.67,
        "disable_rectification": False,
        "allow_model_fallback_to_rule": True,
        "action_allowlist": list(DEFAULT_ACTION_ALLOWLIST),
        "error_tag_allowlist": list(DEFAULT_ERROR_TAG_ALLOWLIST),
    }

    out.update(phase3)
    return out


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if not isinstance(cfg.get("phase3"), dict):
        cfg["phase3"] = get_phase3_cfg(cfg)

    override_map = [
        (args.mode, ["phase3", "mode"]),
        (args.input_jsonl, ["paths", "input_jsonl"]),
        (args.output_root, ["paths", "output_root"]),
        (args.model_path, ["paths", "assist_model_path"]),
        (args.prompt_path, ["paths", "decision_prompt_path"]),
        (args.max_samples, ["runtime", "max_samples"]),
        (args.max_new_tokens, ["inference", "max_new_tokens"]),
        (args.temperature, ["inference", "temperature"]),
        (args.top_p, ["inference", "top_p"]),
        (args.seed, ["seed"]),
        (args.max_parse_retries, ["phase3", "max_parse_retries"]),
        (args.log_level, ["logging", "level"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)

    if args.enable_model_assist:
        set_nested(cfg, ["phase3", "enable_model_assist"], True)
    if args.disable_rectification:
        set_nested(cfg, ["phase3", "disable_rectification"], True)

    if args.roles:
        roles = [x.strip() for x in args.roles.split(",") if x.strip()]
        set_nested(cfg, ["phase3", "expected_roles"], roles)

    if args.splits:
        splits = [x.strip() for x in args.splits.split(",") if x.strip()]
        set_nested(cfg, ["runtime", "default_splits"], splits)

    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg["paths"]

    input_jsonl = resolve_path(workspace_root, config_dir, cfg_paths["input_jsonl"])
    output_root = resolve_path(workspace_root, config_dir, cfg_paths["output_root"])
    model_root = resolve_path(workspace_root, config_dir, cfg_paths["model_root"])

    configured_model_path = resolve_path(
        workspace_root,
        config_dir,
        cfg_paths.get("assist_model_path", str(model_root / "Qwen3-14B")),
    )

    assist_model_path = resolve_existing_model_path(
        configured_model_path=configured_model_path,
        workspace_root=workspace_root,
        config_dir=config_dir,
        model_root=model_root,
        raw_candidates=cfg_paths.get("assist_model_candidates"),
    )

    prompt_raw = cfg_paths.get("decision_prompt_path")
    prompt_path = resolve_path(workspace_root, config_dir, prompt_raw) if isinstance(prompt_raw, str) and prompt_raw else None

    run_id = args.run_id or f"phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root / run_id

    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        input_jsonl=input_jsonl,
        output_root=output_root,
        run_root=run_root,
        model_root=model_root,
        assist_model_path=assist_model_path,
        prompt_path=prompt_path,
    )


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    runtime_cfg = cfg.get("runtime", {})
    splits = runtime_cfg.get("default_splits", ["train", "dev", "test"])
    if not isinstance(splits, list):
        return ["train", "dev", "test"]
    cleaned = [str(x).strip() for x in splits if str(x).strip()]
    return cleaned or ["train", "dev", "test"]


def parse_expected_roles(engine_cfg: Dict[str, Any]) -> List[str]:
    roles = engine_cfg.get("expected_roles", DEFAULT_EXPECTED_ROLES)
    if not isinstance(roles, list):
        roles = list(DEFAULT_EXPECTED_ROLES)
    cleaned = [str(r).strip().lower() for r in roles if str(r).strip()]
    cleaned = dedupe_preserve_order(cleaned, case_insensitive=False)
    return cleaned or list(DEFAULT_EXPECTED_ROLES)


def parse_action_allowlist(engine_cfg: Dict[str, Any]) -> List[str]:
    allowlist = engine_cfg.get("action_allowlist", DEFAULT_ACTION_ALLOWLIST)
    if not isinstance(allowlist, list):
        allowlist = list(DEFAULT_ACTION_ALLOWLIST)
    cleaned: List[str] = []
    for item in allowlist:
        action = canonicalize_action(item)
        if action:
            cleaned.append(action)
    cleaned = dedupe_preserve_order(cleaned, case_insensitive=False)
    return cleaned or list(DEFAULT_ACTION_ALLOWLIST)


def parse_error_tag_allowlist(engine_cfg: Dict[str, Any]) -> List[str]:
    allowlist = engine_cfg.get("error_tag_allowlist", DEFAULT_ERROR_TAG_ALLOWLIST)
    if not isinstance(allowlist, list):
        allowlist = list(DEFAULT_ERROR_TAG_ALLOWLIST)
    cleaned: List[str] = []
    for item in allowlist:
        tag = canonicalize_error_tag(item)
        if tag:
            cleaned.append(tag)
    cleaned = dedupe_preserve_order(cleaned, case_insensitive=False)
    return cleaned or list(DEFAULT_ERROR_TAG_ALLOWLIST)


def normalize_harmful_direction(raw_direction: Any, raw_harmful: Any = None) -> str:
    direction = normalize_string(raw_direction)
    if direction:
        lowered = direction.lower().strip()
        if lowered in VALID_HARMFUL_DIRECTION:
            return lowered
    if raw_harmful in {1, "1", True, "true", "True"}:
        return "harmful"
    if raw_harmful in {0, "0", False, "false", "False"}:
        return "non-harmful"
    return "unknown"


def normalize_tuple_fields(raw_tuple: Dict[str, Any]) -> Dict[str, Any]:
    harmful_direction = normalize_harmful_direction(raw_tuple.get("harmful_direction"), raw_tuple.get("harmful"))
    return {
        "target": normalize_string(raw_tuple.get("target")) or "IMPLICIT_TARGET",
        "argument": normalize_string(raw_tuple.get("argument")),
        "group": normalize_string(raw_tuple.get("group")) or "unknown",
        "harmful_direction": harmful_direction,
        "harmful": direction_to_harmful_binary(harmful_direction),
    }


def direction_to_harmful_binary(direction: str) -> Optional[int]:
    if direction == "harmful":
        return 1
    if direction == "non-harmful":
        return 0
    return None


def truncate_text(value: Optional[str], max_chars: int = 4000) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def canonicalize_error_tag(raw_tag: Any) -> Optional[str]:
    text = normalize_string(raw_tag)
    if text is None:
        return None
    normalized = text.lower().replace("-", "_").replace("/", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    aliases = {
        "implicit_target_miss": "implicit_target_miss",
        "implicit_target": "implicit_target_miss",
        "implicit_target_missing": "implicit_target_miss",
        "cultural_metaphor_sarcasm_misread": "cultural_metaphor_sarcasm_misread",
        "cultural_misread": "cultural_metaphor_sarcasm_misread",
        "group_boundary_error": "group_boundary_error",
        "group_error": "group_boundary_error",
        "harmful_over_under_estimation": "harmful_over_under_estimation",
        "harmful_misestimation": "harmful_over_under_estimation",
        "reasoning_leap": "reasoning_leap",
        "reasoning_error": "reasoning_leap",
    }
    return aliases.get(normalized, normalized)


def canonicalize_action(raw_action: Any) -> Optional[str]:
    text = normalize_string(raw_action)
    if text is None:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    aliases = {
        "reject": "reject_hypothesis",
        "reject_hypothesis": "reject_hypothesis",
        "revise_group": "revise_group",
        "revise_group_label": "revise_group",
        "revise_harmful": "revise_harmful",
        "revise_harmful_direction": "revise_harmful",
        "keep": "keep_candidate",
        "keep_candidate": "keep_candidate",
    }
    return aliases.get(normalized, normalized)


def extract_json_candidate(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model output")

    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    for block in code_blocks:
        candidate = block.strip()
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    list_start = stripped.find("[")
    list_end = stripped.rfind("]")
    if list_start >= 0 and list_end > list_start:
        maybe_list = stripped[list_start : list_end + 1]
        try:
            return json.loads(maybe_list)
        except json.JSONDecodeError:
            pass

    obj_start = stripped.find("{")
    obj_end = stripped.rfind("}")
    if obj_start >= 0 and obj_end > obj_start:
        maybe_obj = stripped[obj_start : obj_end + 1]
        return json.loads(maybe_obj)

    raise ValueError("No parseable JSON in model output")


def prefer_4090_device_order(force_4090_order: bool = False) -> Optional[str]:
    existing = os.getenv("CUDA_VISIBLE_DEVICES")
    if existing and not force_4090_order:
        return existing

    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return existing

    rows: List[Tuple[int, str]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",", 1)]
        if len(parts) != 2:
            continue
        try:
            rows.append((int(parts[0]), parts[1]))
        except ValueError:
            continue

    if not rows:
        return existing

    rows.sort(key=lambda x: (0 if "4090" in x[1] else 1, x[0]))
    ordered = ",".join(str(idx) for idx, _ in rows)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ordered
    return ordered


def build_max_memory_map(torch_module: Any, preferred_ratio: float, other_ratio: float) -> Optional[Dict[int, str]]:
    if not torch_module.cuda.is_available():
        return None

    max_memory: Dict[int, str] = {}
    for idx in range(torch_module.cuda.device_count()):
        props = torch_module.cuda.get_device_properties(idx)
        name = torch_module.cuda.get_device_name(idx)
        total_gib = props.total_memory / (1024 ** 3)
        ratio = preferred_ratio if "4090" in name else other_ratio
        budget = max(4, int(total_gib * ratio))
        max_memory[idx] = f"{budget}GiB"
    return max_memory or None


def normalize_vote_key(field: str, value: Any) -> str:
    if value is None:
        return "__missing__"
    if field in {"harmful_direction", "harmful"}:
        text = str(value).strip().lower()
        return text if text else "__missing__"
    text = normalize_string(value)
    if text is None:
        return "__missing__"
    if CJK_RE.search(text):
        return text
    return text.lower()


def candidate_weight(candidate: Dict[str, Any]) -> float:
    if not isinstance(candidate, dict):
        return 0.0
    return clamp(safe_float(candidate.get("quality_score"), default=0.0))


def choose_best_role_candidate(candidates: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not candidates:
        raise ValueError("choose_best_role_candidate requires at least one candidate")

    def rank_key(row: Dict[str, Any]) -> Tuple[float, int, int]:
        conf = candidate_weight(row)
        trace = row.get("model_trace") if isinstance(row.get("model_trace"), dict) else {}
        parse_ok = 0 if bool(trace.get("parse_fail")) else 1
        fallback_ok = 0 if bool(row.get("fallback_used")) else 1
        return conf, parse_ok, fallback_ok

    ranked = sorted(candidates, key=rank_key, reverse=True)
    return ranked[0]


def build_group_bundles(rows: Sequence[Dict[str, Any]], expected_roles: Sequence[str]) -> Tuple[List[GroupBundle], Dict[str, Any]]:
    expected_set = set(expected_roles)
    grouped: Dict[str, Dict[str, Any]] = {}
    missing_group_id_input = 0

    for row in rows:
        group_id = normalize_string(row.get("group_id"))
        if group_id is None:
            missing_group_id_input += 1
            continue

        bucket = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "sample_id": normalize_string(row.get("sample_id")) or make_sha1(group_id)[:16],
                "source": normalize_string(row.get("source")) or "unknown",
                "split": normalize_string(row.get("split")) or "train",
                "language": normalize_string(row.get("language")) or detect_language(normalize_string(row.get("group_text"))),
                "group_text": normalize_string(row.get("group_text")) or "",
                "sentence_ids": safe_sentence_ids(row.get("sentence_ids")),
                "all_rows": [],
                "role_rows": defaultdict(list),
                "unknown_roles": set(),
            },
        )

        bucket["all_rows"].append(row)

        role_raw = normalize_string(row.get("role")) or "unknown"
        role = role_raw.lower()
        bucket["role_rows"][role].append(row)
        if role not in expected_set:
            bucket["unknown_roles"].add(role)

        if not bucket["group_text"]:
            bucket["group_text"] = normalize_string(row.get("group_text")) or ""
        if not bucket["sentence_ids"]:
            bucket["sentence_ids"] = safe_sentence_ids(row.get("sentence_ids"))

    duplicate_role_rows = 0
    unknown_role_rows = 0
    bundles: List[GroupBundle] = []

    for group_id, bucket in grouped.items():
        role_rows = bucket["role_rows"]
        candidates_by_role: Dict[str, Dict[str, Any]] = {}
        missing_roles: List[str] = []
        duplicate_roles: Dict[str, int] = {}

        for role in expected_roles:
            rows_for_role = role_rows.get(role, [])
            if not rows_for_role:
                missing_roles.append(role)
                continue
            if len(rows_for_role) > 1:
                duplicate_roles[role] = len(rows_for_role) - 1
                duplicate_role_rows += len(rows_for_role) - 1
            candidates_by_role[role] = choose_best_role_candidate(rows_for_role)

        for unknown_role in bucket["unknown_roles"]:
            unknown_role_rows += len(role_rows.get(unknown_role, []))

        sentence_ids = list(bucket["sentence_ids"])
        if not sentence_ids:
            sid_acc: List[int] = []
            for row in bucket["all_rows"]:
                sid_acc.extend(safe_sentence_ids(row.get("sentence_ids")))
            sentence_ids = sorted(set(sid_acc))

        bundle = GroupBundle(
            group_id=group_id,
            sample_id=str(bucket["sample_id"]),
            source=str(bucket["source"]),
            split=str(bucket["split"]),
            language=str(bucket["language"]),
            group_text=str(bucket["group_text"]),
            sentence_ids=sentence_ids,
            candidates_by_role=candidates_by_role,
            all_candidates=list(bucket["all_rows"]),
            missing_roles=missing_roles,
            duplicate_roles=duplicate_roles,
            unknown_roles=sorted(str(x) for x in bucket["unknown_roles"]),
        )
        bundles.append(bundle)

    bundles.sort(key=lambda b: (b.split, b.sample_id, b.group_id))

    ingest_stats = {
        "input_rows": len(rows),
        "group_count": len(bundles),
        "missing_group_id_input": missing_group_id_input,
        "duplicate_role_rows": duplicate_role_rows,
        "unknown_role_rows": unknown_role_rows,
    }
    return bundles, ingest_stats


def get_candidate_field_value(candidate: Dict[str, Any], field: str) -> Any:
    pred_tuple = candidate.get("predicted_tuple") if isinstance(candidate.get("predicted_tuple"), dict) else {}
    if field == "harmful_direction":
        return normalize_harmful_direction(pred_tuple.get("harmful_direction"), pred_tuple.get("harmful"))
    if field == "harmful":
        direction = normalize_harmful_direction(pred_tuple.get("harmful_direction"), pred_tuple.get("harmful"))
        return direction_to_harmful_binary(direction)
    return normalize_string(pred_tuple.get(field))


def conflict_level(disagreement_ratio: float, high_threshold: float = 0.67, medium_threshold: float = 0.34) -> str:
    if disagreement_ratio >= high_threshold:
        return "high"
    if disagreement_ratio >= medium_threshold:
        return "medium"
    if disagreement_ratio > 0.0:
        return "low"
    return "none"


def compute_field_conflict(
    bundle: GroupBundle,
    field: str,
    expected_roles: Sequence[str],
    high_threshold: float,
    medium_threshold: float,
) -> Dict[str, Any]:
    role_values: Dict[str, Any] = {}
    key_counts: Counter[str] = Counter()
    key_scores: Dict[str, float] = defaultdict(float)
    raw_by_key: Dict[str, Any] = {}
    present_roles: List[str] = []

    for role in expected_roles:
        candidate = bundle.candidates_by_role.get(role)
        value = get_candidate_field_value(candidate, field) if candidate else None
        role_values[role] = value

        key = normalize_vote_key(field, value)
        if key == "__missing__":
            continue
        present_roles.append(role)
        key_counts[key] += 1
        key_scores[key] += candidate_weight(candidate) if candidate else 0.0
        if key not in raw_by_key:
            raw_by_key[key] = value

    if not key_counts:
        return {
            "role_values": role_values,
            "vote_counter": {},
            "majority_value": None,
            "disagreement_roles": [],
            "disagreement_ratio": 0.0,
            "conflict_level": "none",
            "present_roles": present_roles,
        }

    ranked_keys = sorted(key_counts.keys(), key=lambda k: (-key_counts[k], -key_scores[k], k))
    majority_key = ranked_keys[0]
    majority_value = raw_by_key.get(majority_key)

    disagreement_roles: List[str] = []
    for role in expected_roles:
        key = normalize_vote_key(field, role_values.get(role))
        if key == "__missing__":
            continue
        if key != majority_key:
            disagreement_roles.append(role)

    disagreement_ratio = round(len(disagreement_roles) / len(present_roles), 4) if present_roles else 0.0
    vote_counter = {str(raw_by_key[k]): int(key_counts[k]) for k in ranked_keys}

    return {
        "role_values": role_values,
        "vote_counter": vote_counter,
        "majority_value": majority_value,
        "disagreement_roles": disagreement_roles,
        "disagreement_ratio": disagreement_ratio,
        "conflict_level": conflict_level(disagreement_ratio, high_threshold=high_threshold, medium_threshold=medium_threshold),
        "present_roles": present_roles,
    }


def compute_conflict_matrix(bundle: GroupBundle, expected_roles: Sequence[str], engine_cfg: Dict[str, Any]) -> Dict[str, Any]:
    high_threshold = safe_float(engine_cfg.get("disagreement_high_threshold", 0.67), default=0.67)
    medium_threshold = safe_float(engine_cfg.get("disagreement_medium_threshold", 0.34), default=0.34)

    fields = {
        "target": compute_field_conflict(bundle, "target", expected_roles, high_threshold, medium_threshold),
        "argument": compute_field_conflict(bundle, "argument", expected_roles, high_threshold, medium_threshold),
        "group": compute_field_conflict(bundle, "group", expected_roles, high_threshold, medium_threshold),
        "harmful_direction": compute_field_conflict(bundle, "harmful_direction", expected_roles, high_threshold, medium_threshold),
    }

    ratios = [safe_float(fields[field]["disagreement_ratio"], default=0.0) for field in fields]
    overall_score = round(sum(ratios) / len(ratios), 4) if ratios else 0.0
    overall_level = conflict_level(overall_score, high_threshold=high_threshold, medium_threshold=medium_threshold)

    high_risk_fields: List[str] = []
    for field_name, payload in fields.items():
        level = str(payload.get("conflict_level"))
        ratio = safe_float(payload.get("disagreement_ratio"), default=0.0)
        if level == "high" or (field_name == "harmful_direction" and ratio >= medium_threshold):
            high_risk_fields.append(field_name)

    return {
        "present_roles": sorted(bundle.candidates_by_role.keys()),
        "missing_roles": list(bundle.missing_roles),
        "fields": fields,
        "overall_conflict_score": overall_score,
        "overall_conflict_level": overall_level,
        "high_risk_fields": high_risk_fields,
    }


def build_group_payload(bundle: GroupBundle) -> Dict[str, Any]:
    segmenter_conf_values: List[float] = []
    source_targets: List[str] = []
    source_arguments: List[str] = []
    source_harmfuls: List[Any] = []

    for cand in bundle.candidates_by_role.values():
        input_package = cand.get("input_package") if isinstance(cand.get("input_package"), dict) else {}
        segmenter_conf_values.append(get_segmenter_confidence(input_package, default=0.0))
        source_target = normalize_string(input_package.get("source_target"))
        source_argument = normalize_string(input_package.get("source_argument"))
        if source_target:
            source_targets.append(source_target)
        if source_argument:
            source_arguments.append(source_argument)
        source_harmfuls.append(input_package.get("source_harmful"))

    return {
        "group_id": bundle.group_id,
        "sample_id": bundle.sample_id,
        "source": bundle.source,
        "split": bundle.split,
        "language": bundle.language,
        "sentence_ids": bundle.sentence_ids,
        "group_text": bundle.group_text,
        "avg_segmenter_confidence": round(sum(segmenter_conf_values) / len(segmenter_conf_values), 4) if segmenter_conf_values else 0.0,
        "source_targets": dedupe_preserve_order(source_targets, case_insensitive=False),
        "source_arguments": dedupe_preserve_order(source_arguments, case_insensitive=False),
        "source_harmful_values": source_harmfuls,
    }


def build_role_outputs(bundle: GroupBundle, expected_roles: Sequence[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for role in expected_roles:
        candidate = bundle.candidates_by_role.get(role)
        if not candidate:
            continue

        pred_tuple = candidate.get("predicted_tuple") if isinstance(candidate.get("predicted_tuple"), dict) else {}

        rows.append(
            {
                "role": role,
                "role_candidate_id": normalize_string(candidate.get("role_candidate_id")),
                "tuple": {
                    "target": normalize_string(pred_tuple.get("target")) or "IMPLICIT_TARGET",
                    "argument": normalize_string(pred_tuple.get("argument")),
                    "group": normalize_string(pred_tuple.get("group")) or "unknown",
                    "harmful": direction_to_harmful_binary(
                        normalize_harmful_direction(pred_tuple.get("harmful_direction"), pred_tuple.get("harmful"))
                    ),
                    "harmful_direction": normalize_harmful_direction(pred_tuple.get("harmful_direction"), pred_tuple.get("harmful")),
                },
                "rationale": normalize_string(candidate.get("rationale")) or "",
                "quality_score": round(candidate_weight(candidate), 4),
                "uncertainty_flag": bool(candidate.get("uncertainty_flag")),
                "escalation_flag": bool(candidate.get("escalation_flag")),
            }
        )
    return rows


def infer_error_tags(bundle: GroupBundle, conflict_matrix: Dict[str, Any], allowed_tags: Set[str]) -> List[str]:
    tags: List[str] = []

    def add_tag(tag: str) -> None:
        if tag in allowed_tags and tag not in tags:
            tags.append(tag)

    fields = conflict_matrix.get("fields") if isinstance(conflict_matrix.get("fields"), dict) else {}
    target_field = fields.get("target") if isinstance(fields.get("target"), dict) else {}
    group_field = fields.get("group") if isinstance(fields.get("group"), dict) else {}
    harmful_field = fields.get("harmful_direction") if isinstance(fields.get("harmful_direction"), dict) else {}

    target_majority = normalize_string(target_field.get("majority_value"))
    if target_majority is None or target_majority.upper() == "IMPLICIT_TARGET":
        add_tag("implicit_target_miss")
    else:
        for cand in bundle.candidates_by_role.values():
            pred = cand.get("predicted_tuple") if isinstance(cand.get("predicted_tuple"), dict) else {}
            target = normalize_string(pred.get("target"))
            if target and target.upper() == "IMPLICIT_TARGET":
                add_tag("implicit_target_miss")
                break

    if safe_float(group_field.get("disagreement_ratio"), default=0.0) >= 0.34:
        add_tag("group_boundary_error")

    if safe_float(harmful_field.get("disagreement_ratio"), default=0.0) >= 0.34:
        add_tag("harmful_over_under_estimation")

    reasoning_issue = False
    for cand in bundle.candidates_by_role.values():
        rationale = normalize_string(cand.get("rationale")) or ""
        if len(rationale) < 12:
            reasoning_issue = True

    if reasoning_issue:
        add_tag("reasoning_leap")

    bystander = bundle.candidates_by_role.get("cultural_bystander")
    if bystander is not None and safe_float(harmful_field.get("disagreement_ratio"), default=0.0) >= 0.34:
        pred = bystander.get("predicted_tuple") if isinstance(bystander.get("predicted_tuple"), dict) else {}
        by_dir = normalize_harmful_direction(pred.get("harmful_direction"), pred.get("harmful"))
        maj_dir = normalize_harmful_direction(harmful_field.get("majority_value"), None)
        if by_dir != maj_dir:
            add_tag("cultural_metaphor_sarcasm_misread")

    return tags


def build_rule_decision_payload(
    bundle: GroupBundle,
    conflict_matrix: Dict[str, Any],
    engine_cfg: Dict[str, Any],
    action_allowlist: Sequence[str],
    error_tag_allowlist: Sequence[str],
) -> Dict[str, Any]:
    allow_actions = set(action_allowlist)
    disable_rectification = parse_bool(engine_cfg.get("disable_rectification", False), default=False)
    if disable_rectification:
        allow_actions = {"keep_candidate"}
    allow_tags = set(error_tag_allowlist)

    confidence_threshold = safe_float(engine_cfg.get("confidence_threshold", 0.8), default=0.8)
    veto_threshold = safe_float(engine_cfg.get("veto_threshold", 0.8), default=0.8)
    high_threshold = safe_float(engine_cfg.get("disagreement_high_threshold", 0.67), default=0.67)

    fields = conflict_matrix.get("fields") if isinstance(conflict_matrix.get("fields"), dict) else {}
    majority_group = normalize_string((fields.get("group") or {}).get("majority_value"))
    majority_target = normalize_string((fields.get("target") or {}).get("majority_value")) or "IMPLICIT_TARGET"
    majority_harmful_dir = normalize_harmful_direction((fields.get("harmful_direction") or {}).get("majority_value"), None)

    tags = infer_error_tags(bundle=bundle, conflict_matrix=conflict_matrix, allowed_tags=allow_tags)

    actions: List[Dict[str, Any]] = []
    revised_candidates: Dict[str, Dict[str, Any]] = {}

    for role, cand in bundle.candidates_by_role.items():
        pred = cand.get("predicted_tuple") if isinstance(cand.get("predicted_tuple"), dict) else {}
        pred_group = normalize_string(pred.get("group")) or "unknown"
        pred_target = normalize_string(pred.get("target")) or "IMPLICIT_TARGET"
        pred_harmful_dir = normalize_harmful_direction(pred.get("harmful_direction"), pred.get("harmful"))
        conf = max(0.5, candidate_weight(cand))
        parse_fail = bool((cand.get("model_trace") or {}).get("parse_fail"))

        action = "keep_candidate"
        patch: Dict[str, Any] = {}
        reason = "No correction needed."

        if parse_fail and conf < confidence_threshold:
            action = "reject_hypothesis"
            reason = "Low-confidence parse-fail role output."
        elif majority_harmful_dir != "unknown" and pred_harmful_dir != majority_harmful_dir and conf < confidence_threshold + 0.05:
            action = "revise_harmful"
            patch = {
                "harmful_direction": majority_harmful_dir,
                "harmful": direction_to_harmful_binary(majority_harmful_dir),
            }
            reason = "Align harmful polarity with majority-supported grounded roles."
        elif majority_group and pred_group != majority_group and conf < confidence_threshold:
            action = "revise_group"
            patch = {"group": majority_group}
            reason = "Align group label with cross-role majority."
        elif pred_target == "IMPLICIT_TARGET" and majority_target != "IMPLICIT_TARGET" and conf < confidence_threshold:
            action = "reject_hypothesis"
            reason = "Implicit target conflicts with stronger grounded explicit target."

        if action not in allow_actions:
            action = "keep_candidate"
            patch = {}
            reason = "Action constrained by allowlist."

        actions.append({"action": action, "role": role, "patch": patch, "reason": reason})
        if action in {"revise_group", "revise_harmful"}:
            revised_candidates[role] = dict(patch)

    if not actions:
        for role in bundle.candidates_by_role.keys():
            actions.append({"action": "keep_candidate", "role": role, "patch": {}, "reason": "No correction needed."})

    field_findings: List[Dict[str, str]] = []
    for field_name in ["target", "argument", "group", "harmful_direction"]:
        field_payload = fields.get(field_name) if isinstance(fields.get(field_name), dict) else {}
        ratio = safe_float(field_payload.get("disagreement_ratio"), default=0.0)
        level = str(field_payload.get("conflict_level") or "none")
        if ratio <= 0.0:
            continue
        field_findings.append(
            {
                "field": field_name,
                "issue": f"Cross-role disagreement ratio {ratio:.2f}",
                "severity": "high" if level == "high" else "medium" if level == "medium" else "low",
            }
        )

    overall_score = safe_float(conflict_matrix.get("overall_conflict_score"), default=0.0)
    audit_confidence = clamp(0.62 + 0.28 * (1.0 - overall_score))
    audit_confidence = round(audit_confidence, 4)

    veto_recommend = bool(
        audit_confidence >= veto_threshold
        and (
            bool(CRITICAL_VETO_TAGS.intersection(set(tags)))
            or overall_score >= high_threshold
        )
    )

    summary = (
        f"Rule decision: overall_conflict={overall_score:.2f}, "
        f"high_risk_fields={','.join(conflict_matrix.get('high_risk_fields', [])) or 'none'}."
    )

    return {
        "source": "rule",
        "summary": summary,
        "error_tags": tags,
        "field_findings": field_findings,
        "actions": actions,
        "revised_candidates": revised_candidates,
        "audit_confidence": audit_confidence,
        "veto_recommend": veto_recommend,
        "invalid_action_count": 0,
        "invalid_error_tags_count": 0,
    }


def normalize_field_findings(raw_findings: Any) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    if not isinstance(raw_findings, list):
        return out
    for item in raw_findings:
        if not isinstance(item, dict):
            continue
        field = normalize_string(item.get("field")) or "unknown"
        issue = normalize_string(item.get("issue")) or normalize_string(item.get("message")) or "unspecified issue"
        severity = normalize_string(item.get("severity")) or "medium"
        severity_low = severity.lower()
        if severity_low not in {"low", "medium", "high"}:
            severity_low = "medium"
        out.append({"field": field, "issue": issue, "severity": severity_low})
        if len(out) >= 16:
            break
    return out


def normalize_revised_candidates(raw_revised: Any, expected_roles: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    expected_set = set(expected_roles)

    def collect_patch(raw_patch: Any) -> Dict[str, Any]:
        patch: Dict[str, Any] = {}
        if not isinstance(raw_patch, dict):
            return patch
        for key in ["target", "argument", "group"]:
            value = normalize_string(raw_patch.get(key))
            if value is not None:
                patch[key] = value
        if "harmful_direction" in raw_patch or "harmful" in raw_patch:
            direction = normalize_harmful_direction(raw_patch.get("harmful_direction"), raw_patch.get("harmful"))
            patch["harmful_direction"] = direction
            harmful = direction_to_harmful_binary(direction)
            if harmful is not None:
                patch["harmful"] = harmful
        return patch

    if isinstance(raw_revised, dict):
        if "role" in raw_revised:
            role = normalize_string(raw_revised.get("role"))
            if role and role.lower() in expected_set:
                out[role.lower()] = collect_patch(raw_revised)
            return out
        for role, patch_obj in raw_revised.items():
            role_name = normalize_string(role)
            if role_name is None:
                continue
            role_low = role_name.lower()
            if role_low not in expected_set:
                continue
            out[role_low] = collect_patch(patch_obj)
        return out

    if isinstance(raw_revised, list):
        for item in raw_revised:
            if not isinstance(item, dict):
                continue
            role = normalize_string(item.get("role"))
            if role is None:
                continue
            role_low = role.lower()
            if role_low not in expected_set:
                continue
            out[role_low] = collect_patch(item)

    return out


def normalize_actions(
    raw_actions: Any,
    expected_roles: Sequence[str],
    action_allowlist: Sequence[str],
) -> Tuple[List[Dict[str, Any]], int]:
    out: List[Dict[str, Any]] = []
    invalid_count = 0
    expected_set = set(expected_roles)
    allow_set = set(action_allowlist)

    if not isinstance(raw_actions, list):
        return out, invalid_count

    for item in raw_actions:
        if not isinstance(item, dict):
            invalid_count += 1
            continue

        action = canonicalize_action(item.get("action"))
        if action is None or action not in allow_set:
            invalid_count += 1
            continue

        role_raw = normalize_string(item.get("role"))
        role = role_raw.lower() if role_raw else None
        if role and role not in expected_set and role not in {"all", "*"}:
            invalid_count += 1
            continue

        patch_source = item.get("patch") if isinstance(item.get("patch"), dict) else item
        patch: Dict[str, Any] = {}
        for key in ["target", "argument", "group"]:
            value = normalize_string(patch_source.get(key))
            if value is not None:
                patch[key] = value

        if "harmful_direction" in patch_source or "harmful" in patch_source:
            direction = normalize_harmful_direction(patch_source.get("harmful_direction"), patch_source.get("harmful"))
            patch["harmful_direction"] = direction
            harmful = direction_to_harmful_binary(direction)
            if harmful is not None:
                patch["harmful"] = harmful

        reason = normalize_string(item.get("reason")) or ""
        out.append({"action": action, "role": role, "patch": patch, "reason": reason})

    return out, invalid_count


def normalize_model_payload(
    parsed: Any,
    expected_roles: Sequence[str],
    action_allowlist: Sequence[str],
    error_tag_allowlist: Sequence[str],
) -> Dict[str, Any]:
    if isinstance(parsed, list):
        if not parsed:
            raise ValueError("Parsed output is an empty list")
        parsed = parsed[0]

    if not isinstance(parsed, dict):
        raise ValueError("Parsed output must be a dict or non-empty list")

    summary = normalize_string(parsed.get("summary"))
    critique = parsed.get("critique") if isinstance(parsed.get("critique"), dict) else {}
    if summary is None:
        summary = normalize_string(critique.get("summary"))
    if summary is None:
        summary = "Model assist returned no summary"

    raw_tags: List[Any] = []
    if isinstance(parsed.get("error_tags"), list):
        raw_tags.extend(parsed.get("error_tags"))
    if isinstance(critique.get("error_tags"), list):
        raw_tags.extend(critique.get("error_tags"))

    allow_tags = set(error_tag_allowlist)
    tags: List[str] = []
    invalid_error_tags_count = 0
    for raw_tag in raw_tags:
        tag = canonicalize_error_tag(raw_tag)
        if tag is None:
            continue
        if tag in allow_tags and tag not in tags:
            tags.append(tag)
        elif tag not in allow_tags:
            invalid_error_tags_count += 1

    field_findings = normalize_field_findings(parsed.get("field_findings") or critique.get("field_findings"))

    actions, invalid_action_count = normalize_actions(
        raw_actions=parsed.get("actions"),
        expected_roles=expected_roles,
        action_allowlist=action_allowlist,
    )
    revised_candidates = normalize_revised_candidates(parsed.get("revised_candidates"), expected_roles)

    conf_raw = parsed.get("audit_confidence")
    audit_confidence = round(clamp(safe_float(conf_raw, default=0.7)), 4) if conf_raw is not None else None

    veto_recommend: Optional[bool] = None
    if "veto_recommend" in parsed:
        veto_recommend = parse_bool(parsed.get("veto_recommend"), default=False)

    return {
        "source": "model",
        "summary": summary,
        "error_tags": tags,
        "field_findings": field_findings,
        "actions": actions,
        "revised_candidates": revised_candidates,
        "audit_confidence": audit_confidence,
        "veto_recommend": veto_recommend,
        "invalid_action_count": int(invalid_action_count),
        "invalid_error_tags_count": int(invalid_error_tags_count),
    }


def merge_rule_and_model_payload(rule_payload: Dict[str, Any], model_payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(rule_payload)
    merged["source"] = "rule+model"

    summary = normalize_string(model_payload.get("summary"))
    if summary:
        merged["summary"] = summary

    rule_tags = list(rule_payload.get("error_tags") or [])
    model_tags = list(model_payload.get("error_tags") or [])
    merged["error_tags"] = dedupe_preserve_order(rule_tags + model_tags, case_insensitive=False)

    model_findings = model_payload.get("field_findings") if isinstance(model_payload.get("field_findings"), list) else []
    if model_findings:
        merged["field_findings"] = model_findings

    model_actions = model_payload.get("actions") if isinstance(model_payload.get("actions"), list) else []
    if model_actions:
        merged["actions"] = model_actions

    revised = dict(rule_payload.get("revised_candidates") or {})
    model_revised = model_payload.get("revised_candidates") if isinstance(model_payload.get("revised_candidates"), dict) else {}
    revised.update(model_revised)
    merged["revised_candidates"] = revised

    model_conf = model_payload.get("audit_confidence")
    if isinstance(model_conf, (int, float)):
        merged["audit_confidence"] = round(clamp(float(model_conf)), 4)

    if model_payload.get("veto_recommend") is not None:
        merged["veto_recommend"] = bool(model_payload.get("veto_recommend"))

    merged["invalid_action_count"] = int(model_payload.get("invalid_action_count", 0))
    merged["invalid_error_tags_count"] = int(model_payload.get("invalid_error_tags_count", 0))

    if not merged.get("actions"):
        merged["actions"] = list(rule_payload.get("actions") or [])

    return merged


def harmonize_predicted_tuple(raw_tuple: Any, fallback_candidate: Dict[str, Any]) -> Dict[str, Any]:
    tuple_obj = raw_tuple if isinstance(raw_tuple, dict) else {}
    input_package = fallback_candidate.get("input_package") if isinstance(fallback_candidate.get("input_package"), dict) else {}

    target = normalize_string(tuple_obj.get("target")) or normalize_string(input_package.get("canonical_target")) or "IMPLICIT_TARGET"
    argument = normalize_string(tuple_obj.get("argument"))
    group = normalize_string(tuple_obj.get("group")) or "unknown"

    harmful_direction = normalize_harmful_direction(tuple_obj.get("harmful_direction"), tuple_obj.get("harmful"))
    harmful = direction_to_harmful_binary(harmful_direction)
    if harmful is None:
        raw_h = tuple_obj.get("harmful")
        if raw_h in {1, "1", True, "true", "True"}:
            harmful = 1
        elif raw_h in {0, "0", False, "false", "False"}:
            harmful = 0

    return {
        "target": target,
        "argument": argument,
        "group": group,
        "harmful": harmful,
        "harmful_direction": harmful_direction,
    }


def apply_decision_actions(
    bundle: GroupBundle,
    decision_payload: Dict[str, Any],
    conflict_matrix: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str], Dict[str, str], Set[str]]:
    present_roles = list(bundle.candidates_by_role.keys())

    base_tuples: Dict[str, Dict[str, Any]] = {}
    revised_tuples: Dict[str, Dict[str, Any]] = {}
    for role, cand in bundle.candidates_by_role.items():
        base = harmonize_predicted_tuple(cand.get("predicted_tuple"), fallback_candidate=cand)
        base_tuples[role] = base
        revised_tuples[role] = dict(base)

    action_by_role: Dict[str, str] = {role: "keep_candidate" for role in present_roles}
    reason_by_role: Dict[str, str] = {role: "No correction needed." for role in present_roles}
    rejected_roles: Set[str] = set()

    majority_group = normalize_string(((conflict_matrix.get("fields") or {}).get("group") or {}).get("majority_value"))
    majority_harmful_direction = normalize_harmful_direction(
        ((conflict_matrix.get("fields") or {}).get("harmful_direction") or {}).get("majority_value"),
        None,
    )

    def expand_roles(role_value: Optional[str]) -> List[str]:
        if role_value is None or role_value in {"all", "*"}:
            return list(present_roles)
        if role_value in present_roles:
            return [role_value]
        return []

    for action_obj in decision_payload.get("actions", []):
        action = str(action_obj.get("action"))
        action_role = action_obj.get("role")
        roles = expand_roles(action_role)
        patch = action_obj.get("patch") if isinstance(action_obj.get("patch"), dict) else {}
        reason = normalize_string(action_obj.get("reason")) or ""

        if not roles:
            continue

        for role in roles:
            if action == "keep_candidate":
                action_by_role[role] = "keep_candidate"
                reason_by_role[role] = reason or "No correction needed."
                continue

            if action == "reject_hypothesis":
                rejected_roles.add(role)
                action_by_role[role] = "reject_hypothesis"
                reason_by_role[role] = reason or "Rejected by decision engine."
                continue

            if action == "revise_group":
                new_group = normalize_string(patch.get("group")) or majority_group
                if new_group is not None:
                    revised_tuples[role]["group"] = new_group
                action_by_role[role] = "revise_group"
                reason_by_role[role] = reason or "Group revised by decision engine."
                continue

            if action == "revise_harmful":
                new_direction = normalize_harmful_direction(patch.get("harmful_direction"), patch.get("harmful"))
                if new_direction == "unknown" and majority_harmful_direction != "unknown":
                    new_direction = majority_harmful_direction
                revised_tuples[role]["harmful_direction"] = new_direction
                revised_tuples[role]["harmful"] = direction_to_harmful_binary(new_direction)
                action_by_role[role] = "revise_harmful"
                reason_by_role[role] = reason or "Harmful polarity revised by decision engine."
                continue

    revised_candidates = decision_payload.get("revised_candidates") if isinstance(decision_payload.get("revised_candidates"), dict) else {}
    for role, patch in revised_candidates.items():
        if role not in revised_tuples:
            continue
        before = dict(revised_tuples[role])
        if isinstance(patch, dict):
            for key in ["target", "argument", "group"]:
                value = normalize_string(patch.get(key))
                if value is not None:
                    revised_tuples[role][key] = value

            if "harmful_direction" in patch or "harmful" in patch:
                new_direction = normalize_harmful_direction(patch.get("harmful_direction"), patch.get("harmful"))
                revised_tuples[role]["harmful_direction"] = new_direction
                revised_tuples[role]["harmful"] = direction_to_harmful_binary(new_direction)

        if action_by_role.get(role) == "keep_candidate":
            if before.get("group") != revised_tuples[role].get("group"):
                action_by_role[role] = "revise_group"
            elif before.get("harmful_direction") != revised_tuples[role].get("harmful_direction"):
                action_by_role[role] = "revise_harmful"

    if present_roles and len(rejected_roles) == len(present_roles):
        keep_role = max(
            present_roles,
            key=lambda role: candidate_weight(bundle.candidates_by_role[role]),
        )
        rejected_roles.remove(keep_role)
        action_by_role[keep_role] = "keep_candidate"
        reason_by_role[keep_role] = "Reject-all safeguard kept the highest-confidence role output."

    return revised_tuples, action_by_role, reason_by_role, rejected_roles


def compute_field_vote_from_tuples(
    tuples_by_role: Dict[str, Dict[str, Any]],
    selected_roles: Sequence[str],
    field: str,
    role_conf_map: Dict[str, float],
) -> Dict[str, Any]:
    key_counts: Counter[str] = Counter()
    key_scores: Dict[str, float] = defaultdict(float)
    raw_by_key: Dict[str, Any] = {}

    for role in selected_roles:
        tup = tuples_by_role.get(role) or {}
        value = tup.get(field)
        key = normalize_vote_key(field, value)
        if key == "__missing__":
            continue
        key_counts[key] += 1
        key_scores[key] += role_conf_map.get(role, 0.0)
        if key not in raw_by_key:
            raw_by_key[key] = value

    if not key_counts:
        return {
            "majority_value": None,
            "vote_counter": {},
            "winner_count": 0,
            "winner_score": 0.0,
        }

    ranked_keys = sorted(key_counts.keys(), key=lambda k: (-key_counts[k], -key_scores[k], k))
    winner = ranked_keys[0]
    vote_counter = {str(raw_by_key[k]): int(key_counts[k]) for k in ranked_keys}

    return {
        "majority_value": raw_by_key[winner],
        "vote_counter": vote_counter,
        "winner_count": int(key_counts[winner]),
        "winner_score": round(float(key_scores[winner]), 4),
    }


def select_veto_candidate_role(
    bundle: GroupBundle,
    revised_tuples: Dict[str, Dict[str, Any]],
    selected_roles: Sequence[str],
    majority_tuple: Dict[str, Any],
) -> Optional[str]:
    if not selected_roles:
        return None

    def evidence_score_for_role(role: str) -> int:
        cand = bundle.candidates_by_role.get(role, {})
        return int(round(candidate_weight(cand) * 100))

    def agreement_score(role: str) -> int:
        tup = revised_tuples.get(role, {})
        score = 0
        for field in ["target", "argument", "group", "harmful_direction"]:
            if normalize_vote_key(field, tup.get(field)) == normalize_vote_key(field, majority_tuple.get(field)):
                score += 1
        return score

    ranked = sorted(
        selected_roles,
        key=lambda role: (
            evidence_score_for_role(role),
            1 if normalize_harmful_direction((revised_tuples.get(role) or {}).get("harmful_direction"), None) != "unknown" else 0,
            agreement_score(role),
            candidate_weight(bundle.candidates_by_role.get(role) or {}),
            role,
        ),
        reverse=True,
    )
    return ranked[0]


def arbitrate_final_decision(
    bundle: GroupBundle,
    decision_payload: Dict[str, Any],
    revised_tuples: Dict[str, Dict[str, Any]],
    action_by_role: Dict[str, str],
    reason_by_role: Dict[str, str],
    rejected_roles: Set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[str], Optional[str]]:
    selected_roles = [role for role in bundle.candidates_by_role.keys() if role not in rejected_roles]
    if not selected_roles:
        keep_role = max(
            bundle.candidates_by_role.keys(),
            key=lambda role: candidate_weight(bundle.candidates_by_role[role]),
        )
        selected_roles = [keep_role]
        rejected_roles = set(bundle.candidates_by_role.keys()) - {keep_role}

    role_conf_map = {
        role: candidate_weight(bundle.candidates_by_role.get(role) or {})
        for role in selected_roles
    }

    vote_target = compute_field_vote_from_tuples(revised_tuples, selected_roles, "target", role_conf_map)
    vote_argument = compute_field_vote_from_tuples(revised_tuples, selected_roles, "argument", role_conf_map)
    vote_group = compute_field_vote_from_tuples(revised_tuples, selected_roles, "group", role_conf_map)
    vote_harm = compute_field_vote_from_tuples(revised_tuples, selected_roles, "harmful_direction", role_conf_map)

    majority_tuple = normalize_tuple_fields({
        "target": normalize_string(vote_target.get("majority_value")) or "IMPLICIT_TARGET",
        "argument": normalize_string(vote_argument.get("majority_value")),
        "group": normalize_string(vote_group.get("majority_value")) or "unknown",
        "harmful_direction": normalize_harmful_direction(vote_harm.get("majority_value"), None),
    })

    veto_flag = bool(decision_payload.get("veto_recommend"))
    veto_reason = ""
    selected_role_for_final: Optional[str] = None
    final_tuple = dict(majority_tuple)

    if veto_flag:
        selected_role_for_final = select_veto_candidate_role(
            bundle=bundle,
            revised_tuples=revised_tuples,
            selected_roles=selected_roles,
            majority_tuple=majority_tuple,
        )
        if selected_role_for_final is not None:
            final_tuple = normalize_tuple_fields(dict(revised_tuples.get(selected_role_for_final) or majority_tuple))
            veto_reason = f"Veto override selected role={selected_role_for_final} based on quality-priority tie-break."
        else:
            veto_reason = "Veto flagged but no valid selected role was available; used majority tuple."

    final_tuple = normalize_tuple_fields(final_tuple)

    consensus_trace = {
        "vote": {
            "target": vote_target,
            "argument": vote_argument,
            "group": vote_group,
            "harmful_direction": vote_harm,
        },
        "selected_roles": selected_roles,
        "rejected_roles": sorted(rejected_roles),
        "veto_flag": veto_flag,
        "veto_reason": veto_reason,
        "selected_role_for_final": selected_role_for_final,
        "action_by_role": action_by_role,
        "reason_by_role": reason_by_role,
        "decision_summary": decision_payload.get("summary"),
    }

    return final_tuple, consensus_trace, selected_roles, selected_role_for_final


def load_prompt_template(prompt_path: Optional[Path]) -> str:
    if prompt_path and prompt_path.exists():
        text = prompt_path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_ASSIST_PROMPT


def render_model_prompt(
    template: str,
    bundle: GroupBundle,
    expected_roles: Sequence[str],
    conflict_matrix: Dict[str, Any],
    rule_payload: Dict[str, Any],
    retry_note: Optional[str],
) -> str:
    output_schema = {
        "summary": "optional short override summary",
        "error_tags": ["reasoning_leap"],
        "actions": [
            {
                "action": "revise_harmful",
                "role": "moderator",
                "reason": "Evidence favors harmful polarity.",
                "patch": {"harmful_direction": "harmful"},
            }
        ],
        "revised_candidates": [
            {
                "role": "moderator",
                "harmful_direction": "harmful",
                "harmful": 1,
            }
        ],
        "audit_confidence": 0.82,
        "veto_recommend": True,
    }

    mapping = {
        "{{GROUP_PAYLOAD_JSON}}": json.dumps(build_group_payload(bundle), ensure_ascii=False, indent=2),
        "{{ROLE_OUTPUTS_JSON}}": json.dumps(build_role_outputs(bundle, expected_roles), ensure_ascii=False, indent=2),
        "{{CONFLICT_MATRIX_JSON}}": json.dumps(conflict_matrix, ensure_ascii=False, indent=2),
        "{{RULE_DECISION_JSON}}": json.dumps(rule_payload, ensure_ascii=False, indent=2),
        "{{OUTPUT_SCHEMA_JSON}}": json.dumps(output_schema, ensure_ascii=False, indent=2),
        "{{RETRY_NOTE}}": retry_note or "",
    }

    rendered = template
    for key, value in mapping.items():
        rendered = rendered.replace(key, value)
    return rendered


class LocalModelAssistLLM:
    def __init__(self, model_path: Path, inference_cfg: Dict[str, Any], prompt_template: str) -> None:
        self.model_path = model_path
        self.inference_cfg = inference_cfg
        self.prompt_template = prompt_template
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self.cuda_available = False
        self.visible_cuda_devices: List[str] = []
        self.model_device: Optional[str] = None

    def initialize(self) -> None:
        if self._ready:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Local model path does not exist: {self.model_path}. "
                "Set --model-path explicitly or update paths.assist_model_path / paths.assist_model_candidates in config."
            )

        if parse_bool(self.inference_cfg.get("prefer_4090_order", True), default=True):
            prefer_4090_device_order(force_4090_order=parse_bool(self.inference_cfg.get("force_4090_order", False), default=False))

        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError("qwen_assist mode requires torch and transformers") from exc

        require_cuda = parse_bool(self.inference_cfg.get("require_cuda", True), default=True)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for qwen_assist mode, but torch.cuda.is_available() is False")

        use_cuda = bool(torch.cuda.is_available())
        self.cuda_available = use_cuda
        self.visible_cuda_devices = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if use_cuda else []

        default_dtype = "float16" if use_cuda else "float32"
        dtype_name = str(self.inference_cfg.get("dtype", default_dtype)).lower()
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype_value = dtype_map.get(dtype_name, torch.float16 if use_cuda else torch.float32)

        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }

        if use_cuda:
            load_kwargs["device_map"] = self.inference_cfg.get("device_map", "balanced")
            preferred_ratio = safe_float(self.inference_cfg.get("preferred_gpu_mem_ratio", 0.97), default=0.97)
            other_ratio = safe_float(self.inference_cfg.get("other_gpu_mem_ratio", 0.25), default=0.25)
            max_memory = build_max_memory_map(torch, preferred_ratio=preferred_ratio, other_ratio=other_ratio)
            if max_memory:
                load_kwargs["max_memory"] = max_memory
            load_kwargs["torch_dtype"] = dtype_value
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        model.eval()

        if hasattr(model, "device"):
            self.model_device = str(getattr(model, "device"))
        else:
            try:
                self.model_device = str(next(model.parameters()).device)
            except StopIteration:
                self.model_device = "unknown"

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._ready = True

    def _generate_text(self, messages: Sequence[Dict[str, str]]) -> str:
        if not self._ready or self._tokenizer is None or self._model is None or self._torch is None:
            raise RuntimeError("LocalModelAssistLLM is not initialized")

        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

        inputs = tokenizer(prompt, return_tensors="pt")

        model_device = getattr(model, "device", None)
        if model_device is None or getattr(model_device, "type", None) in {None, "meta"}:
            try:
                model_device = next(model.parameters()).device
            except StopIteration:
                model_device = None

        if model_device is not None and getattr(model_device, "type", None) not in {"meta", None}:
            try:
                inputs = {k: v.to(model_device) for k, v in inputs.items()}
            except Exception:
                pass

        max_new_tokens = int(self.inference_cfg.get("max_new_tokens", 512))
        temperature = safe_float(self.inference_cfg.get("temperature", 0.0), default=0.0)
        top_p = safe_float(self.inference_cfg.get("top_p", 0.9), default=0.9)
        do_sample = temperature > 0.0
        use_cache = parse_bool(self.inference_cfg.get("use_cache", False), default=False)

        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = eos_token_id

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                use_cache=use_cache,
                eos_token_id=eos_token_id,
                pad_token_id=pad_token_id,
            )

        generated = output_ids[0][inputs["input_ids"].shape[-1] :]
        return tokenizer.decode(generated, skip_special_tokens=True)

    def generate_model_payload(
        self,
        bundle: GroupBundle,
        expected_roles: Sequence[str],
        conflict_matrix: Dict[str, Any],
        rule_payload: Dict[str, Any],
        max_parse_retries: int,
        action_allowlist: Sequence[str],
        error_tag_allowlist: Sequence[str],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        parse_error: Optional[str] = None
        raw_outputs: List[str] = []
        model_payload: Optional[Dict[str, Any]] = None

        attempts = max(0, int(max_parse_retries)) + 1
        for attempt in range(attempts):
            retry_note = parse_error if attempt > 0 else None
            prompt = render_model_prompt(
                template=self.prompt_template,
                bundle=bundle,
                expected_roles=expected_roles,
                conflict_matrix=conflict_matrix,
                rule_payload=rule_payload,
                retry_note=retry_note,
            )
            messages = [
                {
                    "role": "system",
                    "content": "Return strict JSON only. Respect deterministic rule draft and only override if evidence is stronger.",
                },
                {"role": "user", "content": prompt},
            ]
            raw_text = self._generate_text(messages)
            raw_outputs.append(raw_text)

            try:
                parsed = extract_json_candidate(raw_text)
                model_payload = normalize_model_payload(
                    parsed=parsed,
                    expected_roles=expected_roles,
                    action_allowlist=action_allowlist,
                    error_tag_allowlist=error_tag_allowlist,
                )
                parse_error = None
                break
            except Exception as exc:
                parse_error = str(exc)

        if model_payload is None:
            raise ValueError(parse_error or "Failed to parse model-assist output")

        trace = {
            "raw_output": truncate_text(raw_outputs[-1] if raw_outputs else None),
            "raw_outputs_count": len(raw_outputs),
        }
        return model_payload, trace


def should_invoke_model_assist(conflict_matrix: Dict[str, Any], engine_cfg: Dict[str, Any]) -> bool:
    if not parse_bool(engine_cfg.get("enable_model_assist", False), default=False):
        return False

    on_high_risk_only = parse_bool(engine_cfg.get("model_assist_on_high_risk_only", True), default=True)
    if not on_high_risk_only:
        return True

    risk_threshold = safe_float(engine_cfg.get("model_assist_risk_threshold", 0.67), default=0.67)
    overall_score = safe_float(conflict_matrix.get("overall_conflict_score"), default=0.0)
    high_risk_fields = conflict_matrix.get("high_risk_fields") if isinstance(conflict_matrix.get("high_risk_fields"), list) else []

    return bool(overall_score >= risk_threshold or high_risk_fields)


def build_decision_id(bundle: GroupBundle) -> str:
    return f"{bundle.group_id}_d{make_sha1(bundle.group_id, bundle.sample_id)[:10]}"


def build_rectified_rows(
    bundle: GroupBundle,
    decision_id: str,
    decision_payload: Dict[str, Any],
    conflict_matrix: Dict[str, Any],
    revised_tuples: Dict[str, Dict[str, Any]],
    action_by_role: Dict[str, str],
    reason_by_role: Dict[str, str],
    rejected_roles: Set[str],
    selected_roles: Sequence[str],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for role, cand in bundle.candidates_by_role.items():
        original_tuple = harmonize_predicted_tuple(cand.get("predicted_tuple"), fallback_candidate=cand)
        rectified_tuple = revised_tuples.get(role, original_tuple)

        is_revised = (
            json.dumps(original_tuple, ensure_ascii=False, sort_keys=True)
            != json.dumps(rectified_tuple, ensure_ascii=False, sort_keys=True)
        )

        row = dict(cand)
        row["decision_id"] = decision_id
        row["audit_confidence"] = round(clamp(safe_float(decision_payload.get("audit_confidence"), default=0.5)), 4)
        row["error_tags"] = list(decision_payload.get("error_tags") or [])
        row["decision_action"] = action_by_role.get(role, "keep_candidate")
        row["decision_reason"] = reason_by_role.get(role, "")
        row["is_revised"] = bool(is_revised)
        row["rejected_by_engine"] = bool(role in rejected_roles)
        row["selected_for_arbitration"] = bool(role in selected_roles)
        row["veto_recommend"] = bool(decision_payload.get("veto_recommend"))
        row["original_predicted_tuple"] = original_tuple
        row["rectified_tuple"] = rectified_tuple
        row["decision_trace"] = {
            "summary": decision_payload.get("summary"),
            "field_findings": decision_payload.get("field_findings"),
        }
        row["conflict_snapshot"] = {
            "overall_conflict_score": safe_float(conflict_matrix.get("overall_conflict_score"), default=0.0),
            "overall_conflict_level": str(conflict_matrix.get("overall_conflict_level") or "none"),
        }
        rows.append(row)

    return rows


def build_group_decision_record(
    bundle: GroupBundle,
    decision_id: str,
    expected_roles: Sequence[str],
    conflict_matrix: Dict[str, Any],
    decision_payload: Dict[str, Any],
    final_tuple: Dict[str, Any],
    consensus_trace: Dict[str, Any],
    mode_tag: str,
    model_assist_used: bool,
    parse_fail: bool,
    parse_error: Optional[str],
    fallback_used: bool,
    trace: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    role_candidate_ids = {
        role: normalize_string(cand.get("role_candidate_id"))
        for role, cand in bundle.candidates_by_role.items()
    }

    return {
        "decision_id": decision_id,
        "group_id": bundle.group_id,
        "sample_id": bundle.sample_id,
        "source": bundle.source,
        "split": bundle.split,
        "language": bundle.language,
        "sentence_ids": bundle.sentence_ids,
        "group_text": bundle.group_text,
        "expected_roles": list(expected_roles),
        "present_roles": sorted(bundle.candidates_by_role.keys()),
        "missing_roles": list(bundle.missing_roles),
        "duplicate_roles": dict(bundle.duplicate_roles),
        "unknown_roles": list(bundle.unknown_roles),
        "role_candidate_ids": role_candidate_ids,
        "conflict_matrix": conflict_matrix,
        "decision_payload": {
            "source": decision_payload.get("source", "rule"),
            "summary": decision_payload.get("summary"),
            "error_tags": list(decision_payload.get("error_tags") or []),
            "field_findings": list(decision_payload.get("field_findings") or []),
            "actions": list(decision_payload.get("actions") or []),
            "audit_confidence": round(clamp(safe_float(decision_payload.get("audit_confidence"), default=0.5)), 4),
            "veto_recommend": bool(decision_payload.get("veto_recommend")),
        },
        "final_tuple": final_tuple,
        "consensus_trace": consensus_trace,
        "model_trace": {
            "mode": mode_tag,
            "model_assist_used": bool(model_assist_used),
            "parse_fail": bool(parse_fail),
            "parse_error": parse_error,
            "fallback_used": bool(fallback_used),
            "raw_output": truncate_text((trace or {}).get("raw_output")),
            "raw_outputs_count": int((trace or {}).get("raw_outputs_count", 0)),
            "invalid_action_count": int(decision_payload.get("invalid_action_count", 0)),
            "invalid_error_tags_count": int(decision_payload.get("invalid_error_tags_count", 0)),
        },
    }


def build_final_tuple_record(
    bundle: GroupBundle,
    decision_id: str,
    final_tuple: Dict[str, Any],
    consensus_trace: Dict[str, Any],
    decision_payload: Dict[str, Any],
) -> Dict[str, Any]:
    normalized_final_tuple = normalize_tuple_fields(final_tuple)
    return {
        "decision_id": decision_id,
        "group_id": bundle.group_id,
        "sample_id": bundle.sample_id,
        "source": bundle.source,
        "split": bundle.split,
        "language": bundle.language,
        "sentence_ids": bundle.sentence_ids,
        "group_text": bundle.group_text,
        "final_tuple": normalized_final_tuple,
        "veto_flag": bool(consensus_trace.get("veto_flag")),
        "veto_reason": normalize_string(consensus_trace.get("veto_reason")) or "",
        "audit_confidence": round(clamp(safe_float(decision_payload.get("audit_confidence"), default=0.5)), 4),
        "error_tags": list(decision_payload.get("error_tags") or []),
        "decision_summary": normalize_string(decision_payload.get("summary")) or "",
        "consensus_trace": consensus_trace,
    }


def process_groups(
    bundles: Sequence[GroupBundle],
    mode: str,
    expected_roles: Sequence[str],
    engine_cfg: Dict[str, Any],
    inference_cfg: Dict[str, Any],
    paths: Paths,
    prompt_template: str,
    logger: EventLogger,
    action_allowlist: Sequence[str],
    error_tag_allowlist: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], int, int, int]:
    group_decisions: List[Dict[str, Any]] = []
    rectified_rows: List[Dict[str, Any]] = []
    final_rows: List[Dict[str, Any]] = []

    parse_fail_count = 0
    fallback_count = 0
    model_invoked_count = 0

    max_parse_retries = int(max(0, engine_cfg.get("max_parse_retries", 1)))
    allow_model_fallback_to_rule = parse_bool(engine_cfg.get("allow_model_fallback_to_rule", True), default=True)
    require_cuda = parse_bool(inference_cfg.get("require_cuda", True), default=True)
    progress_interval = int(max(1, engine_cfg.get("progress_interval", 200)))

    llm: Optional[LocalModelAssistLLM] = None
    if parse_bool(engine_cfg.get("enable_model_assist", False), default=False) and mode == "qwen_assist":
        llm = LocalModelAssistLLM(model_path=paths.assist_model_path, inference_cfg=inference_cfg, prompt_template=prompt_template)
        try:
            llm.initialize()
            logger.event(
                "decision_engine",
                "model_assist_ready",
                cuda_available=llm.cuda_available,
                visible_cuda_devices=llm.visible_cuda_devices,
                model_device=llm.model_device,
            )
        except Exception as exc:
            if require_cuda and not allow_model_fallback_to_rule:
                raise
            if not allow_model_fallback_to_rule:
                raise
            llm = None
            logger.event("decision_engine", "model_assist_unavailable_fallback", level="WARNING", message=str(exc))

    for index, bundle in enumerate(bundles, start=1):
        conflict_matrix = compute_conflict_matrix(bundle=bundle, expected_roles=expected_roles, engine_cfg=engine_cfg)

        parse_fail = False
        parse_error: Optional[str] = None
        fallback_used = mode == "rule_only"
        model_assist_used = False
        trace: Optional[Dict[str, Any]] = None

        rule_payload = build_rule_decision_payload(
            bundle=bundle,
            conflict_matrix=conflict_matrix,
            engine_cfg=engine_cfg,
            action_allowlist=action_allowlist,
            error_tag_allowlist=error_tag_allowlist,
        )

        decision_payload = dict(rule_payload)

        if llm is not None and should_invoke_model_assist(conflict_matrix=conflict_matrix, engine_cfg=engine_cfg):
            model_invoked_count += 1
            try:
                model_payload, trace = llm.generate_model_payload(
                    bundle=bundle,
                    expected_roles=expected_roles,
                    conflict_matrix=conflict_matrix,
                    rule_payload=rule_payload,
                    max_parse_retries=max_parse_retries,
                    action_allowlist=action_allowlist,
                    error_tag_allowlist=error_tag_allowlist,
                )
                decision_payload = merge_rule_and_model_payload(rule_payload=rule_payload, model_payload=model_payload)
                model_assist_used = True
                fallback_used = False
            except Exception as exc:
                if not allow_model_fallback_to_rule:
                    raise
                parse_fail = True
                parse_error = str(exc)
                parse_fail_count += 1
                fallback_count += 1
                decision_payload = dict(rule_payload)
                fallback_used = True

        decision_id = build_decision_id(bundle)

        revised_tuples, action_by_role, reason_by_role, rejected_roles = apply_decision_actions(
            bundle=bundle,
            decision_payload=decision_payload,
            conflict_matrix=conflict_matrix,
        )

        final_tuple, consensus_trace, selected_roles, _ = arbitrate_final_decision(
            bundle=bundle,
            decision_payload=decision_payload,
            revised_tuples=revised_tuples,
            action_by_role=action_by_role,
            reason_by_role=reason_by_role,
            rejected_roles=rejected_roles,
        )

        group_rectified = build_rectified_rows(
            bundle=bundle,
            decision_id=decision_id,
            decision_payload=decision_payload,
            conflict_matrix=conflict_matrix,
            revised_tuples=revised_tuples,
            action_by_role=action_by_role,
            reason_by_role=reason_by_role,
            rejected_roles=rejected_roles,
            selected_roles=selected_roles,
        )
        rectified_rows.extend(group_rectified)

        group_decision = build_group_decision_record(
            bundle=bundle,
            decision_id=decision_id,
            expected_roles=expected_roles,
            conflict_matrix=conflict_matrix,
            decision_payload=decision_payload,
            final_tuple=final_tuple,
            consensus_trace=consensus_trace,
            mode_tag=mode,
            model_assist_used=model_assist_used,
            parse_fail=parse_fail,
            parse_error=parse_error,
            fallback_used=fallback_used,
            trace=trace,
        )
        group_decisions.append(group_decision)

        final_row = build_final_tuple_record(
            bundle=bundle,
            decision_id=decision_id,
            final_tuple=final_tuple,
            consensus_trace=consensus_trace,
            decision_payload=decision_payload,
        )
        final_rows.append(final_row)

        if index == 1 or index % progress_interval == 0:
            logger.event(
                "decision_engine",
                "progress",
                index=index,
                group_id=bundle.group_id,
                total_group_decisions=len(group_decisions),
                total_final_rows=len(final_rows),
            )

    return group_decisions, rectified_rows, final_rows, parse_fail_count, fallback_count, model_invoked_count


def quality_gates(
    group_decisions: Sequence[Dict[str, Any]],
    rectified_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
    expected_roles: Sequence[str],
    ingest_stats: Dict[str, Any],
    action_allowlist: Sequence[str],
    error_tag_allowlist: Sequence[str],
) -> Dict[str, Any]:
    allow_actions = set(action_allowlist)
    allow_tags = set(error_tag_allowlist)

    duplicate_decision_id = 0
    missing_group_id = int(ingest_stats.get("missing_group_id_input", 0))
    invalid_audit_confidence = 0
    groups_missing_roles = 0
    groups_without_final = 0

    invalid_action_from_model = 0
    invalid_error_tags_from_model = 0

    seen_decision_ids: Set[str] = set()
    for row in group_decisions:
        decision_id = str(row.get("decision_id") or "")
        if decision_id in seen_decision_ids:
            duplicate_decision_id += 1
        seen_decision_ids.add(decision_id)

        if not normalize_string(row.get("group_id")):
            missing_group_id += 1

        confidence = safe_float((row.get("decision_payload") or {}).get("audit_confidence"), default=-1.0)
        if confidence < 0.0 or confidence > 1.0:
            invalid_audit_confidence += 1

        if row.get("missing_roles"):
            groups_missing_roles += 1

        trace = row.get("model_trace") if isinstance(row.get("model_trace"), dict) else {}
        invalid_action_from_model += int(trace.get("invalid_action_count", 0) or 0)
        invalid_error_tags_from_model += int(trace.get("invalid_error_tags_count", 0) or 0)

    duplicate_role_candidate_id = 0
    invalid_rectified_action = 0
    invalid_rectified_error_tags = 0

    seen_rcid: Set[str] = set()
    for row in rectified_rows:
        rcid = normalize_string(row.get("role_candidate_id"))
        if rcid:
            if rcid in seen_rcid:
                duplicate_role_candidate_id += 1
            seen_rcid.add(rcid)

        action = str(row.get("decision_action") or "")
        if action not in allow_actions:
            invalid_rectified_action += 1

        tags = row.get("error_tags")
        if isinstance(tags, list):
            for tag in tags:
                canon = canonicalize_error_tag(tag)
                if canon is None or canon not in allow_tags:
                    invalid_rectified_error_tags += 1

    decision_ids_with_final = set(str(r.get("decision_id") or "") for r in final_rows)
    for row in group_decisions:
        if str(row.get("decision_id") or "") not in decision_ids_with_final:
            groups_without_final += 1

    duplicate_final_decision_id = 0
    seen_final_ids: Set[str] = set()
    for row in final_rows:
        did = str(row.get("decision_id") or "")
        if did in seen_final_ids:
            duplicate_final_decision_id += 1
        seen_final_ids.add(did)

    passed = (
        duplicate_decision_id == 0
        and missing_group_id == 0
        and invalid_audit_confidence == 0
        and groups_missing_roles == 0
        and groups_without_final == 0
        and duplicate_role_candidate_id == 0
        and invalid_rectified_action == 0
        and invalid_rectified_error_tags == 0
        and invalid_action_from_model == 0
        and invalid_error_tags_from_model == 0
        and duplicate_final_decision_id == 0
    )

    return {
        "total_group_decisions": len(group_decisions),
        "total_rectified_candidates": len(rectified_rows),
        "total_final_tuples": len(final_rows),
        "expected_roles": list(expected_roles),
        "ingest_stats": ingest_stats,
        "duplicate_decision_id": duplicate_decision_id,
        "missing_group_id": missing_group_id,
        "invalid_audit_confidence": invalid_audit_confidence,
        "groups_missing_roles": groups_missing_roles,
        "groups_without_final": groups_without_final,
        "duplicate_role_candidate_id": duplicate_role_candidate_id,
        "invalid_rectified_action": invalid_rectified_action,
        "invalid_rectified_error_tags": invalid_rectified_error_tags,
        "invalid_action_from_model": invalid_action_from_model,
        "invalid_error_tags_from_model": invalid_error_tags_from_model,
        "duplicate_final_decision_id": duplicate_final_decision_id,
        "passed": passed,
    }


def build_summary(
    group_decisions: Sequence[Dict[str, Any]],
    rectified_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
    parse_fail_count: int,
    fallback_count: int,
    model_invoked_count: int,
) -> Dict[str, Any]:
    split_dist = dict(sorted(Counter(str(a.get("split")) for a in group_decisions).items()))
    lang_dist = dict(sorted(Counter(str(a.get("language")) for a in group_decisions).items()))
    source_dist = dict(sorted(Counter(str(a.get("source")) for a in group_decisions).items()))

    role_dist = dict(sorted(Counter(str(r.get("role")) for r in rectified_rows).items()))
    action_dist = dict(sorted(Counter(str(r.get("decision_action")) for r in rectified_rows).items()))

    error_tag_dist = dict(
        sorted(
            Counter(
                str(tag)
                for row in group_decisions
                for tag in ((row.get("decision_payload") or {}).get("error_tags") or [])
            ).items()
        )
    )

    veto_count = sum(1 for r in final_rows if bool(r.get("veto_flag")))
    revised_count = sum(1 for r in rectified_rows if bool(r.get("is_revised")))
    rejected_count = sum(1 for r in rectified_rows if bool(r.get("rejected_by_engine")))
    selected_count = sum(1 for r in rectified_rows if bool(r.get("selected_for_arbitration")))

    confidences = [safe_float((r.get("decision_payload") or {}).get("audit_confidence"), default=0.0) for r in group_decisions]
    avg_conf = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

    conflict_scores = [safe_float((r.get("conflict_matrix") or {}).get("overall_conflict_score"), default=0.0) for r in group_decisions]
    avg_conflict = round(sum(conflict_scores) / len(conflict_scores), 4) if conflict_scores else 0.0

    model_assist_used = sum(1 for r in group_decisions if bool((r.get("model_trace") or {}).get("model_assist_used")))

    return {
        "group_decision_count": len(group_decisions),
        "rectified_candidate_count": len(rectified_rows),
        "final_tuple_count": len(final_rows),
        "split_distribution": split_dist,
        "language_distribution": lang_dist,
        "source_distribution": source_dist,
        "role_distribution": role_dist,
        "action_distribution": action_dist,
        "error_tag_distribution": error_tag_dist,
        "veto_count": veto_count,
        "veto_ratio": round(veto_count / len(final_rows), 4) if final_rows else 0.0,
        "revised_candidate_count": revised_count,
        "revised_ratio": round(revised_count / len(rectified_rows), 4) if rectified_rows else 0.0,
        "rejected_candidate_count": rejected_count,
        "selected_for_arbitration_count": selected_count,
        "selected_for_arbitration_ratio": round(selected_count / len(rectified_rows), 4) if rectified_rows else 0.0,
        "model_parse_fail_count": int(parse_fail_count),
        "fallback_count": int(fallback_count),
        "model_invoked_count": int(model_invoked_count),
        "model_assist_used_count": int(model_assist_used),
        "avg_audit_confidence": avg_conf,
        "avg_conflict_score": avg_conflict,
    }


def bucket_rows_by_key(rows: Sequence[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(key))].append(row)
    return buckets


def build_phase4_trace_pack_rows(
    group_decisions: Sequence[Dict[str, Any]],
    rectified_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rectified_by_decision = bucket_rows_by_key(rectified_rows, "decision_id")
    final_by_decision = {str(row.get("decision_id") or ""): row for row in final_rows}

    rows: List[Dict[str, Any]] = []
    for decision in group_decisions:
        decision_id = str(decision.get("decision_id") or "")
        per_role = sorted(
            rectified_by_decision.get(decision_id, []),
            key=lambda x: (str(x.get("role") or ""), str(x.get("role_candidate_id") or "")),
        )
        final_row = final_by_decision.get(decision_id)

        first_input = {}
        if per_role:
            first_input = per_role[0].get("input_package") if isinstance(per_role[0].get("input_package"), dict) else {}

        phase4_decompose = first_input.get("phase4_decompose") if isinstance(first_input.get("phase4_decompose"), dict) else None
        if phase4_decompose is None:
            phase4_decompose = {
                "task": "decompose",
                "sentence_ids": list(decision.get("sentence_ids") or []),
                "group_text": str(decision.get("group_text") or ""),
                "canonical_target": "IMPLICIT_TARGET",
                "canonical_target_id": None,
                "target_surface_mentions": [],
                "harmful_direction": "unknown",
                "harmful_intent": "unknown",
                "segmenter_confidence": round(clamp(get_segmenter_confidence(first_input, default=0.0)), 4),
                "escalation_flag": False,
            }

        role_hypotheses: List[Dict[str, Any]] = []
        rectification_actions: List[Dict[str, Any]] = []
        for role_row in per_role:
            predicted_tuple = role_row.get("original_predicted_tuple") if isinstance(role_row.get("original_predicted_tuple"), dict) else {}
            role_payload = role_row.get("phase4_role_hypothesis") if isinstance(role_row.get("phase4_role_hypothesis"), dict) else {
                "task": str(role_row.get("role") or "unknown"),
                "role": str(role_row.get("role") or "unknown"),
                "role_candidate_id": str(role_row.get("role_candidate_id") or ""),
                "group_id": str(role_row.get("group_id") or ""),
                "sample_id": str(role_row.get("sample_id") or ""),
                "tuple": predicted_tuple,
                "rationale": normalize_string(role_row.get("rationale")) or "",
                "quality_score": round(candidate_weight(role_row), 4),
                "uncertainty_flag": bool(role_row.get("uncertainty_flag")),
                "escalation_flag": bool(role_row.get("escalation_flag")),
            }
            role_hypotheses.append(role_payload)

            rectification_actions.append(
                {
                    "role": str(role_row.get("role") or "unknown"),
                    "role_candidate_id": str(role_row.get("role_candidate_id") or ""),
                    "decision_action": str(role_row.get("decision_action") or "keep_candidate"),
                    "decision_reason": normalize_string(role_row.get("decision_reason")) or "",
                    "is_revised": bool(role_row.get("is_revised")),
                    "rejected_by_engine": bool(role_row.get("rejected_by_engine")),
                    "selected_for_arbitration": bool(role_row.get("selected_for_arbitration")),
                    "original_predicted_tuple": predicted_tuple,
                    "rectified_tuple": role_row.get("rectified_tuple") if isinstance(role_row.get("rectified_tuple"), dict) else predicted_tuple,
                }
            )

        decision_payload = decision.get("decision_payload") if isinstance(decision.get("decision_payload"), dict) else {}
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

        rows.append(
            {
                "decision_id": decision_id,
                "group_id": str(decision.get("group_id") or ""),
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
                    "final_tuple": final_tuple,
                    "audit_confidence": round(clamp(safe_float(decision_payload.get("audit_confidence"), default=0.5)), 4),
                    "veto_recommend": bool(decision_payload.get("veto_recommend")),
                    "veto_flag": veto_flag,
                    "veto_reason": veto_reason,
                    "consensus_trace": consensus_trace,
                },
            }
        )

    return rows


def write_candidate_outputs(
    run_root: Path,
    expected_roles: Sequence[str],
    group_decisions: Sequence[Dict[str, Any]],
    rectified_rows: Sequence[Dict[str, Any]],
    final_rows: Sequence[Dict[str, Any]],
    phase4_trace_pack_rows: Sequence[Dict[str, Any]],
) -> None:
    write_jsonl(run_root / "candidates" / "group_decisions.jsonl", group_decisions)
    write_jsonl(run_root / "candidates" / "rectified_roles.jsonl", rectified_rows)
    write_jsonl(run_root / "candidates" / "final_tuples.jsonl", final_rows)
    write_jsonl(run_root / "candidates" / "phase4_trace_pack.jsonl", phase4_trace_pack_rows)
    write_jsonl(run_root / "phase4" / "phase4_trace_pack.jsonl", phase4_trace_pack_rows)

    rectified_by_role = bucket_rows_by_key(rectified_rows, "role")
    for role in expected_roles:
        write_jsonl(run_root / "candidates" / "roles" / f"{role}.jsonl", rectified_by_role.get(role, []))

    final_by_split = bucket_rows_by_key(final_rows, "split")
    for split in ["train", "dev", "test"]:
        write_jsonl(run_root / "candidates" / "by_split" / f"final_{split}.jsonl", final_by_split.get(split, []))


def run_phase3(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
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

        engine_cfg = get_phase3_cfg(cfg)
        inference_cfg = cfg.get("inference", {})
        runtime_cfg = cfg.get("runtime", {})

        mode = str(engine_cfg.get("mode", "rule_only"))
        if mode not in {"rule_only", "qwen_assist"}:
            raise ValueError(f"Unsupported mode: {mode}")

        expected_roles = parse_expected_roles(engine_cfg)
        action_allowlist = parse_action_allowlist(engine_cfg)
        error_tag_allowlist = parse_error_tag_allowlist(engine_cfg)

        prompt_template = load_prompt_template(paths.prompt_path)

        logger.event(
            "bootstrap",
            "run_started",
            config_path=str(paths.config_path),
            input_jsonl=str(paths.input_jsonl),
            output_root=str(paths.run_root),
            model_root=str(paths.model_root),
            assist_model_path=str(paths.assist_model_path),
            prompt_path=str(paths.prompt_path) if paths.prompt_path else None,
            mode=mode,
            expected_roles=expected_roles,
            enable_model_assist=parse_bool(engine_cfg.get("enable_model_assist", False), default=False),
        )

        all_rows = read_jsonl(paths.input_jsonl)
        logger.event("load", "input_loaded", candidates=len(all_rows))

        allowed_splits = set(parse_allowed_splits(cfg))
        filtered_rows = [r for r in all_rows if str(r.get("split")) in allowed_splits]
        logger.event("load", "filtered_input", candidates=len(filtered_rows), splits=sorted(allowed_splits))

        bundles, ingest_stats = build_group_bundles(filtered_rows, expected_roles=expected_roles)

        max_samples = runtime_cfg.get("max_samples")
        if isinstance(max_samples, int) and max_samples > 0:
            bundles = bundles[:max_samples]

        logger.event("load", "grouped_input", groups=len(bundles), ingest_stats=ingest_stats)

        process_engine_cfg = dict(engine_cfg)
        process_engine_cfg["progress_interval"] = runtime_cfg.get("progress_interval", 200)

        group_decisions, rectified_rows, final_rows, parse_fail_count, fallback_count, model_invoked_count = process_groups(
            bundles=bundles,
            mode=mode,
            expected_roles=expected_roles,
            engine_cfg=process_engine_cfg,
            inference_cfg=inference_cfg,
            paths=paths,
            prompt_template=prompt_template,
            logger=logger,
            action_allowlist=action_allowlist,
            error_tag_allowlist=error_tag_allowlist,
        )

        quality = quality_gates(
            group_decisions=group_decisions,
            rectified_rows=rectified_rows,
            final_rows=final_rows,
            expected_roles=expected_roles,
            ingest_stats=ingest_stats,
            action_allowlist=action_allowlist,
            error_tag_allowlist=error_tag_allowlist,
        )
        write_json(paths.run_root / "reports" / "quality_gates.json", quality)
        logger.event("quality", "quality_report_written", **quality)

        strict_mode = parse_bool(cfg.get("quality", {}).get("strict", False), default=False)
        if strict_mode and not quality.get("passed", False):
            raise RuntimeError(f"Phase3 quality gate failed under strict mode: {quality}")

        summary = build_summary(
            group_decisions=group_decisions,
            rectified_rows=rectified_rows,
            final_rows=final_rows,
            parse_fail_count=parse_fail_count,
            fallback_count=fallback_count,
            model_invoked_count=model_invoked_count,
        )
        write_json(paths.run_root / "reports" / "summary.json", summary)
        logger.event("report", "summary_written", **summary)

        phase4_trace_pack_rows = build_phase4_trace_pack_rows(
            group_decisions=group_decisions,
            rectified_rows=rectified_rows,
            final_rows=final_rows,
        )

        if not cli_args.dry_run:
            write_candidate_outputs(
                run_root=paths.run_root,
                expected_roles=expected_roles,
                group_decisions=group_decisions,
                rectified_rows=rectified_rows,
                final_rows=final_rows,
                phase4_trace_pack_rows=phase4_trace_pack_rows,
            )

            # Stable latest pointer for downstream consumption (for example, phase4 distillation).
            write_candidate_outputs(
                run_root=paths.output_root / "latest",
                expected_roles=expected_roles,
                group_decisions=group_decisions,
                rectified_rows=rectified_rows,
                final_rows=final_rows,
                phase4_trace_pack_rows=phase4_trace_pack_rows,
            )

            logger.event(
                "export",
                "candidates_written",
                group_decisions=len(group_decisions),
                rectified=len(rectified_rows),
                final=len(final_rows),
                phase4_trace_pack=len(phase4_trace_pack_rows),
            )
        else:
            logger.event("export", "dry_run_skip_large_writes")

        run_summary = {
            "run_id": run_id,
            "created_at": utc_now(),
            "config_path": str(paths.config_path),
            "input_jsonl": str(paths.input_jsonl),
            "output_root": str(paths.run_root),
            "mode": mode,
            "assist_model_path": str(paths.assist_model_path),
            "prompt_path": str(paths.prompt_path) if paths.prompt_path else None,
            "expected_roles": expected_roles,
            "counts": {
                "input_candidates": len(filtered_rows),
                "input_groups": len(bundles),
                "output_group_decisions": len(group_decisions),
                "output_rectified_candidates": len(rectified_rows),
                "output_final_tuples": len(final_rows),
                "output_phase4_trace_pack": len(phase4_trace_pack_rows),
            },
            "quality": quality,
            "summary": summary,
        }

        summary_path = paths.run_root / "run_summary.json"
        write_json(summary_path, run_summary)
        logger.event("bootstrap", "run_completed", summary_path=str(summary_path))

        return run_summary
    finally:
        logger.close()


def main() -> None:
    run_phase3()


if __name__ == "__main__":
    main()
