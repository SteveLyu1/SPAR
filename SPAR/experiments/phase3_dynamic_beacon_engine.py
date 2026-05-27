from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import html
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from SPAR.experiments.api_adapter import derive_api_output_namespace, rewrite_output_namespace_path_text


LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
CJK_RE = re.compile(r"[\u4e00-\u9fff]")
FOCUS_TAG_RE = re.compile(r"</?focus>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")
GROUNDING_EDGE_CHARS = " \t\r\n\"'`“”‘’()[]{}<>«»「」『』【】《》,.;:!?/\\|+-_=*&^%$#@~，。；：！？、…"
ROLE_OUTPUT_KEY_MAP = {
    "victim": "victim",
    "moderator": "moderator",
    "cultural_bystander": "bystander",
}
OUTPUT_KEY_ROLE_MAP = {v: k for k, v in ROLE_OUTPUT_KEY_MAP.items()}
VALID_HARMFUL_DIRECTION = {"harmful", "non-harmful", "unknown"}
ZH_HATE_GROUPS = {"Sexism", "Region", "Racism", "LGBTQ", "others", "non-hate", "unknown"}
DEFAULT_EXPECTED_ROLES = ["victim", "moderator", "cultural_bystander"]
DEFAULT_HEURISTIC_SCORING: Dict[str, Dict[str, float]] = {
    "candidate_quality": {
        "argument_grounded": 0.38,
        "target_explicit": 0.12,
        "target_canonical_match": 0.12,
        "target_canonical_weak_match": 0.02,
        "target_canonical_conflict_penalty": 0.08,
        "implicit_target": 0.04,
        "group_consistent": 0.16,
        "rationale_present": 0.16,
        "label_known": 0.08,
        "lexicon_candidate_bonus": 0.05,
    },
    "cluster_similarity": {
        "target": 0.45,
        "argument": 0.45,
        "group": 0.05,
        "label": 0.05,
    },
    "cluster_scoring": {
        "support_role_bonus": 0.12,
        "lexicon_cluster_bonus": 0.08,
    },
    "consensus": {
        "min_top_cluster_score": 0.8,
        "margin_multiplier": 1.35,
        "alignment_similarity_threshold": 0.7,
    },
    "role_weighting": {
        "lexicon_role_bonus": 0.18,
        "victim_microaggression_bonus": 0.35,
        "moderator_strong_bonus": 0.25,
        "invalid_candidate_penalty_multiplier": 0.65,
    },
    "fallback_scoring": {
        "support_role_bonus": 0.15,
        "lexicon_cluster_bonus": 0.12,
        "moderator_floor_bonus": 0.18,
        "victim_microaggression_bonus": 0.18,
    },
}
AUDITOR_ROLE_ALIASES = {
    "victim": "victim",
    "moderator": "moderator",
    "bystander": "cultural_bystander",
    "cultural_bystander": "cultural_bystander",
}
_PHASE2_MODULE: Any = None


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    input_jsonl: Path
    output_root: Path
    run_root: Path
    model_root: Path
    auditor_model_path: Path
    regen_model_path: Path
    auditor_prompt_path: Optional[Path]
    auditor_prompt_map: Dict[str, Path]
    zh_lexicon_path: Path


@dataclass
class Candidate:
    candidate_id: str
    role: str
    role_output_key: str
    role_index: int
    language: str
    tuple_obj: Dict[str, Any]
    rationale: str
    raw_item: Dict[str, Any]
    lexicon_hit: bool
    quality_score: float
    issues: List[str]
    valid_for_fusion: bool


@dataclass
class Cluster:
    cluster_id: str
    members: List[Candidate] = field(default_factory=list)
    canonical_tuple: Dict[str, Any] = field(default_factory=dict)
    support_roles: List[str] = field(default_factory=list)
    cluster_score: float = 0.0
    lexicon_hit: bool = False


@dataclass
class PreparedGroupState:
    working_row: Dict[str, Any]
    candidates: List[Candidate]
    clusters: List[Cluster]
    signals: Dict[str, Any]
    beacons: List[Dict[str, str]]
    regeneration_trace: List[Dict[str, Any]]
    regenerated_roles: List[str]
    fallback_cluster: Optional[Cluster]
    fallback_tuple: Dict[str, Any]
    fallback_rationale: str


@dataclass
class ResumeState:
    group_decisions: List[Dict[str, Any]]
    rectified_rows: List[Dict[str, Any]]
    final_rows: List[Dict[str, Any]]
    completed_groups: int
    prepared_groups: int
    current_bucket: str
    last_completed_group_id: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def normalize_prompt_language_key(language: Optional[str]) -> str:
    lang = str(language or "").strip().lower()
    if lang.startswith("zh"):
        return "zh"
    if lang.startswith("en"):
        return "en"
    return lang or "default"


def dedupe_preserve_order(items: Sequence[str], case_insensitive: bool = False) -> List[str]:
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


def make_sha1(*parts: Any) -> str:
    return hashlib.sha1("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()


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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def cleanup_session_vllm_workers(
    logger: Optional["EventLogger"],
    *,
    reason: str,
    orphan_only: bool,
    wait_seconds: float = 2.0,
) -> List[int]:
    try:
        current_pid = os.getpid()
        current_sid = os.getsid(0)
        current_pgid = os.getpgid(0)
    except Exception as exc:
        if logger is not None:
            logger.event(
                "runtime",
                "vllm_worker_cleanup_failed",
                level="WARNING",
                reason=reason,
                orphan_only=orphan_only,
                error=f"session_lookup_failed: {exc}",
            )
        return []

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,sess=,pgid=,comm="],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        if logger is not None:
            logger.event(
                "runtime",
                "vllm_worker_cleanup_failed",
                level="WARNING",
                reason=reason,
                orphan_only=orphan_only,
                error=f"ps_failed: {exc}",
            )
        return []

    killed: List[int] = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            sess = int(parts[2])
            pgid = int(parts[3])
        except ValueError:
            continue
        command_name = parts[4]
        if pid == current_pid:
            continue
        if sess != current_sid or pgid != current_pgid:
            continue
        if orphan_only and ppid != 1:
            continue
        if not command_name.startswith("VLLM::Worker"):
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    if killed:
        time.sleep(wait_seconds)
        if logger is not None:
            logger.event(
                "runtime",
                "vllm_worker_cleanup",
                reason=reason,
                orphan_only=orphan_only,
                killed_pids=killed,
                session_id=current_sid,
                process_group_id=current_pgid,
            )
    return killed


def cleanup_stale_vllm_workers_for_user(
    logger: Optional["EventLogger"],
    *,
    reason: str,
    min_age_seconds: float = 90.0,
    wait_seconds: float = 2.0,
) -> List[int]:
    current_user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not current_user:
        return []

    try:
        current_pid = os.getpid()
    except Exception:
        current_pid = -1

    try:
        result = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,user=,etimes=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        if logger is not None:
            logger.event(
                "runtime",
                "stale_vllm_worker_cleanup_failed",
                level="WARNING",
                reason=reason,
                error=f"ps_failed: {exc}",
            )
        return []

    killed: List[int] = []
    for raw_line in result.stdout.splitlines():
        parts = raw_line.strip().split(None, 4)
        if len(parts) != 5:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
            etimes = float(parts[3])
        except ValueError:
            continue
        user = parts[2]
        args = parts[4]
        if pid == current_pid or user != current_user:
            continue
        if "VLLM::Worker" not in args:
            continue
        if etimes < float(min_age_seconds):
            continue
        if ppid != 1:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
            killed.append(pid)
        except ProcessLookupError:
            continue
        except Exception:
            continue

    if killed:
        time.sleep(wait_seconds)
        if logger is not None:
            logger.event(
                "runtime",
                "stale_vllm_worker_cleanup",
                reason=reason,
                min_age_seconds=min_age_seconds,
                killed_pids=killed,
                current_user=current_user,
            )
    return killed


def build_vllm_init_retry_cfg(
    base_cfg: Dict[str, Any],
    *,
    engine_kind: str,
    retry_attempt: int,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    if retry_attempt <= 0:
        return cfg
    if str(cfg.get("backend") or "").strip().lower() != "vllm":
        return cfg

    if engine_kind == "regen":
        retry_gpu_util = safe_float(cfg.get("retry_gpu_memory_utilization", 0.82), default=0.82)
        retry_max_num_seqs = int(max(1, cfg.get("retry_max_num_seqs", 4)))
        retry_max_num_batched_tokens = int(max(2048, cfg.get("retry_max_num_batched_tokens", 4096)))
    else:
        retry_gpu_util = safe_float(cfg.get("retry_gpu_memory_utilization", 0.88), default=0.88)
        retry_max_num_seqs = int(max(1, cfg.get("retry_max_num_seqs", 8)))
        retry_max_num_batched_tokens = int(max(4096, cfg.get("retry_max_num_batched_tokens", 12288)))

    current_gpu_util = safe_float(cfg.get("gpu_memory_utilization", 0.9), default=0.9)
    current_max_num_seqs = int(max(1, cfg.get("max_num_seqs", retry_max_num_seqs)))
    current_max_num_batched_tokens = int(max(2048, cfg.get("max_num_batched_tokens", retry_max_num_batched_tokens)))

    cfg["gpu_memory_utilization"] = min(current_gpu_util, retry_gpu_util)
    cfg["max_num_seqs"] = min(current_max_num_seqs, retry_max_num_seqs)
    cfg["max_num_batched_tokens"] = min(current_max_num_batched_tokens, retry_max_num_batched_tokens)
    return cfg


def build_backend_fallback_cfg(
    base_cfg: Dict[str, Any],
    *,
    fallback_backend: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    cfg["backend"] = fallback_backend
    if fallback_backend == "qwen_local":
        cfg.setdefault("device_map", "balanced")
        cfg.setdefault("preferred_gpu_mem_ratio", 0.9)
        cfg.setdefault("other_gpu_mem_ratio", 0.8)
        cfg.setdefault("min_multi_gpu_mem_ratio", 0.8)
    return cfg


def resolve_init_fallback_backend(
    inference_cfg: Dict[str, Any],
    model_path: Path,
) -> str:
    requested = str(inference_cfg.get("fallback_backend_on_init_failure", "")).strip().lower()
    current_backend = str(inference_cfg.get("backend", "")).strip().lower()
    if not requested or requested == current_backend:
        return ""
    model_name = model_path.name
    phase2 = load_phase2_module()
    if phase2.is_qwen35_family_model(model_name):
        return ""
    return requested


def resolve_path(workspace_root: Path, config_dir: Path, raw: str) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if raw.startswith("./") or raw.startswith("../"):
        return (config_dir / path).resolve()
    return (workspace_root / path).resolve()


def set_nested(cfg: Dict[str, Any], keys: Sequence[str], value: Any) -> None:
    node = cfg
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_heuristic_scoring(cfg: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    raw = cfg.get("phase3", {}).get("heuristic_scoring", {})
    if not isinstance(raw, dict):
        raw = {}
    merged = deep_merge_dict(DEFAULT_HEURISTIC_SCORING, raw)
    out: Dict[str, Dict[str, float]] = {}
    for section, values in merged.items():
        if not isinstance(values, dict):
            continue
        out[section] = {str(k): safe_float(v) for k, v in values.items()}
    return out


def normalize_auditor_role_name(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    lowered = text.lower()
    return AUDITOR_ROLE_ALIASES.get(lowered, lowered)


def scrub_placeholder_literal(value: Any) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    lowered = text.lower()
    placeholder_literals = {
        "string",
        "string_or_null",
        "null_or_string",
        "group_or_null",
        "target_or_implicit_target",
        "short_string_under_120_words",
        "short_string",
    }
    if lowered in placeholder_literals:
        return None
    return text


def load_phase2_module() -> Any:
    global _PHASE2_MODULE
    if _PHASE2_MODULE is not None:
        return _PHASE2_MODULE
    phase2_path = Path(__file__).resolve().with_name("phase2_role_generator.py")
    module_name = "spar_phase2_role_generator_dynamic"
    spec = importlib.util.spec_from_file_location(module_name, str(phase2_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load phase2 module from {phase2_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _PHASE2_MODULE = module
    return module


class EventLogger:
    def __init__(self, run_id: str, log_path: Path, level: str = "INFO", append: bool = False) -> None:
        self.run_id = run_id
        self.log_path = log_path
        self.level = level if level in LOG_LEVEL_ORDER else "INFO"
        ensure_dir(log_path.parent)
        self._fh = log_path.open("a" if append else "w", encoding="utf-8")

    def event(self, stage: str, event: str, level: str = "INFO", **payload: Any) -> None:
        safe_level = level if level in LOG_LEVEL_ORDER else "INFO"
        if LOG_LEVEL_ORDER[safe_level] < LOG_LEVEL_ORDER[self.level]:
            return
        self._fh.write(
            json.dumps(
                {
                    "ts": utc_now(),
                    "run_id": self.run_id,
                    "level": safe_level,
                    "stage": stage,
                    "event": event,
                    "payload": payload,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def parse_args() -> argparse.Namespace:
    default_config = str(Path(__file__).resolve().parents[1] / "configs" / "phase3_en_api.json")
    parser = argparse.ArgumentParser(description="SPAR-Hate Phase 3 Arbitrate Dynamic Beacon Engine")
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--api-profile", default=None, help="Override API profile for auditor/regen, e.g. gpt or deepseek")
    parser.add_argument("--input-jsonl", default=None, help="Override phase2 input JSONL")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--auditor-model-path", default=None, help="Override auditor model path")
    parser.add_argument("--regen-model-path", default=None, help="Override regeneration model path")
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N groups")
    parser.add_argument("--splits", default=None, help="Comma-separated splits")
    parser.add_argument("--languages", default=None, help="Comma-separated languages, e.g. zh,en")
    parser.add_argument("--dry-run", action="store_true", help="Skip candidate JSONL writes")
    parser.add_argument("--strict", action="store_true", help="Fail on quality gate failure")
    parser.add_argument("--no-resume-from-checkpoint", action="store_true", help="Ignore existing checkpoint/candidate exports and restart this run id from scratch")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None)
    return parser.parse_args()


def load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    if args.api_profile is not None:
        set_nested(cfg, ["inference", "auditor", "api_profile"], args.api_profile)
        set_nested(cfg, ["inference", "regen", "api_profile"], args.api_profile)
    if args.api_profile is not None:
        set_nested(cfg, ["inference", "auditor", "backend"], "api")
        set_nested(cfg, ["inference", "regen", "backend"], "api")
    overrides = [
        (args.input_jsonl, ["paths", "input_jsonl"]),
        (args.output_root, ["paths", "output_root"]),
        (args.auditor_model_path, ["paths", "auditor_model_path"]),
        (args.regen_model_path, ["paths", "regen_model_path"]),
        (args.max_samples, ["runtime", "max_samples"]),
        (args.log_level, ["logging", "level"]),
    ]
    for value, keys in overrides:
        if value is not None:
            set_nested(cfg, keys, value)
    if args.splits:
        set_nested(cfg, ["runtime", "default_splits"], [s.strip() for s in args.splits.split(",") if s.strip()])
    if args.languages:
        set_nested(
            cfg,
            ["runtime", "default_languages"],
            [normalize_prompt_language_key(s) for s in args.languages.split(",") if str(s).strip()],
        )
    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)
    if args.no_resume_from_checkpoint:
        set_nested(cfg, ["phase3", "resume_from_checkpoint"], False)


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg.get("paths", {})
    auditor_inference_cfg = cfg.get("inference", {}).get("auditor", {}) if isinstance(cfg.get("inference", {}).get("auditor"), dict) else {}
    auditor_api_profiles = auditor_inference_cfg.get("api_profiles") if isinstance(auditor_inference_cfg.get("api_profiles"), dict) else None
    auditor_backend = str(auditor_inference_cfg.get("backend", "transformers")).strip().lower()
    output_root_raw = str(cfg_paths["output_root"])
    input_jsonl_raw = str(cfg_paths["input_jsonl"])
    if auditor_backend == "api":
        namespace = derive_api_output_namespace(auditor_inference_cfg, api_profiles=auditor_api_profiles)
        output_root_raw = str(rewrite_output_namespace_path_text(output_root_raw, namespace))
        input_jsonl_raw = str(rewrite_output_namespace_path_text(input_jsonl_raw, namespace))
    output_root = resolve_path(workspace_root, config_dir, output_root_raw)
    run_id = args.run_id or f"phase3_dynamic_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    auditor_prompt_raw = cfg_paths.get("auditor_prompt_path")
    auditor_prompt_map_raw = cfg_paths.get("auditor_prompt_map")
    auditor_prompt_path = None
    if isinstance(auditor_prompt_raw, str) and auditor_prompt_raw.strip():
        auditor_prompt_path = resolve_path(workspace_root, config_dir, auditor_prompt_raw.strip())
    auditor_prompt_map: Dict[str, Path] = {}
    if isinstance(auditor_prompt_map_raw, dict):
        for raw_lang, raw_path in auditor_prompt_map_raw.items():
            if not isinstance(raw_path, str) or not raw_path.strip():
                continue
            lang_key = normalize_prompt_language_key(str(raw_lang))
            auditor_prompt_map[lang_key] = resolve_path(workspace_root, config_dir, raw_path.strip())
    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        input_jsonl=resolve_path(workspace_root, config_dir, input_jsonl_raw),
        output_root=output_root,
        run_root=output_root / run_id,
        model_root=resolve_path(workspace_root, config_dir, str(cfg_paths["model_root"])),
        auditor_model_path=resolve_path(workspace_root, config_dir, str(cfg_paths["auditor_model_path"])),
        regen_model_path=resolve_path(workspace_root, config_dir, str(cfg_paths["regen_model_path"])),
        auditor_prompt_path=auditor_prompt_path,
        auditor_prompt_map=auditor_prompt_map,
        zh_lexicon_path=resolve_path(workspace_root, config_dir, str(cfg_paths["zh_lexicon_path"])),
    )


def parse_expected_roles(cfg: Dict[str, Any]) -> List[str]:
    roles = cfg.get("phase3", {}).get("expected_roles", list(DEFAULT_EXPECTED_ROLES))
    if not isinstance(roles, list):
        roles = list(DEFAULT_EXPECTED_ROLES)
    cleaned = [str(r).strip().lower() for r in roles if str(r).strip()]
    return dedupe_preserve_order(cleaned, case_insensitive=False) or list(DEFAULT_EXPECTED_ROLES)


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    splits = cfg.get("runtime", {}).get("default_splits", ["train", "dev", "test"])
    if not isinstance(splits, list):
        return ["train", "dev", "test"]
    cleaned = [str(s).strip() for s in splits if str(s).strip()]
    return cleaned or ["train", "dev", "test"]


def parse_allowed_languages(cfg: Dict[str, Any]) -> List[str]:
    languages = cfg.get("runtime", {}).get("default_languages", ["zh", "en"])
    if not isinstance(languages, list):
        return ["zh", "en"]
    cleaned = [normalize_prompt_language_key(x) for x in languages if str(x).strip()]
    cleaned = [x for x in cleaned if x]
    return dedupe_preserve_order(cleaned, case_insensitive=False) or ["zh", "en"]


def detect_language(text: Optional[str]) -> str:
    if text and CJK_RE.search(text):
        return "zh"
    return "en"


def parse_sample_id_from_group_id(group_id: Any) -> Optional[str]:
    gid = normalize_string(group_id)
    if gid is None or "_g" not in gid:
        return None
    sample_id, suffix = gid.rsplit("_g", 1)
    if not sample_id or not suffix.isdigit():
        return None
    return sample_id


def normalize_binary_label(value: Any) -> Optional[int]:
    if value in {1, "1", True, "true", "True", "yes", "YES"}:
        return 1
    if value in {0, "0", False, "false", "False", "no", "NO"}:
        return 0
    return None


def harmful_direction_from_label(value: Any) -> str:
    label = normalize_binary_label(value)
    if label == 1:
        return "harmful"
    if label == 0:
        return "non-harmful"
    return "unknown"


def normalize_final_tuple(tuple_obj: Dict[str, Any], language: str) -> Dict[str, Any]:
    target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    argument = normalize_string(tuple_obj.get("argument"))
    group = normalize_string(tuple_obj.get("group"))
    if language == "zh":
        harmful = normalize_binary_label(tuple_obj.get("hateful") if "hateful" in tuple_obj else tuple_obj.get("harmful"))
        if harmful == 0:
            group = "non-hate"
        else:
            group = group or "unknown"
    else:
        harmful = normalize_binary_label(tuple_obj.get("harmful"))
        group = None
    direction = harmful_direction_from_label(harmful)
    return {
        "target": target,
        "argument": argument,
        "group": group,
        "harmful": harmful,
        "harmful_direction": direction,
    }


def normalized_identity_text(value: Any, language: str) -> str:
    text = normalize_string(value)
    if text is None:
        return "__missing__"
    return text.casefold() if language == "en" else text


def normalize_spacing(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def normalize_grounding_text(value: Any, language: str) -> str:
    text = normalize_string(value) or ""
    if not text:
        return ""
    text = normalize_spacing(FOCUS_TAG_RE.sub("", text))
    return text.casefold() if language == "en" else text


def trimmed_grounding_variants(value: Any, language: str) -> List[str]:
    base = normalize_grounding_text(value, language)
    if not base:
        return []
    variants = [base]
    stripped = base.strip(GROUNDING_EDGE_CHARS)
    if stripped and stripped != base:
        variants.append(stripped)
    return dedupe_preserve_order([item for item in variants if item], case_insensitive=(language == "en"))


def is_argument_grounded_in_focus(argument: Any, focus_inner: str, language: str) -> bool:
    normalized_focus = normalize_grounding_text(focus_inner, language)
    if not normalized_focus:
        return True
    for variant in trimmed_grounding_variants(argument, language):
        if variant and variant in normalized_focus:
            return True
    return False


def final_tuple_identity_key(tuple_obj: Dict[str, Any], language: str) -> Tuple[str, str, str, str]:
    harmful = normalize_binary_label(tuple_obj.get("harmful"))
    group = normalize_string(tuple_obj.get("group"))
    if language == "en":
        group = None
    return (
        normalized_identity_text(tuple_obj.get("target") or "IMPLICIT_TARGET", language),
        normalized_identity_text(tuple_obj.get("argument"), language),
        normalized_identity_text(group, "en"),
        harmful_direction_from_label(harmful),
    )


def extract_focus_inner_text(focus_text: str, fallback: str) -> str:
    match = re.search(r"<focus>([\s\S]*?)</focus>", focus_text, flags=re.IGNORECASE)
    if match:
        return normalize_spacing(match.group(1))
    return normalize_spacing(FOCUS_TAG_RE.sub("", fallback))


def normalize_prediction_keys(row: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    raw = row.get("phase2_predictions") if isinstance(row.get("phase2_predictions"), dict) else {}
    normalized = {"victim": [], "moderator": [], "bystander": []}
    for key, value in raw.items():
        key_text = str(key).strip().lower()
        if key_text == "cultural_bystander":
            key_text = "bystander"
        if key_text not in normalized:
            continue
        normalized[key_text] = value if isinstance(value, list) else []
    return normalized


def normalize_group_row(row: Dict[str, Any]) -> Dict[str, Any]:
    group_id = normalize_string(row.get("group_id")) or make_sha1("group", row)[:16]
    sample_id = normalize_string(row.get("sample_id")) or parse_sample_id_from_group_id(group_id) or group_id
    group_text = normalize_string(row.get("group_text")) or ""
    focus_text = normalize_string(row.get("focus_text")) or (f"<focus>{group_text}</focus>" if group_text else "<focus></focus>")
    language = normalize_string(row.get("language")) or detect_language(group_text or focus_text)
    split = normalize_string(row.get("split")) or "train"
    source = normalize_string(row.get("source")) or "unknown"
    canonical_target = normalize_string(row.get("canonical_target")) or "IMPLICIT_TARGET"
    local_group_id = row.get("local_group_id")
    try:
        local_group_id = int(local_group_id)
    except (TypeError, ValueError):
        local_group_id = 1
    return {
        "group_id": group_id,
        "sample_id": sample_id,
        "original_id": normalize_string(row.get("original_id")) or sample_id,
        "source": source,
        "language": language,
        "split": split,
        "local_group_id": max(1, int(local_group_id)),
        "group_text": group_text,
        "focus_text": focus_text,
        "canonical_target": canonical_target,
        "escalation_flag": bool(row.get("escalation_flag")),
        "sentence_ids": row.get("sentence_ids") if isinstance(row.get("sentence_ids"), list) else [],
        "phase2_predictions": normalize_prediction_keys(row),
    }


def convert_flat_rows_to_group_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        group_id = normalize_string(row.get("group_id"))
        if group_id is None:
            continue
        bucket = grouped.setdefault(
            group_id,
            {
                "group_id": group_id,
                "sample_id": normalize_string(row.get("sample_id")) or parse_sample_id_from_group_id(group_id) or group_id,
                "original_id": normalize_string(row.get("original_id")) or "",
                "source": normalize_string(row.get("source")) or "unknown",
                "language": normalize_string(row.get("language")) or "en",
                "split": normalize_string(row.get("split")) or "train",
                "local_group_id": row.get("local_group_id") or 1,
                "group_text": normalize_string(row.get("group_text")) or "",
                "focus_text": normalize_string(row.get("focus_text")) or "",
                "canonical_target": normalize_string(row.get("canonical_target")) or "IMPLICIT_TARGET",
                "escalation_flag": bool(row.get("escalation_flag")),
                "sentence_ids": row.get("sentence_ids") if isinstance(row.get("sentence_ids"), list) else [],
                "phase2_predictions": {"victim": [], "moderator": [], "bystander": []},
            },
        )
        role = normalize_string(row.get("role")) or "unknown"
        role_key = ROLE_OUTPUT_KEY_MAP.get(role.lower(), role.lower())
        if role_key == "cultural_bystander":
            role_key = "bystander"
        if role_key not in bucket["phase2_predictions"]:
            continue
        if isinstance(row.get("prediction"), dict):
            item = dict(row["prediction"])
        else:
            predicted_tuple = row.get("predicted_tuple") if isinstance(row.get("predicted_tuple"), dict) else {}
            rationale = normalize_string(row.get("rationale")) or ""
            item = {
                "rationale": rationale,
                "target": normalize_string(predicted_tuple.get("target")) or "IMPLICIT_TARGET",
                "argument": normalize_string(predicted_tuple.get("argument")) or "",
            }
            if bucket["language"] == "zh":
                item["group"] = normalize_string(predicted_tuple.get("group")) or "unknown"
                item["hateful"] = normalize_binary_label(predicted_tuple.get("harmful"))
            else:
                item["harmful"] = normalize_binary_label(predicted_tuple.get("harmful"))
        bucket["phase2_predictions"][role_key].append(item)
    return [normalize_group_row(row) for row in grouped.values()]


def load_input_group_rows(path: Path) -> List[Dict[str, Any]]:
    rows = read_jsonl(path)
    if not rows:
        return []
    if any(isinstance(row.get("phase2_predictions"), dict) for row in rows):
        return [normalize_group_row(row) for row in rows]
    return convert_flat_rows_to_group_rows(rows)


def tokenize_for_similarity(text: Optional[str], language: str) -> Set[str]:
    normalized = normalize_string(text)
    if normalized is None:
        return set()
    if language == "zh" or CJK_RE.search(normalized):
        return {ch for ch in normalized if ch.strip()}
    return set(re.findall(r"[a-z0-9]+", normalized.lower()))


def text_similarity(a: Optional[str], b: Optional[str], language: str) -> float:
    ta = normalize_string(a)
    tb = normalize_string(b)
    if ta is None or tb is None:
        return 0.0
    if ta == tb:
        return 1.0
    if ta in tb or tb in ta:
        shorter = min(len(ta), len(tb))
        if shorter >= 2:
            return 0.92
    set_a = tokenize_for_similarity(ta, language)
    set_b = tokenize_for_similarity(tb, language)
    if not set_a or not set_b:
        return 0.0
    return round(len(set_a & set_b) / len(set_a | set_b), 4)


def zh_group_consistent(group: Optional[str], harmful: Optional[int]) -> bool:
    if harmful == 0:
        return normalize_string(group) == "non-hate"
    if harmful == 1:
        return normalize_string(group) in ZH_HATE_GROUPS - {"non-hate"}
    return normalize_string(group) in ZH_HATE_GROUPS


def candidate_quality(
    group_row: Dict[str, Any],
    role: str,
    tuple_obj: Dict[str, Any],
    rationale: str,
    lexicon_hit: bool,
    threshold: float,
    scoring_cfg: Dict[str, Dict[str, float]],
) -> Tuple[float, List[str], bool]:
    issues: List[str] = []
    language = str(group_row.get("language") or "en")
    focus_inner = extract_focus_inner_text(str(group_row.get("focus_text") or ""), str(group_row.get("group_text") or ""))
    argument = normalize_string(tuple_obj.get("argument"))
    target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
    group = normalize_string(tuple_obj.get("group"))
    harmful = normalize_binary_label(tuple_obj.get("harmful"))
    canonical_target = normalize_string(group_row.get("canonical_target")) or "IMPLICIT_TARGET"
    quality_cfg = scoring_cfg.get("candidate_quality", {})

    score = 0.0

    if argument:
        if is_argument_grounded_in_focus(argument=argument, focus_inner=focus_inner, language=language):
            score += quality_cfg.get("argument_grounded", 0.38)
        else:
            issues.append("argument_not_grounded")
    else:
        issues.append("missing_argument")

    if target and target != "IMPLICIT_TARGET":
        score += quality_cfg.get("target_explicit", 0.12)
        if canonical_target != "IMPLICIT_TARGET":
            if target == canonical_target or target in canonical_target or canonical_target in target:
                score += quality_cfg.get("target_canonical_match", 0.12)
            elif text_similarity(target, canonical_target, language) >= 0.5:
                score += quality_cfg.get("target_canonical_weak_match", 0.02)
            else:
                score -= quality_cfg.get("target_canonical_conflict_penalty", 0.0)
                issues.append("target_conflict_with_hint")
    else:
        score += quality_cfg.get("implicit_target", 0.04)
        if canonical_target != "IMPLICIT_TARGET":
            issues.append("implicit_target")

    if language == "zh":
        if zh_group_consistent(group, harmful):
            score += quality_cfg.get("group_consistent", 0.16)
        else:
            issues.append("group_label_inconsistent")
    else:
        score += quality_cfg.get("group_consistent", 0.16)

    if rationale:
        score += quality_cfg.get("rationale_present", 0.16)
        if len(rationale) > 140:
            issues.append("rationale_too_long")
    else:
        issues.append("empty_rationale")

    if harmful is not None:
        score += quality_cfg.get("label_known", 0.08)
    else:
        issues.append("unknown_label")

    if role == "cultural_bystander" and lexicon_hit:
        score += quality_cfg.get("lexicon_candidate_bonus", 0.05)

    score = round(clamp(score), 4)
    fatal_issues = {"missing_argument", "argument_not_grounded", "group_label_inconsistent"}
    valid = score >= threshold and not fatal_issues.intersection(set(issues))
    return score, issues, valid


def build_candidates(group_row: Dict[str, Any], threshold: float, scoring_cfg: Dict[str, Dict[str, float]]) -> List[Candidate]:
    language = str(group_row.get("language") or "en")
    candidates: List[Candidate] = []
    for role_output_key, items in (group_row.get("phase2_predictions") or {}).items():
        role = OUTPUT_KEY_ROLE_MAP.get(role_output_key, role_output_key)
        if not isinstance(items, list):
            continue
        for idx, item in enumerate(items, start=1):
            if not isinstance(item, dict):
                continue
            tuple_obj = normalize_final_tuple(item, language=language)
            rationale = normalize_string(item.get("rationale")) or ""
            lexicon_hit = role == "cultural_bystander" and rationale.startswith("[Lexicon Hit]")
            quality_score, issues, valid = candidate_quality(
                group_row=group_row,
                role=role,
                tuple_obj=tuple_obj,
                rationale=rationale,
                lexicon_hit=lexicon_hit,
                threshold=threshold,
                scoring_cfg=scoring_cfg,
            )
            candidates.append(
                Candidate(
                    candidate_id=f"{group_row['group_id']}:{role}:{idx}",
                    role=role,
                    role_output_key=role_output_key,
                    role_index=idx,
                    language=language,
                    tuple_obj=tuple_obj,
                    rationale=rationale,
                    raw_item=dict(item),
                    lexicon_hit=lexicon_hit,
                    quality_score=quality_score,
                    issues=issues,
                    valid_for_fusion=valid,
                )
            )
    return candidates


def candidate_cluster_similarity(candidate: Candidate, cluster: Cluster, scoring_cfg: Dict[str, Dict[str, float]]) -> float:
    if not cluster.members:
        return 0.0
    language = candidate.language
    ref = max(cluster.members, key=lambda x: x.quality_score)
    target_sim = text_similarity(candidate.tuple_obj.get("target"), ref.tuple_obj.get("target"), language)
    argument_sim = text_similarity(candidate.tuple_obj.get("argument"), ref.tuple_obj.get("argument"), language)
    group_sim = 1.0 if normalize_string(candidate.tuple_obj.get("group")) == normalize_string(ref.tuple_obj.get("group")) else 0.0
    label_sim = 1.0 if candidate.tuple_obj.get("harmful_direction") == ref.tuple_obj.get("harmful_direction") else 0.0
    similarity_cfg = scoring_cfg.get("cluster_similarity", {})
    return round(
        similarity_cfg.get("target", 0.45) * target_sim
        + similarity_cfg.get("argument", 0.45) * argument_sim
        + similarity_cfg.get("group", 0.05) * group_sim
        + similarity_cfg.get("label", 0.05) * label_sim,
        4,
    )


def weighted_vote(values: Sequence[Tuple[Any, float]], language: str) -> Any:
    votes: Dict[str, float] = defaultdict(float)
    raw: Dict[str, Any] = {}
    for value, weight in values:
        text = normalize_string(value)
        if text is None:
            continue
        key = text if language == "zh" else text.lower()
        votes[key] += max(0.01, weight)
        if key not in raw:
            raw[key] = value
    if not votes:
        return None
    winner = sorted(votes.items(), key=lambda x: (-x[1], x[0]))[0][0]
    return raw.get(winner)


def refresh_cluster_fields(cluster: Cluster, scoring_cfg: Dict[str, Dict[str, float]]) -> None:
    members = cluster.members
    if not members:
        cluster.canonical_tuple = {
            "target": "IMPLICIT_TARGET",
            "argument": None,
            "group": None,
            "harmful": None,
            "harmful_direction": "unknown",
        }
        cluster.support_roles = []
        cluster.cluster_score = 0.0
        cluster.lexicon_hit = False
        return
    language = members[0].language
    cluster.canonical_tuple = {
        "target": weighted_vote([(m.tuple_obj.get("target"), m.quality_score) for m in members], language=language) or "IMPLICIT_TARGET",
        "argument": weighted_vote([(m.tuple_obj.get("argument"), m.quality_score) for m in members], language=language),
        "group": weighted_vote([(m.tuple_obj.get("group"), m.quality_score) for m in members], language=language if language == "zh" else "en"),
        "harmful_direction": weighted_vote([(m.tuple_obj.get("harmful_direction"), m.quality_score) for m in members], language="en") or "unknown",
    }
    cluster.canonical_tuple["harmful"] = 1 if cluster.canonical_tuple["harmful_direction"] == "harmful" else 0 if cluster.canonical_tuple["harmful_direction"] == "non-harmful" else None
    cluster.support_roles = sorted({m.role for m in members})
    cluster.lexicon_hit = any(m.lexicon_hit for m in members)
    cluster_cfg = scoring_cfg.get("cluster_scoring", {})
    cluster.cluster_score = round(
        sum(m.quality_score for m in members)
        + cluster_cfg.get("support_role_bonus", 0.12) * len(cluster.support_roles)
        + (cluster_cfg.get("lexicon_cluster_bonus", 0.08) if cluster.lexicon_hit else 0.0),
        4,
    )


def cluster_candidates(candidates: Sequence[Candidate], threshold: float, scoring_cfg: Dict[str, Dict[str, float]]) -> List[Cluster]:
    ordered = sorted(candidates, key=lambda c: (not c.valid_for_fusion, -c.quality_score, c.role, c.role_index))
    clusters: List[Cluster] = []
    for candidate in ordered:
        best_idx = None
        best_score = -1.0
        for idx, cluster in enumerate(clusters):
            score = candidate_cluster_similarity(candidate, cluster, scoring_cfg)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None and best_score >= threshold:
            clusters[best_idx].members.append(candidate)
            refresh_cluster_fields(clusters[best_idx], scoring_cfg)
            continue
        cluster = Cluster(cluster_id=f"c{len(clusters) + 1}", members=[candidate])
        refresh_cluster_fields(cluster, scoring_cfg)
        clusters.append(cluster)
    clusters.sort(key=lambda c: (-c.cluster_score, -len(c.support_roles), c.cluster_id))
    for idx, cluster in enumerate(clusters, start=1):
        cluster.cluster_id = f"c{idx}"
    return clusters


def best_candidate_by_role(candidates: Sequence[Candidate]) -> Dict[str, Candidate]:
    best: Dict[str, Candidate] = {}
    for candidate in sorted(candidates, key=lambda c: (-c.quality_score, c.role, c.role_index)):
        if candidate.role not in best:
            best[candidate.role] = candidate
    return best


def detect_signals(
    group_row: Dict[str, Any],
    candidates: Sequence[Candidate],
    clusters: Sequence[Cluster],
    expected_roles: Sequence[str],
    scoring_cfg: Dict[str, Dict[str, float]],
) -> Dict[str, Any]:
    best = best_candidate_by_role(candidates)
    present_roles = [role for role in expected_roles if role in best]
    missing_roles = [role for role in expected_roles if role not in best]
    harmful_count = sum(1 for role in present_roles if best[role].tuple_obj.get("harmful") == 1)
    non_harmful_count = sum(1 for role in present_roles if best[role].tuple_obj.get("harmful") == 0)
    conflict_type = f"{harmful_count}:{non_harmful_count}" if present_roles else "0:0"

    defective_roles: List[str] = []
    for role in expected_roles:
        role_candidates = [c for c in candidates if c.role == role]
        if not role_candidates or not any(c.valid_for_fusion for c in role_candidates):
            defective_roles.append(role)

    lexicon_hit = any(c.lexicon_hit for c in candidates if c.role == "cultural_bystander")
    victim_best = best.get("victim")
    moderator_best = best.get("moderator")
    bystander_best = best.get("cultural_bystander")
    victim_micro = bool(
        victim_best
        and victim_best.tuple_obj.get("harmful") == 1
        and all(
            best.get(role) is None or best[role].tuple_obj.get("harmful") == 0
            for role in ["moderator", "cultural_bystander"]
        )
    )
    moderator_strong = bool(moderator_best and moderator_best.tuple_obj.get("harmful") == 1)

    focus_text = str(group_row.get("focus_text") or "")
    quote_keywords = ["引用", "反讽", "玩笑", "quoted", "quote", "sarcasm", "joking", "reclaimed"]
    quote_sources = [
        str(group_row.get("focus_text") or "").lower(),
        str(group_row.get("group_text") or "").lower(),
    ]
    lowered_quote_keywords = [k.lower() for k in quote_keywords]
    quote_caution = any(token in source for source in quote_sources for token in lowered_quote_keywords)

    complete_consensus = False
    consensus_aligned_roles: List[str] = []
    if clusters:
        top = clusters[0]
        next_score = clusters[1].cluster_score if len(clusters) > 1 else 0.0
        consensus_cfg = scoring_cfg.get("consensus", {})
        alignment_threshold = consensus_cfg.get("alignment_similarity_threshold", 0.7)
        top_label = top.canonical_tuple.get("harmful")
        for role in present_roles:
            candidate = best.get(role)
            if candidate is None:
                continue
            same_label = candidate.tuple_obj.get("harmful") == top_label
            aligned = candidate in top.members or candidate_cluster_similarity(candidate, top, scoring_cfg) >= alignment_threshold
            if same_label and aligned:
                consensus_aligned_roles.append(role)
        complete_consensus = (
            len(consensus_aligned_roles) == len(present_roles)
            and len(present_roles) >= 2
            and top.cluster_score >= max(consensus_cfg.get("min_top_cluster_score", 0.8), next_score * consensus_cfg.get("margin_multiplier", 1.35))
            and not defective_roles
        )

    lane = "consensus" if complete_consensus else "defective" if defective_roles else "conflict"
    return {
        "present_roles": present_roles,
        "missing_roles": missing_roles,
        "defective_roles": defective_roles,
        "initial_conflict_type": conflict_type,
        "lexicon_hit": lexicon_hit,
        "victim_microaggression": victim_micro,
        "moderator_strong": moderator_strong,
        "quote_caution": quote_caution,
        "lane": lane,
        "complete_consensus": complete_consensus,
        "cluster_count": len(clusters),
        "consensus_aligned_roles": consensus_aligned_roles,
    }


def assemble_beacons(signals: Dict[str, Any]) -> List[Dict[str, str]]:
    beacons: List[Dict[str, str]] = [
        {
            "id": "grounding",
            "text": "Final argument must be grounded in the focus span. Do not invent target or argument beyond the focus text and canonical target anchor.",
        }
    ]
    if signals.get("lexicon_hit"):
        beacons.append(
            {
                "id": "lexicon",
                "text": "Bystander rationale contains [Lexicon Hit]. Give the Chinese bystander materially higher interpretive weight on local coded slang than the other roles, but keep the final ruling auditor-led and cross-check victim harm, moderator floor, and reclaimed or quoted usage.",
            }
        )
    if signals.get("victim_microaggression"):
        beacons.append(
            {
                "id": "victim_microaggression",
                "text": "Only the victim strongly perceives harm. Examine whether implicit exclusion or stereotype logic is coherent even if moderator and bystander are conservative.",
            }
        )
    if signals.get("moderator_strong"):
        beacons.append(
            {
                "id": "moderator_floor",
                "text": "Moderator already detects explicit policy-risk or physical hostility. Use moderator as lower-bound anchor unless there is strong counter-evidence of quotation, irony, or reclaimed usage.",
            }
        )
    if signals.get("quote_caution"):
        beacons.append(
            {
                "id": "quote_caution",
                "text": "Check whether the text is quotation, rebuttal, irony, or reclaimed usage before assigning hateful label.",
            }
        )
    if signals.get("complete_consensus"):
        beacons.append(
            {
                "id": "consensus",
                "text": "Roles already align. Extract the most precise grounded tuple and compress the shared rationale into one short final ruling.",
            }
        )
    return beacons


def role_weight(candidate: Candidate, signals: Dict[str, Any], scoring_cfg: Dict[str, Dict[str, float]]) -> float:
    role_cfg = scoring_cfg.get("role_weighting", {})
    weight = 1.0 + candidate.quality_score
    if candidate.role == "cultural_bystander" and candidate.lexicon_hit and signals.get("lexicon_hit"):
        weight += role_cfg.get("lexicon_role_bonus", 0.18)
    if candidate.role == "victim" and signals.get("victim_microaggression") and candidate.tuple_obj.get("harmful") == 1:
        weight += role_cfg.get("victim_microaggression_bonus", 0.35)
    if candidate.role == "moderator" and signals.get("moderator_strong") and candidate.tuple_obj.get("harmful") == 1:
        weight += role_cfg.get("moderator_strong_bonus", 0.25)
    if not candidate.valid_for_fusion:
        weight *= role_cfg.get("invalid_candidate_penalty_multiplier", 0.65)
    return round(weight, 4)


def choose_cluster_fallback(clusters: Sequence[Cluster], signals: Dict[str, Any], scoring_cfg: Dict[str, Dict[str, float]]) -> Optional[Cluster]:
    if not clusters:
        return None
    fallback_cfg = scoring_cfg.get("fallback_scoring", {})
    scored: List[Tuple[float, Cluster]] = []
    for cluster in clusters:
        score = 0.0
        for member in cluster.members:
            score += role_weight(member, signals, scoring_cfg)
        score += fallback_cfg.get("support_role_bonus", 0.15) * len(cluster.support_roles)
        if cluster.lexicon_hit and signals.get("lexicon_hit"):
            score += fallback_cfg.get("lexicon_cluster_bonus", 0.12)
        if signals.get("moderator_strong") and any(m.role == "moderator" and m.tuple_obj.get("harmful") == 1 for m in cluster.members):
            score += fallback_cfg.get("moderator_floor_bonus", 0.18)
        if signals.get("victim_microaggression") and any(m.role == "victim" and m.tuple_obj.get("harmful") == 1 for m in cluster.members):
            score += fallback_cfg.get("victim_microaggression_bonus", 0.18)
        scored.append((round(score, 4), cluster))
    scored.sort(key=lambda x: (-x[0], x[1].cluster_id))
    return scored[0][1]


def build_deterministic_rationale(cluster: Optional[Cluster], signals: Dict[str, Any], language: str) -> str:
    if cluster is None:
        return "Insufficient grounded role evidence; defaulted to conservative non-harmful judgment."
    roles = "/".join(cluster.support_roles) or "roles"
    tuple_obj = cluster.canonical_tuple
    if language == "zh":
        label_text = "有害" if tuple_obj.get("harmful") == 1 else "无害"
        return f"{roles} 综合指向目标“{tuple_obj.get('target') or 'IMPLICIT_TARGET'}”，核心论据为“{tuple_obj.get('argument') or ''}”，最终判为{label_text}。"
    label_text = "harmful" if tuple_obj.get("harmful") == 1 else "non-harmful"
    return f"{roles} jointly ground the target '{tuple_obj.get('target') or 'IMPLICIT_TARGET'}' on argument '{tuple_obj.get('argument') or ''}', so the final ruling is {label_text}."


def cluster_member_quality_avg(cluster: Cluster) -> float:
    if not cluster.members:
        return 0.0
    return round(sum(member.quality_score for member in cluster.members) / len(cluster.members), 4)


def build_cluster_final_tuple(cluster: Cluster, focus_text: str, language: str) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    normalized = normalize_final_tuple(cluster.canonical_tuple, language=language)
    return refine_final_tuple_spans(
        final_tuple=normalized,
        selected_cluster=cluster,
        focus_text=focus_text,
        language=language,
    )


def select_retained_clusters(
    clusters: Sequence[Cluster],
    primary_cluster: Optional[Cluster],
    signals: Dict[str, Any],
    cfg: Dict[str, Any],
) -> List[Cluster]:
    phase3_cfg = cfg.get("phase3", {}) if isinstance(cfg.get("phase3"), dict) else {}
    retention_cfg = phase3_cfg.get("multi_cluster_retention", {})
    if not isinstance(retention_cfg, dict):
        retention_cfg = {}

    enabled = parse_bool(retention_cfg.get("enabled", True), default=True)
    if not enabled:
        return [primary_cluster] if primary_cluster is not None else []

    lane = str(signals.get("lane") or "")
    allow_on_consensus = parse_bool(retention_cfg.get("allow_on_consensus", False), default=False)
    if lane == "consensus" and not allow_on_consensus:
        return [primary_cluster] if primary_cluster is not None else []

    max_clusters = max(1, int(retention_cfg.get("max_clusters", 2) or 2))
    min_cluster_score = safe_float(retention_cfg.get("min_cluster_score", 1.05), default=1.05)
    min_avg_member_quality = safe_float(retention_cfg.get("min_avg_member_quality", 0.78), default=0.78)
    relative_score_floor = safe_float(retention_cfg.get("relative_score_floor", 0.4), default=0.4)
    min_support_roles = max(1, int(retention_cfg.get("min_support_roles", 2) or 2))
    keep_explicit_target_harmful_singletons = parse_bool(
        retention_cfg.get("keep_explicit_target_harmful_singletons", True),
        default=True,
    )
    keep_lexicon_singletons = parse_bool(retention_cfg.get("keep_lexicon_singletons", False), default=False)
    keep_nonharmful_implicit_singletons = parse_bool(
        retention_cfg.get("keep_nonharmful_implicit_singletons", False),
        default=False,
    )
    require_valid_member = parse_bool(retention_cfg.get("require_valid_member", True), default=True)
    reject_target_conflict_singletons = parse_bool(
        retention_cfg.get("reject_target_conflict_singletons", True),
        default=True,
    )

    retained: List[Cluster] = []
    seen_keys: Set[Tuple[str, str, str, str]] = set()
    language = primary_cluster.members[0].language if primary_cluster and primary_cluster.members else "en"
    top_score = primary_cluster.cluster_score if primary_cluster is not None else (clusters[0].cluster_score if clusters else 0.0)

    if primary_cluster is not None:
        retained.append(primary_cluster)
        seen_keys.add(final_tuple_identity_key(primary_cluster.canonical_tuple, language))

    for cluster in clusters:
        if primary_cluster is not None and cluster.cluster_id == primary_cluster.cluster_id:
            continue
        if len(retained) >= max_clusters:
            break

        tuple_obj = cluster.canonical_tuple
        tuple_key = final_tuple_identity_key(tuple_obj, language)
        if tuple_key in seen_keys:
            continue

        target = normalize_string(tuple_obj.get("target")) or "IMPLICIT_TARGET"
        harmful = normalize_binary_label(tuple_obj.get("harmful"))
        explicit_target = target != "IMPLICIT_TARGET"
        support_count = len(cluster.support_roles)
        avg_member_quality = cluster_member_quality_avg(cluster)
        relative_ok = top_score <= 0.0 or cluster.cluster_score >= top_score * relative_score_floor
        support_ok = support_count >= min_support_roles
        has_valid_member = any(member.valid_for_fusion for member in cluster.members)
        singleton_harmful_ok = (
            keep_explicit_target_harmful_singletons
            and support_count == 1
            and harmful == 1
            and explicit_target
        )
        singleton_lexicon_ok = keep_lexicon_singletons and support_count == 1 and harmful == 1 and cluster.lexicon_hit
        nonharmful_implicit_singleton = support_count == 1 and harmful == 0 and not explicit_target

        if require_valid_member and not has_valid_member:
            continue
        if nonharmful_implicit_singleton and not keep_nonharmful_implicit_singletons:
            continue
        if (
            reject_target_conflict_singletons
            and support_count == 1
            and any("target_conflict_with_hint" in member.issues for member in cluster.members)
        ):
            continue
        if cluster.cluster_score < min_cluster_score or avg_member_quality < min_avg_member_quality:
            continue
        if not (relative_ok or support_ok or singleton_harmful_ok or singleton_lexicon_ok):
            continue

        retained.append(cluster)
        seen_keys.add(tuple_key)

    return retained


def compact_candidate(candidate: Candidate) -> Dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "role": candidate.role,
        "tuple": candidate.tuple_obj,
        "rationale": candidate.rationale,
        "quality_score": candidate.quality_score,
        "issues": candidate.issues,
        "valid_for_fusion": candidate.valid_for_fusion,
        "lexicon_hit": candidate.lexicon_hit,
    }


def truncate_compact_text(value: Any, max_chars: int) -> Optional[str]:
    text = normalize_string(value)
    if text is None:
        return None
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def dump_prompt_json(obj: Any, compact: bool) -> str:
    if compact:
        return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(obj, ensure_ascii=False, indent=2)


def compact_candidate_for_auditor(
    candidate: Candidate,
    *,
    include_candidate_id: bool,
    include_full_issues: bool,
    rationale_max_chars: int,
) -> Dict[str, Any]:
    payload = {
        "role": candidate.role,
        "tuple": candidate.tuple_obj,
        "rationale": truncate_compact_text(candidate.rationale, rationale_max_chars),
        "quality_score": candidate.quality_score,
        "valid_for_fusion": candidate.valid_for_fusion,
        "lexicon_hit": candidate.lexicon_hit,
    }
    if include_candidate_id:
        payload["candidate_id"] = candidate.candidate_id
    if candidate.issues:
        payload["issues"] = list(candidate.issues) if include_full_issues else list(candidate.issues[:2])
    return payload


def compact_cluster_member_for_auditor(
    candidate: Candidate,
    *,
    include_candidate_id: bool,
    include_full_issues: bool,
    rationale_max_chars: int,
) -> Dict[str, Any]:
    return compact_candidate_for_auditor(
        candidate,
        include_candidate_id=include_candidate_id,
        include_full_issues=include_full_issues,
        rationale_max_chars=rationale_max_chars,
    )


def parse_auditor_compact_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    phase3_cfg = cfg.get("phase3", {}) if isinstance(cfg.get("phase3"), dict) else {}
    raw = phase3_cfg.get("auditor_compact_request", {})
    compact_cfg = raw if isinstance(raw, dict) else {}
    enabled = parse_bool(compact_cfg.get("enabled", False), default=False)
    return {
        "enabled": enabled,
        "role_top_candidates": int(max(1, compact_cfg.get("role_top_candidates", 1) or 1)),
        "cluster_top_k": int(max(1, compact_cfg.get("cluster_top_k", 2) or 2)),
        "cluster_member_top_k": int(max(1, compact_cfg.get("cluster_member_top_k", 3) or 3)),
        "rationale_max_chars": int(max(24, compact_cfg.get("rationale_max_chars", 120) or 120)),
        "include_candidate_id": parse_bool(compact_cfg.get("include_candidate_id", False), default=False),
        "include_full_issues": parse_bool(compact_cfg.get("include_full_issues", False), default=False),
        "beacons_id_only": parse_bool(compact_cfg.get("beacons_id_only", False), default=False),
        "drop_empty_all_candidates": parse_bool(compact_cfg.get("drop_empty_all_candidates", True), default=True),
        "prefer_short_prompt_template": parse_bool(compact_cfg.get("prefer_short_prompt_template", False), default=False),
        "render_minified_json": parse_bool(compact_cfg.get("render_minified_json", enabled), default=enabled),
        "auto_fit_context": parse_bool(compact_cfg.get("auto_fit_context", enabled), default=enabled),
        "prompt_token_buffer": int(max(64, compact_cfg.get("prompt_token_buffer", 256) or 256)),
        "max_auto_compaction_stage": int(max(0, compact_cfg.get("max_auto_compaction_stage", 4) or 4)),
    }


def rank_cluster_for_auditor(cluster: Cluster) -> Tuple[float, int, str]:
    return (-cluster.cluster_score, -len(cluster.support_roles), cluster.cluster_id)


def select_cluster_members_for_auditor(cluster: Cluster, member_top_k: int) -> List[Candidate]:
    selected: List[Candidate] = []
    seen_roles: Set[str] = set()
    ordered = sorted(cluster.members, key=lambda c: (-c.quality_score, c.role, c.role_index))
    for member in ordered:
        if member.role in seen_roles:
            continue
        selected.append(member)
        seen_roles.add(member.role)
        if len(selected) >= member_top_k:
            return selected
    for member in ordered:
        if member in selected:
            continue
        selected.append(member)
        if len(selected) >= member_top_k:
            break
    return selected


def derive_safe_max_input_tokens(
    inference_cfg: Dict[str, Any],
    *,
    reserve_tokens: int,
) -> int:
    max_model_len = int(max(1024, inference_cfg.get("max_model_len", 8192)))
    candidate_generation_caps: List[int] = []
    for key in (
        "think_retry_max_new_tokens",
        "think_max_new_tokens",
        "no_think_retry_max_new_tokens",
        "no_think_max_new_tokens",
    ):
        raw = inference_cfg.get(key)
        if raw is None:
            continue
        try:
            candidate_generation_caps.append(int(raw))
        except (TypeError, ValueError):
            continue
    generation_cap = max(candidate_generation_caps) if candidate_generation_caps else 1024
    effective_reserve = int(max(64, reserve_tokens))
    budget = max_model_len - generation_cap - effective_reserve
    return int(max(512, budget))


def estimate_token_count(tokenizer: Any, text: str) -> int:
    if tokenizer is None:
        return len(text)
    try:
        return int(len(tokenizer.encode(text, add_special_tokens=False)))
    except Exception:
        pass
    try:
        encoded = tokenizer(text, add_special_tokens=False)
        input_ids = encoded.get("input_ids") if isinstance(encoded, dict) else getattr(encoded, "input_ids", None)
        if isinstance(input_ids, list):
            if input_ids and isinstance(input_ids[0], list):
                return int(len(input_ids[0]))
            return int(len(input_ids))
    except Exception:
        pass
    return len(text)


def slim_candidate_payload_for_prompt(
    payload: Any,
    *,
    rationale_max_chars: int,
    keep_rationale: bool,
) -> Any:
    if not isinstance(payload, dict):
        return payload
    out: Dict[str, Any] = {}
    for key in ("role", "tuple", "quality_score", "valid_for_fusion", "lexicon_hit"):
        if key in payload:
            out[key] = copy.deepcopy(payload[key])
    if keep_rationale:
        rationale = truncate_compact_text(payload.get("rationale"), rationale_max_chars)
        if rationale is not None:
            out["rationale"] = rationale
    return out


def compact_group_row_for_prompt(
    group_row: Dict[str, Any],
    *,
    minimal: bool,
    group_text_max_chars: int,
) -> Dict[str, Any]:
    if not isinstance(group_row, dict):
        return {}
    if minimal:
        keys = ["group_id", "sample_id", "language", "focus_text", "canonical_target", "escalation_flag"]
    else:
        keys = [
            "group_id",
            "sample_id",
            "source",
            "split",
            "language",
            "group_text",
            "focus_text",
            "canonical_target",
            "escalation_flag",
        ]
    out: Dict[str, Any] = {}
    for key in keys:
        if key not in group_row:
            continue
        value = copy.deepcopy(group_row.get(key))
        if key == "group_text" and isinstance(value, str) and group_text_max_chars > 0:
            truncated = truncate_compact_text(value, group_text_max_chars)
            value = truncated if truncated is not None else value
        out[key] = value
    return out


def apply_auditor_request_compaction_stage(request: Dict[str, Any], stage: int) -> Dict[str, Any]:
    if stage <= 0:
        return copy.deepcopy(request)

    compacted = copy.deepcopy(request)
    role_payload = compacted.get("role_payload")
    if not isinstance(role_payload, list):
        role_payload = []
        compacted["role_payload"] = role_payload
    clusters_payload = compacted.get("clusters_payload")
    if not isinstance(clusters_payload, list):
        clusters_payload = []
        compacted["clusters_payload"] = clusters_payload
    beacons = compacted.get("beacons")
    if not isinstance(beacons, list):
        beacons = []
        compacted["beacons"] = beacons

    if stage >= 1:
        compacted["beacons"] = [{"id": str(b.get("id") or "").strip()} for b in beacons if str(b.get("id") or "").strip()]
        for role_entry in role_payload:
            if isinstance(role_entry, dict):
                role_entry.pop("all_candidates", None)

    if stage >= 2:
        for role_entry in role_payload:
            if not isinstance(role_entry, dict):
                continue
            if role_entry.get("top_candidate") is not None:
                role_entry["top_candidate"] = slim_candidate_payload_for_prompt(
                    role_entry.get("top_candidate"),
                    rationale_max_chars=48,
                    keep_rationale=True,
                )
        compacted_clusters: List[Dict[str, Any]] = []
        for cluster in clusters_payload[:2]:
            if not isinstance(cluster, dict):
                continue
            cluster_payload: Dict[str, Any] = {
                "cluster_id": copy.deepcopy(cluster.get("cluster_id")),
                "canonical_tuple": copy.deepcopy(cluster.get("canonical_tuple")),
                "support_roles": copy.deepcopy(cluster.get("support_roles")),
                "cluster_score": copy.deepcopy(cluster.get("cluster_score")),
                "lexicon_hit": copy.deepcopy(cluster.get("lexicon_hit")),
                "member_count": copy.deepcopy(cluster.get("member_count")),
            }
            members = cluster.get("members")
            if isinstance(members, list) and members:
                cluster_payload["members"] = [
                    slim_candidate_payload_for_prompt(member, rationale_max_chars=48, keep_rationale=True)
                    for member in members[:1]
                ]
            compacted_clusters.append(cluster_payload)
        compacted["clusters_payload"] = compacted_clusters

    if stage >= 3:
        for role_entry in role_payload:
            if not isinstance(role_entry, dict):
                continue
            if role_entry.get("top_candidate") is not None:
                role_entry["top_candidate"] = slim_candidate_payload_for_prompt(
                    role_entry.get("top_candidate"),
                    rationale_max_chars=0,
                    keep_rationale=False,
                )
        compacted["clusters_payload"] = [
            {
                "cluster_id": copy.deepcopy(cluster.get("cluster_id")),
                "canonical_tuple": copy.deepcopy(cluster.get("canonical_tuple")),
                "support_roles": copy.deepcopy(cluster.get("support_roles")),
                "cluster_score": copy.deepcopy(cluster.get("cluster_score")),
                "lexicon_hit": copy.deepcopy(cluster.get("lexicon_hit")),
                "member_count": copy.deepcopy(cluster.get("member_count")),
            }
            for cluster in compacted.get("clusters_payload", [])[:1]
            if isinstance(cluster, dict)
        ]
        compacted["group_row"] = compact_group_row_for_prompt(
            compacted.get("group_row", {}),
            minimal=False,
            group_text_max_chars=280,
        )

    if stage >= 4:
        compacted["group_row"] = compact_group_row_for_prompt(
            compacted.get("group_row", {}),
            minimal=True,
            group_text_max_chars=0,
        )
        triage_payload = compacted.get("triage_payload")
        if isinstance(triage_payload, dict):
            signals = triage_payload.get("signals")
            compacted_signals: Dict[str, Any] = {}
            if isinstance(signals, dict):
                for key in (
                    "lexicon_hit",
                    "victim_microaggression",
                    "moderator_strong",
                    "quote_caution",
                    "complete_consensus",
                    "consensus_aligned_roles",
                ):
                    if key in signals:
                        compacted_signals[key] = copy.deepcopy(signals.get(key))
            compacted["triage_payload"] = {
                "lane": copy.deepcopy(triage_payload.get("lane")),
                "initial_conflict_type": copy.deepcopy(triage_payload.get("initial_conflict_type")),
                "signals": compacted_signals,
            }
        compacted["beacons"] = [{"id": "grounding"}]

    return compacted


def load_role_prompt_overrides(cfg: Dict[str, Any], workspace_root: Path, config_dir: Path) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    raw_map = cfg.get("paths", {}).get("role_prompt_map", {})
    if not isinstance(raw_map, dict):
        return out
    for role, lang_map in raw_map.items():
        if not isinstance(lang_map, dict):
            continue
        out[role] = {}
        for language, raw in lang_map.items():
            if not isinstance(raw, str) or not raw.strip():
                continue
            path = resolve_path(workspace_root, config_dir, raw.strip())
            out[role][language] = path.read_text(encoding="utf-8")
    return out


def extract_json_object_candidate(text: str) -> Dict[str, Any]:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty model output")
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", raw, flags=re.IGNORECASE).strip()
    stripped = re.sub(r"<think>[\s\S]*", "", stripped, flags=re.IGNORECASE).strip()
    for source in [stripped, raw]:
        if not source:
            continue
        code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", source, flags=re.IGNORECASE)
        for block in code_blocks:
            cand = block.strip()
            if not cand:
                continue
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except Exception:
                continue
        try:
            parsed = json.loads(source)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
        obj_start = source.find("{")
        obj_end = source.rfind("}")
        if obj_start >= 0 and obj_end > obj_start:
            parsed = json.loads(source[obj_start : obj_end + 1])
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("No parseable JSON object in model output")


def normalize_auditor_payload(
    payload: Dict[str, Any],
    language: str,
    focus_text: str,
    fallback_cluster: Optional[Cluster],
) -> Dict[str, Any]:
    selected_cluster_id = normalize_string(payload.get("selected_cluster_id"))
    final_target = scrub_placeholder_literal(payload.get("final_target"))
    final_argument = scrub_placeholder_literal(payload.get("final_argument"))
    final_group = scrub_placeholder_literal(payload.get("final_group"))
    final_label = normalize_binary_label(payload.get("final_label"))
    fused_rationale = scrub_placeholder_literal(payload.get("fused_rationale")) or ""
    confidence = round(clamp(safe_float(payload.get("confidence"), default=0.75)), 4)
    downweighted_roles = [
        role
        for role in (normalize_auditor_role_name(x) for x in payload.get("downweighted_roles", []))
        if role
    ]
    rejected_roles = [
        role
        for role in (normalize_auditor_role_name(x) for x in payload.get("rejected_roles", []))
        if role
    ]
    correction_trace: List[Dict[str, Any]] = []
    raw_correction_trace = payload.get("correction_trace", [])
    if isinstance(raw_correction_trace, list):
        for item in raw_correction_trace:
            if not isinstance(item, dict):
                continue
            role = normalize_auditor_role_name(item.get("role"))
            action = scrub_placeholder_literal(item.get("action")) or "note"
            reason = scrub_placeholder_literal(item.get("reason"))
            if not reason:
                continue
            correction_trace.append(
                {
                    "role": role,
                    "action": action.lower(),
                    "reason": reason,
                }
            )
    focus_inner = extract_focus_inner_text(focus_text, focus_text)
    if final_argument and final_argument not in focus_inner:
        final_argument = fallback_cluster.canonical_tuple.get("argument") if fallback_cluster else None
    if language == "zh":
        if final_label == 0:
            final_group = "non-hate"
        else:
            final_group = final_group or (fallback_cluster.canonical_tuple.get("group") if fallback_cluster else "unknown")
    else:
        final_group = None
    return {
        "selected_cluster_id": selected_cluster_id,
        "final_target": final_target or (fallback_cluster.canonical_tuple.get("target") if fallback_cluster else "IMPLICIT_TARGET"),
        "final_argument": final_argument or (fallback_cluster.canonical_tuple.get("argument") if fallback_cluster else None),
        "final_group": final_group,
        "final_label": final_label if final_label is not None else (fallback_cluster.canonical_tuple.get("harmful") if fallback_cluster else 0),
        "fused_rationale": fused_rationale or build_deterministic_rationale(fallback_cluster, {}, language),
        "downweighted_roles": dedupe_preserve_order(downweighted_roles, case_insensitive=False),
        "rejected_roles": dedupe_preserve_order(rejected_roles, case_insensitive=False),
        "correction_trace": correction_trace,
        "confidence": confidence,
    }


def select_span_refinement_member(
    selected_cluster: Optional[Cluster],
    final_tuple: Dict[str, Any],
    language: str,
) -> Optional[Candidate]:
    if selected_cluster is None:
        return None

    final_target = normalize_string(final_tuple.get("target")) or "IMPLICIT_TARGET"
    final_group = normalize_string(final_tuple.get("group"))
    final_label = normalize_binary_label(final_tuple.get("harmful"))

    compatible_members: List[Candidate] = []
    for member in sorted(selected_cluster.members, key=lambda c: (-c.quality_score, c.role, c.role_index)):
        member_tuple = member.tuple_obj
        member_argument = normalize_string(member_tuple.get("argument"))
        if member_argument is None:
            continue

        member_label = normalize_binary_label(member_tuple.get("harmful"))
        if final_label is not None and member_label is not None and member_label != final_label:
            continue

        if language == "zh" and final_label == 1:
            member_group = normalize_string(member_tuple.get("group"))
            if final_group not in {None, "unknown"} and member_group not in {final_group, None, "unknown"}:
                continue

        if final_target != "IMPLICIT_TARGET":
            member_target = normalize_string(member_tuple.get("target")) or "IMPLICIT_TARGET"
            if text_similarity(member_target, final_target, language) < 0.5:
                continue

        compatible_members.append(member)

    if final_target != "IMPLICIT_TARGET":
        exact_target_members: List[Candidate] = []
        has_expanded_same_target = False
        for member in compatible_members:
            member_target = normalize_string(member.tuple_obj.get("target")) or "IMPLICIT_TARGET"
            member_argument = normalize_string(member.tuple_obj.get("argument")) or ""
            if member_target == "IMPLICIT_TARGET" or text_similarity(member_target, final_target, language) < 0.9:
                continue
            if member_argument == member_target:
                exact_target_members.append(member)
                continue
            if final_target in member_argument and len(member_argument) > len(final_target):
                has_expanded_same_target = True
        if exact_target_members and has_expanded_same_target:
            return sorted(exact_target_members, key=lambda c: (-c.quality_score, c.role, c.role_index))[0]

    if compatible_members:
        return compatible_members[0]
    return None


def refine_final_tuple_spans(
    final_tuple: Dict[str, Any],
    selected_cluster: Optional[Cluster],
    focus_text: str,
    language: str,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    refinement_member = select_span_refinement_member(
        selected_cluster=selected_cluster,
        final_tuple=final_tuple,
        language=language,
    )
    if refinement_member is None:
        return normalize_final_tuple(final_tuple, language=language), []

    refined = dict(final_tuple)
    refinement_trace: List[Dict[str, Any]] = []
    focus_inner = extract_focus_inner_text(focus_text, focus_text)

    member_argument = normalize_string(refinement_member.tuple_obj.get("argument"))
    current_argument = normalize_string(refined.get("argument"))
    if member_argument and member_argument in focus_inner and member_argument != current_argument:
        refined["argument"] = member_argument
        refinement_trace.append(
            {
                "role": refinement_member.role,
                "action": "span_refined",
                "reason": "Used the highest-quality grounded argument span from the selected cluster to avoid an over-broad final argument.",
            }
        )

    member_target = normalize_string(refinement_member.tuple_obj.get("target")) or "IMPLICIT_TARGET"
    current_target = normalize_string(refined.get("target")) or "IMPLICIT_TARGET"
    if member_target != "IMPLICIT_TARGET":
        should_replace_target = current_target == "IMPLICIT_TARGET"
        if not should_replace_target and text_similarity(member_target, current_target, language) >= 0.9 and len(member_target) < len(current_target):
            should_replace_target = True
        if should_replace_target and member_target != current_target:
            refined["target"] = member_target
            refinement_trace.append(
                {
                    "role": refinement_member.role,
                    "action": "span_refined",
                    "reason": "Filled or tightened the final target span using the strongest compatible member from the selected cluster.",
                }
            )

    return normalize_final_tuple(refined, language=language), refinement_trace


class RegenerationEngine:
    def __init__(self, model_path: Path, inference_cfg: Dict[str, Any], role_prompt_overrides: Dict[str, Dict[str, str]], zh_lexicon_path: Path, logger: EventLogger) -> None:
        phase2 = load_phase2_module()
        self.phase2 = phase2
        self.model_path = model_path
        self.inference_cfg = inference_cfg
        self.role_prompt_overrides = role_prompt_overrides
        self.zh_lexicon_terms = phase2.load_zh_annotated_lexicon(zh_lexicon_path)
        self.logger = logger
        self.llm: Optional[Any] = None

    def _ensure_llm(self) -> Any:
        if self.llm is None:
            backend = str(self.inference_cfg.get("backend", "")).strip().lower()
            if backend == "api":
                candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=self.inference_cfg, prompt_template="")
                candidate_llm.initialize()
                self.llm = candidate_llm
                return self.llm
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    cleanup_stale_vllm_workers_for_user(
                        self.logger,
                        reason=f"regen_init_attempt_{attempt + 1}",
                    )
                    cleanup_session_vllm_workers(
                        self.logger,
                        reason=f"regen_init_attempt_{attempt + 1}",
                        orphan_only=False,
                    )
                    init_cfg = build_vllm_init_retry_cfg(
                        self.inference_cfg,
                        engine_kind="regen",
                        retry_attempt=attempt,
                    )
                    candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=init_cfg, prompt_template="")
                    candidate_llm.initialize()
                    self.llm = candidate_llm
                    break
                except Exception as exc:
                    last_error = exc
                    candidate = locals().get("candidate_llm")
                    if candidate is not None:
                        try:
                            candidate.shutdown()
                        except Exception:
                            pass
                    self.llm = None
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    if attempt == 0:
                        self.logger.event(
                            "runtime",
                            "regen_llm_init_retry",
                            model_path=str(self.model_path),
                            retry_inference_cfg=build_vllm_init_retry_cfg(
                                self.inference_cfg,
                                engine_kind="regen",
                                retry_attempt=1,
                            ),
                            error=str(exc),
                        )
                        continue
                    self.logger.event(
                        "runtime",
                        "regen_llm_init_retry_exhausted",
                        level="WARNING",
                        model_path=str(self.model_path),
                        attempted_backend=str(self.inference_cfg.get("backend", "")).strip().lower(),
                        error=str(exc),
                    )
                    break
            if self.llm is None and last_error is not None:
                fallback_backend = resolve_init_fallback_backend(self.inference_cfg, self.model_path)
                if fallback_backend:
                    fallback_cfg = build_backend_fallback_cfg(self.inference_cfg, fallback_backend=fallback_backend)
                    self.logger.event(
                        "runtime",
                        "regen_llm_backend_fallback",
                        model_path=str(self.model_path),
                        backend=fallback_backend,
                        error=str(last_error),
                    )
                    candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=fallback_cfg, prompt_template="")
                    candidate_llm.initialize()
                    self.llm = candidate_llm
                else:
                    raise last_error
        return self.llm

    def release(self) -> None:
        if self.llm is None:
            return
        try:
            self.llm.shutdown()
        except Exception:
            pass
        self.llm = None

    def regenerate(self, group_row: Dict[str, Any], role_names: Sequence[str], max_parse_retries: int) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        self.logger.event(
            "runtime",
            "regen_begin",
            group_id=group_row.get("group_id"),
            roles=list(role_names),
        )
        llm = self._ensure_llm()
        outputs: Dict[str, List[Dict[str, Any]]] = {}
        traces: List[Dict[str, Any]] = []
        for role_name in role_names:
            language = str(group_row.get("language") or "en")
            template = self.role_prompt_overrides.get(role_name, {}).get(language)
            if not template:
                outputs[ROLE_OUTPUT_KEY_MAP[role_name]] = []
                traces.append({"role": role_name, "status": "missing_prompt"})
                continue
            lexicon_knowledge = ""
            if role_name == "cultural_bystander" and language == "zh":
                hits = self.phase2.lookup_zh_lexicon_hits(row=group_row, lexicon_terms=self.zh_lexicon_terms)
                lexicon_knowledge = self.phase2.build_lexicon_knowledge_block(hits)
            pending_error: Optional[str] = None
            raw_output = ""
            normalized: List[Dict[str, Any]] = []
            success = False
            for attempt in range(max_parse_retries + 1):
                prompt = self.phase2.render_role_prompt_text(
                    template=template,
                    focus_text=str(group_row.get("focus_text") or ""),
                    group_text=str(group_row.get("group_text") or ""),
                    lexicon_knowledge=lexicon_knowledge,
                    retry_note=pending_error if attempt > 0 else None,
                    target_hint=normalize_string(group_row.get("canonical_target")) if normalize_string(group_row.get("canonical_target")) != "IMPLICIT_TARGET" else "",
                )
                force_no_think_retry = parse_bool(self.inference_cfg.get("force_no_think_retry", True), default=True)
                force_no_think_this_attempt = bool(llm.enable_thinking and force_no_think_retry and attempt > 0)
                messages_batch = [
                    [
                        {
                            "role": "system",
                            "content": self.phase2.build_generation_system_message(
                                enable_thinking=bool(llm.enable_thinking),
                                force_no_think=force_no_think_this_attempt,
                                model_name=llm.model_path.name,
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                ]
                try:
                    generation_max_new_tokens = self.phase2.resolve_attempt_max_new_tokens(
                        inference_cfg=self.inference_cfg,
                        enable_thinking=bool(llm.enable_thinking),
                        attempt=attempt,
                        force_no_think_retry=force_no_think_retry,
                    )
                    result = self.phase2.generate_with_oom_split_retry(
                        llm=llm,
                        messages_batch=messages_batch,
                        role_name=role_name,
                        logger=self.logger,
                        enable_split_retry=parse_bool(self.inference_cfg.get("enable_oom_split_retry", True), default=True),
                        max_depth=int(self.inference_cfg.get("oom_split_retry_max_depth", 4)),
                        max_new_tokens_override=generation_max_new_tokens,
                        force_no_think=force_no_think_this_attempt,
                    )
                    raw_output = result[0] if result else ""
                    parsed = self.phase2.extract_json_array_candidate(raw_output)
                    normalized = self.phase2.normalize_prediction_array_by_language(
                        parsed=parsed,
                        language=language,
                        focus_text=str(group_row.get("focus_text") or ""),
                        group_text=str(group_row.get("group_text") or ""),
                    )
                    success = True
                    break
                except Exception as exc:
                    pending_error = str(exc)
            output_key = ROLE_OUTPUT_KEY_MAP[role_name]
            outputs[output_key] = normalized if success else []
            traces.append(
                {
                    "role": role_name,
                    "status": "regenerated" if success else "failed",
                    "raw_output": raw_output[:2000],
                    "error": pending_error,
                    "count": len(normalized),
                }
            )
        self.logger.event(
            "runtime",
            "regen_end",
            group_id=group_row.get("group_id"),
            roles=list(role_names),
        )
        return outputs, traces


class AuditorEngine:
    def __init__(
        self,
        model_path: Path,
        inference_cfg: Dict[str, Any],
        prompt_templates: Dict[str, str],
        logger: EventLogger,
        runtime_cfg: Optional[Dict[str, Any]] = None,
    ) -> None:
        phase2 = load_phase2_module()
        self.phase2 = phase2
        self.model_path = model_path
        self.inference_cfg = inference_cfg
        self.prompt_templates = dict(prompt_templates)
        self.logger = logger
        self.runtime_cfg = dict(runtime_cfg or {})
        self.llm: Optional[Any] = None

    def _ensure_llm(self) -> Any:
        if self.llm is None:
            backend = str(self.inference_cfg.get("backend", "")).strip().lower()
            if backend == "api":
                candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=self.inference_cfg, prompt_template="")
                candidate_llm.initialize()
                self.llm = candidate_llm
                return self.llm
            last_error: Optional[Exception] = None
            for attempt in range(2):
                try:
                    cleanup_stale_vllm_workers_for_user(
                        self.logger,
                        reason=f"auditor_init_attempt_{attempt + 1}",
                    )
                    cleanup_session_vllm_workers(
                        self.logger,
                        reason=f"auditor_init_attempt_{attempt + 1}",
                        orphan_only=False,
                    )
                    init_cfg = build_vllm_init_retry_cfg(
                        self.inference_cfg,
                        engine_kind="auditor",
                        retry_attempt=attempt,
                    )
                    candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=init_cfg, prompt_template="")
                    candidate_llm.initialize()
                    self.llm = candidate_llm
                    break
                except Exception as exc:
                    last_error = exc
                    candidate = locals().get("candidate_llm")
                    if candidate is not None:
                        try:
                            candidate.shutdown()
                        except Exception:
                            pass
                    self.llm = None
                    gc.collect()
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if hasattr(torch.cuda, "ipc_collect"):
                                torch.cuda.ipc_collect()
                    except Exception:
                        pass
                    if attempt == 0:
                        self.logger.event(
                            "runtime",
                            "auditor_llm_init_retry",
                            model_path=str(self.model_path),
                            retry_inference_cfg=build_vllm_init_retry_cfg(
                                self.inference_cfg,
                                engine_kind="auditor",
                                retry_attempt=1,
                            ),
                            error=str(exc),
                        )
                        continue
                    self.logger.event(
                        "runtime",
                        "auditor_llm_init_retry_exhausted",
                        level="WARNING",
                        model_path=str(self.model_path),
                        attempted_backend=str(self.inference_cfg.get("backend", "")).strip().lower(),
                        error=str(exc),
                    )
                    break
            if self.llm is None and last_error is not None:
                fallback_backend = resolve_init_fallback_backend(self.inference_cfg, self.model_path)
                if fallback_backend:
                    fallback_cfg = build_backend_fallback_cfg(self.inference_cfg, fallback_backend=fallback_backend)
                    self.logger.event(
                        "runtime",
                        "auditor_llm_backend_fallback",
                        model_path=str(self.model_path),
                        backend=fallback_backend,
                        error=str(last_error),
                    )
                    candidate_llm = self.phase2.LocalRoleLLM(model_path=self.model_path, inference_cfg=fallback_cfg, prompt_template="")
                    candidate_llm.initialize()
                    self.llm = candidate_llm
                else:
                    raise last_error
        return self.llm

    def release(self) -> None:
        if self.llm is None:
            return
        try:
            self.llm.shutdown()
        except Exception:
            pass
        self.llm = None

    def _empty_trace(self) -> Dict[str, Any]:
        return {
            "used": True,
            "prompt_language": None,
            "parse_fail": False,
            "raw_output": None,
            "error": None,
            "attempt_count": 0,
            "retry_count": 0,
            "successful_attempt": None,
            "parse_errors": [],
            "prompt_token_count": None,
            "prompt_token_budget": None,
            "prompt_compaction_stage": 0,
            "prompt_auto_compacted": False,
        }

    def _select_prompt_template(self, language: str) -> str:
        lang_key = normalize_prompt_language_key(language)
        compact_cfg = parse_auditor_compact_cfg({"phase3": self.runtime_cfg})
        if parse_bool(compact_cfg.get("enabled", False), default=False) and parse_bool(compact_cfg.get("prefer_short_prompt_template", False), default=False):
            short_template = self.prompt_templates.get("default")
            if short_template:
                return short_template
        if lang_key in self.prompt_templates:
            return self.prompt_templates[lang_key]
        if "default" in self.prompt_templates:
            return self.prompt_templates["default"]
        if self.prompt_templates:
            return next(iter(self.prompt_templates.values()))
        raise ValueError("No auditor prompt template available")

    def _build_output_schema(self, language: str) -> Dict[str, Any]:
        lang_key = normalize_prompt_language_key(language)
        if lang_key == "zh":
            return {
                "selected_cluster_id": "c1",
                "final_target": "北京人",
                "final_argument": "说河南人偷井盖的",
                "final_group": "Region",
                "final_label": 1,
                "fused_rationale": "聚焦片段将地域刻板印象归因到明确群体，保留更贴近文本的候选并下调误读角色。",
                "downweighted_roles": ["moderator"],
                "rejected_roles": [],
                "correction_trace": [
                    {
                        "role": "moderator",
                        "action": "downweighted",
                        "reason": "过度依赖表层否定句式，忽略了焦点片段中的真实归因对象。",
                    }
                ],
                "confidence": 0.85,
            }
        return {
            "selected_cluster_id": "c1",
            "final_target": "men",
            "final_argument": "trash",
            "final_group": None,
            "final_label": 1,
            "fused_rationale": "The focus span contains a direct group generalization, so the grounded harmful cluster is kept and the weaker alternative is downweighted.",
            "downweighted_roles": ["victim"],
            "rejected_roles": [],
            "correction_trace": [
                {
                    "role": "victim",
                    "action": "downweighted",
                    "reason": "It softened a clear group-directed insult that remains explicit in the focus span.",
                }
            ],
            "confidence": 0.85,
        }

    def _render_prompt(
        self,
        group_row: Dict[str, Any],
        role_payload: List[Dict[str, Any]],
        clusters_payload: List[Dict[str, Any]],
        triage_payload: Dict[str, Any],
        beacons: Sequence[Dict[str, str]],
        retry_note: Optional[str],
    ) -> str:
        language = normalize_prompt_language_key(str(group_row.get("language") or "en"))
        prompt_template = self._select_prompt_template(language)
        beacons_text_lines: List[str] = []
        for beacon in beacons:
            beacon_id = str(beacon.get("id") or "").strip()
            beacon_text = str(beacon.get("text") or "").strip()
            if beacon_id and beacon_text:
                beacons_text_lines.append(f"- [{beacon_id}] {beacon_text}")
            elif beacon_id:
                beacons_text_lines.append(f"- [{beacon_id}]")
        beacons_text = "\n".join(beacons_text_lines)
        output_schema = self._build_output_schema(language)
        compact_cfg = parse_auditor_compact_cfg({"phase3": self.runtime_cfg})
        render_minified_json = parse_bool(compact_cfg.get("render_minified_json", False), default=False)
        return (
            prompt_template
            .replace("{{GROUP_PAYLOAD_JSON}}", dump_prompt_json(group_row, render_minified_json))
            .replace("{{ROLE_OVERVIEW_JSON}}", dump_prompt_json(role_payload, render_minified_json))
            .replace("{{CLUSTERS_JSON}}", dump_prompt_json(clusters_payload, render_minified_json))
            .replace("{{TRIAGE_JSON}}", dump_prompt_json(triage_payload, render_minified_json))
            .replace("{{APPLIED_BEACONS_TEXT}}", beacons_text or "- [grounding] Keep all finals grounded to focus_text.")
            .replace("{{OUTPUT_SCHEMA_JSON}}", dump_prompt_json(output_schema, render_minified_json))
            .replace("{{RETRY_NOTE}}", retry_note or "")
        )

    def _prepare_request_for_prompt(
        self,
        request: Dict[str, Any],
        system_content: str,
        retry_note: Optional[str],
        enable_thinking_override: bool,
    ) -> Tuple[Dict[str, Any], int, int, int, bool]:
        compact_cfg = parse_auditor_compact_cfg({"phase3": self.runtime_cfg})
        budget = derive_safe_max_input_tokens(
            self.inference_cfg,
            reserve_tokens=int(compact_cfg.get("prompt_token_buffer", 256) or 256),
        )
        if not parse_bool(compact_cfg.get("auto_fit_context", False), default=False):
            prompt = self._render_prompt(
                group_row=request["group_row"],
                role_payload=request["role_payload"],
                clusters_payload=request["clusters_payload"],
                triage_payload=request["triage_payload"],
                beacons=request["beacons"],
                retry_note=retry_note,
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            llm = self._ensure_llm()
            prompt_text = llm._messages_to_prompt(messages, enable_thinking_override=enable_thinking_override)
            token_count = estimate_token_count(llm._tokenizer, prompt_text)
            return copy.deepcopy(request), token_count, budget, 0, False

        max_stage = int(max(0, compact_cfg.get("max_auto_compaction_stage", 4) or 4))
        llm = self._ensure_llm()
        for stage in range(max_stage + 1):
            candidate_request = apply_auditor_request_compaction_stage(request, stage)
            prompt = self._render_prompt(
                group_row=candidate_request["group_row"],
                role_payload=candidate_request["role_payload"],
                clusters_payload=candidate_request["clusters_payload"],
                triage_payload=candidate_request["triage_payload"],
                beacons=candidate_request["beacons"],
                retry_note=retry_note,
            )
            messages = [
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ]
            prompt_text = llm._messages_to_prompt(messages, enable_thinking_override=enable_thinking_override)
            token_count = estimate_token_count(llm._tokenizer, prompt_text)
            if token_count <= budget or stage >= max_stage:
                return candidate_request, token_count, budget, stage, stage > 0
        return copy.deepcopy(request), budget, budget, 0, False

    def audit_batch(
        self,
        requests: Sequence[Dict[str, Any]],
        max_parse_retries: int,
    ) -> List[Tuple[Optional[Dict[str, Any]], Dict[str, Any]]]:
        if not requests:
            return []

        self.logger.event(
            "runtime",
            "auditor_batch_begin",
            batch_size=len(requests),
            group_ids=[(request.get("group_row") or {}).get("group_id") for request in requests],
        )
        llm = self._ensure_llm()
        traces: List[Dict[str, Any]] = [self._empty_trace() for _ in requests]
        payloads: List[Optional[Dict[str, Any]]] = [None for _ in requests]
        retry_notes: List[Optional[str]] = [None for _ in requests]
        pending_indices: List[int] = list(range(len(requests)))

        for attempt in range(max_parse_retries + 1):
            if not pending_indices:
                break

            force_no_think_retry = parse_bool(self.inference_cfg.get("force_no_think_retry", True), default=True)
            force_no_think_this_attempt = bool(llm.enable_thinking and force_no_think_retry and attempt > 0)
            system_content = self.phase2.build_generation_system_message(
                enable_thinking=bool(llm.enable_thinking),
                force_no_think=force_no_think_this_attempt,
                model_name=llm.model_path.name,
            ).replace("JSON array", "JSON object")
            messages_batch: List[List[Dict[str, str]]] = []

            for request_idx in pending_indices:
                request = requests[request_idx]
                effective_request, prompt_token_count, prompt_token_budget, prompt_compaction_stage, prompt_auto_compacted = self._prepare_request_for_prompt(
                    request=request,
                    system_content=system_content,
                    retry_note=retry_notes[request_idx] if attempt > 0 else None,
                    enable_thinking_override=bool(llm.enable_thinking and not force_no_think_this_attempt),
                )
                prompt = self._render_prompt(
                    group_row=effective_request["group_row"],
                    role_payload=effective_request["role_payload"],
                    clusters_payload=effective_request["clusters_payload"],
                    triage_payload=effective_request["triage_payload"],
                    beacons=effective_request["beacons"],
                    retry_note=retry_notes[request_idx] if attempt > 0 else None,
                )
                traces[request_idx]["prompt_language"] = normalize_prompt_language_key(str(effective_request["group_row"].get("language") or "en"))
                traces[request_idx]["prompt_token_count"] = prompt_token_count
                traces[request_idx]["prompt_token_budget"] = prompt_token_budget
                traces[request_idx]["prompt_compaction_stage"] = prompt_compaction_stage
                traces[request_idx]["prompt_auto_compacted"] = prompt_auto_compacted
                messages_batch.append(
                    [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": prompt},
                    ]
                )
                traces[request_idx]["attempt_count"] = attempt + 1
                traces[request_idx]["retry_count"] = attempt

            try:
                generation_max_new_tokens = self.phase2.resolve_attempt_max_new_tokens(
                    inference_cfg=self.inference_cfg,
                    enable_thinking=bool(llm.enable_thinking),
                    attempt=attempt,
                    force_no_think_retry=force_no_think_retry,
                )
                outputs = self.phase2.generate_with_oom_split_retry(
                    llm=llm,
                    messages_batch=messages_batch,
                    role_name="auditor",
                    logger=self.logger,
                    enable_split_retry=parse_bool(self.inference_cfg.get("enable_oom_split_retry", True), default=True),
                    max_depth=int(self.inference_cfg.get("oom_split_retry_max_depth", 4)),
                    max_new_tokens_override=generation_max_new_tokens,
                    force_no_think=force_no_think_this_attempt,
                )
            except Exception as exc:
                error_text = str(exc)
                for request_idx in pending_indices:
                    retry_notes[request_idx] = error_text
                    traces[request_idx]["parse_fail"] = True
                    traces[request_idx]["error"] = error_text
                    traces[request_idx]["parse_errors"].append(error_text)
                continue

            next_pending: List[int] = []
            for request_idx, raw_output in zip(pending_indices, outputs):
                trace = traces[request_idx]
                trace["raw_output"] = (raw_output or "")[:4000]
                try:
                    payload = extract_json_object_candidate(raw_output or "")
                except Exception as exc:
                    error_text = str(exc)
                    retry_notes[request_idx] = error_text
                    trace["parse_fail"] = True
                    trace["error"] = error_text
                    trace["parse_errors"].append(error_text)
                    next_pending.append(request_idx)
                    continue

                payloads[request_idx] = payload
                trace["successful_attempt"] = attempt + 1
                trace["parse_fail"] = False
                trace["error"] = None

            pending_indices = next_pending

        self.logger.event(
            "runtime",
            "auditor_batch_end",
            batch_size=len(requests),
            group_ids=[(request.get("group_row") or {}).get("group_id") for request in requests],
            resolved=sum(1 for payload in payloads if payload is not None),
        )
        return list(zip(payloads, traces))

    def audit(
        self,
        group_row: Dict[str, Any],
        role_payload: List[Dict[str, Any]],
        clusters_payload: List[Dict[str, Any]],
        triage_payload: Dict[str, Any],
        beacons: Sequence[Dict[str, str]],
        max_parse_retries: int,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        results = self.audit_batch(
            requests=[
                {
                    "group_row": group_row,
                    "role_payload": role_payload,
                    "clusters_payload": clusters_payload,
                    "triage_payload": triage_payload,
                    "beacons": list(beacons),
                }
            ],
            max_parse_retries=max_parse_retries,
        )
        if not results:
            return None, self._empty_trace()
        return results[0]


def load_prompt_templates(default_path: Optional[Path], prompt_map: Dict[str, Path]) -> Dict[str, str]:
    templates: Dict[str, str] = {}
    if default_path is not None:
        templates["default"] = default_path.read_text(encoding="utf-8")
    for lang_key, path in prompt_map.items():
        templates[normalize_prompt_language_key(lang_key)] = path.read_text(encoding="utf-8")
    if not templates:
        raise ValueError("No auditor prompt template configured")
    return templates


def should_invoke_auditor(signals: Dict[str, Any], cfg: Dict[str, Any]) -> bool:
    phase3_cfg = cfg.get("phase3", {})
    lane = signals.get("lane")
    if lane == "consensus" and signals.get("complete_consensus"):
        fastpath_enabled = parse_bool(phase3_cfg.get("enable_consensus_fastpath", True), default=True)
        fastpath_max_clusters = max(1, int(phase3_cfg.get("consensus_fastpath_max_cluster_count", 1) or 1))
        fastpath_blocked = bool(
            signals.get("lexicon_hit")
            or signals.get("victim_microaggression")
            or signals.get("quote_caution")
        )
        if fastpath_enabled and int(signals.get("cluster_count", 0) or 0) <= fastpath_max_clusters and not fastpath_blocked:
            return False
    if lane == "consensus":
        if parse_bool(phase3_cfg.get("auditor_on_consensus", False), default=False):
            return True
        if not signals.get("lexicon_hit") and not signals.get("victim_microaggression"):
            return False
    if lane == "defective" and parse_bool(phase3_cfg.get("auditor_on_defective", True), default=True):
        return True
    if lane == "conflict" and parse_bool(phase3_cfg.get("auditor_on_conflict", True), default=True):
        return True
    if signals.get("lexicon_hit") and parse_bool(phase3_cfg.get("auditor_on_lexicon_hit", True), default=True):
        return True
    if signals.get("victim_microaggression") and parse_bool(phase3_cfg.get("auditor_on_victim_microaggression", True), default=True):
        return True
    return False


def build_role_payload(
    candidates: Sequence[Candidate],
    expected_roles: Sequence[str],
    compact_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    by_role: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_role[candidate.role].append(candidate)
    compact_cfg = compact_cfg or {}
    compact_enabled = parse_bool(compact_cfg.get("enabled", False), default=False)
    role_top_candidates = int(max(1, compact_cfg.get("role_top_candidates", 1) or 1))
    include_candidate_id = parse_bool(compact_cfg.get("include_candidate_id", False), default=False)
    include_full_issues = parse_bool(compact_cfg.get("include_full_issues", False), default=False)
    rationale_max_chars = int(max(24, compact_cfg.get("rationale_max_chars", 120) or 120))
    drop_empty_all_candidates = parse_bool(compact_cfg.get("drop_empty_all_candidates", True), default=True)
    payload: List[Dict[str, Any]] = []
    for role in expected_roles:
        role_candidates = sorted(by_role.get(role, []), key=lambda c: (-c.quality_score, c.role_index))
        top_candidate_payload = None
        all_candidates_payload: List[Dict[str, Any]] = []
        if role_candidates:
            if compact_enabled:
                top_candidate_payload = compact_candidate_for_auditor(
                    role_candidates[0],
                    include_candidate_id=include_candidate_id,
                    include_full_issues=include_full_issues,
                    rationale_max_chars=rationale_max_chars,
                )
                all_candidates_payload = [
                    compact_candidate_for_auditor(
                        candidate,
                        include_candidate_id=include_candidate_id,
                        include_full_issues=include_full_issues,
                        rationale_max_chars=rationale_max_chars,
                    )
                    for candidate in role_candidates[:role_top_candidates]
                ]
            else:
                top_candidate_payload = compact_candidate(role_candidates[0])
                all_candidates_payload = [compact_candidate(c) for c in role_candidates[:4]]
        entry = {
            "role": role,
            "candidate_count": len(role_candidates),
            "top_candidate": top_candidate_payload,
        }
        if all_candidates_payload or not drop_empty_all_candidates:
            entry["all_candidates"] = all_candidates_payload
        payload.append(entry)
    return payload


def build_clusters_payload(
    clusters: Sequence[Cluster],
    compact_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    compact_cfg = compact_cfg or {}
    compact_enabled = parse_bool(compact_cfg.get("enabled", False), default=False)
    cluster_top_k = int(max(1, compact_cfg.get("cluster_top_k", 2) or 2))
    cluster_member_top_k = int(max(1, compact_cfg.get("cluster_member_top_k", 3) or 3))
    include_candidate_id = parse_bool(compact_cfg.get("include_candidate_id", False), default=False)
    include_full_issues = parse_bool(compact_cfg.get("include_full_issues", False), default=False)
    rationale_max_chars = int(max(24, compact_cfg.get("rationale_max_chars", 120) or 120))
    selected_clusters = (
        sorted(clusters, key=rank_cluster_for_auditor)[:cluster_top_k]
        if compact_enabled
        else list(clusters)
    )
    out: List[Dict[str, Any]] = []
    for cluster in selected_clusters:
        members_payload = []
        selected_members = (
            select_cluster_members_for_auditor(cluster, cluster_member_top_k)
            if compact_enabled
            else list(cluster.members[:6])
        )
        for member in selected_members:
            if compact_enabled:
                members_payload.append(
                    compact_cluster_member_for_auditor(
                        member,
                        include_candidate_id=include_candidate_id,
                        include_full_issues=include_full_issues,
                        rationale_max_chars=rationale_max_chars,
                    )
                )
            else:
                members_payload.append(compact_candidate(member))
        out.append(
            {
                "cluster_id": cluster.cluster_id,
                "canonical_tuple": cluster.canonical_tuple,
                "support_roles": cluster.support_roles,
                "cluster_score": cluster.cluster_score,
                "lexicon_hit": cluster.lexicon_hit,
                "member_count": len(cluster.members),
                "members": members_payload,
            }
        )
    return out


def empty_auditor_trace() -> Dict[str, Any]:
    return {
        "used": False,
        "parse_fail": False,
        "raw_output": None,
        "error": None,
        "attempt_count": 0,
        "retry_count": 0,
        "successful_attempt": None,
        "parse_errors": [],
        "prompt_token_count": None,
        "prompt_token_budget": None,
        "prompt_compaction_stage": 0,
        "prompt_auto_compacted": False,
    }


def prepare_group_state(
    group_row: Dict[str, Any],
    expected_roles: Sequence[str],
    cfg: Dict[str, Any],
    scoring_cfg: Dict[str, Dict[str, float]],
    regen_engine: Optional[RegenerationEngine],
) -> PreparedGroupState:
    phase3_cfg = cfg.get("phase3", {})
    quality_threshold = safe_float(phase3_cfg.get("candidate_quality_threshold", 0.55), default=0.55)
    cluster_threshold = safe_float(phase3_cfg.get("cluster_similarity_threshold", 0.62), default=0.62)
    max_parse_retries = int(phase3_cfg.get("max_parse_retries", 1))
    enable_regeneration = parse_bool(phase3_cfg.get("enable_regeneration", True), default=True)
    max_regenerate_roles = int(max(0, phase3_cfg.get("max_regenerate_roles_per_group", 2)))

    working_row = copy.deepcopy(group_row)
    candidates = build_candidates(working_row, threshold=quality_threshold, scoring_cfg=scoring_cfg)
    clusters = cluster_candidates(candidates, threshold=cluster_threshold, scoring_cfg=scoring_cfg)
    signals = detect_signals(working_row, candidates, clusters, expected_roles, scoring_cfg)
    beacons = assemble_beacons(signals)

    regeneration_trace: List[Dict[str, Any]] = []
    regenerated_roles: List[str] = []
    if regen_engine is not None and enable_regeneration and signals.get("defective_roles"):
        roles_to_regen = list(signals["defective_roles"])[:max_regenerate_roles]
        regenerated_outputs, regeneration_trace = regen_engine.regenerate(
            group_row=working_row,
            role_names=roles_to_regen,
            max_parse_retries=max_parse_retries,
        )
        for output_key, new_items in regenerated_outputs.items():
            if new_items:
                working_row["phase2_predictions"][output_key] = new_items
                regenerated_roles.append(OUTPUT_KEY_ROLE_MAP.get(output_key, output_key))
        candidates = build_candidates(working_row, threshold=quality_threshold, scoring_cfg=scoring_cfg)
        clusters = cluster_candidates(candidates, threshold=cluster_threshold, scoring_cfg=scoring_cfg)
        signals = detect_signals(working_row, candidates, clusters, expected_roles, scoring_cfg)
        beacons = assemble_beacons(signals)

    fallback_cluster = choose_cluster_fallback(clusters, signals, scoring_cfg)
    fallback_tuple = normalize_final_tuple(fallback_cluster.canonical_tuple if fallback_cluster else {}, language=working_row["language"])
    fallback_rationale = build_deterministic_rationale(fallback_cluster, signals, working_row["language"])

    return PreparedGroupState(
        working_row=working_row,
        candidates=list(candidates),
        clusters=list(clusters),
        signals=signals,
        beacons=list(beacons),
        regeneration_trace=regeneration_trace,
        regenerated_roles=regenerated_roles,
        fallback_cluster=fallback_cluster,
        fallback_tuple=fallback_tuple,
        fallback_rationale=fallback_rationale,
    )


def build_auditor_request(
    prepared: PreparedGroupState,
    expected_roles: Sequence[str],
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    working_row = prepared.working_row
    compact_cfg = parse_auditor_compact_cfg(cfg or {})
    beacons_payload: List[Dict[str, Any]]
    if parse_bool(compact_cfg.get("enabled", False), default=False) and parse_bool(compact_cfg.get("beacons_id_only", False), default=False):
        beacons_payload = [{"id": b["id"]} for b in prepared.beacons]
    else:
        beacons_payload = list(prepared.beacons)
    return {
        "group_row": {
            "group_id": working_row["group_id"],
            "sample_id": working_row["sample_id"],
            "source": working_row["source"],
            "split": working_row["split"],
            "language": working_row["language"],
            "group_text": working_row["group_text"],
            "focus_text": working_row["focus_text"],
            "canonical_target": working_row["canonical_target"],
            "escalation_flag": working_row["escalation_flag"],
        },
        "role_payload": build_role_payload(prepared.candidates, expected_roles, compact_cfg=compact_cfg),
        "clusters_payload": build_clusters_payload(prepared.clusters, compact_cfg=compact_cfg),
        "triage_payload": {
            "lane": prepared.signals["lane"],
            "initial_conflict_type": prepared.signals["initial_conflict_type"],
            "signals": {
                k: v
                for k, v in prepared.signals.items()
                if k not in {"present_roles", "missing_roles", "defective_roles"}
            },
            "present_roles": prepared.signals["present_roles"],
            "missing_roles": prepared.signals["missing_roles"],
            "defective_roles": prepared.signals["defective_roles"],
            "regenerated_roles": prepared.regenerated_roles,
        },
        "beacons": beacons_payload,
    }


def finalize_group_state(
    prepared: PreparedGroupState,
    expected_roles: Sequence[str],
    normalized_auditor: Optional[Dict[str, Any]],
    auditor_trace: Optional[Dict[str, Any]],
    auditor_used: bool,
    cfg: Dict[str, Any],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    working_row = prepared.working_row
    candidates = prepared.candidates
    clusters = prepared.clusters
    signals = prepared.signals
    beacons = prepared.beacons
    fallback_cluster = prepared.fallback_cluster
    fallback_tuple = prepared.fallback_tuple
    fallback_rationale = prepared.fallback_rationale
    regeneration_trace = prepared.regeneration_trace
    regenerated_roles = prepared.regenerated_roles

    selected_cluster_id = normalized_auditor.get("selected_cluster_id") if normalized_auditor else None
    selected_cluster = next((c for c in clusters if c.cluster_id == selected_cluster_id), None) if selected_cluster_id else None
    if selected_cluster is None:
        selected_cluster = fallback_cluster

    if normalized_auditor:
        final_tuple = {
            "target": normalized_auditor["final_target"],
            "argument": normalized_auditor["final_argument"],
            "group": normalized_auditor["final_group"],
            "harmful": normalized_auditor["final_label"],
        }
        final_tuple = normalize_final_tuple(final_tuple, language=working_row["language"])
        fused_rationale = normalized_auditor["fused_rationale"]
        audit_confidence = normalized_auditor["confidence"]
        downweighted_roles = normalized_auditor["downweighted_roles"]
        rejected_roles = normalized_auditor["rejected_roles"]
        correction_trace = normalized_auditor["correction_trace"]
    else:
        final_tuple = fallback_tuple
        fused_rationale = fallback_rationale
        audit_confidence = (
            round(clamp(0.55 + 0.12 * len(selected_cluster.support_roles) + (0.08 if selected_cluster and selected_cluster.lexicon_hit else 0.0)), 4)
            if selected_cluster
            else 0.55
        )
        downweighted_roles = []
        rejected_roles = []
        correction_trace = []

    final_tuple, span_refinement_trace = refine_final_tuple_spans(
        final_tuple=final_tuple,
        selected_cluster=selected_cluster,
        focus_text=working_row["focus_text"],
        language=working_row["language"],
    )
    if span_refinement_trace:
        correction_trace = list(correction_trace) + span_refinement_trace

    retained_clusters = select_retained_clusters(
        clusters=clusters,
        primary_cluster=selected_cluster,
        signals=signals,
        cfg=cfg,
    )
    retained_cluster_ids = [cluster.cluster_id for cluster in retained_clusters]

    final_tuples: List[Dict[str, Any]] = []
    final_tuple_keys: Set[Tuple[str, str, str, str]] = set()

    def append_final_tuple_entry(
        tuple_obj: Dict[str, Any],
        cluster: Optional[Cluster],
        derivation: str,
        tuple_correction_trace: Sequence[Dict[str, Any]],
    ) -> None:
        key = final_tuple_identity_key(tuple_obj, working_row["language"])
        if key in final_tuple_keys:
            return
        final_tuple_keys.add(key)
        entry = dict(tuple_obj)
        entry["cluster_id"] = cluster.cluster_id if cluster is not None else None
        entry["derivation"] = derivation
        entry["cluster_score"] = cluster.cluster_score if cluster is not None else None
        entry["support_roles"] = list(cluster.support_roles) if cluster is not None else []
        entry["lexicon_hit"] = bool(cluster.lexicon_hit) if cluster is not None else False
        if tuple_correction_trace:
            entry["correction_trace"] = list(tuple_correction_trace)
        final_tuples.append(entry)

    append_final_tuple_entry(
        tuple_obj=final_tuple,
        cluster=selected_cluster,
        derivation="auditor_primary" if normalized_auditor else "deterministic_primary",
        tuple_correction_trace=correction_trace,
    )
    for cluster in retained_clusters:
        if selected_cluster is not None and cluster.cluster_id == selected_cluster.cluster_id:
            continue
        supplemental_tuple, supplemental_trace = build_cluster_final_tuple(
            cluster=cluster,
            focus_text=working_row["focus_text"],
            language=working_row["language"],
        )
        append_final_tuple_entry(
            tuple_obj=supplemental_tuple,
            cluster=cluster,
            derivation="supplemental_cluster",
            tuple_correction_trace=supplemental_trace,
        )

    decision_id = make_sha1("phase3", working_row["group_id"], working_row["sample_id"], fused_rationale)[:16]

    rectified_rows: List[Dict[str, Any]] = []
    by_role: Dict[str, List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_role[candidate.role].append(candidate)
    for role in expected_roles:
        role_candidates = sorted(by_role.get(role, []), key=lambda c: (-c.quality_score, c.role_index))
        rectified_rows.append(
            {
                "decision_id": decision_id,
                "group_id": working_row["group_id"],
                "sample_id": working_row["sample_id"],
                "source": working_row["source"],
                "split": working_row["split"],
                "language": working_row["language"],
                "role": role,
                "role_status": "regenerated" if role in regenerated_roles else ("missing" if not role_candidates else "original"),
                "downweighted_by_auditor": role in downweighted_roles,
                "rejected_by_auditor": role in rejected_roles,
                "candidate_count": len(role_candidates),
                "candidates": [compact_candidate(c) for c in role_candidates[:6]],
            }
        )

    trace_payload = auditor_trace or empty_auditor_trace()
    group_decision = {
        "decision_id": decision_id,
        "group_id": working_row["group_id"],
        "sample_id": working_row["sample_id"],
        "original_id": working_row["original_id"],
        "source": working_row["source"],
        "split": working_row["split"],
        "language": working_row["language"],
        "local_group_id": working_row["local_group_id"],
        "group_text": working_row["group_text"],
        "focus_text": working_row["focus_text"],
        "canonical_target": working_row["canonical_target"],
        "escalation_flag": working_row["escalation_flag"],
        "phase2_predictions": working_row["phase2_predictions"],
        "triage": {
            "lane": signals["lane"],
            "initial_conflict_type": signals["initial_conflict_type"],
            "present_roles": signals["present_roles"],
            "missing_roles": signals["missing_roles"],
            "defective_roles": signals["defective_roles"],
            "regenerated_roles": regenerated_roles,
        },
        "signals": {
            "lexicon_hit": signals["lexicon_hit"],
            "victim_microaggression": signals["victim_microaggression"],
            "moderator_strong": signals["moderator_strong"],
            "quote_caution": signals["quote_caution"],
            "complete_consensus": signals["complete_consensus"],
            "consensus_aligned_roles": signals.get("consensus_aligned_roles") or [],
        },
        "applied_beacons": [b["id"] for b in beacons],
        "beacons": list(beacons),
        "clusters": build_clusters_payload(clusters),
        "regeneration_trace": regeneration_trace,
        "auditor_trace": trace_payload,
        "auditor_used": auditor_used,
        "selected_cluster_id": selected_cluster.cluster_id if selected_cluster else None,
        "retained_cluster_ids": retained_cluster_ids,
        "final_tuple": final_tuple,
        "final_tuples": final_tuples,
        "final_verdict": {
            "final_target": final_tuple["target"],
            "final_argument": final_tuple["argument"],
            "final_group": final_tuple["group"],
            "final_label": final_tuple["harmful"],
            "fused_rationale": fused_rationale,
            "correction_trace": correction_trace,
        },
        "audit_confidence": audit_confidence,
    }

    final_row = {
        "decision_id": decision_id,
        "group_id": working_row["group_id"],
        "sample_id": working_row["sample_id"],
        "local_group_id": working_row["local_group_id"],
        "source": working_row["source"],
        "split": working_row["split"],
        "language": working_row["language"],
        "sentence_ids": working_row.get("sentence_ids") or [],
        "group_text": working_row["group_text"],
        "final_tuple": final_tuple,
        "final_tuples": final_tuples,
        "final_target": final_tuple["target"],
        "final_argument": final_tuple["argument"],
        "final_group": final_tuple["group"],
        "final_label": final_tuple["harmful"],
        "fused_rationale": fused_rationale,
        "initial_conflict_type": signals["initial_conflict_type"],
        "applied_beacons": [b["id"] for b in beacons],
        "audit_confidence": audit_confidence,
        "veto_flag": False,
        "veto_reason": "",
        "decision_summary": fused_rationale,
        "correction_trace": correction_trace,
        "consensus_trace": {
            "lane": signals["lane"],
            "selected_cluster_id": selected_cluster.cluster_id if selected_cluster else None,
            "retained_cluster_ids": retained_cluster_ids,
            "regenerated_roles": regenerated_roles,
            "downweighted_roles": downweighted_roles,
            "rejected_roles": rejected_roles,
            "consensus_aligned_roles": signals.get("consensus_aligned_roles") or [],
        },
    }
    return group_decision, rectified_rows, final_row


def process_group(
    group_row: Dict[str, Any],
    expected_roles: Sequence[str],
    cfg: Dict[str, Any],
    scoring_cfg: Dict[str, Dict[str, float]],
    auditor_engine: Optional[AuditorEngine],
    regen_engine: Optional[RegenerationEngine],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any]]:
    prepared = prepare_group_state(
        group_row=group_row,
        expected_roles=expected_roles,
        cfg=cfg,
        scoring_cfg=scoring_cfg,
        regen_engine=regen_engine,
    )
    max_parse_retries = int(cfg.get("phase3", {}).get("max_parse_retries", 1))

    normalized_auditor: Optional[Dict[str, Any]] = None
    auditor_trace = empty_auditor_trace()
    auditor_used = False
    if auditor_engine is not None and should_invoke_auditor(prepared.signals, cfg):
        auditor_used = True
        request = build_auditor_request(prepared, expected_roles, cfg=cfg)
        raw_payload, auditor_trace = auditor_engine.audit(
            group_row=request["group_row"],
            role_payload=request["role_payload"],
            clusters_payload=request["clusters_payload"],
            triage_payload=request["triage_payload"],
            beacons=request["beacons"],
            max_parse_retries=max_parse_retries,
        )
        if raw_payload is not None:
            normalized_auditor = normalize_auditor_payload(
                payload=raw_payload,
                language=prepared.working_row["language"],
                focus_text=prepared.working_row["focus_text"],
                fallback_cluster=prepared.fallback_cluster,
            )

    return finalize_group_state(
        prepared=prepared,
        expected_roles=expected_roles,
        normalized_auditor=normalized_auditor,
        auditor_trace=auditor_trace,
        auditor_used=auditor_used,
        cfg=cfg,
    )


def build_phase4_trace_pack_rows(group_decisions: Sequence[Dict[str, Any]], rectified_rows: Sequence[Dict[str, Any]], final_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rectified_by_decision: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    final_by_decision: Dict[str, Dict[str, Any]] = {}
    for row in rectified_rows:
        rectified_by_decision[str(row.get("decision_id") or "")].append(row)
    for row in final_rows:
        final_by_decision[str(row.get("decision_id") or "")] = row

    out: List[Dict[str, Any]] = []
    for decision in group_decisions:
        decision_id = str(decision.get("decision_id") or "")
        final_row = final_by_decision.get(decision_id, {})
        out.append(
            {
                "sample_id": str(decision.get("sample_id") or ""),
                "group_id": str(decision.get("group_id") or ""),
                "decision_id": decision_id,
                "source": str(decision.get("source") or "unknown"),
                "split": str(decision.get("split") or "train"),
                "language": str(decision.get("language") or "en"),
                "step_a_decomposition_evidence": {
                    "group_text": decision.get("group_text"),
                    "focus_text": decision.get("focus_text"),
                    "canonical_target": decision.get("canonical_target"),
                    "escalation_flag": bool(decision.get("escalation_flag")),
                },
                "step_b_role_hypotheses": [
                    {
                        "role": row.get("role"),
                        "role_status": row.get("role_status"),
                        "candidate_count": row.get("candidate_count"),
                        "candidates": row.get("candidates"),
                    }
                    for row in sorted(rectified_by_decision.get(decision_id, []), key=lambda r: str(r.get("role") or ""))
                ],
                "step_c_conflict_diagnosis": {
                    "triage": decision.get("triage"),
                    "signals": decision.get("signals"),
                    "applied_beacons": decision.get("applied_beacons"),
                    "clusters": decision.get("clusters"),
                },
                "step_d_rectification_actions": {
                    "regeneration_trace": decision.get("regeneration_trace"),
                    "auditor_trace": decision.get("auditor_trace"),
                },
                "step_e_final_arbitration": {
                    "selected_cluster_id": decision.get("selected_cluster_id"),
                    "retained_cluster_ids": decision.get("retained_cluster_ids") or [],
                    "final_tuple": final_row.get("final_tuple") if isinstance(final_row.get("final_tuple"), dict) else decision.get("final_tuple"),
                    "final_tuples": final_row.get("final_tuples") if isinstance(final_row.get("final_tuples"), list) else [],
                    "fused_rationale": final_row.get("fused_rationale") or (decision.get("final_verdict") or {}).get("fused_rationale"),
                    "correction_trace": final_row.get("correction_trace") or (decision.get("final_verdict") or {}).get("correction_trace") or [],
                    "audit_confidence": final_row.get("audit_confidence") or decision.get("audit_confidence"),
                },
            }
        )
    return out


def quality_gates(group_decisions: Sequence[Dict[str, Any]], final_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    invalid_final_tuple = 0
    duplicate_decision_id = 0
    seen: Set[str] = set()
    for row in final_rows:
        did = str(row.get("decision_id") or "")
        if did in seen:
            duplicate_decision_id += 1
        seen.add(did)
        final_tuple = row.get("final_tuple") if isinstance(row.get("final_tuple"), dict) else {}
        if normalize_string(final_tuple.get("target")) is None:
            invalid_final_tuple += 1
    passed = invalid_final_tuple == 0 and duplicate_decision_id == 0 and len(group_decisions) == len(final_rows)
    return {
        "total_group_decisions": len(group_decisions),
        "total_final_tuples": len(final_rows),
        "invalid_final_tuple": invalid_final_tuple,
        "duplicate_decision_id": duplicate_decision_id,
        "passed": passed,
    }


def build_summary(group_decisions: Sequence[Dict[str, Any]], final_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    lane_dist = dict(sorted(Counter(str((r.get("triage") or {}).get("lane") or "unknown") for r in group_decisions).items()))
    lang_dist = dict(sorted(Counter(str(r.get("language") or "unknown") for r in group_decisions).items()))
    beacon_dist = Counter()
    regenerated = 0
    auditor_used = 0
    emitted_tuple_count = 0
    for row in group_decisions:
        for beacon in row.get("applied_beacons") or []:
            beacon_dist[str(beacon)] += 1
        regenerated += len((row.get("triage") or {}).get("regenerated_roles") or [])
        auditor_used += 1 if row.get("auditor_used") else 0
    for row in final_rows:
        tuple_list = row.get("final_tuples") if isinstance(row.get("final_tuples"), list) else None
        emitted_tuple_count += len(tuple_list) if tuple_list else 1
    return {
        "group_decision_count": len(group_decisions),
        "final_tuple_count": len(final_rows),
        "emitted_tuple_count": emitted_tuple_count,
        "avg_emitted_tuples_per_group": round(emitted_tuple_count / len(final_rows), 4) if final_rows else 0.0,
        "lane_distribution": lane_dist,
        "language_distribution": lang_dist,
        "beacon_distribution": dict(sorted(beacon_dist.items())),
        "regenerated_role_count": regenerated,
        "auditor_used_count": auditor_used,
    }


def bucket_rows_by_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("split") or "train")].append(row)
    return buckets


def bucket_rows_by_language_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        language = normalize_prompt_language_key(str(row.get("language") or "unknown"))
        split = str(row.get("split") or "train")
        buckets[f"{language}_{split}"].append(row)
    return buckets


def ordered_bucket_values(observed_values: Sequence[str], preferred_values: Sequence[str]) -> List[str]:
    observed = dedupe_preserve_order([str(value).strip() for value in observed_values if str(value).strip()])
    preferred = [str(value).strip() for value in preferred_values if str(value).strip()]
    out = [value for value in preferred if value in observed]
    out.extend(value for value in observed if value not in out)
    return out


def build_processing_buckets(rows: Sequence[Dict[str, Any]]) -> List[Tuple[str, str, List[Dict[str, Any]]]]:
    buckets: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    observed_languages: List[str] = []
    observed_splits: List[str] = []
    for row in rows:
        language = normalize_prompt_language_key(str(row.get("language") or "unknown"))
        split = str(row.get("split") or "train")
        buckets[(language, split)].append(row)
        observed_languages.append(language)
        observed_splits.append(split)

    ordered_languages = ordered_bucket_values(observed_languages, ["zh", "en"])
    ordered_splits = ordered_bucket_values(observed_splits, ["test", "train", "dev"])

    out: List[Tuple[str, str, List[Dict[str, Any]]]] = []
    for language in ordered_languages:
        for split in ordered_splits:
            bucket_rows = buckets.get((language, split))
            if bucket_rows:
                out.append((language, split, bucket_rows))
    return out


def iter_row_chunks(rows: Sequence[Dict[str, Any]], chunk_size: int) -> Iterable[List[Dict[str, Any]]]:
    size = max(1, int(chunk_size))
    for start in range(0, len(rows), size):
        yield list(rows[start : start + size])


def write_rows_with_views(base_dir: Path, stem: str, rows: Sequence[Dict[str, Any]]) -> None:
    write_jsonl(base_dir / f"{stem}.jsonl", rows)

    rows_by_split = bucket_rows_by_split(rows)
    split_order = ordered_bucket_values(["train", "dev", "test", *rows_by_split.keys()], ["train", "dev", "test"])
    for split in split_order:
        write_jsonl(
            base_dir / f"{stem}_by_split" / f"{split}.jsonl",
            rows_by_split.get(split, []),
        )

    rows_by_language_split = bucket_rows_by_language_split(rows)
    observed_language_order: List[str] = []
    for key in rows_by_language_split.keys():
        language, _, _ = key.partition("_")
        observed_language_order.append(language)
    language_order = ordered_bucket_values(["zh", "en", *observed_language_order], ["zh", "en"])
    for language in language_order:
        for split in split_order:
            key = f"{language}_{split}"
            write_jsonl(
                base_dir / f"{stem}_by_lang_split" / f"{key}.jsonl",
                rows_by_language_split.get(key, []),
            )


def write_candidate_outputs(run_root: Path, group_decisions: Sequence[Dict[str, Any]], rectified_rows: Sequence[Dict[str, Any]], final_rows: Sequence[Dict[str, Any]], phase4_rows: Sequence[Dict[str, Any]]) -> None:
    candidates_dir = run_root / "candidates"
    write_rows_with_views(candidates_dir, "group_decisions", group_decisions)
    write_rows_with_views(candidates_dir, "rectified_roles", rectified_rows)
    write_rows_with_views(candidates_dir, "final_tuples", final_rows)
    write_rows_with_views(candidates_dir, "phase4_trace_pack", phase4_rows)
    write_rows_with_views(run_root / "phase4", "phase4_trace_pack", phase4_rows)


def load_resume_state(paths: Paths, filtered_rows: Sequence[Dict[str, Any]], logger: EventLogger) -> Optional[ResumeState]:
    checkpoint_path = paths.run_root / "reports" / "checkpoint_progress.json"
    decisions_path = paths.run_root / "candidates" / "group_decisions.jsonl"
    rectified_path = paths.run_root / "candidates" / "rectified_roles.jsonl"
    final_path = paths.run_root / "candidates" / "final_tuples.jsonl"
    if not checkpoint_path.exists():
        return None
    if not decisions_path.exists() or not final_path.exists():
        logger.event(
            "checkpoint",
            "resume_unavailable",
            level="WARNING",
            reason="missing_candidate_exports",
            checkpoint_path=str(checkpoint_path),
            decisions_path=str(decisions_path),
            final_path=str(final_path),
        )
        return None

    try:
        checkpoint = read_json(checkpoint_path)
    except Exception as exc:
        logger.event(
            "checkpoint",
            "resume_unavailable",
            level="WARNING",
            reason="invalid_checkpoint_json",
            checkpoint_path=str(checkpoint_path),
            error=str(exc),
        )
        return None

    requested_completed = max(0, int(checkpoint.get("completed_groups") or 0))
    if requested_completed <= 0:
        return None

    loaded_decisions = read_jsonl(decisions_path)
    loaded_final_rows = read_jsonl(final_path)
    loaded_rectified_rows = read_jsonl(rectified_path) if rectified_path.exists() else []
    if not loaded_decisions or not loaded_final_rows:
        logger.event(
            "checkpoint",
            "resume_unavailable",
            level="WARNING",
            reason="empty_candidate_exports",
            requested_completed=requested_completed,
            loaded_decisions=len(loaded_decisions),
            loaded_final_rows=len(loaded_final_rows),
        )
        return None

    final_by_decision: Dict[str, Dict[str, Any]] = {}
    for row in loaded_final_rows:
        decision_id = str(row.get("decision_id") or "")
        if decision_id and decision_id not in final_by_decision:
            final_by_decision[decision_id] = row

    stable_group_decisions: List[Dict[str, Any]] = []
    stable_final_rows: List[Dict[str, Any]] = []
    stable_decision_ids: Set[str] = set()
    max_candidate_count = min(requested_completed, len(loaded_decisions), len(filtered_rows))
    for index in range(max_candidate_count):
        decision = loaded_decisions[index]
        decision_id = str(decision.get("decision_id") or "")
        group_id = str(decision.get("group_id") or "")
        expected_group_id = str(filtered_rows[index].get("group_id") or "")
        if not decision_id or decision_id in stable_decision_ids:
            logger.event(
                "checkpoint",
                "resume_truncated",
                level="WARNING",
                reason="invalid_or_duplicate_decision_id",
                stable_completed=len(stable_group_decisions),
                decision_id=decision_id,
                group_id=group_id,
            )
            break
        if group_id != expected_group_id:
            logger.event(
                "checkpoint",
                "resume_truncated",
                level="WARNING",
                reason="group_id_mismatch",
                stable_completed=len(stable_group_decisions),
                found_group_id=group_id,
                expected_group_id=expected_group_id,
            )
            break
        final_row = final_by_decision.get(decision_id)
        if final_row is None:
            logger.event(
                "checkpoint",
                "resume_truncated",
                level="WARNING",
                reason="missing_final_row",
                stable_completed=len(stable_group_decisions),
                decision_id=decision_id,
                group_id=group_id,
            )
            break
        stable_group_decisions.append(decision)
        stable_final_rows.append(final_row)
        stable_decision_ids.add(decision_id)

    stable_completed = len(stable_group_decisions)
    if stable_completed <= 0:
        logger.event(
            "checkpoint",
            "resume_unavailable",
            level="WARNING",
            reason="no_stable_completed_groups",
            requested_completed=requested_completed,
        )
        return None

    stable_rectified_rows = [
        row
        for row in loaded_rectified_rows
        if str(row.get("decision_id") or "") in stable_decision_ids
    ]

    next_bucket = ""
    if stable_completed < len(filtered_rows):
        next_row = filtered_rows[stable_completed]
        next_bucket = f"{normalize_prompt_language_key(next_row.get('language'))}_{str(next_row.get('split') or 'train')}"
    elif stable_group_decisions:
        last_row = stable_group_decisions[-1]
        next_bucket = f"{normalize_prompt_language_key(last_row.get('language'))}_{str(last_row.get('split') or 'train')}"
    current_bucket = normalize_string(checkpoint.get("current_bucket")) or next_bucket
    if next_bucket:
        current_bucket = next_bucket

    last_completed_group_id = str(stable_group_decisions[-1].get("group_id") or "")
    logger.event(
        "checkpoint",
        "resume_loaded",
        checkpoint_path=str(checkpoint_path),
        requested_completed=requested_completed,
        resumed_completed=stable_completed,
        loaded_decisions=len(loaded_decisions),
        loaded_final_rows=len(loaded_final_rows),
        loaded_rectified_rows=len(loaded_rectified_rows),
        resumed_rectified_rows=len(stable_rectified_rows),
        current_bucket=current_bucket,
        last_completed_group_id=last_completed_group_id,
    )
    return ResumeState(
        group_decisions=stable_group_decisions,
        rectified_rows=stable_rectified_rows,
        final_rows=stable_final_rows,
        completed_groups=stable_completed,
        prepared_groups=stable_completed,
        current_bucket=current_bucket,
        last_completed_group_id=last_completed_group_id,
    )


def slice_processing_buckets_for_resume(
    processing_buckets: Sequence[Tuple[str, str, List[Dict[str, Any]]]],
    completed_groups: int,
) -> List[Tuple[int, str, str, List[Dict[str, Any]]]]:
    remaining: List[Tuple[int, str, str, List[Dict[str, Any]]]] = []
    groups_to_skip = max(0, int(completed_groups))
    for bucket_index, (language, split, rows) in enumerate(processing_buckets, start=1):
        if groups_to_skip >= len(rows):
            groups_to_skip -= len(rows)
            continue
        if groups_to_skip > 0:
            rows = rows[groups_to_skip:]
            groups_to_skip = 0
        if rows:
            remaining.append((bucket_index, language, split, rows))
    if groups_to_skip > 0:
        raise ValueError(
            f"Resume completed_groups exceeds available rows: remaining_skip={groups_to_skip}, total_rows={sum(len(rows) for _, _, rows in processing_buckets)}"
        )
    return remaining


def run_phase3(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cli_args = args or parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, cli_args)
    paths = build_paths(cli_args, cfg, workspace_root)
    run_id = paths.run_root.name
    phase3_cfg = cfg.get("phase3", {}) if isinstance(cfg.get("phase3"), dict) else {}
    resume_enabled = parse_bool(phase3_cfg.get("resume_from_checkpoint", True), default=True)
    resume_log_append = resume_enabled and (paths.run_root / "reports" / "checkpoint_progress.json").exists()

    logger = EventLogger(
        run_id=run_id,
        log_path=paths.run_root / "logs" / "events.jsonl",
        level=str(cfg.get("logging", {}).get("level", "INFO")),
        append=resume_log_append,
    )
    try:
        if not paths.input_jsonl.exists():
            raise FileNotFoundError(f"Input JSONL not found: {paths.input_jsonl}")

        expected_roles = parse_expected_roles(cfg)
        scoring_cfg = resolve_heuristic_scoring(cfg)
        allowed_splits = set(parse_allowed_splits(cfg))
        allowed_languages = set(parse_allowed_languages(cfg))
        all_group_rows = load_input_group_rows(paths.input_jsonl)
        filtered_rows = [
            row
            for row in all_group_rows
            if str(row.get("split") or "") in allowed_splits
            and normalize_prompt_language_key(str(row.get("language") or "en")) in allowed_languages
        ]
        max_samples = cfg.get("runtime", {}).get("max_samples")
        if isinstance(max_samples, int) and max_samples > 0:
            filtered_rows = filtered_rows[:max_samples]

        processing_buckets = build_processing_buckets(filtered_rows)
        bucket_counts = {f"{language}_{split}": len(rows) for language, split, rows in processing_buckets}
        group_decisions: List[Dict[str, Any]] = []
        rectified_rows: List[Dict[str, Any]] = []
        final_rows: List[Dict[str, Any]] = []
        completed_groups = 0
        prepared_groups = 0
        current_bucket_key = ""
        last_completed_group_id = ""
        if resume_enabled and not cli_args.dry_run:
            resume_state = load_resume_state(paths=paths, filtered_rows=filtered_rows, logger=logger)
            if resume_state is not None:
                group_decisions = list(resume_state.group_decisions)
                rectified_rows = list(resume_state.rectified_rows)
                final_rows = list(resume_state.final_rows)
                completed_groups = int(resume_state.completed_groups)
                prepared_groups = int(resume_state.prepared_groups)
                current_bucket_key = str(resume_state.current_bucket or "")
                last_completed_group_id = str(resume_state.last_completed_group_id or "")
        resumable_buckets = slice_processing_buckets_for_resume(processing_buckets, completed_groups)
        remaining_bucket_counts = {f"{language}_{split}": len(rows) for _, language, split, rows in resumable_buckets}
        remaining_groups = max(0, len(filtered_rows) - completed_groups)

        logger.event(
            "bootstrap",
            "run_started",
            config_path=str(paths.config_path),
            input_jsonl=str(paths.input_jsonl),
            output_root=str(paths.run_root),
            auditor_model_path=str(paths.auditor_model_path),
            regen_model_path=str(paths.regen_model_path),
            groups=len(filtered_rows),
            expected_roles=expected_roles,
            allowed_languages=sorted(allowed_languages),
            heuristic_scoring=scoring_cfg,
            resume_enabled=resume_enabled,
            resumed_completed_groups=completed_groups,
            remaining_groups=remaining_groups,
        )

        config_dir = paths.config_path.parent
        role_prompt_overrides: Dict[str, Dict[str, str]] = {}
        auditor_prompt_templates: Dict[str, str] = {}
        auditor_engine: Optional[AuditorEngine] = None
        regen_engine: Optional[RegenerationEngine] = None
        if remaining_groups > 0:
            role_prompt_overrides = load_role_prompt_overrides(cfg, workspace_root=workspace_root, config_dir=config_dir)
            auditor_prompt_templates = load_prompt_templates(paths.auditor_prompt_path, paths.auditor_prompt_map)
            if parse_bool(phase3_cfg.get("enable_auditor", True), default=True):
                auditor_engine = AuditorEngine(
                    model_path=paths.auditor_model_path,
                    inference_cfg=cfg.get("inference", {}).get("auditor", {}),
                    prompt_templates=auditor_prompt_templates,
                    logger=logger,
                    runtime_cfg=phase3_cfg,
                )
            if parse_bool(phase3_cfg.get("enable_regeneration", True), default=True):
                regen_engine = RegenerationEngine(
                    model_path=paths.regen_model_path,
                    inference_cfg=cfg.get("inference", {}).get("regen", {}),
                    role_prompt_overrides=role_prompt_overrides,
                    zh_lexicon_path=paths.zh_lexicon_path,
                    logger=logger,
                )

        progress_interval = int(max(1, phase3_cfg.get("progress_interval", cfg.get("runtime", {}).get("progress_interval", 200))))
        checkpoint_interval = int(max(progress_interval, phase3_cfg.get("checkpoint_interval", 200)))
        processing_chunk_size = int(max(1, phase3_cfg.get("processing_chunk_size", checkpoint_interval)))
        auditor_micro_batch_size = int(max(1, phase3_cfg.get("auditor_micro_batch_size", 4)))
        max_parse_retries = int(phase3_cfg.get("max_parse_retries", 1))
        shared_model_runtime = (
            auditor_engine is not None
            and regen_engine is not None
            and paths.auditor_model_path.resolve() == paths.regen_model_path.resolve()
        )
        logger.event(
            "bootstrap",
            "bucket_plan",
            bucket_count=len(processing_buckets),
            bucket_counts=bucket_counts,
            remaining_bucket_count=len(resumable_buckets),
            remaining_bucket_counts=remaining_bucket_counts,
            checkpoint_interval=checkpoint_interval,
            processing_chunk_size=processing_chunk_size,
            processing_order=list(bucket_counts.keys()),
            remaining_processing_order=list(remaining_bucket_counts.keys()),
        )

        def append_outputs(decision: Dict[str, Any], role_rows: List[Dict[str, Any]], final_row: Dict[str, Any]) -> None:
            nonlocal completed_groups, last_completed_group_id
            group_decisions.append(decision)
            rectified_rows.extend(role_rows)
            final_rows.append(final_row)
            completed_groups += 1
            last_completed_group_id = str(decision.get("group_id") or "")
            if completed_groups == 1 or completed_groups % progress_interval == 0:
                logger.event(
                    "phase3",
                    "progress",
                    index=completed_groups,
                    group_id=decision.get("group_id"),
                    lane=(decision.get("triage") or {}).get("lane"),
                )

        def write_checkpoint(reason: str, *, force: bool = False) -> None:
            if not force and completed_groups == 0:
                return
            phase4_rows = build_phase4_trace_pack_rows(group_decisions, rectified_rows, final_rows)
            if not cli_args.dry_run:
                write_candidate_outputs(paths.run_root, group_decisions, rectified_rows, final_rows, phase4_rows)
                write_candidate_outputs(paths.output_root / "latest", group_decisions, rectified_rows, final_rows, phase4_rows)
            checkpoint_payload = {
                "run_id": run_id,
                "updated_at": utc_now(),
                "reason": reason,
                "current_bucket": current_bucket_key,
                "completed_groups": completed_groups,
                "prepared_groups": prepared_groups,
                "total_groups": len(filtered_rows),
                "last_completed_group_id": last_completed_group_id,
                "counts": {
                    "group_decisions": len(group_decisions),
                    "rectified_roles": len(rectified_rows),
                    "final_tuples": len(final_rows),
                    "phase4_trace_pack": len(phase4_rows),
                },
            }
            write_json(paths.run_root / "reports" / "checkpoint_progress.json", checkpoint_payload)
            logger.event(
                "export",
                "checkpoint_written",
                reason=reason,
                bucket_key=current_bucket_key,
                completed_groups=completed_groups,
                prepared_groups=prepared_groups,
                last_completed_group_id=last_completed_group_id,
                group_decisions=len(group_decisions),
                rectified=len(rectified_rows),
                final_tuples=len(final_rows),
                phase4_trace_pack=len(phase4_rows),
            )

        def flush_pending_audits(pending_batch: List[PreparedGroupState]) -> None:
            if not pending_batch:
                return

            requests = [build_auditor_request(prepared, expected_roles, cfg=cfg) for prepared in pending_batch]
            if auditor_engine is not None:
                audit_results = auditor_engine.audit_batch(requests=requests, max_parse_retries=max_parse_retries)
            else:
                audit_results = [(None, empty_auditor_trace()) for _ in pending_batch]

            for prepared, (raw_payload, auditor_trace) in zip(pending_batch, audit_results):
                normalized_auditor: Optional[Dict[str, Any]] = None
                if raw_payload is not None:
                    normalized_auditor = normalize_auditor_payload(
                        payload=raw_payload,
                        language=prepared.working_row["language"],
                        focus_text=prepared.working_row["focus_text"],
                        fallback_cluster=prepared.fallback_cluster,
                    )
                decision, role_rows, final_row = finalize_group_state(
                    prepared=prepared,
                    expected_roles=expected_roles,
                    normalized_auditor=normalized_auditor,
                    auditor_trace=auditor_trace,
                    auditor_used=True,
                    cfg=cfg,
                )
                append_outputs(decision, role_rows, final_row)

        for bucket_index, bucket_language, bucket_split, bucket_rows in resumable_buckets:
            current_bucket_key = f"{bucket_language}_{bucket_split}"
            logger.event(
                "phase3",
                "bucket_started",
                bucket_index=bucket_index,
                bucket_count=len(processing_buckets),
                bucket_key=current_bucket_key,
                language=bucket_language,
                split=bucket_split,
                groups=len(bucket_rows),
                completed_groups=completed_groups,
            )

            for chunk_index, chunk_rows in enumerate(iter_row_chunks(bucket_rows, processing_chunk_size), start=1):
                if shared_model_runtime and auditor_engine is not None:
                    auditor_engine.release()

                prepared_chunk: List[PreparedGroupState] = []
                for group_row in chunk_rows:
                    logger.event(
                        "runtime",
                        "prepare_begin",
                        group_id=group_row.get("group_id"),
                        bucket_key=current_bucket_key,
                        chunk_index=chunk_index,
                        prepared_groups=prepared_groups,
                    )
                    prepared = prepare_group_state(
                        group_row=group_row,
                        expected_roles=expected_roles,
                        cfg=cfg,
                        scoring_cfg=scoring_cfg,
                        regen_engine=regen_engine,
                    )
                    logger.event(
                        "runtime",
                        "prepare_end",
                        group_id=group_row.get("group_id"),
                        bucket_key=current_bucket_key,
                        chunk_index=chunk_index,
                        lane=(prepared.signals or {}).get("lane"),
                        defective_roles=(prepared.signals or {}).get("defective_roles"),
                    )
                    prepared_chunk.append(prepared)
                    prepared_groups += 1
                    if prepared_groups == 1 or prepared_groups % progress_interval == 0:
                        logger.event(
                            "phase3",
                            "preprocess_progress",
                            index=prepared_groups,
                            group_id=group_row.get("group_id"),
                            bucket_key=current_bucket_key,
                            chunk_index=chunk_index,
                        )

                if shared_model_runtime and regen_engine is not None:
                    regen_engine.release()

                pending_audits: List[PreparedGroupState] = []
                for prepared in prepared_chunk:
                    if auditor_engine is not None and should_invoke_auditor(prepared.signals, cfg):
                        pending_audits.append(prepared)
                        if len(pending_audits) >= auditor_micro_batch_size:
                            flush_pending_audits(pending_audits)
                            pending_audits = []
                        if completed_groups > 0 and completed_groups % checkpoint_interval == 0:
                            write_checkpoint("interval")
                        continue

                    if pending_audits:
                        flush_pending_audits(pending_audits)
                        pending_audits = []
                        if completed_groups > 0 and completed_groups % checkpoint_interval == 0:
                            write_checkpoint("interval")

                    decision, role_rows, final_row = finalize_group_state(
                        prepared=prepared,
                        expected_roles=expected_roles,
                        normalized_auditor=None,
                        auditor_trace=empty_auditor_trace(),
                        auditor_used=False,
                        cfg=cfg,
                    )
                    append_outputs(decision, role_rows, final_row)
                    if completed_groups > 0 and completed_groups % checkpoint_interval == 0:
                        write_checkpoint("interval")

                flush_pending_audits(pending_audits)
                write_checkpoint("chunk_completed", force=True)
                logger.event(
                    "phase3",
                    "chunk_completed",
                    bucket_key=current_bucket_key,
                    chunk_index=chunk_index,
                    chunk_groups=len(chunk_rows),
                    completed_groups=completed_groups,
                    prepared_groups=prepared_groups,
                )

            write_checkpoint("bucket_completed", force=True)
            logger.event(
                "phase3",
                "bucket_completed",
                bucket_index=bucket_index,
                bucket_count=len(processing_buckets),
                bucket_key=current_bucket_key,
                completed_groups=completed_groups,
                prepared_groups=prepared_groups,
            )

        if regen_engine is not None:
            regen_engine.release()

        if auditor_engine is not None:
            auditor_engine.release()

        quality = quality_gates(group_decisions, final_rows)
        write_json(paths.run_root / "reports" / "quality_gates.json", quality)
        logger.event("quality", "quality_report_written", **quality)

        strict_mode = parse_bool(cfg.get("quality", {}).get("strict", False), default=False)
        if strict_mode and not quality.get("passed", False):
            raise RuntimeError(f"Phase3 quality gate failed: {quality}")

        summary = build_summary(group_decisions, final_rows)
        write_json(paths.run_root / "reports" / "summary.json", summary)
        write_json(paths.run_root / "reports" / "heuristic_scoring.json", scoring_cfg)
        logger.event("report", "summary_written", **summary)

        phase4_rows = build_phase4_trace_pack_rows(group_decisions, rectified_rows, final_rows)

        if not cli_args.dry_run:
            write_candidate_outputs(paths.run_root, group_decisions, rectified_rows, final_rows, phase4_rows)
            write_candidate_outputs(paths.output_root / "latest", group_decisions, rectified_rows, final_rows, phase4_rows)
            logger.event("export", "candidates_written", group_decisions=len(group_decisions), rectified=len(rectified_rows), final_tuples=len(final_rows), phase4_trace_pack=len(phase4_rows))

        run_summary = {
            "run_id": run_id,
            "created_at": utc_now(),
            "config_path": str(paths.config_path),
            "input_jsonl": str(paths.input_jsonl),
            "output_root": str(paths.run_root),
            "allowed_languages": sorted(allowed_languages),
            "counts": {
                "input_groups": len(filtered_rows),
                "output_group_decisions": len(group_decisions),
                "output_rectified_roles": len(rectified_rows),
                "output_final_tuples": len(final_rows),
                "output_emitted_tuples": sum(
                    len(row.get("final_tuples")) if isinstance(row.get("final_tuples"), list) and row.get("final_tuples") else 1
                    for row in final_rows
                ),
                "output_phase4_trace_pack": len(phase4_rows),
            },
            "quality": quality,
            "summary": summary,
            "heuristic_scoring": scoring_cfg,
        }
        write_json(paths.run_root / "run_summary.json", run_summary)
        logger.event("bootstrap", "run_completed", summary_path=str(paths.run_root / "run_summary.json"))
        return run_summary
    finally:
        logger.close()


def main() -> None:
    run_phase3()


if __name__ == "__main__":
    main()
