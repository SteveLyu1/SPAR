from __future__ import annotations

import argparse
import html
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from SPAR.experiments.api_adapter import (
    ApiTextGenerationClient,
    derive_api_output_namespace,
    render_messages_as_text,
    resolve_api_backend_config,
    rewrite_output_namespace_path_text,
)

try:
    import fcntl
except ImportError:
    fcntl = None


CJK_RE = re.compile(r"[\u4e00-\u9fff]")
SENTENCE_RE = re.compile(r"[^.!?;\u3002\uff01\uff1f\uff1b\n]+[.!?;\u3002\uff01\uff1f\uff1b\n]*", re.MULTILINE)
TARGET_SPLIT_RE = re.compile(r"[,，/|;；、]+|\band\b|\bor\b|\bvs\b|\s*和\s*|\s*与\s*|\s*及\s*|\s*以及\s*", flags=re.IGNORECASE)

LOG_LEVEL_ORDER = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}

LANGUAGE_SPLIT_ORDER: Tuple[Tuple[str, str], ...] = (
    ("zh", "train"),
    ("zh", "test"),
    ("en", "train"),
    ("en", "test"),
)

LANGUAGE_SPLIT_FILENAME_MAP: Dict[str, str] = {
    "zh_train": "cleaned_zh_train.jsonl",
    "zh_test": "cleaned_zh_test.jsonl",
    "en_train": "cleaned_en_train.jsonl",
    "en_test": "cleaned_en_test.jsonl",
}

PHASE1_OUTPUT_FILENAME_MAP: Dict[str, str] = {
    "zh_train": "phase1_zh_train.jsonl",
    "zh_test": "phase1_zh_test.jsonl",
    "en_train": "phase1_en_train.jsonl",
    "en_test": "phase1_en_test.jsonl",
}

PHASE1_OUTPUT_LOCK_FILENAME = "phase1_output.lock"


@dataclass
class Paths:
    workspace_root: Path
    config_path: Path
    input_jsonl: Path
    output_root: Path
    run_root: Path
    model_root: Path
    segmenter_model_path: Path
    segmenter_prompt_path: Optional[Path]
    segmenter_prompt_zh_path: Optional[Path]
    segmenter_prompt_en_path: Optional[Path]


@dataclass
class SentenceChunk:
    sentence_id: int
    text: str
    start: int
    end: int


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_sha1(*parts: Any) -> str:
    joined = "||".join(str(p) for p in parts)
    return hashlib.sha1(joined.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_vllm_dtype(dtype_name: str) -> str:
    normalized = str(dtype_name or "auto").strip().lower()
    if normalized in {"float16", "half", "fp16"}:
        return "float16"
    if normalized in {"bfloat16", "bf16"}:
        return "bfloat16"
    if normalized in {"float32", "fp32"}:
        return "float32"
    return "auto"


def ensure_tokenizer_vllm_compat(tokenizer: Any) -> Any:
    """Patch tokenizer attributes expected by vLLM but missing in some fast tokenizer backends."""
    if not hasattr(tokenizer, "all_special_tokens_extended"):
        try:
            tokenizer.all_special_tokens_extended = list(getattr(tokenizer, "all_special_tokens", []))
        except Exception:
            tokenizer.all_special_tokens_extended = []
    return tokenizer


def patch_transformers_tokenizer_classes_for_vllm() -> None:
    """Patch tokenizer classes so vLLM-created tokenizer instances stay compatible."""
    candidates: List[Any] = []
    try:
        from transformers.tokenization_utils_tokenizers import TokenizersBackend

        candidates.append(TokenizersBackend)
    except Exception:
        pass
    try:
        from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer

        candidates.append(Qwen2Tokenizer)
    except Exception:
        pass
    try:
        from transformers.models.qwen2.tokenization_qwen2_fast import Qwen2TokenizerFast

        candidates.append(Qwen2TokenizerFast)
    except Exception:
        pass

    for tokenizer_cls in candidates:
        if tokenizer_cls is None or hasattr(tokenizer_cls, "all_special_tokens_extended"):
            continue

        def _all_special_tokens_extended(self: Any) -> List[str]:
            return list(getattr(self, "all_special_tokens", []))

        setattr(tokenizer_cls, "all_special_tokens_extended", property(_all_special_tokens_extended))


def acquire_output_root_lock(
    output_root: Path,
    run_id: str,
    timeout_seconds: float = 900.0,
    poll_interval_seconds: float = 0.2,
) -> Dict[str, Any]:
    if fcntl is None:
        raise RuntimeError("Output write lock requires fcntl support on this platform.")

    ensure_dir(output_root)
    lock_path = output_root / PHASE1_OUTPUT_LOCK_FILENAME
    lock_file = lock_path.open("a+", encoding="utf-8")

    started = time.monotonic()
    safe_timeout = max(0.0, float(timeout_seconds))
    safe_poll = max(0.05, float(poll_interval_seconds))

    while True:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            waited = time.monotonic() - started
            if waited > safe_timeout:
                lock_file.close()
                raise RuntimeError(
                    f"Could not acquire output write lock within {safe_timeout:.1f}s: {lock_path}"
                )
            time.sleep(safe_poll)

    waited_seconds = round(time.monotonic() - started, 3)
    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(
        json.dumps(
            {
                "run_id": run_id,
                "pid": os.getpid(),
                "acquired_at": utc_now(),
                "waited_seconds": waited_seconds,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    lock_file.flush()

    return {
        "path": lock_path,
        "fh": lock_file,
        "timeout_seconds": safe_timeout,
        "waited_seconds": waited_seconds,
    }


def release_output_root_lock(lock_handle: Optional[Dict[str, Any]]) -> None:
    if not lock_handle:
        return
    lock_file = lock_handle.get("fh")
    if lock_file is None:
        return
    try:
        lock_file.seek(0)
        lock_file.truncate(0)
        lock_file.write(
            json.dumps(
                {
                    "released_at": utc_now(),
                    "pid": os.getpid(),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        lock_file.flush()
    except Exception:
        pass

    if fcntl is not None:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
    try:
        lock_file.close()
    except Exception:
        pass


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


def detect_language(text: Optional[str]) -> str:
    if text and CJK_RE.search(text):
        return "zh"
    return "en"


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    return default


def clamp(value: float, low: float = 0.01, high: float = 0.99) -> float:
    return max(low, min(high, value))


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


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_text_for_match(text: Optional[str], language: str) -> str:
    raw = normalize_string(text) or ""
    cleaned = html.unescape(raw)
    cleaned = (
        cleaned.replace("‘", "'")
        .replace("’", "'")
        .replace("`", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("–", "-")
        .replace("—", "-")
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if language == "en":
        cleaned = cleaned.lower()
    return cleaned


def compact_text_for_alignment(text: Optional[str], language: str) -> str:
    normalized = normalize_text_for_match(text, language=language)
    chars: List[str] = []
    for ch in normalized:
        if ch.isalnum():
            chars.append(ch)
    return "".join(chars)


def build_compact_source_items(source_text: str, language: str) -> List[Tuple[int, str]]:
    source_items: List[Tuple[int, str]] = []
    for idx, ch in enumerate(source_text):
        normalized_char = normalize_text_for_match(ch, language=language)
        for sub in normalized_char:
            if sub.isalnum():
                source_items.append((idx, sub))
    return source_items


def extract_compact_substring_from_source(group_text: Optional[str], source_text: Optional[str], language: str) -> Optional[str]:
    phrase = normalize_string(group_text)
    source = normalize_string(source_text)
    if phrase is None or source is None:
        return None

    phrase_compact = compact_text_for_alignment(phrase, language=language)
    if not phrase_compact:
        return None

    source_items = build_compact_source_items(source, language=language)
    source_compact = "".join(ch for _, ch in source_items)
    pos = source_compact.find(phrase_compact)
    if pos < 0:
        return None

    start = source_items[pos][0]
    end = source_items[pos + len(phrase_compact) - 1][0] + 1
    recovered = source[start:end].strip()
    return recovered or None


def extract_approximate_source_span(
    group_text: Optional[str],
    source_text: Optional[str],
    language: str,
) -> Tuple[Optional[str], float, float]:
    phrase = normalize_string(group_text)
    source = normalize_string(source_text)
    if phrase is None or source is None:
        return None, 0.0, 0.0

    phrase_compact = compact_text_for_alignment(phrase, language=language)
    if not phrase_compact:
        return None, 0.0, 0.0

    source_items = build_compact_source_items(source, language=language)
    if not source_items:
        return None, 0.0, 0.0

    source_compact = "".join(ch for _, ch in source_items)
    matcher = SequenceMatcher(None, phrase_compact, source_compact)
    blocks = [block for block in matcher.get_matching_blocks() if block.size > 0]
    if not blocks:
        return None, 0.0, matcher.ratio()

    first_block = next((block for block in blocks if block.size >= 2), blocks[0])
    last_block = next((block for block in reversed(blocks) if block.size >= 2), blocks[-1])
    start_pos = first_block.b
    end_pos = last_block.b + last_block.size - 1
    if start_pos < 0 or end_pos < start_pos or end_pos >= len(source_items):
        return None, 0.0, matcher.ratio()

    recovered = source[source_items[start_pos][0] : source_items[end_pos][0] + 1].strip()
    coverage = sum(block.size for block in blocks) / max(1, len(phrase_compact))
    ratio = matcher.ratio()
    if not recovered:
        return None, coverage, ratio
    return recovered, coverage, ratio


def longest_common_substring(text_a: str, text_b: str) -> str:
    if not text_a or not text_b:
        return ""
    matcher = SequenceMatcher(None, text_a, text_b)
    match = matcher.find_longest_match(0, len(text_a), 0, len(text_b))
    if match.size <= 0:
        return ""
    return text_b[match.b : match.b + match.size]


def recover_group_text_from_compact_overlap(
    group_text: Optional[str],
    source_text: Optional[str],
    language: str,
) -> Optional[str]:
    phrase = normalize_string(group_text)
    source = normalize_string(source_text)
    if phrase is None or source is None:
        return None

    phrase_compact = compact_text_for_alignment(phrase, language=language)
    if not phrase_compact:
        return None

    source_items = build_compact_source_items(source, language=language)
    if not source_items:
        return None

    source_compact = "".join(ch for _, ch in source_items)
    overlap = longest_common_substring(phrase_compact, source_compact)
    if len(overlap) < 4:
        return None

    pos = source_compact.find(overlap)
    if pos < 0:
        return None

    start = source_items[pos][0]
    end = source_items[pos + len(overlap) - 1][0] + 1
    recovered = source[start:end].strip()
    if not recovered:
        return None

    overlap_ratio = len(overlap) / max(1, len(phrase_compact))
    if overlap_ratio >= 0.65:
        return recovered
    if language != "en" and len(overlap) >= 6:
        return recovered
    return None


def recover_group_text_from_source(
    group_text: Optional[str],
    source_text: Optional[str],
    sentences: Sequence["SentenceChunk"],
    language: str,
) -> Optional[str]:
    phrase = normalize_string(group_text)
    source = normalize_string(source_text)
    if phrase is None or source is None:
        return phrase
    if is_group_text_in_source(phrase, source, language=language):
        return phrase

    recovered = extract_compact_substring_from_source(phrase, source, language=language)
    if recovered:
        return recovered

    single_sentence_source = len([s for s in sentences if s.text.strip()]) <= 1
    recovered = recover_group_text_from_compact_overlap(phrase, source, language=language)
    if recovered and is_group_text_in_source(recovered, source, language=language):
        return recovered

    recovered, coverage, ratio = extract_approximate_source_span(phrase, source, language=language)
    if recovered and is_group_text_in_source(recovered, source, language=language):
        if coverage >= 0.72 or ratio >= 0.68:
            return recovered
        if single_sentence_source and (coverage >= 0.46 or ratio >= 0.42):
            return recovered

    phrase_compact = compact_text_for_alignment(phrase, language=language)
    if not phrase_compact:
        return phrase

    sentence_texts = [s.text.strip() for s in sentences if s.text.strip()]
    candidates: List[str] = []
    seen: Set[str] = set()
    for text in sentence_texts + [source]:
        if text and text not in seen:
            seen.add(text)
            candidates.append(text)

    best_candidate: Optional[str] = None
    best_coverage = 0.0
    best_ratio = 0.0
    for candidate in candidates:
        candidate_compact = compact_text_for_alignment(candidate, language=language)
        if not candidate_compact:
            continue
        matcher = SequenceMatcher(None, phrase_compact, candidate_compact)
        coverage = sum(block.size for block in matcher.get_matching_blocks()) / max(1, len(phrase_compact))
        ratio = matcher.ratio()
        if coverage > best_coverage or (abs(coverage - best_coverage) < 1e-9 and ratio > best_ratio):
            best_candidate = candidate
            best_coverage = coverage
            best_ratio = ratio

    if best_candidate is None:
        return phrase

    if best_coverage >= 0.88 or best_ratio >= 0.82:
        return best_candidate
    if single_sentence_source and best_coverage >= 0.60 and best_ratio >= 0.48:
        return best_candidate
    return phrase


def is_group_text_in_source(group_text: Optional[str], source_text: Optional[str], language: str) -> bool:
    g = normalize_text_for_match(group_text, language=language)
    s = normalize_text_for_match(source_text, language=language)
    if not g or not s:
        return False
    if g in s:
        return True

    if language == "en":
        g_compact = re.sub(r"[^a-z0-9]+", "", g)
        s_compact = re.sub(r"[^a-z0-9]+", "", s)
        return bool(g_compact) and g_compact in s_compact

    g_compact = re.sub(r"\s+", "", g)
    s_compact = re.sub(r"\s+", "", s)
    return bool(g_compact) and g_compact in s_compact


def estimate_eta_iso(start_iso: str, processed: int, total: int) -> Optional[str]:
    if processed <= 0 or total <= processed:
        return None
    try:
        started_at = datetime.fromisoformat(start_iso)
    except Exception:
        return None
    now = datetime.now(timezone.utc)
    elapsed = (now - started_at).total_seconds()
    if elapsed <= 0:
        return None
    remaining = total - processed
    eta_seconds = elapsed * (remaining / processed)
    eta = now + timedelta(seconds=max(0.0, eta_seconds))
    return eta.isoformat()


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
    parser = argparse.ArgumentParser(description="SPAR-Hate Phase 1 Segment pipeline")
    default_config = str(Path(__file__).resolve().parents[1] / "configs" / "phase1_en_api.json")
    parser.add_argument("--config", default=default_config, help=f"Config path. Default: {default_config}")
    parser.add_argument("--run-id", default=None, help="Optional run id")
    parser.add_argument("--mode", choices=["heuristic", "qwen_local", "vllm_local", "api", "stub"], default=None)
    parser.add_argument("--api-profile", default=None, help="Override inference.api.profile, e.g. gpt or deepseek")
    parser.add_argument("--input-jsonl", default=None, help="Override phase0 unified input JSONL")
    parser.add_argument("--output-root", default=None, help="Override output root")
    parser.add_argument("--model-path", default=None, help="Override local segmenter model path")
    parser.add_argument("--prompt-path", default=None, help="Override segmenter prompt template path")
    parser.add_argument("--prompt-path-zh", default=None, help="Override zh segmenter prompt template path")
    parser.add_argument("--prompt-path-en", default=None, help="Override en segmenter prompt template path")
    parser.add_argument("--max-samples", type=int, default=None, help="Process at most N samples")
    parser.add_argument("--max-samples-per-job", type=int, default=None, help="Process at most N samples for each language/split job")
    parser.add_argument("--splits", default=None, help="Comma-separated splits, e.g. train,dev,test")
    parser.add_argument("--batch-size", type=int, default=None, help="Override inference batch size for qwen_local")
    parser.add_argument("--candidate-output-profile", choices=["full", "compact"], default=None, help="Candidate export schema profile")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Compatibility option (unused in heuristic backend)")
    parser.add_argument("--temperature", type=float, default=None, help="Compatibility option (unused in heuristic backend)")
    parser.add_argument("--top-p", type=float, default=None, help="Compatibility option (unused in heuristic backend)")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
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
            import yaml
        except ImportError as exc:
            raise RuntimeError("PyYAML is required for YAML config. Use JSON config or install pyyaml.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"Expected mapping config in {config_path}, got {type(loaded)}")
        return loaded
    raise ValueError(f"Unsupported config format: {config_path}")


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


def apply_cli_overrides(cfg: Dict[str, Any], args: argparse.Namespace) -> None:
    override_map = [
        (args.mode, ["segmenter", "mode"]),
        (args.api_profile, ["inference", "api_profile"]),
        (args.input_jsonl, ["paths", "input_jsonl"]),
        (args.output_root, ["paths", "output_root"]),
        (args.model_path, ["paths", "segmenter_model_path"]),
        (args.prompt_path, ["paths", "segmenter_prompt_path"]),
        (args.prompt_path_zh, ["paths", "segmenter_prompt_path_zh"]),
        (args.prompt_path_en, ["paths", "segmenter_prompt_path_en"]),
        (args.seed, ["seed"]),
        (args.log_level, ["logging", "level"]),
        (args.batch_size, ["inference", "batch_size"]),
        (args.candidate_output_profile, ["runtime", "candidate_output_profile"]),
        (args.max_samples_per_job, ["runtime", "max_samples_per_job"]),
        (args.max_new_tokens, ["inference", "max_new_tokens"]),
        (args.temperature, ["inference", "temperature"]),
        (args.top_p, ["inference", "top_p"]),
    ]
    for value, keys in override_map:
        if value is not None:
            set_nested(cfg, keys, value)

    mode_to_backend = {
        "qwen_local": "transformers",
        "vllm_local": "vllm",
        "api": "api",
    }
    if args.mode in mode_to_backend:
        set_nested(cfg, ["inference", "backend"], mode_to_backend[str(args.mode)])

    if args.splits:
        set_nested(cfg, ["runtime", "default_splits"], [x.strip() for x in args.splits.split(",") if x.strip()])
    if args.strict:
        set_nested(cfg, ["quality", "strict"], True)


def get_segmenter_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg.get("segmenter"), dict):
        return dict(cfg["segmenter"])
    return {}


def build_paths(args: argparse.Namespace, cfg: Dict[str, Any], workspace_root: Path) -> Paths:
    config_path = Path(args.config).resolve()
    config_dir = config_path.parent
    cfg_paths = cfg["paths"]
    inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
    api_profiles = inference_cfg.get("api_profiles") if isinstance(inference_cfg.get("api_profiles"), dict) else None
    segmenter_cfg = get_segmenter_cfg(cfg)
    mode = str(segmenter_cfg.get("mode", "heuristic")).strip().lower()

    input_jsonl = resolve_path(workspace_root, config_dir, str(cfg_paths["input_jsonl"]))
    output_root = resolve_path(workspace_root, config_dir, str(cfg_paths["output_root"]))
    model_root = resolve_path(workspace_root, config_dir, str(cfg_paths["model_root"]))

    model_raw = cfg_paths.get("segmenter_model_path") or str(model_root)
    prompt_raw = cfg_paths.get("segmenter_prompt_path")
    prompt_zh_raw = cfg_paths.get("segmenter_prompt_path_zh")
    prompt_en_raw = cfg_paths.get("segmenter_prompt_path_en")

    model_path = resolve_path(workspace_root, config_dir, str(model_raw))
    prompt_path = resolve_path(workspace_root, config_dir, str(prompt_raw)) if isinstance(prompt_raw, str) and prompt_raw else None
    prompt_zh_path = resolve_path(workspace_root, config_dir, str(prompt_zh_raw)) if isinstance(prompt_zh_raw, str) and prompt_zh_raw else None
    prompt_en_path = resolve_path(workspace_root, config_dir, str(prompt_en_raw)) if isinstance(prompt_en_raw, str) and prompt_en_raw else None

    if prompt_path is not None:
        prompt_parent = prompt_path.parent
        if prompt_zh_path is None:
            candidate_zh = prompt_parent / "phase1_segmenter_prompt_zh_fewshot.txt"
            if candidate_zh.exists():
                prompt_zh_path = candidate_zh
        if prompt_en_path is None:
            candidate_en = prompt_parent / "phase1_segmenter_prompt_en_fewshot.txt"
            if candidate_en.exists():
                prompt_en_path = candidate_en

    if mode == "api":
        namespace = derive_api_output_namespace(inference_cfg, api_profiles=api_profiles)
        output_root = resolve_path(
            workspace_root,
            config_dir,
            str(rewrite_output_namespace_path_text(str(cfg_paths["output_root"]), namespace)),
        )

    run_id = args.run_id or f"phase1_segmenter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_root = output_root

    return Paths(
        workspace_root=workspace_root,
        config_path=config_path,
        input_jsonl=input_jsonl,
        output_root=output_root,
        run_root=run_root,
        model_root=model_root,
        segmenter_model_path=model_path,
        segmenter_prompt_path=prompt_path,
        segmenter_prompt_zh_path=prompt_zh_path,
        segmenter_prompt_en_path=prompt_en_path,
    )


def parse_allowed_splits(cfg: Dict[str, Any]) -> List[str]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    splits = runtime_cfg.get("default_splits", ["train", "test"])
    if not isinstance(splits, list):
        return ["train", "test"]
    cleaned = [str(x).strip() for x in splits if str(x).strip()]
    return cleaned or ["train", "test"]


def truncate_text(value: Optional[str], max_chars: int = 4000) -> Optional[str]:
    if value is None:
        return None
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 3] + "..."


def safe_sentence_ids(value: Any) -> List[int]:
    ids: List[int] = []
    if isinstance(value, list):
        for item in value:
            if isinstance(item, int) and item >= 0:
                ids.append(item)
            elif isinstance(item, str) and item.isdigit():
                ids.append(int(item))
    elif isinstance(value, int) and value >= 0:
        ids.append(value)
    elif isinstance(value, str) and value.isdigit():
        ids.append(int(value))
    return sorted(set(ids))


def _extract_balanced_json_arrays(text: str) -> List[str]:
    arrays: List[str] = []
    if not text:
        return arrays

    n = len(text)
    for start_idx, ch in enumerate(text):
        if ch != "[":
            continue

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start_idx, n):
            cur = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue

            if cur == '"':
                in_string = True
            elif cur == "[":
                depth += 1
            elif cur == "]":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1].strip()
                    if candidate:
                        arrays.append(candidate)
                    break

    return dedupe_preserve_order(arrays, case_insensitive=False)


def _extract_balanced_json_objects(text: str) -> List[str]:
    objects: List[str] = []
    if not text:
        return objects

    n = len(text)
    for start_idx, ch in enumerate(text):
        if ch != "{":
            continue

        depth = 0
        in_string = False
        escaped = False
        for idx in range(start_idx, n):
            cur = text[idx]
            if in_string:
                if escaped:
                    escaped = False
                elif cur == "\\":
                    escaped = True
                elif cur == '"':
                    in_string = False
                continue

            if cur == '"':
                in_string = True
            elif cur == "{":
                depth += 1
            elif cur == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start_idx : idx + 1].strip()
                    if candidate:
                        objects.append(candidate)
                    break

    return dedupe_preserve_order(objects, case_insensitive=False)


def _decode_json_string_fragment(raw_value: str) -> str:
    value = raw_value.strip()
    if not value:
        return ""
    try:
        return json.loads('"' + value + '"')
    except Exception:
        try:
            return bytes(value, "utf-8").decode("unicode_escape")
        except Exception:
            return value


def _extract_string_field_from_object(obj_text: str, key: str) -> Optional[str]:
    patterns = [
        rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"',
        rf"'{re.escape(key)}'\s*:\s*'((?:\\.|[^'\\])*)'",
    ]
    for pattern in patterns:
        m = re.search(pattern, obj_text, flags=re.DOTALL)
        if m:
            parsed = normalize_string(_decode_json_string_fragment(m.group(1)))
            if parsed is not None:
                return parsed

    bare_patterns = [
        rf'"{re.escape(key)}"\s*:\s*([^,\n}}]+)',
        rf"'{re.escape(key)}'\s*:\s*([^,\n}}]+)",
    ]
    for pattern in bare_patterns:
        m = re.search(pattern, obj_text, flags=re.DOTALL)
        if m:
            candidate = m.group(1).strip().strip('"').strip("'").strip()
            parsed = normalize_string(candidate)
            if parsed is not None:
                return parsed
    return None


def _extract_int_field_from_object(obj_text: str, key: str) -> Optional[int]:
    pattern = rf'["\']?{re.escape(key)}["\']?\s*:\s*([0-9]+)'
    m = re.search(pattern, obj_text)
    if not m:
        return None
    try:
        value = int(m.group(1))
    except (TypeError, ValueError):
        return None
    if value <= 0:
        return None
    return value


def recover_groups_from_malformed_json(text: str) -> List[Dict[str, Any]]:
    if not text or not text.strip():
        return []

    candidates = _extract_balanced_json_objects(text)
    if not candidates:
        candidates = [text.strip()]

    recovered: List[Dict[str, Any]] = []
    for idx, obj in enumerate(candidates, start=1):
        group_text = _extract_string_field_from_object(obj, "group_text")
        canonical_target = _extract_string_field_from_object(obj, "canonical_target")
        if group_text is None or canonical_target is None:
            continue

        local_group_id = _extract_int_field_from_object(obj, "local_group_id")
        if local_group_id is None:
            local_group_id = idx

        recovered.append(
            {
                "local_group_id": int(local_group_id),
                "group_text": group_text,
                "canonical_target": canonical_target,
            }
        )

    if not recovered:
        return []

    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for item in recovered:
        signature = (str(item.get("group_text") or ""), str(item.get("canonical_target") or ""))
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(item)

    for new_idx, item in enumerate(deduped, start=1):
        item["local_group_id"] = new_idx
    return deduped


def _json_repair_candidates(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []

    candidates: List[str] = [stripped]
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    candidates.extend(block.strip() for block in code_blocks if block.strip())
    candidates.extend(_extract_balanced_json_arrays(stripped))

    for block in code_blocks:
        block = block.strip()
        if block:
            candidates.extend(_extract_balanced_json_arrays(block))

    repaired: List[str] = []
    for item in candidates:
        fixed = item
        fixed = fixed.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
        fixed = re.sub(r",\s*([\]}])", r"\1", fixed)
        repaired.append(fixed)
    return dedupe_preserve_order(repaired, case_insensitive=False)


def extract_json_candidate(text: str) -> Any:
    if not text or not text.strip():
        raise ValueError("Empty model output")

    last_error: Optional[Exception] = None
    decoder = json.JSONDecoder()
    for candidate in _json_repair_candidates(text):
        try:
            stripped = candidate.lstrip()
            first_bracket_positions = [idx for idx in (stripped.find("["), stripped.find("{")) if idx >= 0]
            if not first_bracket_positions:
                continue

            first_idx = min(first_bracket_positions)
            prefix = stripped[:first_idx].strip().lower()
            if prefix:


                suspicious_markers = (
                    "# role",
                    "# task",
                    "few-shot",
                    "input package",
                    "original text",
                    "sentence list",
                    "sample_id",
                    "{{",
                )
                if any(marker in prefix for marker in suspicious_markers):
                    continue
                if len(prefix) > 120:
                    continue

            payload = stripped[first_idx:]
            parsed, end = decoder.raw_decode(payload)
            trailing = payload[end:].strip()
            if trailing and not trailing.startswith("```"):
                continue
            if isinstance(parsed, list):
                return parsed
            last_error = ValueError("Model output root must be a JSON array")
        except Exception as exc:
            last_error = exc

    recovered = recover_groups_from_malformed_json(text)
    if recovered:
        return recovered

    raise ValueError(f"No parseable JSON in model output: {last_error}")


def enforce_strict_model_group_schema(raw_groups: Sequence[Any]) -> None:
    allowed_keys = {"local_group_id", "group_text", "canonical_target"}
    for idx, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            raise ValueError(f"Group item at index {idx} is not an object")
        keys = set(item.keys())
        if "group_text" not in keys or "canonical_target" not in keys:
            raise ValueError(f"Group item at index {idx} misses required keys group_text/canonical_target")
        extra_keys = keys - allowed_keys
        if extra_keys:
            raise ValueError(f"Group item at index {idx} contains unsupported keys: {sorted(extra_keys)}")


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


def load_segmenter_prompt_templates(paths: Paths) -> Dict[str, str]:
    fallback_zh = (
        "# Role\n"
        "You are the SPAR-Hate Segment Agent specializing in Chinese text. \n"
        "Your ONLY output is a strict JSON array.\n\n"
        "# Rules\n"
        "1. 按目标和意图拆分：如果文本中包含对不同目标的攻击，或者包含截然不同的意图，必须拆分为不同组。\n"
        "2. group_text 必须从原文逐字复制，不可改写。\n"
        "3. canonical_target 必须是明确目标；若目标缺失或隐式，输出 IMPLICIT_TARGET。\n"
        "4. 只允许输出 local_group_id, group_text, canonical_target。\n\n"
        "# Current Task\n"
        "Original text:\n"
        "{{TEXT}}\n\n"
        "Output:\n["
    )
    fallback_en = (
        "# Role\n"
        "You are the SPAR-Hate Segment Agent specializing in English text. \n"
        "Your ONLY output is a strict JSON array.\n\n"
        "# Rules\n"
        "1. Divide by target and intent.\n"
        "2. group_text MUST be copied EXACTLY from original text.\n"
        "3. If target is absent/implicit, output IMPLICIT_TARGET.\n"
        "4. Only output local_group_id, group_text, canonical_target.\n\n"
        "# Current Task\n"
        "Original text:\n"
        "{{TEXT}}\n\n"
        "Output:\n["
    )

    def _read(path: Optional[Path], fallback: str) -> str:
        if path is not None and path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        return fallback

    return {
        "zh": _read(paths.segmenter_prompt_zh_path or paths.segmenter_prompt_path, fallback_zh),
        "en": _read(paths.segmenter_prompt_en_path or paths.segmenter_prompt_path, fallback_en),
    }


def resolve_hint_mode(segmenter_cfg: Dict[str, Any]) -> str:
    mode = str(segmenter_cfg.get("hint_mode", "train_only")).strip().lower()
    if mode in {"always", "all"}:
        return "always"
    if mode in {"none", "off", "disabled"}:
        return "none"
    return "train_only"


def render_segmenter_prompt(
    template: str,
    row: Dict[str, Any],
    retry_note: Optional[str],
    previous_raw_output: Optional[str],
) -> str:
    prompt = template.replace("{{TEXT}}", normalize_string(row.get("text")) or "")

    if retry_note:
        prompt += "\n\nRetry requirement: previous response was invalid JSON/schema. Error: " + retry_note
    if previous_raw_output:
        prompt += "\n\nPrevious invalid output (fix it, do not explain):\n" + truncate_text(previous_raw_output, max_chars=1800)
    return prompt


def render_json_repair_prompt(previous_raw_output: str, error_message: str) -> str:
    return (
        "Convert the following text into strict JSON array only.\n"
        "Keep only fields: local_group_id, group_text, canonical_target.\n"
        "Output must start with [ and end with ]. No markdown.\n"
        f"Parse error: {error_message}\n\n"
        "Raw output:\n"
        f"{previous_raw_output}"
    )


def split_sentences_with_offsets(text: str) -> List[SentenceChunk]:
    chunks: List[SentenceChunk] = []
    for match in SENTENCE_RE.finditer(text):
        raw = match.group(0)
        if not raw.strip():
            continue
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        start = match.start() + left
        end = match.start() + right
        sentence = text[start:end]
        if sentence:
            chunks.append(SentenceChunk(sentence_id=len(chunks), text=sentence, start=start, end=end))
    if not chunks and text.strip():
        stripped = text.strip()
        start = text.find(stripped)
        end = start + len(stripped)
        chunks.append(SentenceChunk(sentence_id=0, text=stripped, start=start, end=end))
    return chunks


def split_target_hints(target: Optional[str]) -> List[str]:
    raw = normalize_string(target)
    if raw is None:
        return []
    parts = [p.strip() for p in TARGET_SPLIT_RE.split(raw) if p and p.strip()]
    return dedupe_preserve_order(parts or [raw], case_insensitive=False)


def parse_row_tuples(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    tuples: List[Dict[str, Any]] = []
    gt = row.get("ground_truth")
    if isinstance(gt, list):
        for item in gt:
            if not isinstance(item, dict):
                continue
            tuples.append(
                {
                    "target": normalize_string(item.get("target")),
                    "argument": normalize_string(item.get("argument")),
                    "group": normalize_string(item.get("group")),
                    "harmful": item.get("hateful") if "hateful" in item else item.get("harmful"),
                }
            )

    if tuples:
        return tuples

    tuples.append(
        {
            "target": normalize_string(row.get("target")),
            "argument": normalize_string(row.get("argument")),
            "group": normalize_string(row.get("group")),
            "harmful": row.get("hateful") if "hateful" in row else row.get("harmful"),
        }
    )
    return tuples


def build_tuple_hints(row: Dict[str, Any], hint_mode: str) -> List[Dict[str, Any]]:
    if hint_mode == "none":
        return []
    if hint_mode == "train_only" and str(row.get("split") or "train") != "train":
        return []

    hints: List[Dict[str, Any]] = []
    for tup in parse_row_tuples(row):
        target = normalize_string(tup.get("target"))
        argument = normalize_string(tup.get("argument"))
        harmful = tup.get("harmful")
        if target is None and argument is None and harmful is None:
            continue
        hints.append(
            {
                "target": target,
                "argument": argument,
                "harmful": harmful,
                "group": normalize_string(tup.get("group")),
            }
        )
    return hints


def infer_sentence_ids_for_group_text(group_text: str, sentences: Sequence[SentenceChunk], language: str) -> List[int]:
    phrase = normalize_string(group_text)
    if phrase is None:
        return [0] if sentences else []

    matched: List[int] = []
    phrase_cmp = phrase.lower() if language == "en" else phrase
    for sentence in sentences:
        text_cmp = sentence.text.lower() if language == "en" else sentence.text
        if phrase_cmp in text_cmp or text_cmp in phrase_cmp:
            matched.append(sentence.sentence_id)
    if matched:
        return sorted(set(matched))
    return [0] if sentences else []


def normalize_model_groups(parsed: Any, row: Dict[str, Any], sentences: Sequence[SentenceChunk], language: str) -> List[Dict[str, Any]]:
    if not isinstance(parsed, list):
        raise ValueError("Model output must be a JSON array")
    raw_groups = parsed
    enforce_strict_model_group_schema(raw_groups)

    sentence_map = {s.sentence_id: s for s in sentences}
    normalized: List[Dict[str, Any]] = []

    for idx, item in enumerate(raw_groups):
        if not isinstance(item, dict):
            continue

        raw_lid = item.get("local_group_id")
        if raw_lid is None:
            raw_lid = item.get("group_id")
        try:
            local_group_id = int(raw_lid)
        except (TypeError, ValueError):
            local_group_id = idx + 1
        if local_group_id <= 0:
            local_group_id = idx + 1

        sentence_ids = safe_sentence_ids(item.get("sentence_ids"))
        if not sentence_ids:
            sentence_ids = infer_sentence_ids_for_group_text(normalize_string(item.get("group_text")) or "", sentences=sentences, language=language)

        group_text = recover_group_text_from_source(
            group_text=item.get("group_text"),
            source_text=row.get("text"),
            sentences=sentences,
            language=language,
        )
        if not sentence_ids:
            sentence_ids = infer_sentence_ids_for_group_text(group_text or "", sentences=sentences, language=language)
        sentence_ids = [sid for sid in sentence_ids if sid in sentence_map]
        if not sentence_ids and sentences:
            sentence_ids = [0]

        if group_text is None:
            group_text = " ".join(sentence_map[sid].text for sid in sentence_ids if sid in sentence_map).strip() or (normalize_string(row.get("text")) or "")

        canonical_target = normalize_string(item.get("canonical_target") or item.get("target")) or "IMPLICIT_TARGET"
        anchor_sentence_id = min(sentence_ids) if sentence_ids else 0
        normalized.append(
            {
                "local_group_id": local_group_id,
                "anchor_sentence_id": anchor_sentence_id,
                "sentence_ids": sentence_ids,
                "group_text": group_text,
                "canonical_target": canonical_target,
            }
        )

    if not normalized:
        text = normalize_string(row.get("text")) or ""
        return [
            {
                "local_group_id": 1,
                "anchor_sentence_id": 0,
                "sentence_ids": [0] if text else [],
                "group_text": text,
                "canonical_target": "IMPLICIT_TARGET",
            }
        ]

    deduped: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, Tuple[int, ...], str]] = set()
    for item in sorted(normalized, key=lambda x: (x["anchor_sentence_id"], x["local_group_id"])):
        tkey = item["canonical_target"].lower() if language == "en" else item["canonical_target"]
        gkey = item["group_text"].lower() if language == "en" else item["group_text"]
        sign = (tkey, tuple(item["sentence_ids"]), gkey)
        if sign in seen:
            continue
        seen.add(sign)
        deduped.append(item)

    source_text = normalize_string(row.get("text")) or ""
    filtered = [
        item
        for item in deduped
        if is_group_text_in_source(item.get("group_text"), source_text, language=language)
    ]
    if filtered:
        deduped = filtered
    elif source_text:
        deduped = [
            {
                "local_group_id": 1,
                "anchor_sentence_id": 0,
                "sentence_ids": [0] if sentences else [],
                "group_text": source_text,
                "canonical_target": "IMPLICIT_TARGET",
            }
        ]

    for idx, item in enumerate(deduped, start=1):
        item["local_group_id"] = idx
        if not item["sentence_ids"] and sentences:
            item["sentence_ids"] = [0]
        item["anchor_sentence_id"] = min(item["sentence_ids"]) if item["sentence_ids"] else 0

    return deduped


def validate_model_groups_against_source(
    groups: Sequence[Dict[str, Any]],
    row: Dict[str, Any],
    language: str,
    max_group_text_mismatch_per_sample: int,
) -> Optional[str]:
    source_text = normalize_string(row.get("text")) or ""
    mismatch_count = 0
    for item in groups:
        group_text = normalize_string(item.get("group_text"))
        if group_text is None:
            mismatch_count += 1
            continue
        if not is_group_text_in_source(group_text, source_text, language=language):
            mismatch_count += 1

    if mismatch_count > max(0, int(max_group_text_mismatch_per_sample)):
        sample_id = str(row.get("sample_id") or "")
        return (
            f"group_text/source mismatch for sample {sample_id}: "
            f"{mismatch_count} groups exceed threshold {max_group_text_mismatch_per_sample}"
        )
    return None


class LocalSegmenterLLM:
    def __init__(
        self,
        model_path: Path,
        inference_cfg: Dict[str, Any],
        prompt_templates: Dict[str, str],
        hint_mode: str,
        max_group_text_mismatch_per_sample: int = 0,
    ) -> None:
        self.model_path = model_path
        self.inference_cfg = inference_cfg
        self.prompt_templates = prompt_templates
        self.hint_mode = hint_mode
        self.max_group_text_mismatch_per_sample = max(0, int(max_group_text_mismatch_per_sample))
        self.enable_thinking = parse_bool(self.inference_cfg.get("enable_thinking", False), default=False)
        self.backend = str(self.inference_cfg.get("backend", "transformers")).strip().lower()
        self._ready = False
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._sampling_params_cls = None
        self._api_client: Optional[ApiTextGenerationClient] = None
        self.model_device: Optional[str] = None

    def initialize(self) -> None:
        if self._ready:
            return
        if self.backend == "api":
            self._initialize_api()
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Local model path does not exist: {self.model_path}")

        if parse_bool(self.inference_cfg.get("prefer_4090_order", True), default=True):
            prefer_4090_device_order(force_4090_order=parse_bool(self.inference_cfg.get("force_4090_order", False), default=False))

        if self.backend == "vllm":
            self._initialize_vllm()
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("qwen_local mode requires torch and transformers") from exc

        require_cuda = parse_bool(self.inference_cfg.get("require_cuda", True), default=True)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for qwen_local mode, but torch.cuda.is_available() is False")

        use_cuda = bool(torch.cuda.is_available())
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
            other_ratio = safe_float(self.inference_cfg.get("other_gpu_mem_ratio", 0.90), default=0.90)
            if torch.cuda.device_count() >= 2 and str(load_kwargs.get("device_map", "")).startswith("balanced"):
                min_multi_gpu_ratio = safe_float(self.inference_cfg.get("min_multi_gpu_mem_ratio", 0.85), default=0.85)
                other_ratio = max(other_ratio, min_multi_gpu_ratio)
            max_memory = build_max_memory_map(torch, preferred_ratio=preferred_ratio, other_ratio=other_ratio)
            if max_memory:
                load_kwargs["max_memory"] = max_memory
            load_kwargs["torch_dtype"] = dtype_value
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(str(self.model_path), **load_kwargs)
        model.eval()

        try:
            self.model_device = str(next(model.parameters()).device)
        except StopIteration:
            self.model_device = "unknown"

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._ready = True

    def _initialize_vllm(self) -> None:
        try:
            import torch
            from transformers import AutoTokenizer
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vllm backend requires vllm, torch, and transformers") from exc

        require_cuda = parse_bool(self.inference_cfg.get("require_cuda", True), default=True)
        if require_cuda and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for vllm backend, but torch.cuda.is_available() is False")

        patch_transformers_tokenizer_classes_for_vllm()
        tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), trust_remote_code=True)
        tokenizer = ensure_tokenizer_vllm_compat(tokenizer)
        if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"

        dtype_name = resolve_vllm_dtype(str(self.inference_cfg.get("dtype", "auto")))
        tensor_parallel_size = int(max(1, self.inference_cfg.get("tensor_parallel_size", 1)))
        gpu_memory_utilization = float(self.inference_cfg.get("gpu_memory_utilization", 0.9))
        max_model_len = int(max(1024, self.inference_cfg.get("max_model_len", 4096)))
        max_num_batched_tokens = int(max_model_len if self.inference_cfg.get("max_num_batched_tokens") is None else self.inference_cfg.get("max_num_batched_tokens"))
        llm_kwargs: Dict[str, Any] = {
            "model": str(self.model_path),
            "tokenizer": str(self.model_path),
            "dtype": dtype_name,
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "max_model_len": max_model_len,
            "max_num_batched_tokens": max_num_batched_tokens,
            "trust_remote_code": parse_bool(self.inference_cfg.get("trust_remote_code", True), default=True),
            "enforce_eager": parse_bool(self.inference_cfg.get("enforce_eager", False), default=False),
            "disable_custom_all_reduce": parse_bool(self.inference_cfg.get("disable_custom_all_reduce", False), default=False),
            "enable_chunked_prefill": parse_bool(self.inference_cfg.get("enable_chunked_prefill", True), default=True),
        }
        block_size = self.inference_cfg.get("block_size")
        if block_size is not None:
            llm_kwargs["block_size"] = int(block_size)
        max_num_seqs = self.inference_cfg.get("max_num_seqs")
        if max_num_seqs is not None:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)
        seed = self.inference_cfg.get("seed")
        if seed is not None:
            llm_kwargs["seed"] = int(seed)

        model = LLM(**llm_kwargs)
        self.model_device = f"vllm-tp{tensor_parallel_size}"
        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch
        self._sampling_params_cls = SamplingParams
        self._ready = True

    def _initialize_api(self) -> None:
        api_profiles = self.inference_cfg.get("api_profiles") if isinstance(self.inference_cfg.get("api_profiles"), dict) else None
        cfg = resolve_api_backend_config(self.inference_cfg, api_profiles=api_profiles)
        self._api_client = ApiTextGenerationClient(cfg)
        self._tokenizer = None
        self._model = self._api_client
        self._torch = None
        self._sampling_params_cls = None
        self.model_device = f"api:{cfg.provider}:{cfg.api_format}:{cfg.model}"
        self._ready = True

    def _messages_to_prompt(self, messages: Sequence[Dict[str, str]]) -> str:
        if self.backend == "api":
            return render_messages_as_text(messages)
        if self._tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        tokenizer = self._tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            kwargs: Dict[str, Any] = {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            if "Qwen3.5" in self.model_path.name:
                kwargs["enable_thinking"] = bool(self.enable_thinking)
            return tokenizer.apply_chat_template(messages, **kwargs)
        return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in messages) + "\nASSISTANT:"

    def _generate_batch_text(self, prompts: Sequence[str]) -> List[str]:
        if not self._ready or self._model is None:
            raise RuntimeError("LocalSegmenterLLM is not initialized")
        if not prompts:
            return []

        if self.backend == "api":
            raise RuntimeError("API backend requires message-based generation path")

        tokenizer = self._tokenizer
        if tokenizer is None:
            raise RuntimeError("Tokenizer is not initialized")
        if self.backend == "vllm":
            if self._sampling_params_cls is None:
                raise RuntimeError("vLLM SamplingParams is not initialized")
            max_input_tokens = int(self.inference_cfg.get("max_input_tokens", 2048))
            max_new_tokens = int(self.inference_cfg.get("max_new_tokens", 768))
            temperature = safe_float(self.inference_cfg.get("temperature", 0.2), default=0.2)
            top_p = safe_float(self.inference_cfg.get("top_p", 0.9), default=0.9)
            top_k = int(self.inference_cfg.get("top_k", 20) or 0)
            repetition_penalty = safe_float(self.inference_cfg.get("repetition_penalty", 1.0), default=1.0)
            sampling_kwargs: Dict[str, Any] = {
                "max_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "repetition_penalty": repetition_penalty,
                "skip_special_tokens": True,
            }
            if top_k > 0:
                sampling_kwargs["top_k"] = top_k
            if max_input_tokens > 0:
                sampling_kwargs["truncate_prompt_tokens"] = max_input_tokens
            try:
                sampling_params = self._sampling_params_cls(**sampling_kwargs)
            except TypeError as exc:
                if "truncate_prompt_tokens" not in str(exc):
                    raise
                sampling_kwargs.pop("truncate_prompt_tokens", None)
                sampling_params = self._sampling_params_cls(**sampling_kwargs)
            outputs = self._model.generate(list(prompts), sampling_params=sampling_params, use_tqdm=False)
            texts: List[str] = []
            for item in outputs:
                if not getattr(item, "outputs", None):
                    texts.append("")
                    continue
                texts.append(str(item.outputs[0].text))
            return texts

        if self._torch is None:
            raise RuntimeError("Torch backend state is not initialized")
        model = self._model
        torch = self._torch

        max_input_tokens = int(self.inference_cfg.get("max_input_tokens", 2048))
        tokenize_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
        if max_input_tokens > 0:
            tokenize_kwargs["truncation"] = True
            tokenize_kwargs["max_length"] = max_input_tokens
        inputs = tokenizer(list(prompts), **tokenize_kwargs)

        model_device = getattr(model, "device", None)
        if model_device is not None and getattr(model_device, "type", None) not in {None, "meta"}:
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

        max_new_tokens = int(self.inference_cfg.get("max_new_tokens", 768))
        temperature = safe_float(self.inference_cfg.get("temperature", 0.2), default=0.2)
        top_p = safe_float(self.inference_cfg.get("top_p", 0.9), default=0.9)
        do_sample = temperature > 0.0
        use_cache = parse_bool(self.inference_cfg.get("use_cache", True), default=True)

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else None,
                top_p=top_p if do_sample else None,
                use_cache=use_cache,
                eos_token_id=getattr(tokenizer, "eos_token_id", None),
                pad_token_id=getattr(tokenizer, "pad_token_id", getattr(tokenizer, "eos_token_id", None)),
            )

        prompt_padded_len = int(inputs["input_ids"].shape[1])
        outputs: List[str] = []
        for row_idx in range(output_ids.shape[0]):


            generated = output_ids[row_idx][prompt_padded_len:]
            outputs.append(tokenizer.decode(generated, skip_special_tokens=True))
        return outputs

    def generate_from_messages_batch(self, messages_batch: Sequence[Sequence[Dict[str, str]]]) -> List[str]:
        if self.backend == "api":
            if self._api_client is None:
                raise RuntimeError("API client is not initialized")
            return self._api_client.generate_from_messages_batch(
                messages_batch,
                max_new_tokens=int(self.inference_cfg.get("max_new_tokens", 768)),
                temperature=safe_float(self.inference_cfg.get("temperature", 0.2), default=0.2),
                top_p=safe_float(self.inference_cfg.get("top_p", 0.9), default=0.9),
                enable_thinking=self.enable_thinking,
            )
        prompts = [self._messages_to_prompt(messages) for messages in messages_batch]
        return self._generate_batch_text(prompts)

    def shutdown(self) -> None:
        api_client = self._api_client
        self._api_client = None
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._sampling_params_cls = None
        self._ready = False
        if api_client is not None:
            try:
                api_client.shutdown()
            except Exception:
                pass

    def generate_groups_batch(
        self,
        rows: Sequence[Dict[str, Any]],
        max_parse_retries: int,
        max_repair_retries: int,
        forced_language: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not rows:
            return []

        pending: List[int] = list(range(len(rows)))
        parse_errors: List[Optional[str]] = [None for _ in rows]
        parsed_groups: List[Optional[List[Dict[str, Any]]]] = [None for _ in rows]
        raw_outputs: List[List[str]] = [[] for _ in rows]

        sentences_cache: List[List[SentenceChunk]] = []
        language_cache: List[str] = []
        for row in rows:
            text = normalize_string(row.get("text")) or ""
            sentences = split_sentences_with_offsets(text)
            if not sentences and text:
                sentences = [SentenceChunk(sentence_id=0, text=text, start=0, end=len(text))]
            language = forced_language or str(row.get("language") or detect_language(text))
            sentences_cache.append(sentences)
            language_cache.append(language)

        for attempt in range(max(0, max_parse_retries) + 1):
            if not pending:
                break
            snapshot = list(pending)
            prompts: List[str] = []

            for idx in snapshot:
                language = language_cache[idx]
                template = self.prompt_templates.get(language, self.prompt_templates.get("en", ""))
                previous_raw = raw_outputs[idx][-1] if (attempt > 0 and raw_outputs[idx]) else None
                prompt = render_segmenter_prompt(
                    template=template,
                    row=rows[idx],
                    retry_note=parse_errors[idx] if attempt > 0 else None,
                    previous_raw_output=previous_raw,
                )
                messages = [
                    {"role": "system", "content": "Return strict JSON array only."},
                    {"role": "user", "content": prompt},
                ]
                prompts.append(self._messages_to_prompt(messages))

            outputs = self.generate_from_messages_batch(messages_batch=[
                [
                    {"role": "system", "content": "Return strict JSON array only."},
                    {"role": "user", "content": render_segmenter_prompt(
                        template=self.prompt_templates.get(language_cache[idx], self.prompt_templates.get("en", "")),
                        row=rows[idx],
                        retry_note=parse_errors[idx] if attempt > 0 else None,
                        previous_raw_output=raw_outputs[idx][-1] if (attempt > 0 and raw_outputs[idx]) else None,
                    )},
                ]
                for idx in snapshot
            ])
            if len(outputs) != len(snapshot):
                raise RuntimeError("Batch generation output size mismatch in phase1 parse pass")

            next_pending: List[int] = []
            for local_idx, global_idx in enumerate(snapshot):
                raw_text = outputs[local_idx]
                raw_outputs[global_idx].append(raw_text)
                try:
                    parsed = extract_json_candidate(raw_text)
                    normalized_groups = normalize_model_groups(
                        parsed=parsed,
                        row=rows[global_idx],
                        sentences=sentences_cache[global_idx],
                        language=language_cache[global_idx],
                    )
                    semantic_error = validate_model_groups_against_source(
                        groups=normalized_groups,
                        row=rows[global_idx],
                        language=language_cache[global_idx],
                        max_group_text_mismatch_per_sample=self.max_group_text_mismatch_per_sample,
                    )
                    if semantic_error:
                        raise ValueError(semantic_error)
                    parsed_groups[global_idx] = normalized_groups
                    parse_errors[global_idx] = None
                except Exception as exc:
                    parse_errors[global_idx] = str(exc)
                    next_pending.append(global_idx)
            pending = next_pending

        for _ in range(max(0, max_repair_retries)):
            if not pending:
                break
            snapshot = list(pending)
            prompts: List[str] = []
            for idx in snapshot:
                previous_raw = raw_outputs[idx][-1] if raw_outputs[idx] else ""
                repair_prompt = render_json_repair_prompt(previous_raw_output=previous_raw, error_message=parse_errors[idx] or "json parse failure")
                messages = [
                    {"role": "system", "content": "Return strict JSON array only."},
                    {"role": "user", "content": repair_prompt},
                ]
                prompts.append(self._messages_to_prompt(messages))

            outputs = self._generate_batch_text(prompts)
            next_pending: List[int] = []
            for local_idx, global_idx in enumerate(snapshot):
                raw_text = outputs[local_idx]
                raw_outputs[global_idx].append(raw_text)
                try:
                    parsed = extract_json_candidate(raw_text)
                    normalized_groups = normalize_model_groups(
                        parsed=parsed,
                        row=rows[global_idx],
                        sentences=sentences_cache[global_idx],
                        language=language_cache[global_idx],
                    )
                    semantic_error = validate_model_groups_against_source(
                        groups=normalized_groups,
                        row=rows[global_idx],
                        language=language_cache[global_idx],
                        max_group_text_mismatch_per_sample=self.max_group_text_mismatch_per_sample,
                    )
                    if semantic_error:
                        raise ValueError(semantic_error)
                    parsed_groups[global_idx] = normalized_groups
                    parse_errors[global_idx] = None
                except Exception as exc:
                    parse_errors[global_idx] = str(exc)
                    next_pending.append(global_idx)
            pending = next_pending

        results: List[Dict[str, Any]] = []
        for idx, groups in enumerate(parsed_groups):
            trace = {
                "raw_output": truncate_text(raw_outputs[idx][-1] if raw_outputs[idx] else None),
                "raw_outputs_count": len(raw_outputs[idx]),
            }
            if groups is None:
                results.append({"groups": None, "trace": trace, "error": parse_errors[idx] or "Failed to parse segmenter output"})
            else:
                results.append({"groups": groups, "trace": trace, "error": None})
        return results


def assign_sentence_ids(sentences: Sequence[SentenceChunk], targets: Sequence[str], argument: Optional[str], text: str) -> List[int]:
    sid_set: Set[int] = set()
    arg = normalize_string(argument)
    for sentence in sentences:
        sent_text = sentence.text
        sent_text_lower = sent_text.lower()
        for target in targets:
            if target and ((target.lower() in sent_text_lower) or (target in sent_text)):
                sid_set.add(sentence.sentence_id)
        if arg and ((arg.lower() in sent_text_lower) or (arg in sent_text)):
            sid_set.add(sentence.sentence_id)

    if sid_set:
        return sorted(sid_set)

    if sentences:
        return [0]

    return [0] if text.strip() else []


def make_group_id(sample_id: str, local_group_id: int) -> str:
    safe_local_group_id = max(1, int(local_group_id))
    return f"{sample_id}_g{safe_local_group_id}"


def parse_sample_id_from_group_id(group_id: Any) -> Optional[str]:
    gid = normalize_string(group_id)
    if gid is None or "_g" not in gid:
        return None
    sample_id, suffix = gid.rsplit("_g", 1)
    if not sample_id or not suffix.isdigit():
        return None
    return sample_id


def assert_groups_match_job_partition(groups: Sequence[Dict[str, Any]], expected_language: str, expected_split: str) -> None:
    for idx, group in enumerate(groups):
        glang = str(group.get("language") or "").strip().lower()
        gsplit = str(group.get("split") or "").strip().lower()
        if glang != expected_language or gsplit != expected_split:
            raise RuntimeError(
                f"Cross-dataset leakage detected at index {idx}: "
                f"expected {expected_language}_{expected_split}, got {glang}_{gsplit}"
            )


def build_candidates_for_row(row: Dict[str, Any], mode_tag: str) -> List[Dict[str, Any]]:
    text = normalize_string(row.get("text")) or ""
    sample_id = str(row.get("sample_id") or row.get("id") or make_sha1(text, row.get("source"), row.get("split"))[:16])
    original_id = str(row.get("original_id") or sample_id)
    source = str(row.get("source") or "unknown")
    split = str(row.get("split") or "train")
    language = str(row.get("language") or detect_language(text))

    local_group_id = 1
    fallback_group_id = make_group_id(sample_id, local_group_id)
    return [
        {
            "group_id": fallback_group_id,
            "sample_id": sample_id,
            "original_id": original_id,
            "source": source,
            "split": split,
            "language": language,
            "local_group_id": local_group_id,
            "group_text": text,
            "canonical_target": "IMPLICIT_TARGET",
            "escalation_flag": True,
            "fallback_used": True,
            "model_trace": {
                "mode": mode_tag,
                "parse_fail": False,
                "parse_error": None,
                "raw_output": None,
            },
        }
    ]


def build_candidates_for_row_from_model(
    row: Dict[str, Any],
    model_groups: Sequence[Dict[str, Any]],
    mode_tag: str,
    trace: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    text = normalize_string(row.get("text")) or ""
    sample_id = str(row.get("sample_id") or row.get("id") or make_sha1(text, row.get("source"), row.get("split"))[:16])
    original_id = str(row.get("original_id") or sample_id)
    source = str(row.get("source") or "unknown")
    split = str(row.get("split") or "train")
    language = str(row.get("language") or detect_language(text))

    candidates: List[Dict[str, Any]] = []

    for item in model_groups:
        local_group_id = int(item.get("local_group_id") or (len(candidates) + 1))
        canonical_target = normalize_string(item.get("canonical_target")) or "IMPLICIT_TARGET"
        group_text = normalize_string(item.get("group_text")) or text
        escalation_flag = canonical_target == "IMPLICIT_TARGET"

        group_id = make_group_id(sample_id, local_group_id)
        candidate = {
            "group_id": group_id,
            "sample_id": sample_id,
            "original_id": original_id,
            "source": source,
            "split": split,
            "language": language,
            "local_group_id": local_group_id,
            "group_text": group_text,
            "canonical_target": canonical_target,
            "escalation_flag": bool(escalation_flag),
            "fallback_used": False,
            "model_trace": {
                "mode": mode_tag,
                "parse_fail": False,
                "parse_error": None,
                "raw_output": truncate_text((trace or {}).get("raw_output")),
            },
        }
        candidates.append(candidate)

    if candidates:
        return candidates
    return build_candidates_for_row(row=row, mode_tag="stub")


def quality_gates(groups: Sequence[Dict[str, Any]], run_stats: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    missing_group_id = 0
    duplicate_group_id = 0
    missing_original_id = 0
    missing_source = 0
    invalid_language = 0
    missing_group_text = 0
    missing_canonical_target = 0
    invalid_split = 0
    missing_local_group_id = 0
    invalid_escalation_flag = 0

    seen: Set[str] = set()
    for group in groups:
        gid = str(group.get("group_id") or "")
        if not gid:
            missing_group_id += 1
        if gid in seen:
            duplicate_group_id += 1
        seen.add(gid)

        if not normalize_string(group.get("original_id")):
            missing_original_id += 1
        if not normalize_string(group.get("source")):
            missing_source += 1
        if str(group.get("language")) not in {"zh", "en"}:
            invalid_language += 1

        if not normalize_string(group.get("group_text")):
            missing_group_text += 1
        if not normalize_string(group.get("canonical_target")):
            missing_canonical_target += 1
        if str(group.get("split")) not in {"train", "dev", "test"}:
            invalid_split += 1

        local_group_id = group.get("local_group_id")
        if not isinstance(local_group_id, int) or local_group_id <= 0:
            missing_local_group_id += 1
        if not isinstance(group.get("escalation_flag"), bool):
            invalid_escalation_flag += 1

    parse_fail_count = int((run_stats or {}).get("model_parse_fail_count", 0))
    fallback_count = int((run_stats or {}).get("fallback_count", 0))
    fallback_group_count = int((run_stats or {}).get("fallback_group_count", -1))
    if fallback_group_count < 0:
        fallback_group_count = sum(1 for g in groups if bool(g.get("fallback_used")))
    semantic_mismatch_rejection_count = int((run_stats or {}).get("semantic_mismatch_rejection_count", 0))

    passed = (
        missing_group_id == 0
        and duplicate_group_id == 0
        and missing_original_id == 0
        and missing_source == 0
        and invalid_language == 0
        and missing_group_text == 0
        and missing_canonical_target == 0
        and invalid_split == 0
        and missing_local_group_id == 0
        and invalid_escalation_flag == 0
    )

    return {
        "total_groups": len(groups),
        "missing_group_id": missing_group_id,
        "duplicate_group_id": duplicate_group_id,
        "missing_original_id": missing_original_id,
        "missing_source": missing_source,
        "invalid_language": invalid_language,
        "missing_group_text": missing_group_text,
        "missing_canonical_target": missing_canonical_target,
        "invalid_split": invalid_split,
        "missing_local_group_id": missing_local_group_id,
        "invalid_escalation_flag": invalid_escalation_flag,
        "model_parse_fail_count": parse_fail_count,
        "fallback_count": fallback_count,
        "fallback_group_count": fallback_group_count,
        "semantic_mismatch_rejection_count": semantic_mismatch_rejection_count,
        "passed": passed,
    }


def build_summary(groups: Sequence[Dict[str, Any]], run_stats: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    split_dist = dict(sorted(Counter(str(g.get("split")) for g in groups).items()))
    lang_dist = dict(sorted(Counter(str(g.get("language")) for g in groups).items()))
    source_dist = dict(sorted(Counter(str(g.get("source")) for g in groups).items()))

    escalation_count = sum(
        1
        for g in groups
        if bool(g.get("escalation_flag")) or str(g.get("canonical_target") or "") == "IMPLICIT_TARGET"
    )
    fallback_count = int((run_stats or {}).get("fallback_group_count", -1))
    if fallback_count < 0:
        fallback_count = sum(1 for g in groups if bool(g.get("fallback_used")))

    return {
        "group_count": len(groups),
        "split_distribution": split_dist,
        "language_distribution": lang_dist,
        "source_distribution": source_dist,
        "escalation_count": escalation_count,
        "escalation_ratio": round(escalation_count / len(groups), 4) if groups else 0.0,
        "fallback_count": fallback_count,
        "fallback_row_count": int((run_stats or {}).get("fallback_count", 0)),
        "model_parse_fail_count": int((run_stats or {}).get("model_parse_fail_count", 0)),
        "semantic_mismatch_rejection_count": int((run_stats or {}).get("semantic_mismatch_rejection_count", 0)),
    }


def bucket_rows_by_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get("split") or "train")].append(row)
    return buckets


def bucket_rows_by_language_split(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        language = str(row.get("language") or "").strip().lower()
        split = str(row.get("split") or "train").strip().lower()
        key = f"{language}_{split}"
        buckets[key].append(row)
    return buckets


COMPACT_CANDIDATE_KEYS: Tuple[str, ...] = (
    "group_id",
    "original_id",
    "source",
    "language",
    "split",
    "local_group_id",
    "group_text",
    "canonical_target",
    "escalation_flag",
)


def to_compact_candidate_row(row: Dict[str, Any]) -> Dict[str, Any]:
    compact: Dict[str, Any] = {key: row[key] for key in COMPACT_CANDIDATE_KEYS if key in row}

    sample_id = parse_sample_id_from_group_id(compact.get("group_id"))
    if sample_id is None:
        sample_id = normalize_string(row.get("sample_id")) or normalize_string(row.get("id"))

    if not normalize_string(compact.get("original_id")):
        inferred_original_id = None
        if sample_id and "_" in sample_id:
            inferred_original_id = sample_id.split("_", 1)[1]
        compact["original_id"] = inferred_original_id or sample_id or "unknown"

    if "local_group_id" not in compact:
        try:
            compact["local_group_id"] = int(row.get("local_group_id") or 1)
        except (TypeError, ValueError):
            compact["local_group_id"] = 1
    try:
        compact["local_group_id"] = max(1, int(compact.get("local_group_id") or 1))
    except (TypeError, ValueError):
        compact["local_group_id"] = 1

    if not normalize_string(compact.get("group_id")) and sample_id:
        compact["group_id"] = make_group_id(sample_id, int(compact["local_group_id"]))
    compact["source"] = normalize_string(compact.get("source")) or "unknown"
    compact["language"] = str(compact.get("language") or "en").strip().lower()
    compact["split"] = str(compact.get("split") or "train").strip().lower()
    compact["group_text"] = normalize_string(compact.get("group_text")) or ""
    compact["canonical_target"] = normalize_string(compact.get("canonical_target")) or "IMPLICIT_TARGET"
    compact["escalation_flag"] = bool(compact.get("canonical_target") == "IMPLICIT_TARGET" or compact.get("escalation_flag"))

    return compact


def build_candidate_rows(groups: Sequence[Dict[str, Any]], candidate_output_profile: str) -> List[Dict[str, Any]]:
    _profile = str(candidate_output_profile or "compact").strip().lower()
    return [to_compact_candidate_row(dict(row)) for row in groups]


def extract_dict_objects_from_text(text: str) -> List[Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return []

    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    cursor = 0
    while cursor < len(stripped):
        while cursor < len(stripped) and stripped[cursor].isspace():
            cursor += 1
        if cursor >= len(stripped):
            break
        try:
            parsed, end_idx = decoder.raw_decode(stripped, cursor)
            cursor = end_idx
            if isinstance(parsed, dict):
                objects.append(parsed)
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict):
                        objects.append(item)
            continue
        except json.JSONDecodeError:
            next_dict = stripped.find("{", cursor + 1)
            next_list = stripped.find("[", cursor + 1)
            candidates = [idx for idx in (next_dict, next_list) if idx >= 0]
            if not candidates:
                break
            cursor = min(candidates)
    return objects


def normalize_compact_output_row_for_repair(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(row, dict):
        return None
    compact = to_compact_candidate_row(row)

    if set(compact.keys()) != set(COMPACT_CANDIDATE_KEYS):
        return None
    if not normalize_string(compact.get("group_id")):
        return None
    if "_g" not in str(compact.get("group_id")):
        return None
    if not normalize_string(compact.get("original_id")):
        return None
    if not normalize_string(compact.get("source")):
        return None
    if not normalize_string(compact.get("group_text")):
        return None
    if not normalize_string(compact.get("canonical_target")):
        return None

    language = str(compact.get("language") or "").strip().lower()
    if language not in {"zh", "en"}:
        return None
    split = str(compact.get("split") or "").strip().lower()
    if split not in {"train", "test"}:
        return None

    try:
        local_group_id = int(compact.get("local_group_id") or 0)
    except (TypeError, ValueError):
        return None
    if local_group_id <= 0:
        return None

    normalized: Dict[str, Any] = {
        "group_id": str(compact.get("group_id")),
        "original_id": str(compact.get("original_id")),
        "source": str(compact.get("source")),
        "language": language,
        "split": split,
        "local_group_id": local_group_id,
        "group_text": str(compact.get("group_text")),
        "canonical_target": str(compact.get("canonical_target")),
        "escalation_flag": bool(compact.get("escalation_flag")),
    }
    return normalized


def selfcheck_and_repair_jsonl(
    path: Path,
    expected_language: Optional[str] = None,
    expected_split: Optional[str] = None,
) -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "exists": path.exists(),
        "rows_before": 0,
        "rows_after": 0,
        "unparseable_lines": 0,
        "concatenated_objects_recovered": 0,
        "invalid_rows_dropped": 0,
        "partition_mismatch_dropped": 0,
        "duplicate_group_id_dropped": 0,
        "repaired": False,
    }
    if not path.exists():
        return report

    valid_rows: List[Dict[str, Any]] = []
    nonempty_lines = 0
    raw_text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        nonempty_lines += 1
        objects = extract_dict_objects_from_text(line)
        if not objects:
            report["unparseable_lines"] += 1
            continue
        if len(objects) > 1:
            report["concatenated_objects_recovered"] += len(objects) - 1

        for obj in objects:
            repaired = normalize_compact_output_row_for_repair(obj)
            if repaired is None:
                report["invalid_rows_dropped"] += 1
                continue
            if expected_language and str(repaired.get("language")) != expected_language:
                report["partition_mismatch_dropped"] += 1
                continue
            if expected_split and str(repaired.get("split")) != expected_split:
                report["partition_mismatch_dropped"] += 1
                continue
            valid_rows.append(repaired)

    deduped_rows: List[Dict[str, Any]] = []
    seen_group_ids: Set[str] = set()
    for row in valid_rows:
        group_id = str(row.get("group_id") or "")
        if not group_id:
            report["invalid_rows_dropped"] += 1
            continue
        if group_id in seen_group_ids:
            report["duplicate_group_id_dropped"] += 1
            continue
        seen_group_ids.add(group_id)
        deduped_rows.append(row)

    report["rows_before"] = nonempty_lines
    report["rows_after"] = len(deduped_rows)

    repair_needed = bool(
        report["unparseable_lines"]
        or report["concatenated_objects_recovered"]
        or report["invalid_rows_dropped"]
        or report["partition_mismatch_dropped"]
        or report["duplicate_group_id_dropped"]
        or len(deduped_rows) != nonempty_lines
    )

    if repair_needed:
        write_jsonl(path, deduped_rows)
        report["repaired"] = True

    return report


def selfcheck_and_repair_phase1_outputs(output_paths_by_job: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    reports: Dict[str, Dict[str, Any]] = {}
    for job_key, output_path in output_paths_by_job.items():
        language, split = job_key.split("_", 1)
        reports[job_key] = selfcheck_and_repair_jsonl(
            output_path,
            expected_language=language,
            expected_split=split,
        )
    return reports


def write_candidate_outputs(output_root: Path, groups: Sequence[Dict[str, Any]], candidate_output_profile: str = "full") -> None:
    full_rows = [dict(row) for row in groups]
    candidate_rows = build_candidate_rows(full_rows, candidate_output_profile=candidate_output_profile)

    by_lang_split = bucket_rows_by_language_split(candidate_rows)
    for key in ["zh_train", "zh_test", "en_train", "en_test"]:
        filename = PHASE1_OUTPUT_FILENAME_MAP[key]
        write_jsonl(output_root / filename, by_lang_split.get(key, []))


def default_phase1_cleaned_input_paths(workspace_root: Path) -> Dict[str, Path]:
    base = workspace_root / "SPAR/outputs/main/phase0"
    return {key: base / filename for key, filename in LANGUAGE_SPLIT_FILENAME_MAP.items()}


def resolve_phase1_language_split_inputs(paths: Paths, cfg: Dict[str, Any]) -> Dict[str, Path]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
    config_dir = paths.config_path.parent
    backend = str(inference_cfg.get("backend", "transformers")).strip().lower()
    api_profiles = inference_cfg.get("api_profiles") if isinstance(inference_cfg.get("api_profiles"), dict) else None

    resolved = default_phase1_cleaned_input_paths(paths.workspace_root)
    raw_map = runtime_cfg.get("language_split_inputs")
    if isinstance(raw_map, dict):
        for key in LANGUAGE_SPLIT_FILENAME_MAP.keys():
            raw_path = raw_map.get(key)
            if isinstance(raw_path, str) and raw_path.strip():
                effective_raw = raw_path.strip()
                if backend == "api":
                    namespace = derive_api_output_namespace(inference_cfg, api_profiles=api_profiles)
                    effective_raw = str(rewrite_output_namespace_path_text(effective_raw, namespace))
                resolved[key] = resolve_path(paths.workspace_root, config_dir, effective_raw)
    return resolved


def normalize_row_for_language_split(row: Dict[str, Any], language: str, split: str) -> Dict[str, Any]:
    text = normalize_string(row.get("text")) or ""
    source = normalize_string(row.get("source")) or "unknown"

    sample_id = normalize_string(row.get("sample_id")) or normalize_string(row.get("id"))
    if sample_id is None:
        sample_id = make_sha1(text, language, split, source)[:16]

    original_id = normalize_string(row.get("original_id"))
    if original_id is None and "_" in sample_id:
        original_id = sample_id.split("_", 1)[1]
    if original_id is None:
        original_id = sample_id


    return {
        "sample_id": sample_id,
        "id": sample_id,
        "original_id": original_id,
        "source": source,
        "language": language,
        "split": split,
        "text": text,
    }


def load_phase1_rows_by_language_split(
    paths: Paths,
    cfg: Dict[str, Any],
    logger: EventLogger,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, str]]:
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    require_inputs = parse_bool(runtime_cfg.get("require_language_split_inputs", True), default=True)

    input_map = resolve_phase1_language_split_inputs(paths=paths, cfg=cfg)
    rows_by_key: Dict[str, List[Dict[str, Any]]] = {}
    loaded_paths: Dict[str, str] = {}

    for key in LANGUAGE_SPLIT_FILENAME_MAP.keys():
        language, split = key.split("_", 1)
        path = input_map[key]
        loaded_paths[key] = str(path)
        if not path.exists():
            if require_inputs:
                raise FileNotFoundError(f"Required input JSONL missing for {key}: {path}")
            logger.event(
                "load",
                "language_split_input_missing",
                level="WARNING",
                key=key,
                path=str(path),
            )
            rows_by_key[key] = []
            continue

        raw_rows = read_jsonl(path)
        rows_by_key[key] = [normalize_row_for_language_split(row, language=language, split=split) for row in raw_rows]

    return rows_by_key, loaded_paths


def load_checkpoint_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        return None
    return None


def prune_phase1_output_root(output_root: Path, preserve: Optional[Set[str]] = None) -> None:
    ensure_dir(output_root)
    preserve_names = set(preserve or set())
    for child in output_root.iterdir():
        if child.name in preserve_names:
            continue
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            try:
                child.unlink()
            except FileNotFoundError:
                continue


def run_phase1(args: Optional[argparse.Namespace] = None) -> Dict[str, Any]:
    cli_args = args or parse_args()
    workspace_root = Path(__file__).resolve().parents[2]
    config_path = Path(cli_args.config).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    cfg = load_config(config_path)
    apply_cli_overrides(cfg, cli_args)
    paths = build_paths(args=cli_args, cfg=cfg, workspace_root=workspace_root)
    run_id = cli_args.run_id or f"phase1_segmenter_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ensure_dir(paths.output_root)

    runtime_preview = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
    lock_timeout_seconds = safe_float(runtime_preview.get("output_write_lock_timeout_sec", 900.0), default=900.0)
    lock_handle = acquire_output_root_lock(
        paths.output_root,
        run_id=run_id,
        timeout_seconds=lock_timeout_seconds,
    )
    checkpoint_preview = paths.output_root / "phase1_checkpoint.json"
    resume_preview = (
        parse_bool(runtime_preview.get("resume_from_checkpoint", True), default=True)
        and not cli_args.dry_run
        and checkpoint_preview.exists()
    )
    if not cli_args.dry_run and not resume_preview:
        prune_phase1_output_root(paths.output_root, preserve={PHASE1_OUTPUT_LOCK_FILENAME})

    logger = EventLogger(
        run_id=run_id,
        log_path=paths.output_root / "phase1_events.jsonl",
        level=str(cfg.get("logging", {}).get("level", "INFO")),
    )
    logger.event(
        "lock",
        "output_root_locked",
        lock_path=str(lock_handle.get("path")),
        lock_timeout_seconds=lock_handle.get("timeout_seconds"),
        lock_waited_seconds=lock_handle.get("waited_seconds"),
    )

    try:
        segmenter_cfg = get_segmenter_cfg(cfg)
        inference_cfg = cfg.get("inference", {}) if isinstance(cfg.get("inference"), dict) else {}
        runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
        quality_cfg = cfg.get("quality", {}) if isinstance(cfg.get("quality"), dict) else {}

        mode = str(segmenter_cfg.get("mode", "heuristic"))
        if mode not in {"heuristic", "stub", "qwen_local", "vllm_local", "api"}:
            raise ValueError(f"Unsupported mode: {mode}")

        hint_mode = resolve_hint_mode(segmenter_cfg)
        max_parse_retries = int(max(0, segmenter_cfg.get("max_parse_retries", 2)))
        max_repair_retries = int(max(0, segmenter_cfg.get("max_repair_retries", 1)))
        allow_qwen_fallback = parse_bool(segmenter_cfg.get("allow_qwen_fallback_to_heuristic", True), default=True)
        continue_on_parse_fail = parse_bool(segmenter_cfg.get("continue_on_parse_fail", True), default=True)
        _requested_profile = str(runtime_cfg.get("candidate_output_profile", "compact")).strip().lower()
        candidate_output_profile = "compact"
        max_group_text_mismatch_per_sample = int(max(0, quality_cfg.get("max_group_text_mismatch_per_sample", 0)))

        checkpoint_interval_rows = int(max(1, runtime_cfg.get("checkpoint_interval_rows", 50)))
        checkpoint_path = paths.output_root / "phase1_checkpoint.json"
        resume_from_checkpoint = (
            parse_bool(runtime_cfg.get("resume_from_checkpoint", True), default=True)
            and not cli_args.dry_run
            and checkpoint_path.exists()
        )

        prompt_paths = {
            "default": str(paths.segmenter_prompt_path) if paths.segmenter_prompt_path else None,
            "zh": str(paths.segmenter_prompt_zh_path) if paths.segmenter_prompt_zh_path else None,
            "en": str(paths.segmenter_prompt_en_path) if paths.segmenter_prompt_en_path else None,
        }

        logger.event(
            "bootstrap",
            "run_started",
            config_path=str(paths.config_path),
            input_jsonl=str(paths.input_jsonl),
            output_root=str(paths.output_root),
            model_root=str(paths.model_root),
            segmenter_model_path=str(paths.segmenter_model_path),
            segmenter_prompt_paths=prompt_paths,
            mode=mode,
            hint_mode=hint_mode,
            max_parse_retries=max_parse_retries,
            max_repair_retries=max_repair_retries,
            allow_qwen_fallback_to_heuristic=allow_qwen_fallback,
            continue_on_parse_fail=continue_on_parse_fail,
            candidate_output_profile=candidate_output_profile,
            max_group_text_mismatch_per_sample=max_group_text_mismatch_per_sample,
            checkpoint_interval_rows=checkpoint_interval_rows,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        rows_by_job, loaded_inputs = load_phase1_rows_by_language_split(paths=paths, cfg=cfg, logger=logger)
        logger.event(
            "load",
            "input_loaded",
            samples_by_job={k: len(v) for k, v in rows_by_job.items()},
            loaded_inputs=loaded_inputs,
        )

        allowed_splits = {split.lower() for split in parse_allowed_splits(cfg)}
        selected_job_keys = [
            f"{language}_{split}"
            for language, split in LANGUAGE_SPLIT_ORDER
            if split in allowed_splits
        ]
        if not selected_job_keys:
            raise RuntimeError(f"No phase1 jobs selected for allowed_splits={sorted(allowed_splits)}")

        max_samples = runtime_cfg.get("max_samples")
        max_samples_per_job = runtime_cfg.get("max_samples_per_job")
        if isinstance(cli_args.max_samples, int) and cli_args.max_samples > 0:
            max_samples = cli_args.max_samples
        if isinstance(cli_args.max_samples_per_job, int) and cli_args.max_samples_per_job > 0:
            max_samples_per_job = cli_args.max_samples_per_job

        planned_rows_by_job: Dict[str, List[Dict[str, Any]]] = {}
        remaining = int(max_samples) if isinstance(max_samples, int) and max_samples > 0 else None
        for key in selected_job_keys:
            rows = list(rows_by_job.get(key, []))
            if isinstance(max_samples_per_job, int) and max_samples_per_job > 0:
                rows = rows[: max_samples_per_job]
            if remaining is not None:
                rows = rows[: max(0, remaining)]
                remaining = max(0, remaining - len(rows))
            planned_rows_by_job[key] = rows

        planned_sample_ids_by_job: Dict[str, Set[str]] = {
            key: {str(row.get("sample_id") or "") for row in planned_rows_by_job.get(key, []) if str(row.get("sample_id") or "")}
            for key in selected_job_keys
        }
        total_rows = sum(len(rows) for rows in planned_rows_by_job.values())

        logger.event(
            "load",
            "filtered_input",
            total_rows=total_rows,
            jobs=selected_job_keys,
            rows_by_job={k: len(planned_rows_by_job.get(k, [])) for k in selected_job_keys},
            splits=sorted(allowed_splits),
            max_samples=max_samples,
            max_samples_per_job=max_samples_per_job,
        )

        run_stats: Dict[str, int] = {
            "model_parse_fail_count": 0,
            "fallback_count": 0,
            "fallback_group_count": 0,
            "semantic_mismatch_rejection_count": 0,
        }
        groups_by_job: Dict[str, List[Dict[str, Any]]] = {key: [] for key in selected_job_keys}
        processed_sample_ids_by_job: Dict[str, Set[str]] = {key: set() for key in selected_job_keys}
        job_stats: Dict[str, Dict[str, int]] = {
            key: {
                "processed_rows": 0,
                "parse_fail_count": 0,
                "fallback_count": 0,
                "fallback_group_count": 0,
                "semantic_mismatch_rejection_count": 0,
            }
            for key in selected_job_keys
        }

        checkpoint_state = load_checkpoint_state(checkpoint_path) if resume_from_checkpoint else None
        run_started_at = str((checkpoint_state or {}).get("started_at") or utc_now())

        if checkpoint_state and isinstance(checkpoint_state.get("run_stats"), dict):
            previous_stats = checkpoint_state["run_stats"]
            for key in run_stats.keys():
                try:
                    run_stats[key] = int(previous_stats.get(key, run_stats[key]))
                except (TypeError, ValueError):
                    continue

        if checkpoint_state and isinstance(checkpoint_state.get("job_stats"), dict):
            previous_job_stats = checkpoint_state["job_stats"]
            for key, snapshot in previous_job_stats.items():
                if key not in job_stats or not isinstance(snapshot, dict):
                    continue
                for stat_key in job_stats[key].keys():
                    try:
                        job_stats[key][stat_key] = int(snapshot.get(stat_key, job_stats[key][stat_key]))
                    except (TypeError, ValueError):
                        continue

        output_paths_by_job: Dict[str, Path] = {
            key: paths.output_root / PHASE1_OUTPUT_FILENAME_MAP[key]
            for key in selected_job_keys
        }
        parse_fail_samples_path = paths.output_root / "phase1_parse_fail_samples.jsonl"

        line_integrity_startup_reports: Dict[str, Dict[str, Any]] = {}
        line_integrity_post_reports: Dict[str, Dict[str, Any]] = {}
        if not cli_args.dry_run:
            line_integrity_startup_reports = selfcheck_and_repair_phase1_outputs(output_paths_by_job)
            for key, report in line_integrity_startup_reports.items():
                logger.event(
                    "selfcheck",
                    "startup_output_checked",
                    job=key,
                    output_path=str(output_paths_by_job[key]),
                    **report,
                )

        for key in selected_job_keys:
            output_path = output_paths_by_job[key]
            if resume_from_checkpoint and output_path.exists():
                existing_rows = read_jsonl(output_path)
                groups_by_job[key].extend(existing_rows)
                restored_ids: Set[str] = set()
                for row in existing_rows:
                    sample_id = normalize_string(row.get("sample_id")) or parse_sample_id_from_group_id(row.get("group_id"))
                    if sample_id:
                        restored_ids.add(sample_id)
                processed_sample_ids_by_job[key].update(restored_ids)
                logger.event(
                    "resume",
                    "loaded_existing_output",
                    job=key,
                    output_path=str(output_path),
                    groups=len(existing_rows),
                    processed_rows=len(processed_sample_ids_by_job[key]),
                )
            elif not cli_args.dry_run:
                write_jsonl(output_path, [])
        if not cli_args.dry_run and (not resume_from_checkpoint or not parse_fail_samples_path.exists()):
            write_jsonl(parse_fail_samples_path, [])

        def processed_rows_for_job(job_key: str) -> int:
            return len(planned_sample_ids_by_job[job_key] & processed_sample_ids_by_job[job_key])

        processed_rows_total = sum(processed_rows_for_job(key) for key in selected_job_keys)
        progress_interval = int(max(1, runtime_cfg.get("progress_interval", 200)))
        rows_since_checkpoint = 0

        def build_checkpoint_payload(status: str = "running") -> Dict[str, Any]:
            eta_utc = estimate_eta_iso(run_started_at, processed_rows_total, total_rows)
            jobs_payload: Dict[str, Any] = {}
            for key in selected_job_keys:
                processed_rows = processed_rows_for_job(key)
                jobs_payload[key] = {
                    "input_path": loaded_inputs.get(key),
                    "output_path": str(output_paths_by_job[key]),
                    "total_rows": len(planned_rows_by_job.get(key, [])),
                    "processed_rows": processed_rows,
                    "completed": processed_rows >= len(planned_rows_by_job.get(key, [])),
                }
            return {
                "run_id": run_id,
                "status": status,
                "mode": mode,
                "started_at": run_started_at,
                "updated_at": utc_now(),
                "total_rows": total_rows,
                "processed_rows_total": processed_rows_total,
                "eta_utc": eta_utc,
                "run_stats": run_stats,
                "job_stats": job_stats,
                "jobs": jobs_payload,
            }

        def persist_checkpoint(force: bool = False, status: str = "running") -> None:
            nonlocal rows_since_checkpoint
            if cli_args.dry_run:
                return
            if not force and rows_since_checkpoint < checkpoint_interval_rows:
                return
            payload = build_checkpoint_payload(status=status)
            write_json(checkpoint_path, payload)
            write_json(
                paths.output_root / "phase1_eta.json",
                {
                    "run_id": run_id,
                    "status": status,
                    "updated_at": payload.get("updated_at"),
                    "processed_rows_total": payload.get("processed_rows_total"),
                    "total_rows": payload.get("total_rows"),
                    "eta_utc": payload.get("eta_utc"),
                },
            )
            rows_since_checkpoint = 0
            logger.event(
                "checkpoint",
                "saved",
                status=status,
                processed_rows=payload.get("processed_rows_total"),
                total_rows=payload.get("total_rows"),
                eta_utc=payload.get("eta_utc"),
            )
            print(
                json.dumps(
                    {
                        "event": "phase1_checkpoint",
                        "status": status,
                        "processed_rows_total": payload.get("processed_rows_total"),
                        "total_rows": payload.get("total_rows"),
                        "eta_utc": payload.get("eta_utc"),
                    },
                    ensure_ascii=False,
                )
            )

        persist_checkpoint(force=True, status="running")

        llm: Optional[LocalSegmenterLLM] = None
        if mode in {"qwen_local", "vllm_local", "api"}:
            pending_total = 0
            for key in selected_job_keys:
                for row in planned_rows_by_job.get(key, []):
                    sample_id = str(row.get("sample_id") or "")
                    if sample_id not in processed_sample_ids_by_job[key]:
                        pending_total += 1
            if pending_total > 0:
                prompt_templates = load_segmenter_prompt_templates(paths)
                llm = LocalSegmenterLLM(
                    model_path=paths.segmenter_model_path,
                    inference_cfg=inference_cfg,
                    prompt_templates=prompt_templates,
                    hint_mode=hint_mode,
                    max_group_text_mismatch_per_sample=max_group_text_mismatch_per_sample,
                )
                llm.initialize()
                logger.event(
                    "segmenter",
                    "model_backend_initialized",
                    mode=mode,
                    backend=llm.backend,
                    model_device=llm.model_device,
                    pending_rows=pending_total,
                )
            else:
                logger.event("resume", "no_pending_rows_skip_model_init")

        for job_key in selected_job_keys:
            language, split = job_key.split("_", 1)
            rows_for_job = planned_rows_by_job.get(job_key, [])
            pending_rows = [
                row
                for row in rows_for_job
                if str(row.get("sample_id") or "") not in processed_sample_ids_by_job[job_key]
            ]

            logger.event(
                "segmenter",
                "job_started",
                job=job_key,
                language=language,
                split=split,
                total_rows=len(rows_for_job),
                resumed_rows=len(rows_for_job) - len(pending_rows),
                pending_rows=len(pending_rows),
            )

            if not pending_rows:
                logger.event("segmenter", "job_completed", job=job_key, groups_total=len(groups_by_job[job_key]), pending_rows=0)
                continue

            if mode in {"qwen_local", "vllm_local", "api"}:
                if llm is None:
                    raise RuntimeError(f"{mode} mode requires initialized LocalSegmenterLLM")

                batch_size = int(max(1, inference_cfg.get("batch_size", 1)))
                for batch_start in range(0, len(pending_rows), batch_size):
                    batch_rows = pending_rows[batch_start : batch_start + batch_size]
                    results = llm.generate_groups_batch(
                        rows=batch_rows,
                        max_parse_retries=max_parse_retries,
                        max_repair_retries=max_repair_retries,
                        forced_language=language,
                    )

                    for offset, row in enumerate(batch_rows):
                        sample_id = str(row.get("sample_id") or "")
                        result = results[offset]
                        row_groups: List[Dict[str, Any]]

                        if result.get("error"):
                            parse_error = str(result.get("error"))
                            run_stats["model_parse_fail_count"] += 1
                            job_stats[job_key]["parse_fail_count"] += 1
                            trace = result.get("trace") if isinstance(result.get("trace"), dict) else {}
                            raw_output = truncate_text(trace.get("raw_output"), max_chars=1200)
                            if "group_text/source mismatch" in parse_error:
                                run_stats["semantic_mismatch_rejection_count"] += 1
                                job_stats[job_key]["semantic_mismatch_rejection_count"] += 1

                            logger.event(
                                "segmenter",
                                "parse_fail",
                                level="WARNING",
                                job=job_key,
                                sample_id=sample_id,
                                error=parse_error,
                                raw_output=raw_output,
                            )

                            if not cli_args.dry_run:
                                append_jsonl(
                                    parse_fail_samples_path,
                                    [
                                        {
                                            "run_id": run_id,
                                            "job": job_key,
                                            "sample_id": sample_id,
                                            "language": language,
                                            "split": split,
                                            "error": parse_error,
                                            "raw_output": raw_output,
                                            "raw_outputs_count": int(trace.get("raw_outputs_count") or 0),
                                            "ts": utc_now(),
                                        }
                                    ],
                                )

                            should_raise_on_parse_fail = (not allow_qwen_fallback) and (not continue_on_parse_fail)
                            if should_raise_on_parse_fail:
                                raise RuntimeError(
                                    f"qwen_local non-compliant output for sample {sample_id} after retries: {parse_error}"
                                )

                            run_stats["fallback_count"] += 1
                            job_stats[job_key]["fallback_count"] += 1
                            fallback_mode = "heuristic"
                            row_groups = build_candidates_for_row(row=row, mode_tag=fallback_mode)
                            run_stats["fallback_group_count"] += len(row_groups)
                            job_stats[job_key]["fallback_group_count"] += len(row_groups)

                            for local_idx, group in enumerate(row_groups, start=1):
                                group["fallback_used"] = True
                                group["local_group_id"] = local_idx
                                group["model_trace"] = {
                                    "mode": mode,
                                    "parse_fail": True,
                                    "parse_error": parse_error,
                                    "raw_output": truncate_text(trace.get("raw_output")),
                                }
                        else:
                            parsed_groups = result.get("groups") if isinstance(result.get("groups"), list) else []
                            row_groups = build_candidates_for_row_from_model(
                                row=row,
                                model_groups=parsed_groups,
                                mode_tag=mode,
                                trace=result.get("trace") if isinstance(result.get("trace"), dict) else None,
                            )

                        assert_groups_match_job_partition(row_groups, expected_language=language, expected_split=split)
                        groups_by_job[job_key].extend(row_groups)
                        if not cli_args.dry_run:
                            compact_rows = build_candidate_rows(row_groups, candidate_output_profile=candidate_output_profile)
                            append_jsonl(output_paths_by_job[job_key], compact_rows)

                        if sample_id and sample_id not in processed_sample_ids_by_job[job_key]:
                            processed_sample_ids_by_job[job_key].add(sample_id)
                            if sample_id in planned_sample_ids_by_job[job_key]:
                                processed_rows_total += 1
                        job_stats[job_key]["processed_rows"] = processed_rows_for_job(job_key)

                        rows_since_checkpoint += 1
                        if processed_rows_total == 1 or processed_rows_total % progress_interval == 0:
                            logger.event(
                                "segmenter",
                                "progress",
                                job=job_key,
                                processed_rows=processed_rows_total,
                                total_rows=total_rows,
                                groups_total=sum(len(groups_by_job[k]) for k in selected_job_keys),
                                parse_fail_count=run_stats["model_parse_fail_count"],
                                eta_utc=estimate_eta_iso(run_started_at, processed_rows_total, total_rows),
                            )
                        persist_checkpoint(force=False)
            else:
                mode_tag = "stub" if mode == "stub" else "heuristic"
                for row in pending_rows:
                    row_groups = build_candidates_for_row(row=row, mode_tag=mode_tag)
                    for local_idx, group in enumerate(row_groups, start=1):
                        group["local_group_id"] = local_idx

                    assert_groups_match_job_partition(row_groups, expected_language=language, expected_split=split)
                    groups_by_job[job_key].extend(row_groups)
                    if not cli_args.dry_run:
                        compact_rows = build_candidate_rows(row_groups, candidate_output_profile=candidate_output_profile)
                        append_jsonl(output_paths_by_job[job_key], compact_rows)

                    sample_id = str(row.get("sample_id") or "")
                    if sample_id and sample_id not in processed_sample_ids_by_job[job_key]:
                        processed_sample_ids_by_job[job_key].add(sample_id)
                        if sample_id in planned_sample_ids_by_job[job_key]:
                            processed_rows_total += 1
                    job_stats[job_key]["processed_rows"] = processed_rows_for_job(job_key)

                    rows_since_checkpoint += 1
                    if processed_rows_total == 1 or processed_rows_total % progress_interval == 0:
                        logger.event(
                            "segmenter",
                            "progress",
                            job=job_key,
                            processed_rows=processed_rows_total,
                            total_rows=total_rows,
                            groups_total=sum(len(groups_by_job[k]) for k in selected_job_keys),
                            eta_utc=estimate_eta_iso(run_started_at, processed_rows_total, total_rows),
                        )
                    persist_checkpoint(force=False)

            logger.event(
                "segmenter",
                "job_completed",
                job=job_key,
                processed_rows=job_stats[job_key]["processed_rows"],
                total_rows=len(rows_for_job),
                groups_total=len(groups_by_job[job_key]),
                parse_fail_count=job_stats[job_key]["parse_fail_count"],
                fallback_count=job_stats[job_key]["fallback_count"],
            )

        persist_checkpoint(force=True, status="processed")

        if not cli_args.dry_run:
            line_integrity_post_reports = selfcheck_and_repair_phase1_outputs(output_paths_by_job)
            for key, report in line_integrity_post_reports.items():
                logger.event(
                    "selfcheck",
                    "postrun_output_checked",
                    job=key,
                    output_path=str(output_paths_by_job[key]),
                    **report,
                )
            for key in selected_job_keys:
                groups_by_job[key] = read_jsonl(output_paths_by_job[key])

        groups: List[Dict[str, Any]] = []
        for key in selected_job_keys:
            groups.extend(groups_by_job[key])

        quality = quality_gates(groups=groups, run_stats=run_stats)

        require_no_model_parse_fail = parse_bool(quality_cfg.get("require_no_model_parse_fail", False), default=False)
        require_no_fallback_when_qwen = parse_bool(quality_cfg.get("require_no_fallback_when_qwen", False), default=False)
        require_no_group_text_mismatch = parse_bool(quality_cfg.get("require_no_group_text_mismatch", False), default=False)
        parse_fail_gate_passed = True
        fallback_gate_passed = True
        group_text_mismatch_gate_passed = True
        if mode in {"qwen_local", "vllm_local", "api"} and require_no_model_parse_fail:
            parse_fail_gate_passed = int(quality.get("model_parse_fail_count", 0)) == 0
        if mode in {"qwen_local", "vllm_local", "api"} and require_no_fallback_when_qwen:
            fallback_gate_passed = int(quality.get("fallback_count", 0)) == 0
        if mode in {"qwen_local", "vllm_local", "api"} and require_no_group_text_mismatch:
            group_text_mismatch_gate_passed = int(quality.get("semantic_mismatch_rejection_count", 0)) == 0

        quality["require_no_model_parse_fail"] = require_no_model_parse_fail
        quality["require_no_fallback_when_qwen"] = require_no_fallback_when_qwen
        quality["require_no_group_text_mismatch"] = require_no_group_text_mismatch
        quality["parse_fail_gate_passed"] = parse_fail_gate_passed
        quality["fallback_gate_passed"] = fallback_gate_passed
        quality["group_text_mismatch_gate_passed"] = group_text_mismatch_gate_passed
        quality["passed"] = bool(
            quality.get("passed", False)
            and parse_fail_gate_passed
            and fallback_gate_passed
            and group_text_mismatch_gate_passed
        )

        write_json(paths.output_root / "phase1_quality_gates.json", quality)
        logger.event("quality", "quality_report_written", **quality)

        summary = build_summary(groups=groups, run_stats=run_stats)
        write_json(paths.output_root / "phase1_summary.json", summary)
        logger.event("report", "summary_written", **summary)

        strict_mode = parse_bool(quality_cfg.get("strict", False), default=False)
        strict_raise_exception = parse_bool(quality_cfg.get("strict_raise_exception", False), default=False)
        if strict_mode and not quality.get("passed", False):
            logger.event(
                "quality",
                "strict_mode_not_passed",
                level="ERROR",
                strict_raise_exception=strict_raise_exception,
                quality=quality,
            )
            if strict_raise_exception:
                raise RuntimeError(f"Phase1 quality gate failed under strict mode: {quality}")

        if not cli_args.dry_run:
            write_candidate_outputs(paths.output_root, groups, candidate_output_profile=candidate_output_profile)
            logger.event(
                "export",
                "candidates_written",
                total=len(groups),
                outputs={k: str(paths.output_root / PHASE1_OUTPUT_FILENAME_MAP[k]) for k in selected_job_keys},
            )
        else:
            logger.event("export", "dry_run_skip_large_writes")

        jobs_summary = {
            key: {
                "input_path": loaded_inputs.get(key),
                "output_path": str(paths.output_root / PHASE1_OUTPUT_FILENAME_MAP[key]),
                "total_rows": len(planned_rows_by_job.get(key, [])),
                "processed_rows": processed_rows_for_job(key),
                "group_count": len(groups_by_job.get(key, [])),
            }
            for key in selected_job_keys
        }

        run_summary = {
            "run_id": run_id,
            "created_at": utc_now(),
            "started_at": run_started_at,
            "config_path": str(paths.config_path),
            "input_jsonl": str(paths.input_jsonl),
            "input_jsonl_by_lang_split": loaded_inputs,
            "output_root": str(paths.output_root),
            "mode": mode,
            "model_path": str(paths.segmenter_model_path),
            "candidate_output_profile": candidate_output_profile,
            "prompt_paths": prompt_paths,
            "loaded_inputs": loaded_inputs,
            "jobs": jobs_summary,
            "checkpoint": {
                "path": str(checkpoint_path),
                "enabled": not cli_args.dry_run,
                "interval_rows": checkpoint_interval_rows,
            },
            "parse_fail_samples_path": str(parse_fail_samples_path),
            "output_write_lock": {
                "path": str(lock_handle.get("path")),
                "timeout_seconds": lock_handle.get("timeout_seconds"),
                "waited_seconds": lock_handle.get("waited_seconds"),
            },
            "line_integrity": {
                "startup": line_integrity_startup_reports,
                "post_run": line_integrity_post_reports,
            },
            "eta_utc": estimate_eta_iso(run_started_at, processed_rows_total, total_rows),
            "run_stats": run_stats,
            "quality": quality,
            "summary": summary,
        }
        write_json(paths.output_root / "phase1_run_summary.json", run_summary)
        persist_checkpoint(force=True, status="completed")
        logger.event("bootstrap", "run_finished", status="ok", groups=len(groups))
        return run_summary
    except Exception as exc:
        try:
            if "persist_checkpoint" in locals():
                persist_checkpoint(force=True, status="failed")
        except Exception:
            pass
        logger.event("bootstrap", "run_failed", level="ERROR", error=str(exc))
        raise
    finally:
        if "llm" in locals() and llm is not None:
            try:
                llm.shutdown()
            except Exception:
                pass
        logger.close()
        release_output_root_lock(lock_handle)


def main() -> None:
    run_phase1()


if __name__ == "__main__":
    main()
