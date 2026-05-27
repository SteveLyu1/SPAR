from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIB_ROOT = PROJECT_ROOT / "lib"
EVAL_ROOT = PROJECT_ROOT / "evaluation"
for path in [str(PROJECT_ROOT), str(LIB_ROOT), str(EVAL_ROOT)]:
    if path not in sys.path:
        sys.path.insert(0, path)

from target_level_eval import evaluate_input
from api_adapter import ApiTextGenerationClient, resolve_api_backend_config
from json_utils import (
    normalize_hate_label,
    salvage_json_objects_from_array_text,
    sanitize_json_string_control_chars,
)


ARRAY_BLOCK_RE = re.compile(r"\[[\s\S]*\]", flags=re.MULTILINE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\u3002\uFF01\uFF1F!?;\uFF1B\n])")
LONG_CLAUSE_SPLIT_RE = re.compile(r"(?<=[,\uFF0C])")

GROUPING_PROMPTS = {
    "en": """You are a Target-Based Thought Grouping Agent. Your task is to process the input document into structured target-based thought groups while ensuring accuracy, relevance, and completeness.

# Definition of Targets
Targets should capture:
- People, social groups, or entities that are being described, criticized, attacked, or judged.
- Explicit target mentions that appear in the text.
- Use IMPLICIT_TARGET only when the text contains a judgment or attack but no explicit target string.

# Follow these steps:
1. Divide: Segment the input document into thought groups, where each group focuses on a single target.
2. Split: If a sentence contains multiple targets or multiple independent attacks, divide it into smaller thought groups.
3. Collect: If multiple consecutive sentences represent the same target, combine them into one thought group.
4. Extract: Identify and extract explicit target terms as exact words or phrases from the text. Do not generate target terms. Use IMPLICIT_TARGET only if necessary.

Output the results in the json format:
[
    {
        "sentences": ["<sentence1>", "<sentence2>", ...],
        "target": "<target>"
    }
]

Input Document:
{{TEXT}}""",
    "zh": """你是一个“基于目标的思想分组 Agent”。你的任务是把输入文本处理成结构化的、基于目标的 thought groups，并尽量保证准确、相关、完整。

# Target 定义
target 应该刻画：
- 文本中被描述、被评价、被指责、被攻击的对象；
- 明确出现在文本中的人群、地区、人、身份、实体；
- 只有在文本存在评价、攻击或判断，但没有明确目标字符串时，才使用 IMPLICIT_TARGET。

# 请按以下步骤完成
1. Divide：把输入文本切分成若干 thought groups，每个 group 只围绕一个 target。
2. Split：如果一句话里包含多个 target，或者包含多个相互独立的攻击/评价，应拆成更小的 thought groups。
3. Collect：如果连续句子都在表达同一个 target，可以合并到同一个 thought group。
4. Extract：从原文中提取明确的 target 词或短语，必须是原文子串，不要生成新 target。只有必要时才用 IMPLICIT_TARGET。

按如下 json 格式输出：
[
    {
        "sentences": ["<sentence1>", "<sentence2>", ...],
        "target": "<target>"
    }
]

输入文本：
{{TEXT}}""",
}

ARGUMENT_PROMPTS = {
    "en": """You are an Argument Extraction Agent. Extract words or phrases reflecting the core allegation, hostile claim, attack span, or evaluative argument from the given text span. Arguments must be explicit substrings from the text, preserving punctuation and original spelling.

Output the results in the following format:
[
    {"argument": "<argument_1>"},
    {"argument": "<argument_2>"}
]

Requirements:
1. Each argument must come from the text span.
2. If the text span contains multiple independent attacks or evaluations, split them into multiple arguments.
3. Do not generate arguments that are not present in the text.

Text Span:
{{TEXT}}""",
    "zh": """你是一个“论点提取 Agent”。请从给定文本片段中提取能够反映核心指控、攻击片段、贬损判断、敌意评价或论证内容的 argument。argument 必须是原文中的显式子串，保留原始标点和写法。

输出格式：
[
    {"argument": "<argument_1>"},
    {"argument": "<argument_2>"}
]

要求：
1. argument 必须来自输入文本片段；
2. 如果文本片段中有多个独立的攻击/评价片段，应拆成多个 argument；
3. 不要生成原文中不存在的 argument。

文本片段：
{{TEXT}}""",
}

POLARITY_PROMPTS = {
    "en": """You are a Harmfulness Analysis Agent. Read the text span and identify target-argument-harmful predictions.

Output the results in the following json format:
[
    {
        "target": "<target>",
        "argument": "<argument>",
        "harmful": "<0 or 1>"
    }
]

Label definition:
- 1: harmful / hateful / abusive target-directed attack
- 0: non-harmful / non-hateful / merely descriptive or non-abusive evaluation

Rules:
1. target and argument must be exact substrings from the text span unless target should be IMPLICIT_TARGET.
2. Each object should express one target-argument-harmful prediction.
3. Do not output ids.

Text Span:
{{TEXT}}""",
    "zh": """你是一个“仇恨性判别 Agent”。阅读输入文本片段，识别其中的 target-argument-hateful 预测结果。

输出格式：
[
    {
        "target": "<target>",
        "argument": "<argument>",
        "hateful": "<0 or 1>"
    }
]

标签定义：
- 1：hateful，表示存在针对 target 的仇恨性、攻击性或其他有害表达；
- 0：non-hateful，表示只是描述、一般评价、非敌意表达，或不足以构成仇恨攻击。

规则：
1. target 和 argument 必须是文本片段中的原文子串；如果没有明确 target，target 可写 IMPLICIT_TARGET。
2. 每个对象表示一个 target-argument-hateful 预测。
3. 不要输出任何 id。

文本片段：
{{TEXT}}""",
}


@dataclass
class JsonArrayResult:
    raw_text: str
    items: List[Any]
    parse_ok: bool
    error: Optional[str]
    attempts: int


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def parse_iso_datetime(value: Any) -> Optional[datetime]:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


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


def build_messages(user_content: str, retry: bool = False) -> List[Dict[str, str]]:
    system = (
        "Return strict JSON array only. Do not output markdown, code fences, or explanatory text."
        if not retry
        else "Return strict JSON array only. Start with [ and end with ]. No markdown. No prose. No comments."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def extract_json_array(raw_text: str) -> List[Any]:
    stripped = (raw_text or "").strip()
    if not stripped:
        raise ValueError("Empty model output")

    candidates = [stripped]
    match = ARRAY_BLOCK_RE.search(stripped)
    if match:
        block = match.group(0).strip()
        if block not in candidates:
            candidates.append(block)

    for candidate in candidates:
        try:
            parsed = json.loads(sanitize_json_string_control_chars(candidate))
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed

    salvaged = salvage_json_objects_from_array_text(stripped)
    if salvaged:
        return salvaged
    raise ValueError("Could not parse model output as JSON array")


def build_inference_cfg(
    *,
    provider: str,
    api_format: str,
    model: str,
    api_key_env: Optional[str],
    api_key: Optional[str],
    base_url: str,
    timeout_sec: float,
    max_retries: int,
    retry_backoff_sec: float,
    max_concurrency: int,
    reasoning_effort: Optional[str],
    output_namespace: str,
) -> Dict[str, Any]:
    profile_payload: Dict[str, Any] = {
        "provider": provider,
        "format": api_format,
        "model": model,
        "base_url": base_url,
        "timeout_sec": timeout_sec,
        "max_retries": max_retries,
        "retry_backoff_sec": retry_backoff_sec,
        "max_concurrency": max_concurrency,
        "headers": {},
        "extra_body": {},
        "output_namespace": output_namespace,
    }
    if api_key:
        profile_payload["api_key"] = api_key
    elif api_key_env:
        profile_payload["api_key_env"] = api_key_env
    if reasoning_effort:
        profile_payload["reasoning_effort"] = reasoning_effort
    return {
        "api_profile": "dance_core_api",
        "api_profiles": {
            "dance_core_api": profile_payload,
        },
        "api": {
            "profile": "dance_core_api",
        },
    }


def generate_json_arrays_batch(
    *,
    client: ApiTextGenerationClient,
    user_contents: Sequence[str],
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> List[JsonArrayResult]:
    if not user_contents:
        return []

    first_messages = [build_messages(content, retry=False) for content in user_contents]
    try:
        first_outputs = client.generate_from_messages_batch(
            first_messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    except Exception:
        first_outputs = []
        for messages in first_messages:
            try:
                output = client.generate_from_messages_batch(
                    [messages],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )[0]
                first_outputs.append(output)
            except Exception as exc:
                first_outputs.append(f"__API_ERROR__::{type(exc).__name__}: {exc}")

    results: List[JsonArrayResult] = []
    retry_indices: List[int] = []
    retry_messages: List[List[Dict[str, str]]] = []

    for idx, raw_text in enumerate(first_outputs):
        if raw_text.startswith("__API_ERROR__::"):
            results.append(
                JsonArrayResult(
                    raw_text=raw_text,
                    items=[],
                    parse_ok=False,
                    error=raw_text,
                    attempts=1,
                )
            )
            continue
        try:
            parsed = extract_json_array(raw_text)
            results.append(JsonArrayResult(raw_text=raw_text, items=parsed, parse_ok=True, error=None, attempts=1))
        except Exception as exc:
            results.append(
                JsonArrayResult(raw_text=raw_text, items=[], parse_ok=False, error=str(exc), attempts=1)
            )
            retry_indices.append(idx)
            retry_messages.append(build_messages(user_contents[idx], retry=True))

    if not retry_indices:
        return results

    try:
        retry_outputs = client.generate_from_messages_batch(
            retry_messages,
            max_new_tokens=retry_max_new_tokens,
            temperature=0.0,
            top_p=1.0,
            enable_thinking=False,
        )
    except Exception:
        retry_outputs = []
        for messages in retry_messages:
            try:
                output = client.generate_from_messages_batch(
                    [messages],
                    max_new_tokens=retry_max_new_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    enable_thinking=False,
                )[0]
                retry_outputs.append(output)
            except Exception as exc:
                retry_outputs.append(f"__API_ERROR__::{type(exc).__name__}: {exc}")

    for local_idx, global_idx in enumerate(retry_indices):
        raw_text = retry_outputs[local_idx]
        if raw_text.startswith("__API_ERROR__::"):
            results[global_idx] = JsonArrayResult(
                raw_text=raw_text,
                items=[],
                parse_ok=False,
                error=raw_text,
                attempts=2,
            )
            continue
        try:
            parsed = extract_json_array(raw_text)
            results[global_idx] = JsonArrayResult(
                raw_text=raw_text,
                items=parsed,
                parse_ok=True,
                error=None,
                attempts=2,
            )
        except Exception as exc:
            results[global_idx] = JsonArrayResult(
                raw_text=raw_text,
                items=[],
                parse_ok=False,
                error=str(exc),
                attempts=2,
            )

    return results


def casefold_for_lookup(text: str, language: str) -> str:
    normalized = html.unescape(text)
    return normalized.casefold() if language == "en" else normalized


def find_span_position(text: str, span: str, language: str) -> int:
    haystack = casefold_for_lookup(text, language)
    needle = casefold_for_lookup(span, language)
    return haystack.find(needle)


def dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def split_text_into_units(text: str, language: str) -> List[str]:
    chunks = [piece.strip() for piece in SENTENCE_SPLIT_RE.split(text) if piece and piece.strip()]
    if not chunks:
        chunks = [text.strip()]

    refined: List[str] = []
    for chunk in chunks:
        if len(chunk) <= 60:
            refined.append(chunk)
            continue
        sub_chunks = [piece.strip() for piece in LONG_CLAUSE_SPLIT_RE.split(chunk) if piece and piece.strip()]
        if len(sub_chunks) <= 1:
            refined.append(chunk)
            continue
        if language == "zh" or len(chunk) > 100:
            refined.extend(sub_chunks)
        else:
            refined.append(chunk)
    return [piece for piece in refined if piece]


def normalize_grouping_items(parsed: Sequence[Any], text: str, language: str) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    seen = set()
    fallback_sentences = split_text_into_units(text, language)

    for idx, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            continue
        target = normalize_string(item.get("target"))
        if target is None:
            continue
        if target != "IMPLICIT_TARGET" and find_span_position(text, target, language) < 0:
            continue

        raw_sentences = item.get("sentences")
        sentences: List[str] = []
        if isinstance(raw_sentences, list):
            for sentence in raw_sentences:
                normalized = normalize_string(sentence)
                if normalized is None:
                    continue
                if find_span_position(text, normalized, language) >= 0:
                    sentences.append(normalized)
        elif isinstance(raw_sentences, str):
            normalized = normalize_string(raw_sentences)
            if normalized is not None and find_span_position(text, normalized, language) >= 0:
                sentences.append(normalized)

        if not sentences:
            if target == "IMPLICIT_TARGET":
                sentences = fallback_sentences[:1] if fallback_sentences else [text]
            else:
                local_units = [unit for unit in fallback_sentences if find_span_position(unit, target, language) >= 0]
                sentences = local_units[:1] if local_units else (fallback_sentences[:1] if fallback_sentences else [text])

        signature = (target, tuple(sentences))
        if signature in seen:
            continue
        seen.add(signature)
        groups.append(
            {
                "sentences": sentences,
                "target": target,
            }
        )
    normalized_groups: List[Dict[str, Any]] = []
    for idx, group in enumerate(groups, start=1):
        normalized_groups.append(
            {
                "group_id": idx,
                "sentences": list(group["sentences"]),
                "target": group["target"],
                "group_text": " ".join(group["sentences"]).strip(),
            }
        )
    return normalized_groups


def normalize_argument_items(parsed: Sequence[Any], text: str, language: str) -> List[str]:
    arguments: List[str] = []
    for item in parsed:
        raw_argument = item if isinstance(item, str) else item.get("argument") if isinstance(item, dict) else None
        argument = normalize_string(raw_argument)
        if argument is None:
            continue
        if find_span_position(text, argument, language) < 0:
            continue
        arguments.append(argument)
    return dedupe_preserve_order(arguments)


def normalize_polarity_predictions(parsed: Sequence[Any], text: str, language: str) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    label_key = "hateful" if language == "zh" else "harmful"
    seen = set()
    for item in parsed:
        if not isinstance(item, dict):
            continue
        target = normalize_string(item.get("target"))
        argument = normalize_string(item.get("argument"))
        if target is None or argument is None:
            continue
        label = normalize_hate_label(item.get(label_key))
        if label is None:
            label = normalize_hate_label(item.get("label"))
        if label is None:
            continue
        if target != "IMPLICIT_TARGET" and find_span_position(text, target, language) < 0:
            continue
        if find_span_position(text, argument, language) < 0:
            continue
        signature = (target, argument, label)
        if signature in seen:
            continue
        seen.add(signature)
        predictions.append(
            {
                "target": target,
                "argument": argument,
                "label": label,
            }
        )
    return predictions


def build_tuple_candidates(groups: Sequence[Dict[str, Any]], group_arguments: Dict[int, List[str]], language: str) -> List[Dict[str, Any]]:
    tuples: List[Dict[str, Any]] = []
    seen = set()
    for group in groups:
        group_id = int(group["group_id"])
        target = group["target"]
        sentences = list(group.get("sentences") or [])
        joined_text = str(group.get("group_text") or " ".join(sentences))
        arguments = group_arguments.get(group_id, [])
        for arg_index, argument in enumerate(arguments, start=1):
            if find_span_position(joined_text, argument, language) < 0:
                continue
            signature = (group_id, target, argument)
            if signature in seen:
                continue
            seen.add(signature)
            tuples.append(
                {
                    "tuple_id": f"g{group_id}_a{arg_index}",
                    "group_id": group_id,
                    "sentences": sentences,
                    "group_text": joined_text,
                    "target": target,
                    "argument": argument,
                }
            )
    return tuples


def build_labeled_predictions_from_polarity_groups(
    groups: Sequence[Dict[str, Any]],
    group_polarity_predictions: Dict[int, List[Dict[str, Any]]],
    language: str,
) -> List[Dict[str, Any]]:
    labeled: List[Dict[str, Any]] = []
    seen = set()
    groups_by_id = {int(group["group_id"]): group for group in groups}
    for group_id, items in group_polarity_predictions.items():
        group = groups_by_id.get(int(group_id))
        if group is None:
            continue
        default_target = normalize_string(group.get("target")) or "IMPLICIT_TARGET"
        for item in items:
            target = normalize_string(item.get("target")) or default_target
            argument = normalize_string(item.get("argument"))
            label = item.get("label")
            if argument is None or label is None:
                continue
            payload = {
                "target": target,
                "argument": argument,
                "label": int(label),
            }
            signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
            if signature in seen:
                continue
            seen.add(signature)
            labeled.append(payload)
    return labeled


def assign_labels_to_candidates(
    candidates: Sequence[Dict[str, Any]],
    polarity_predictions: Sequence[Dict[str, Any]],
    language: str,
) -> List[Dict[str, Any]]:
    by_pair: Dict[Tuple[str, str], int] = {}
    by_argument: Dict[str, int] = {}
    for item in polarity_predictions:
        target_key = normalize_string(item.get("target")) or ""
        argument_key = normalize_string(item.get("argument")) or ""
        label = item.get("label")
        if label is None:
            continue
        by_pair[(target_key, argument_key)] = int(label)
        by_argument[argument_key] = int(label)

    labeled: List[Dict[str, Any]] = []
    for candidate in candidates:
        target = normalize_string(candidate.get("target")) or ""
        argument = normalize_string(candidate.get("argument")) or ""
        label = by_pair.get((target, argument))
        if label is None:
            label = by_argument.get(argument)
        payload = dict(candidate)
        payload["label"] = label
        labeled.append(payload)
    return labeled


def convert_pair_to_prediction(pair: Dict[str, Any], language: str) -> Dict[str, Any]:
    target = None if pair.get("target") == "IMPLICIT_TARGET" else pair.get("target")
    label = pair.get("label")
    if language == "zh":
        return {
            "target": target,
            "argument": pair.get("argument"),
            "hateful": label,
        }
    return {
        "target": target,
        "argument": pair.get("argument"),
        "harmful": label,
    }


def build_phase0_like_row(source_row: Dict[str, Any], predictions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    row = {
        "id": source_row.get("id"),
        "original_id": source_row.get("original_id"),
        "source": source_row.get("source"),
        "language": source_row.get("language"),
        "split": source_row.get("split"),
        "text": source_row.get("text"),
        "meta_info": source_row.get("meta_info") if isinstance(source_row.get("meta_info"), dict) else {},
        "predictions": list(predictions),
    }
    if isinstance(source_row.get("ground_truth"), list):
        row["ground_truth"] = source_row.get("ground_truth")
    return row


def summarize_predictions(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    total_predictions = 0
    unlabeled_predictions = 0
    argumentless_predictions = 0
    for row in rows:
        preds = row.get("predictions") or []
        total_predictions += len(preds)
        for pred in preds:
            if pred.get("argument") is None:
                argumentless_predictions += 1
            if pred.get("harmful") is None and pred.get("hateful") is None:
                unlabeled_predictions += 1
    return {
        "row_count": len(rows),
        "total_predictions": total_predictions,
        "avg_predictions_per_row": round(total_predictions / len(rows), 4) if rows else 0.0,
        "unlabeled_predictions": unlabeled_predictions,
        "argumentless_predictions": argumentless_predictions,
    }


def build_progress_payload(
    *,
    run_id: str,
    args: argparse.Namespace,
    total_rows: int,
    completed_rows: int,
    started_at: datetime,
    phase0_like_rows: Sequence[Dict[str, Any]],
    parse_fail_counts: Dict[str, int],
    phase0_like_path: Path,
    trace_path: Path,
    status: str,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    now_utc = datetime.now(timezone.utc)
    elapsed_sec = max(0.0, (now_utc - started_at).total_seconds())
    rate_rows_per_min = (completed_rows / (elapsed_sec / 60.0)) if elapsed_sec > 0 and completed_rows > 0 else 0.0
    remaining_rows = max(0, total_rows - completed_rows)
    eta_utc: Optional[datetime] = None
    if status not in {"completed", "failed"} and completed_rows > 0 and remaining_rows > 0 and rate_rows_per_min > 0:
        eta_utc = now_utc + timedelta(minutes=(remaining_rows / rate_rows_per_min))
    progress_percent = round((completed_rows * 100.0 / total_rows), 2) if total_rows > 0 else 0.0

    return {
        "run_id": run_id,
        "language": args.language,
        "provider": args.provider,
        "api_format": args.api_format,
        "model": args.model,
        "base_url": args.base_url,
        "output_namespace": args.output_namespace,
        "status": status,
        "error": error,
        "row_count_total": total_rows,
        "completed_rows": completed_rows,
        "remaining_rows": remaining_rows,
        "progress_percent": progress_percent,
        "elapsed_seconds": round(elapsed_sec, 2),
        "elapsed_minutes": round(elapsed_sec / 60.0, 2),
        "rate_rows_per_min": round(rate_rows_per_min, 4),
        "started_at_utc": started_at.isoformat(),
        "updated_at_utc": now_utc.isoformat(),
        "eta_utc": eta_utc.isoformat() if eta_utc is not None else None,
        "phase0_like_path": str(phase0_like_path),
        "trace_path": str(trace_path),
        "parse_fail_counts": dict(parse_fail_counts),
        "prediction_summary": summarize_predictions(phase0_like_rows),
    }


def evaluate_phase0_like_bundle(path: Path, requested_splits: Sequence[str]) -> Dict[str, Any]:
    return evaluate_input(
        input_path=path,
        input_format="phase0_like",
        requested_splits=requested_splits or ["test"],
        scoring_mode="standard",
    )


def build_metric_summary(metrics: Dict[str, Any]) -> Dict[str, Any]:
    overall = metrics.get("scopes", {}).get("overall", {})
    metric_root = overall.get("metrics", {})
    summary: Dict[str, Any] = {}
    for metric_name in ("target", "argument", "ta_pair", "tah_tri"):
        metric_payload = metric_root.get(metric_name, {})
        if not metric_payload or not metric_payload.get("applicable", False):
            continue
        summary[f"{metric_name}_hard_f1"] = float(metric_payload["hard"]["f1"])
        summary[f"{metric_name}_soft_f1"] = float(metric_payload["soft"]["f1"])
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal Dance-style target/argument/polarity API baseline.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--language", choices=["en", "zh"], required=True)
    parser.add_argument("--splits", default="test")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--provider", default="deepseek")
    parser.add_argument("--api-format", default="chat_completions")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--base-url", default="https://api.deepseek.com")
    parser.add_argument("--output-namespace", default="dance-core")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=384)
    parser.add_argument("--retry-max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    requested_splits = [item.strip() for item in str(args.splits or "").split(",") if item.strip()]
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"dance_core_{args.language}_{run_stamp}"
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    inference_cfg = build_inference_cfg(
        provider=args.provider,
        api_format=args.api_format,
        model=args.model,
        api_key_env=args.api_key_env,
        api_key=args.api_key,
        base_url=args.base_url,
        timeout_sec=args.timeout_sec,
        max_retries=args.max_retries,
        retry_backoff_sec=args.retry_backoff_sec,
        max_concurrency=args.max_concurrency,
        reasoning_effort=args.reasoning_effort,
        output_namespace=args.output_namespace,
    )
    api_cfg = resolve_api_backend_config(inference_cfg, api_profiles=inference_cfg.get("api_profiles"))
    client = ApiTextGenerationClient(api_cfg)

    rows = read_jsonl(input_path)
    rows = [
        row
        for row in rows
        if str(row.get("language") or "") == args.language
        and (not requested_splits or str(row.get("split") or "") in requested_splits)
    ]
    if args.max_samples is not None:
        rows = rows[: max(0, int(args.max_samples))]

    phase0_like_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    parse_fail_counts = {"target": 0, "argument": 0, "polarity": 0}
    started_at = datetime.now(timezone.utc)
    phase0_like_path = run_root / "phase0_like.jsonl"
    trace_path = run_root / "trace.jsonl"
    progress_path = run_root / "progress.json"
    total_rows = len(rows)

    if (not args.no_resume) and phase0_like_path.exists():
        phase0_like_rows = read_jsonl(phase0_like_path)
        if trace_path.exists():
            trace_rows = read_jsonl(trace_path)
        existing_progress = load_json(progress_path) or {}
        existing_counts = existing_progress.get("parse_fail_counts")
        if isinstance(existing_counts, dict):
            for key in parse_fail_counts:
                parse_fail_counts[key] = int(existing_counts.get(key) or 0)
        started_at = parse_iso_datetime(existing_progress.get("started_at_utc")) or started_at
        completed_rows = len(phase0_like_rows)
        if completed_rows > 0:
            rows = rows[completed_rows:]
    else:
        write_jsonl(phase0_like_path, [])
        write_jsonl(trace_path, [])

    def flush_progress(status: str, error: Optional[str] = None) -> None:
        payload = build_progress_payload(
            run_id=run_id,
            args=args,
            total_rows=total_rows,
            completed_rows=len(phase0_like_rows),
            started_at=started_at,
            phase0_like_rows=phase0_like_rows,
            parse_fail_counts=parse_fail_counts,
            phase0_like_path=phase0_like_path,
            trace_path=trace_path,
            status=status,
            error=error,
        )
        write_json(progress_path, payload)

    try:
        for start in range(0, len(rows), max(1, args.batch_size)):
            batch_rows = rows[start : start + max(1, args.batch_size)]
            sample_states: List[Dict[str, Any]] = []

            grouping_user_contents = []
            for row in batch_rows:
                text = str(row.get("text") or "")
                sample_states.append(
                    {
                        "row": row,
                        "text": text,
                        "groups": [],
                        "tuple_candidates": [],
                        "raw": {},
                        "parse_ok": {},
                        "errors": {},
                    }
                )
                grouping_user_contents.append(GROUPING_PROMPTS[args.language].replace("{{TEXT}}", text))

            grouping_results = generate_json_arrays_batch(
                client=client,
                user_contents=grouping_user_contents,
                max_new_tokens=args.max_new_tokens,
                retry_max_new_tokens=args.retry_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.enable_thinking,
            )
            for state, result in zip(sample_states, grouping_results):
                state["raw"]["grouping"] = result.raw_text
                state["parse_ok"]["grouping"] = result.parse_ok
                state["errors"]["grouping"] = result.error
                state["attempts_grouping"] = result.attempts
                state["groups"] = normalize_grouping_items(result.items, state["text"], args.language)
                if not result.parse_ok:
                    parse_fail_counts["target"] += 1

            argument_inputs = []
            argument_state_refs: List[Tuple[Dict[str, Any], int]] = []
            for state in sample_states:
                state["group_arguments"] = {}
                for group in state["groups"]:
                    group_text = str(group.get("group_text") or " ".join(group.get("sentences") or []))
                    if not group_text.strip():
                        continue
                    argument_inputs.append(group_text)
                    argument_state_refs.append((state, int(group["group_id"])))

            argument_user_contents = [
                ARGUMENT_PROMPTS[args.language].replace("{{TEXT}}", payload)
                for payload in argument_inputs
            ]
            argument_results = generate_json_arrays_batch(
                client=client,
                user_contents=argument_user_contents,
                max_new_tokens=args.max_new_tokens,
                retry_max_new_tokens=args.retry_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.enable_thinking,
            )
            touched_argument_states = set()
            for ref, result in zip(argument_state_refs, argument_results):
                state, group_id = ref
                touched_argument_states.add(id(state))
                state["raw"]["argument"] = result.raw_text
                state["parse_ok"]["argument"] = result.parse_ok
                state["errors"]["argument"] = result.error
                state["attempts_argument"] = result.attempts
                group_obj = next((group for group in state["groups"] if int(group["group_id"]) == group_id), None)
                group_text = str(group_obj.get("group_text") or "") if group_obj is not None else ""
                state["group_arguments"][group_id] = normalize_argument_items(result.items, group_text, args.language)
                if not result.parse_ok:
                    parse_fail_counts["argument"] += 1

            for state in sample_states:
                if state.get("groups"):
                    state["tuple_candidates"] = build_tuple_candidates(
                        state["groups"],
                        state.get("group_arguments", {}),
                        args.language,
                    )

            for state in sample_states:
                if "argument" not in state["parse_ok"]:
                    state["parse_ok"]["argument"] = True
                    state["errors"]["argument"] = None
                    state["raw"]["argument"] = "[]"
                    state["attempts_argument"] = 0
                    state["tuple_candidates"] = []

            polarity_inputs: List[str] = []
            polarity_state_refs: List[Tuple[Dict[str, Any], int]] = []
            for state in sample_states:
                for group in state["groups"]:
                    group_id = int(group["group_id"])
                    group_text = str(group.get("group_text") or "")
                    if not group_text.strip():
                        continue
                    polarity_inputs.append(group_text)
                    polarity_state_refs.append((state, group_id))

            polarity_user_contents = [
                POLARITY_PROMPTS[args.language].replace("{{TEXT}}", payload)
                for payload in polarity_inputs
            ]

            polarity_results = generate_json_arrays_batch(
                client=client,
                user_contents=polarity_user_contents,
                max_new_tokens=args.max_new_tokens,
                retry_max_new_tokens=args.retry_max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                enable_thinking=args.enable_thinking,
            )
            for ref, result in zip(polarity_state_refs, polarity_results):
                state, group_id = ref
                state["raw"]["polarity"] = result.raw_text
                state["parse_ok"]["polarity"] = result.parse_ok
                state["errors"]["polarity"] = result.error
                state["attempts_polarity"] = result.attempts
                group_obj = next((group for group in state["groups"] if int(group["group_id"]) == group_id), None)
                group_text = str(group_obj.get("group_text") or "") if group_obj is not None else ""
                state.setdefault("group_polarity_predictions", {})[group_id] = normalize_polarity_predictions(
                    result.items,
                    group_text,
                    args.language,
                )
                if not result.parse_ok:
                    parse_fail_counts["polarity"] += 1

            for state in sample_states:
                if state["tuple_candidates"]:
                    labeled_candidates: List[Dict[str, Any]] = []
                    for group in state["groups"]:
                        group_id = int(group["group_id"])
                        group_candidates = [
                            pair for pair in state["tuple_candidates"] if int(pair["group_id"]) == group_id
                        ]
                        labeled_candidates.extend(
                            assign_labels_to_candidates(
                                group_candidates,
                                state.get("group_polarity_predictions", {}).get(group_id, []),
                                args.language,
                            )
                        )
                    state["tuple_candidates"] = labeled_candidates

            for state in sample_states:
                if "polarity" not in state["parse_ok"]:
                    state["parse_ok"]["polarity"] = True
                    state["errors"]["polarity"] = None
                    state["raw"]["polarity"] = "[]"
                    state["attempts_polarity"] = 0
                    state.setdefault("group_polarity_predictions", {})

            for sample_state in sample_states:
                row = sample_state["row"]
                final_labeled = build_labeled_predictions_from_polarity_groups(
                    sample_state["groups"],
                    sample_state.get("group_polarity_predictions", {}),
                    args.language,
                )
                merged_predictions: List[Dict[str, Any]] = []
                seen = set()
                for pair in final_labeled:
                    prediction = convert_pair_to_prediction(pair, args.language)
                    signature = json.dumps(prediction, ensure_ascii=False, sort_keys=True)
                    if signature in seen:
                        continue
                    seen.add(signature)
                    merged_predictions.append(prediction)

                phase0_like_rows.append(build_phase0_like_row(row, merged_predictions))
                trace_rows.append(
                    {
                        "id": row.get("id"),
                        "language": row.get("language"),
                        "split": row.get("split"),
                        "text": row.get("text"),
                        "groups": sample_state["groups"],
                        "tuple_candidates": sample_state["tuple_candidates"],
                        "raw": sample_state["raw"],
                        "parse_ok": sample_state["parse_ok"],
                        "errors": sample_state["errors"],
                        "final_predictions": merged_predictions,
                    }
                )
            batch_phase0_rows = phase0_like_rows[-len(sample_states):]
            batch_trace_rows = trace_rows[-len(sample_states):]
            append_jsonl(phase0_like_path, batch_phase0_rows)
            append_jsonl(trace_path, batch_trace_rows)
            flush_progress(status="running")
    except Exception as exc:
        flush_progress(status="failed", error=str(exc))
        raise
    finally:
        client.shutdown()

    has_ground_truth = all(isinstance(row.get("ground_truth"), list) for row in phase0_like_rows)
    metrics: Dict[str, Any] = {}
    metric_summary: Dict[str, Any] = {}
    if has_ground_truth and phase0_like_rows:
        metrics = evaluate_phase0_like_bundle(phase0_like_path, requested_splits=requested_splits)
        metric_summary = build_metric_summary(metrics)
        write_json(run_root / "metrics.json", metrics)

    summary = {
        "run_id": run_id,
        "language": args.language,
        "input_jsonl": str(input_path),
        "output_root": str(run_root),
        "provider": args.provider,
        "api_format": args.api_format,
        "model": args.model,
        "row_count": len(phase0_like_rows),
        "splits": requested_splits,
        "framework": {
            "name": "dance_core_api",
            "description": "Minimal Dance-style pipeline with local splitting plus target/argument/polarity API agents.",
        },
        "parse_fail_counts": parse_fail_counts,
        "prediction_summary": summarize_predictions(phase0_like_rows),
        "metric_summary": metric_summary,
        "artifacts": {
            "phase0_like_path": str(phase0_like_path),
            "trace_path": str(trace_path),
            "metrics_path": str(run_root / "metrics.json") if metrics else None,
        },
    }
    write_json(run_root / "summary.json", summary)
    flush_progress(status="completed")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
