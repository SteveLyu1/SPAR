from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .api_adapter import ApiTextGenerationClient, resolve_api_backend_config
from .evaluation import evaluate_input
from .json_utils import (
    normalize_hate_label,
    normalize_string,
    salvage_json_objects_from_array_text,
    sanitize_json_string_control_chars,
)
from .syntax_parser import SentenceInfo, parse_text_to_conll


ARRAY_BLOCK_RE = re.compile(r"\[[\s\S]*\]", flags=re.MULTILINE)

CONLL_DESCRIPTION_EN = (
    "The following is the CoNLL-U style format for the syntactic dependency relationship of this text. "
    "Each row represents a token. The line '# sent_id =' gives the sentence index. ID is the token position within the sentence, "
    "TEXT is the token itself, LEMMA is the base form, POS is the universal part-of-speech tag, TAG is the detailed "
    "part-of-speech tag, FEATS is blank here, HEAD is the head token ID within the same sentence, DEPREL is the "
    "dependency relation label, DEPS is blank here, and MISC is blank here."
)

CONLL_DESCRIPTION_ZH = (
    "下面给出的是该文本的 CoNLL-U 风格句法依存格式。每一行表示一个 token。'# sent_id =' 这一行表示句子编号，"
    "ID 表示该 token 在句内的位置，TEXT 表示词本身，LEMMA 表示词元，POS 表示通用词性，"
    "TAG 表示细粒度词性，FEATS 此处为空，HEAD 表示该 token 在同一句内所依存的中心词 ID，"
    "DEPREL 表示依存关系标签，DEPS 和 MISC 此处为空。"
)

TARGET_PROMPTS = {
    "en": """Given the text "{text}",
"{structure}"
{conll_description}
Based on the syntactic dependency information of the text, analyze all possible targets that are being described, criticized, attacked, or judged in the text. Extract every explicit target as an exact text span from the original text. Use IMPLICIT_TARGET only when the text contains a judgment or attack but no explicit target string.

Output JSON array only:
[
  {{
    "target": "<target>",
    "sentence_id": <integer>
  }}
]""",
    "zh": """给定文本“{text}”，
“{structure}”
{conll_description}
请基于该文本的句法依存信息，分析文本中所有被描述、被评价、被攻击、被判断的 target。请从原文中提取每一个显式 target，必须尽量保持为原文子串。只有当文本中存在评价或攻击但没有明确 target 字符串时，才使用 IMPLICIT_TARGET。

只输出 JSON 数组：
[
  {{
    "target": "<target>",
    "sentence_id": <integer>
  }}
]""",
}

ARGUMENT_PROMPTS = {
    "en": """Given the text "{text}",
"{structure}"
{conll_description}
Based on the syntactic dependency information of the text, analyze all possible opinion, allegation, attack, insulting, or evaluative expressions in the text. Extract every explicit argument span as an exact text span from the original text.

Output JSON array only:
[
  {{
    "argument": "<argument>",
    "sentence_id": <integer>
  }}
]""",
    "zh": """给定文本“{text}”，
“{structure}”
{conll_description}
请基于该文本的句法依存信息，分析文本中所有显式的评价、攻击、指控、贬损、判断性表达。请从原文中提取每一个 argument，必须尽量保持为原文子串。

只输出 JSON 数组：
[
  {{
    "argument": "<argument>",
    "sentence_id": <integer>
  }}
]""",
}

LABEL_PROMPTS = {
    "en": """Given the text "{text}",
"{structure}"
{conll_description}
The following candidate target-argument pairs are extracted from the text:
{pairs_json}

Based on the common sense, the syntactic dependency information, and such target-argument pairs, determine whether each pair expresses a harmful target-directed attack.

Output JSON array only:
[
  {{
    "target": "<target>",
    "argument": "<argument>",
    "harmful": 0
  }}
]""",
    "zh": """给定文本“{text}”，
“{structure}”
{conll_description}
下面是从该文本中抽取出的候选 target-argument 对：
{pairs_json}

请基于常识、句法依存信息以及这些 target-argument 对，判断每一个 pair 是否构成针对 target 的仇恨性或有害攻击。

只输出 JSON 数组：
[
  {{
    "target": "<target>",
    "argument": "<argument>",
    "hateful": 0
  }}
]""",
}

TARGET_SYSTEM_PROMPTS = {
    "en": "You are an AI assistant that helps people find information. Please refine your reply and ensure its accuracy. The reply length is limited to 200 words or less. Return strict JSON array only. Do not output markdown.",
    "zh": "你是一个帮助人们分析信息的 AI 助手。请尽量保证回复准确、精炼。回复长度限制在 200 词以内。只返回严格的 JSON 数组，不要输出 markdown。",
}

ARGUMENT_SYSTEM_PROMPTS = TARGET_SYSTEM_PROMPTS

LABEL_SYSTEM_PROMPTS = {
    "en": "You are a sentiment analysis expert. I will provide you with a text and target-argument pairs mentioned in the text. Please analyze the harmfulness polarity of each target-argument pair in the given text. Return strict JSON array only.",
    "zh": "你是一名仇恨与有害表达分析专家。我会给你一段文本以及文本中的 target-argument 对。请判断每一个 target-argument 对的 hateful/harmful 极性。只返回严格的 JSON 数组。",
}


class JsonArrayResult:
    def __init__(self, raw_text: str, items: List[Any], parse_ok: bool, error: Optional[str], attempts: int) -> None:
        self.raw_text = raw_text
        self.items = items
        self.parse_ok = parse_ok
        self.error = error
        self.attempts = attempts


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


def casefold_for_lookup(text: str, language: str) -> str:
    normalized = html.unescape(text)
    return normalized.casefold() if language == "en" else normalized


def find_span_position(text: str, span: str, language: str) -> int:
    return casefold_for_lookup(text, language).find(casefold_for_lookup(span, language))


def build_messages(system_content: str, user_content: str, retry: bool = False) -> List[Dict[str, str]]:
    if retry:
        system_content = system_content + " Return strict JSON array only. Start with [ and end with ]. No prose."
    return [
        {"role": "system", "content": system_content},
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


def generate_json_arrays_batch(
    *,
    client: ApiTextGenerationClient,
    message_batch: Sequence[Sequence[Dict[str, str]]],
    max_new_tokens: int,
    retry_max_new_tokens: int,
    temperature: float,
    top_p: float,
    enable_thinking: bool,
) -> List[JsonArrayResult]:
    if not message_batch:
        return []
    try:
        first_outputs = client.generate_from_messages_batch(
            message_batch,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
    except Exception:
        first_outputs = []
        for messages in message_batch:
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
    retry_messages: List[Sequence[Dict[str, str]]] = []

    for idx, raw_text in enumerate(first_outputs):
        if raw_text.startswith("__API_ERROR__::"):
            results.append(JsonArrayResult(raw_text, [], False, raw_text, 1))
            continue
        try:
            parsed = extract_json_array(raw_text)
            results.append(JsonArrayResult(raw_text, parsed, True, None, 1))
        except Exception as exc:
            results.append(JsonArrayResult(raw_text, [], False, str(exc), 1))
            retry_indices.append(idx)
            retry_messages.append(
                build_messages(
                    str(message_batch[idx][0].get("content") or ""),
                    str(message_batch[idx][1].get("content") or ""),
                    retry=True,
                )
            )

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
            results[global_idx] = JsonArrayResult(raw_text, [], False, raw_text, 2)
            continue
        try:
            parsed = extract_json_array(raw_text)
            results[global_idx] = JsonArrayResult(raw_text, parsed, True, None, 2)
        except Exception as exc:
            results[global_idx] = JsonArrayResult(raw_text, [], False, str(exc), 2)
    return results


def locate_sentence_for_span(*, text: str, span: str, language: str, sentences: Sequence[SentenceInfo], preferred_sentence_id: Optional[int] = None) -> Optional[int]:
    if preferred_sentence_id is not None:
        for sentence in sentences:
            if sentence.sentence_id == preferred_sentence_id and find_span_position(sentence.text, span, language) >= 0:
                return sentence.sentence_id
    for sentence in sentences:
        if find_span_position(sentence.text, span, language) >= 0:
            return sentence.sentence_id
    if span == "IMPLICIT_TARGET" and sentences:
        return sentences[0].sentence_id
    if find_span_position(text, span, language) >= 0 and sentences:
        return sentences[0].sentence_id
    return None


def normalize_target_items(parsed: Sequence[Any], *, text: str, language: str, sentences: Sequence[SentenceInfo]) -> List[Dict[str, Any]]:
    targets: List[Dict[str, Any]] = []
    seen = set()
    for item in parsed:
        raw_target = item if isinstance(item, str) else item.get("target") if isinstance(item, dict) else None
        target = normalize_string(raw_target)
        if target is None:
            continue
        if target != "IMPLICIT_TARGET" and find_span_position(text, target, language) < 0:
            continue
        preferred_sentence_id = None
        if isinstance(item, dict):
            try:
                preferred_sentence_id = int(item.get("sentence_id")) if item.get("sentence_id") is not None else None
            except (TypeError, ValueError):
                preferred_sentence_id = None
        sentence_id = locate_sentence_for_span(text=text, span=target, language=language, sentences=sentences, preferred_sentence_id=preferred_sentence_id)
        payload = {"target": target, "sentence_id": sentence_id}
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        targets.append(payload)
    return targets


def normalize_argument_items(parsed: Sequence[Any], *, text: str, language: str, sentences: Sequence[SentenceInfo]) -> List[Dict[str, Any]]:
    arguments: List[Dict[str, Any]] = []
    seen = set()
    for item in parsed:
        raw_argument = item if isinstance(item, str) else item.get("argument") if isinstance(item, dict) else None
        argument = normalize_string(raw_argument)
        if argument is None:
            continue
        if find_span_position(text, argument, language) < 0:
            continue
        preferred_sentence_id = None
        if isinstance(item, dict):
            try:
                preferred_sentence_id = int(item.get("sentence_id")) if item.get("sentence_id") is not None else None
            except (TypeError, ValueError):
                preferred_sentence_id = None
        sentence_id = locate_sentence_for_span(text=text, span=argument, language=language, sentences=sentences, preferred_sentence_id=preferred_sentence_id)
        payload = {"argument": argument, "sentence_id": sentence_id}
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        arguments.append(payload)
    return arguments


def sentence_text_by_id(sentences: Sequence[SentenceInfo], sentence_id: Optional[int]) -> str:
    if sentence_id is None:
        return ""
    for sentence in sentences:
        if sentence.sentence_id == sentence_id:
            return sentence.text
    return ""


def choose_best_target_for_argument(*, argument: Dict[str, Any], explicit_targets: Sequence[Dict[str, Any]], text: str, language: str, sentences: Sequence[SentenceInfo]) -> Dict[str, Any]:
    argument_text = str(argument.get("argument") or "")
    same_sentence = [item for item in explicit_targets if item.get("sentence_id") is not None and item.get("sentence_id") == argument.get("sentence_id")]
    candidates = same_sentence or list(explicit_targets)
    if len(candidates) == 1:
        return candidates[0]
    best = candidates[0]
    best_distance: Optional[int] = None
    sentence_scope = sentence_text_by_id(sentences, argument.get("sentence_id")) or text
    argument_pos = find_span_position(sentence_scope, argument_text, language)
    for candidate in candidates:
        target_text = str(candidate.get("target") or "")
        target_pos = find_span_position(sentence_scope, target_text, language)
        if argument_pos < 0 or target_pos < 0:
            target_pos = find_span_position(text, target_text, language)
            argument_pos = find_span_position(text, argument_text, language)
        distance = abs(target_pos - argument_pos) if target_pos >= 0 and argument_pos >= 0 else 10**9
        if best_distance is None or distance < best_distance:
            best = candidate
            best_distance = distance
    return best


def merge_target_argument_candidates(*, targets: Sequence[Dict[str, Any]], arguments: Sequence[Dict[str, Any]], text: str, language: str, sentences: Sequence[SentenceInfo]) -> List[Dict[str, Any]]:
    if not arguments:
        return []
    explicit_targets = [item for item in targets if item.get("target") and item.get("target") != "IMPLICIT_TARGET"]
    has_implicit_target = any(item.get("target") == "IMPLICIT_TARGET" for item in targets)
    merged: List[Dict[str, Any]] = []
    seen = set()
    for argument in arguments:
        if explicit_targets:
            chosen_target = choose_best_target_for_argument(argument=argument, explicit_targets=explicit_targets, text=text, language=language, sentences=sentences)
            target_value = str(chosen_target.get("target") or "IMPLICIT_TARGET")
            sentence_id = argument.get("sentence_id") or chosen_target.get("sentence_id")
        else:
            target_value = "IMPLICIT_TARGET" if has_implicit_target or not explicit_targets else str(explicit_targets[0].get("target") or "IMPLICIT_TARGET")
            sentence_id = argument.get("sentence_id")
        payload = {"target": target_value, "argument": argument.get("argument"), "sentence_id": sentence_id}
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        merged.append(payload)
    return merged


def normalize_polarity_predictions(parsed: Sequence[Any], *, text: str, language: str) -> List[Dict[str, Any]]:
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
        payload = {"target": target, "argument": argument, "label": int(label)}
        signature = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        if signature in seen:
            continue
        seen.add(signature)
        predictions.append(payload)
    return predictions


def assign_labels_to_candidates(candidates: Sequence[Dict[str, Any]], polarity_predictions: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_pair: Dict[Tuple[str, str], int] = {}
    for item in polarity_predictions:
        target_key = normalize_string(item.get("target")) or ""
        argument_key = normalize_string(item.get("argument")) or ""
        label = item.get("label")
        if label is None:
            continue
        by_pair[(target_key, argument_key)] = int(label)
    labeled: List[Dict[str, Any]] = []
    for candidate in candidates:
        target = normalize_string(candidate.get("target")) or ""
        argument = normalize_string(candidate.get("argument")) or ""
        label = by_pair.get((target, argument))
        payload = dict(candidate)
        payload["label"] = label
        labeled.append(payload)
    return labeled


def convert_pair_to_prediction(pair: Dict[str, Any], language: str) -> Dict[str, Any]:
    target = None if pair.get("target") == "IMPLICIT_TARGET" else pair.get("target")
    if language == "zh":
        return {"target": target, "argument": pair.get("argument"), "hateful": pair.get("label")}
    return {"target": target, "argument": pair.get("argument"), "harmful": pair.get("label")}


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
    parser = argparse.ArgumentParser(description="Standalone SynChain-style hate triplet inference.")
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
    parser.add_argument("--output-namespace", default="synchain-hate-triplet")
    parser.add_argument("--timeout-sec", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-backoff-sec", type=float, default=2.0)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=448)
    parser.add_argument("--retry-max-new-tokens", type=int, default=640)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_jsonl).resolve()
    output_root = Path(args.output_root).resolve()
    requested_splits = [item.strip() for item in str(args.splits or "").split(",") if item.strip()]
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_id or f"synchain_hate_triplet_{args.language}_{run_stamp}"
    run_root = output_root / run_id
    run_root.mkdir(parents=True, exist_ok=True)

    inference_cfg = {
        "api_profile": "release_api",
        "api": {
            "profile": "release_api",
            "provider": args.provider,
            "format": args.api_format,
            "model": args.model,
            "base_url": args.base_url,
            "api_key": args.api_key,
            "api_key_env": args.api_key_env,
            "timeout_sec": args.timeout_sec,
            "max_retries": args.max_retries,
            "retry_backoff_sec": args.retry_backoff_sec,
            "max_concurrency": args.max_concurrency,
            "reasoning_effort": args.reasoning_effort,
            "output_namespace": args.output_namespace,
        },
    }
    api_cfg = resolve_api_backend_config(inference_cfg)
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
    phase0_like_path = run_root / "phase0_like.jsonl"
    trace_path = run_root / "trace.jsonl"
    total_rows = len(rows)

    for start in range(0, len(rows), max(1, args.batch_size)):
        batch_rows = rows[start : start + max(1, args.batch_size)]
        sample_states: List[Dict[str, Any]] = []
        target_message_batch: List[Sequence[Dict[str, str]]] = []
        argument_message_batch: List[Sequence[Dict[str, str]]] = []

        for row in batch_rows:
            text = str(row.get("text") or "")
            prompt_text = html.unescape(text)
            parsed_syntax = parse_text_to_conll(prompt_text, args.language)
            conll_description = CONLL_DESCRIPTION_ZH if args.language == "zh" else CONLL_DESCRIPTION_EN
            target_prompt = TARGET_PROMPTS[args.language].format(text=prompt_text, structure=parsed_syntax.conll, conll_description=conll_description)
            argument_prompt = ARGUMENT_PROMPTS[args.language].format(text=prompt_text, structure=parsed_syntax.conll, conll_description=conll_description)
            sample_states.append(
                {
                    "row": row,
                    "text": text,
                    "prompt_text": prompt_text,
                    "syntax_conll": parsed_syntax.conll,
                    "sentences": parsed_syntax.sentences,
                    "targets": [],
                    "arguments": [],
                    "merged_pairs": [],
                    "raw": {},
                    "parse_ok": {},
                    "errors": {},
                }
            )
            target_message_batch.append(build_messages(TARGET_SYSTEM_PROMPTS[args.language], target_prompt))
            argument_message_batch.append(build_messages(ARGUMENT_SYSTEM_PROMPTS[args.language], argument_prompt))

        target_results = generate_json_arrays_batch(client=client, message_batch=target_message_batch, max_new_tokens=args.max_new_tokens, retry_max_new_tokens=args.retry_max_new_tokens, temperature=args.temperature, top_p=args.top_p, enable_thinking=args.enable_thinking)
        for state, result in zip(sample_states, target_results):
            state["raw"]["target"] = result.raw_text
            state["parse_ok"]["target"] = result.parse_ok
            state["errors"]["target"] = result.error
            state["targets"] = normalize_target_items(result.items, text=state["text"], language=args.language, sentences=state["sentences"])
            if not result.parse_ok:
                parse_fail_counts["target"] += 1

        argument_results = generate_json_arrays_batch(client=client, message_batch=argument_message_batch, max_new_tokens=args.max_new_tokens, retry_max_new_tokens=args.retry_max_new_tokens, temperature=args.temperature, top_p=args.top_p, enable_thinking=args.enable_thinking)
        for state, result in zip(sample_states, argument_results):
            state["raw"]["argument"] = result.raw_text
            state["parse_ok"]["argument"] = result.parse_ok
            state["errors"]["argument"] = result.error
            state["arguments"] = normalize_argument_items(result.items, text=state["text"], language=args.language, sentences=state["sentences"])
            state["merged_pairs"] = merge_target_argument_candidates(targets=state["targets"], arguments=state["arguments"], text=state["text"], language=args.language, sentences=state["sentences"])
            if not result.parse_ok:
                parse_fail_counts["argument"] += 1

        label_message_batch: List[Sequence[Dict[str, str]]] = []
        for state in sample_states:
            conll_description = CONLL_DESCRIPTION_ZH if args.language == "zh" else CONLL_DESCRIPTION_EN
            pairs_json = json.dumps([{"target": item.get("target"), "argument": item.get("argument")} for item in state["merged_pairs"]], ensure_ascii=False, indent=2)
            label_prompt = LABEL_PROMPTS[args.language].format(text=state["prompt_text"], structure=state["syntax_conll"], conll_description=conll_description, pairs_json=pairs_json)
            label_message_batch.append(build_messages(LABEL_SYSTEM_PROMPTS[args.language], label_prompt))

        label_results = generate_json_arrays_batch(client=client, message_batch=label_message_batch, max_new_tokens=args.max_new_tokens, retry_max_new_tokens=args.retry_max_new_tokens, temperature=args.temperature, top_p=args.top_p, enable_thinking=args.enable_thinking)

        phase0_batch: List[Dict[str, Any]] = []
        trace_batch: List[Dict[str, Any]] = []
        for state, result in zip(sample_states, label_results):
            state["raw"]["label"] = result.raw_text
            state["parse_ok"]["label"] = result.parse_ok
            state["errors"]["label"] = result.error
            polarity_predictions = normalize_polarity_predictions(result.items, text=state["text"], language=args.language)
            if not result.parse_ok:
                parse_fail_counts["polarity"] += 1
            labeled_pairs = assign_labels_to_candidates(state["merged_pairs"], polarity_predictions)
            final_predictions = [convert_pair_to_prediction(item, args.language) for item in labeled_pairs if item.get("label") is not None and normalize_string(item.get("argument")) is not None]
            phase0_row = build_phase0_like_row(state["row"], final_predictions)
            trace_row = {
                "id": state["row"].get("id"),
                "language": state["row"].get("language"),
                "split": state["row"].get("split"),
                "text": state["text"],
                "syntax_conll": state["syntax_conll"],
                "target_candidates": state["targets"],
                "argument_candidates": state["arguments"],
                "merged_pairs": state["merged_pairs"],
                "raw": state["raw"],
                "parse_ok": state["parse_ok"],
                "errors": state["errors"],
                "final_predictions": final_predictions,
            }
            phase0_batch.append(phase0_row)
            trace_batch.append(trace_row)

        append_jsonl(phase0_like_path, phase0_batch)
        append_jsonl(trace_path, trace_batch)
        phase0_like_rows.extend(phase0_batch)
        trace_rows.extend(trace_batch)
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {run_id} completed {len(phase0_like_rows)}/{total_rows} rows", flush=True)

    metrics = evaluate_input(input_path=phase0_like_path, requested_splits=requested_splits or ["test"])
    metric_summary = build_metric_summary(metrics)
    summary = {
        "run_id": run_id,
        "language": args.language,
        "input_jsonl": str(input_path),
        "output_root": str(run_root),
        "provider": args.provider,
        "api_format": args.api_format,
        "model": args.model,
        "row_count": len(phase0_like_rows),
        "splits": requested_splits or ["test"],
        "framework": {
            "name": "synchain_hate_triplet_release",
            "description": "Standalone SynChain-style syntax-first target and argument extraction with final harmful/hateful classification.",
        },
        "parse_fail_counts": parse_fail_counts,
        "prediction_summary": summarize_predictions(phase0_like_rows),
        "metric_summary": metric_summary,
        "artifacts": {
            "phase0_like_path": str(phase0_like_path),
            "trace_path": str(trace_path),
            "metrics_path": str(run_root / "metrics.json"),
        },
    }
    write_json(run_root / "metrics.json", metrics)
    write_json(run_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
