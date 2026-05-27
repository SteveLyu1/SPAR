from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def sanitize_json_string_control_chars(text: str) -> str:
    chars: List[str] = []
    in_string = False
    escaped = False
    for ch in text:
        if in_string:
            if escaped:
                chars.append(ch)
                escaped = False
                continue
            if ch == "\\":
                chars.append(ch)
                escaped = True
                continue
            if ch == '"':
                chars.append(ch)
                in_string = False
                escaped = False
                continue
            if ord(ch) < 32:
                chars.append(" ")
                continue
            chars.append(ch)
            continue
        chars.append(ch)
        if ch == '"':
            in_string = True
            escaped = False
    return "".join(chars)


def salvage_json_objects_from_array_text(text: str) -> List[Dict[str, Any]]:
    if not text:
        return []
    start = text.find("[")
    if start < 0:
        return []

    recovered: List[Dict[str, Any]] = []
    object_start: Optional[int] = None
    brace_depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(text)):
        ch = text[idx]
        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            escaped = False
            continue
        if ch == "{":
            if brace_depth == 0:
                object_start = idx
            brace_depth += 1
            continue
        if ch == "}":
            if brace_depth <= 0:
                continue
            brace_depth -= 1
            if brace_depth == 0 and object_start is not None:
                candidate = text[object_start : idx + 1]
                try:
                    parsed = json.loads(sanitize_json_string_control_chars(candidate))
                except Exception:
                    object_start = None
                    continue
                if isinstance(parsed, dict):
                    recovered.append(parsed)
                object_start = None
    return recovered


def normalize_hate_label(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value)
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "hate", "hateful", "harmful", "true"}:
        return 1
    if text in {"0", "non-hate", "non_hate", "non-harmful", "non_harmful", "false"}:
        return 0
    return None
