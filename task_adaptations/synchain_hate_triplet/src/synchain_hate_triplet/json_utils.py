from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


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


def normalize_hate_label(value: Any) -> Optional[int]:
    if value in {1, "1", True, "true", "True", "yes", "YES"}:
        return 1
    if value in {0, "0", False, "false", "False", "no", "NO"}:
        return 0
    return None


def sanitize_json_string_control_chars(text: str) -> str:
    if not text:
        return text
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
                continue
            if ch in {"\n", "\r", "\t"} or ord(ch) < 32:
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
