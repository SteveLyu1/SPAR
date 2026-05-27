from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence


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


def deep_merge_dict(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


@dataclass
class ApiBackendConfig:
    profile_name: Optional[str]
    provider: str
    api_format: str
    model: str
    base_url: str
    endpoint: str
    api_key: str
    api_key_header: str
    api_key_prefix: str
    timeout_sec: float
    max_retries: int
    retry_backoff_sec: float
    max_concurrency: int
    headers: Dict[str, str]
    extra_body: Dict[str, Any]
    organization: Optional[str]
    project: Optional[str]
    reasoning_effort: Optional[str]
    output_namespace: str


def _default_base_url(provider: str, api_format: str) -> str:
    normalized_provider = str(provider or "").strip().lower()
    normalized_format = str(api_format or "").strip().lower()
    if normalized_provider == "deepseek":
        return "https://api.deepseek.com"
    if normalized_provider == "openai" or normalized_format in {"responses", "openai_responses"}:
        return "https://api.openai.com/v1"
    return "https://api.openai.com/v1"


def _default_endpoint(api_format: str) -> str:
    normalized = str(api_format or "").strip().lower()
    if normalized in {"responses", "openai_responses"}:
        return "/responses"
    return "/chat/completions"


def _normalize_api_format(raw: Optional[str]) -> str:
    text = normalize_string(raw) or "chat_completions"
    normalized = text.lower()
    alias_map = {
        "openai_chat_completions": "chat_completions",
        "openai_compatible_chat": "chat_completions",
        "chat": "chat_completions",
        "chat_completion": "chat_completions",
        "chat_completions": "chat_completions",
        "openai_responses": "responses",
        "responses": "responses",
        "response": "responses",
    }
    return alias_map.get(normalized, normalized)


def _resolve_api_profile_payload(
    inference_cfg: Dict[str, Any],
    api_profiles: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw = inference_cfg.get("api", {}) if isinstance(inference_cfg.get("api"), dict) else {}
    profile_name = normalize_string(raw.get("profile") or raw.get("api_profile") or inference_cfg.get("api_profile"))
    profile_payload: Dict[str, Any] = {}
    if profile_name and isinstance(api_profiles, dict):
        candidate = api_profiles.get(profile_name)
        if isinstance(candidate, dict):
            profile_payload = dict(candidate)
    return deep_merge_dict(profile_payload, raw)


def _sanitize_output_namespace(raw: Any) -> Optional[str]:
    text = normalize_string(raw)
    if text is None:
        return None
    compact = "".join(ch.lower() if ch.isalnum() else "_" for ch in text)
    compact = compact.strip("_")
    while "__" in compact:
        compact = compact.replace("__", "_")
    return compact or None


def derive_api_output_namespace(
    inference_cfg: Dict[str, Any],
    api_profiles: Optional[Dict[str, Any]] = None,
    *,
    default: str = "api",
) -> str:
    merged = _resolve_api_profile_payload(inference_cfg, api_profiles=api_profiles)
    explicit = _sanitize_output_namespace(merged.get("output_namespace"))
    if explicit is not None:
        return explicit

    profile_name = _sanitize_output_namespace(
        merged.get("profile") or merged.get("api_profile") or inference_cfg.get("api_profile")
    )
    if profile_name in {"gpt", "deepseek"}:
        return str(profile_name)

    provider = (normalize_string(merged.get("provider")) or "").lower()
    model = (normalize_string(merged.get("model")) or "").lower()
    base_url = (normalize_string(merged.get("base_url")) or "").lower()

    if "deepseek" in provider or "deepseek" in model or "deepseek" in base_url:
        return "deepseek"
    if provider in {"openai", "openai_compatible"} or model.startswith("gpt") or "openai" in base_url:
        return "gpt"

    provider_namespace = _sanitize_output_namespace(provider)
    return provider_namespace or default


def rewrite_output_namespace_path_text(raw: Any, namespace: str) -> Any:
    if not isinstance(raw, str):
        return raw
    text = raw.strip()
    if not text:
        return raw

    marker = "SPAR/outputs/"
    idx = text.find(marker)
    if idx < 0:
        return raw

    prefix = text[: idx + len(marker)]
    suffix = text[idx + len(marker) :]
    parts = [part for part in suffix.split("/") if part]
    if not parts:
        return raw

    phases = {"phase1", "phase2", "phase3", "phase35", "phase3.5"}
    providers = {"main", "gpt", "deepseek"}
    phase_idx = 0
    if parts[0] in providers:
        if len(parts) < 2:
            return raw
        phase_idx = 1

    phase_name = parts[phase_idx]
    if phase_name not in phases:
        return raw

    rebuilt = [namespace, phase_name] + parts[phase_idx + 1 :]
    return prefix + "/".join(rebuilt)


def resolve_api_backend_config(
    inference_cfg: Dict[str, Any],
    api_profiles: Optional[Dict[str, Any]] = None,
) -> ApiBackendConfig:
    raw = _resolve_api_profile_payload(inference_cfg, api_profiles=api_profiles)
    profile_name = normalize_string(raw.get("profile") or raw.get("api_profile") or inference_cfg.get("api_profile"))
    provider = (normalize_string(raw.get("provider")) or "openai_compatible").lower()
    api_format = _normalize_api_format(raw.get("format") or raw.get("api_format"))
    model = normalize_string(raw.get("model"))
    if model is None:
        raise ValueError("API backend requires inference.api.model")

    base_url = normalize_string(raw.get("base_url"))
    if base_url is None:
        base_url_env = normalize_string(raw.get("base_url_env"))
        if base_url_env:
            base_url = normalize_string(os.getenv(base_url_env))
    if base_url is None:
        base_url = _default_base_url(provider, api_format)
    base_url = base_url.rstrip("/")

    endpoint = normalize_string(raw.get("endpoint")) or _default_endpoint(api_format)
    if not endpoint.startswith("/"):
        endpoint = f"/{endpoint}"

    api_key = normalize_string(raw.get("api_key"))
    api_key_env = normalize_string(raw.get("api_key_env"))
    if api_key is None and api_key_env:
        api_key = normalize_string(os.getenv(api_key_env))
    if api_key is None:
        default_env = "OPENAI_API_KEY" if provider != "deepseek" else "DEEPSEEK_API_KEY"
        api_key = normalize_string(os.getenv(default_env))
    if api_key is None:
        raise ValueError("API backend requires an API key via inference.api.api_key or inference.api.api_key_env")

    raw_headers = raw.get("headers")
    headers: Dict[str, str] = {}
    if isinstance(raw_headers, dict):
        headers = {str(k): str(v) for k, v in raw_headers.items() if normalize_string(k) and v is not None}

    raw_extra_body = raw.get("extra_body")
    extra_body: Dict[str, Any] = raw_extra_body if isinstance(raw_extra_body, dict) else {}

    timeout_sec = max(1.0, safe_float(raw.get("timeout_sec"), default=120.0))
    max_retries = int(max(0, raw.get("max_retries", 3) or 0))
    retry_backoff_sec = max(0.0, safe_float(raw.get("retry_backoff_sec"), default=2.0))
    max_concurrency = int(max(1, raw.get("max_concurrency", 1) or 1))

    return ApiBackendConfig(
        profile_name=profile_name,
        provider=provider,
        api_format=api_format,
        model=model,
        base_url=base_url,
        endpoint=endpoint,
        api_key=api_key,
        api_key_header=normalize_string(raw.get("api_key_header")) or "Authorization",
        api_key_prefix=str(raw.get("api_key_prefix", "Bearer ")),
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
        max_concurrency=max_concurrency,
        headers=headers,
        extra_body=extra_body,
        organization=normalize_string(raw.get("organization")),
        project=normalize_string(raw.get("project")),
        reasoning_effort=normalize_string(raw.get("reasoning_effort")),
        output_namespace=derive_api_output_namespace(inference_cfg, api_profiles=api_profiles),
    )


def render_messages_as_text(messages: Sequence[Dict[str, str]]) -> str:
    lines: List[str] = []
    for message in messages:
        role = normalize_string(message.get("role")) or "user"
        content = str(message.get("content") or "")
        lines.append(f"{role.upper()}: {content}")
    lines.append("ASSISTANT:")
    return "\n".join(lines)


class ApiTextGenerationClient:
    def __init__(self, config: ApiBackendConfig) -> None:
        self.config = config

    def shutdown(self) -> None:
        return

    def generate_from_messages_batch(
        self,
        messages_batch: Sequence[Sequence[Dict[str, str]]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        enable_thinking: bool = False,
    ) -> List[str]:
        if not messages_batch:
            return []
        if self.config.max_concurrency <= 1 or len(messages_batch) <= 1:
            return [
                self._generate_one(
                    messages=messages,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    enable_thinking=enable_thinking,
                )
                for messages in messages_batch
            ]

        results: List[Optional[str]] = [None for _ in messages_batch]
        with ThreadPoolExecutor(max_workers=self.config.max_concurrency) as executor:
            futures = []
            for idx, messages in enumerate(messages_batch):
                futures.append(
                    (
                        idx,
                        executor.submit(
                            self._generate_one,
                            messages=messages,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            enable_thinking=enable_thinking,
                        ),
                    )
                )
            for idx, future in futures:
                results[idx] = future.result()
        return [str(item or "") for item in results]

    def _generate_one(
        self,
        *,
        messages: Sequence[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        enable_thinking: bool,
    ) -> str:
        payload = self._build_payload(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_thinking=enable_thinking,
        )
        response_obj = self._post_json(payload)
        text = self._extract_text(response_obj)
        if normalize_string(text) is None:
            raise ValueError("API returned empty text content")
        return text

    def _build_payload(
        self,
        *,
        messages: Sequence[Dict[str, str]],
        max_new_tokens: Optional[int],
        temperature: Optional[float],
        top_p: Optional[float],
        enable_thinking: bool,
    ) -> Dict[str, Any]:
        if self.config.api_format == "responses":
            payload: Dict[str, Any] = {
                "model": self.config.model,
                "input": [
                    {
                        "role": normalize_string(message.get("role")) or "user",
                        "content": [{"type": "input_text", "text": str(message.get("content") or "")}],
                    }
                    for message in messages
                ],
            }
            if isinstance(max_new_tokens, int) and max_new_tokens > 0:
                payload["max_output_tokens"] = int(max_new_tokens)
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if top_p is not None:
                payload["top_p"] = float(top_p)
            if enable_thinking and self.config.reasoning_effort is not None:
                payload["reasoning"] = {"effort": self.config.reasoning_effort}
            return deep_merge_dict(payload, self.config.extra_body)

        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": normalize_string(message.get("role")) or "user",
                    "content": str(message.get("content") or ""),
                }
                for message in messages
            ],
        }
        if isinstance(max_new_tokens, int) and max_new_tokens > 0:
            payload["max_tokens"] = int(max_new_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if self.config.provider == "deepseek":
            payload["thinking"] = {"type": "enabled" if enable_thinking else "disabled"}
            if enable_thinking and self.config.reasoning_effort is not None:
                payload["reasoning_effort"] = self.config.reasoning_effort
        return deep_merge_dict(payload, self.config.extra_body)

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "spar-api-adapter/1.0",
            self.config.api_key_header: f"{self.config.api_key_prefix}{self.config.api_key}",
        }
        if self.config.organization:
            headers["OpenAI-Organization"] = self.config.organization
        if self.config.project:
            headers["OpenAI-Project"] = self.config.project
        headers.update(self.config.headers)
        return headers

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url}{self.config.endpoint}"
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = self._build_headers()
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                    raw_text = response.read().decode("utf-8")
                decoded = json.loads(raw_text)
                if not isinstance(decoded, dict):
                    raise ValueError(f"API response must be JSON object, got {type(decoded)}")
                return decoded
            except urllib.error.HTTPError as exc:
                error_body = exc.read().decode("utf-8", errors="replace")
                retryable = exc.code in {408, 409, 429} or 500 <= exc.code < 600
                last_error = RuntimeError(f"API HTTP {exc.code}: {error_body[:1200]}")
                if not retryable or attempt >= self.config.max_retries:
                    raise last_error
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"API network error: {exc}")
                if attempt >= self.config.max_retries:
                    raise last_error
            except Exception as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    raise
            time.sleep(self.config.retry_backoff_sec * (2 ** attempt))

        if last_error is not None:
            raise last_error
        raise RuntimeError("API request failed without explicit error")

    def _extract_text(self, response_obj: Dict[str, Any]) -> str:
        direct_output_text = normalize_string(response_obj.get("output_text"))
        if direct_output_text is not None:
            return direct_output_text

        choice_text = self._extract_text_from_choices(response_obj.get("choices"))
        if choice_text is not None:
            return choice_text

        output_text = self._extract_text_from_output_items(response_obj.get("output"))
        if output_text is not None:
            return output_text

        message_text = self._extract_text_from_message(response_obj.get("message"))
        if message_text is not None:
            return message_text

        raise ValueError(f"Unable to extract text from API response keys={sorted(response_obj.keys())}")

    def _extract_text_from_choices(self, choices: Any) -> Optional[str]:
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if isinstance(message, dict):
            return self._extract_text_from_message(message)
        text = normalize_string(first.get("text"))
        if text is not None:
            return text
        return None

    def _extract_text_from_message(self, message: Any) -> Optional[str]:
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        text = self._extract_text_from_content(content)
        if text is not None:
            return text
        reasoning_content = message.get("reasoning_content")
        reasoning_text = self._extract_text_from_content(reasoning_content)
        if reasoning_text is not None:
            return reasoning_text
        reasoning_text = normalize_string(reasoning_content)
        if reasoning_text is not None:
            return reasoning_text
        return normalize_string(message.get("text"))

    def _extract_text_from_output_items(self, output_items: Any) -> Optional[str]:
        if not isinstance(output_items, list):
            return None
        parts: List[str] = []
        for item in output_items:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            text = self._extract_text_from_content(content)
            if text is not None:
                parts.append(text)
                continue
            nested_text = normalize_string(item.get("text"))
            if nested_text is not None:
                parts.append(nested_text)
        if not parts:
            return None
        return "\n".join(parts).strip()

    def _extract_text_from_content(self, content: Any) -> Optional[str]:
        if isinstance(content, str):
            return normalize_string(content)
        if not isinstance(content, list):
            return None
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                text = normalize_string(item)
                if text is not None:
                    parts.append(text)
                continue
            if not isinstance(item, dict):
                continue
            item_type = normalize_string(item.get("type")) or ""
            if item_type in {"text", "output_text", "input_text"}:
                text_value = item.get("text")
                if isinstance(text_value, dict):
                    text_value = text_value.get("value")
                text = normalize_string(text_value)
                if text is not None:
                    parts.append(text)
                    continue
            nested_text = normalize_string(item.get("content"))
            if nested_text is not None:
                parts.append(nested_text)
        if not parts:
            return None
        return "\n".join(parts).strip()
