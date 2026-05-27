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


def resolve_api_backend_config(inference_cfg: Dict[str, Any]) -> ApiBackendConfig:
    raw = inference_cfg.get("api", {}) if isinstance(inference_cfg.get("api"), dict) else {}
    profile_name = normalize_string(raw.get("profile") or inference_cfg.get("api_profile"))
    provider = (normalize_string(raw.get("provider")) or "openai_compatible").lower()
    api_format = _normalize_api_format(raw.get("format") or raw.get("api_format"))
    model = normalize_string(raw.get("model"))
    if model is None:
        raise ValueError("API backend requires model")

    base_url = normalize_string(raw.get("base_url")) or _default_base_url(provider, api_format)
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
        raise ValueError("API backend requires an API key")

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
        headers={},
        extra_body={},
        organization=normalize_string(raw.get("organization")),
        project=normalize_string(raw.get("project")),
        reasoning_effort=normalize_string(raw.get("reasoning_effort")),
        output_namespace=normalize_string(raw.get("output_namespace")) or "synchain-hate-triplet",
    )


class ApiTextGenerationClient:
    def __init__(self, config: ApiBackendConfig) -> None:
        self.config = config

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
            "User-Agent": "synchain-hate-triplet/1.0",
            self.config.api_key_header: f"{self.config.api_key_prefix}{self.config.api_key}",
        }
        headers.update(self.config.headers)
        return headers

    def _post_json(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.config.base_url}{self.config.endpoint}"
        headers = self._build_headers()
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Optional[Exception] = None
        for attempt in range(self.config.max_retries + 1):
            try:
                request = urllib.request.Request(url=url, data=body, headers=headers, method="POST")
                with urllib.request.urlopen(request, timeout=self.config.timeout_sec) as response:
                    raw = response.read().decode("utf-8")
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    raise RuntimeError("API did not return a JSON object")
                return parsed
            except urllib.error.HTTPError as exc:
                raw = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(f"API HTTP {exc.code}: {raw}")
            except urllib.error.URLError as exc:
                last_error = RuntimeError(f"API URL error: {exc}")
            except Exception as exc:
                last_error = exc
            if attempt < self.config.max_retries:
                time.sleep(self.config.retry_backoff_sec * (attempt + 1))
        if last_error is None:
            raise RuntimeError("API request failed without explicit error")
        raise last_error

    def _extract_text(self, response_obj: Dict[str, Any]) -> str:
        if self.config.api_format == "responses":
            output = response_obj.get("output")
            if isinstance(output, list):
                texts: List[str] = []
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    content = item.get("content")
                    if not isinstance(content, list):
                        continue
                    for part in content:
                        if isinstance(part, dict):
                            text = normalize_string(part.get("text"))
                            if text is not None:
                                texts.append(text)
                return "\n".join(texts).strip()
            return normalize_string(response_obj.get("output_text")) or ""

        choices = response_obj.get("choices")
        if isinstance(choices, list) and choices:
            choice0 = choices[0]
            if isinstance(choice0, dict):
                message = choice0.get("message")
                if isinstance(message, dict):
                    return normalize_string(message.get("content")) or ""
        return ""
