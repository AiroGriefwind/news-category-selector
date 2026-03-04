from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

import requests


class LLMClientError(Exception):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return json.loads(stripped)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("response does not contain a valid JSON object")
    return json.loads(stripped[start : end + 1])


@dataclass
class LLMConfig:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.2
    timeout_seconds: int = 45
    max_retries: int = 2


class OpenAICompatibleClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    @classmethod
    def from_env(cls) -> "OpenAICompatibleClient":
        base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
        api_key = os.getenv("LLM_API_KEY", "").strip()
        model = os.getenv("LLM_MODEL", "gpt-4o-mini")
        temperature = float(os.getenv("LLM_TEMPERATURE", "0.2"))
        timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "2"))
        return cls(
            LLMConfig(
                base_url=base_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )
        )

    def _request(self, prompt: str, model: str) -> dict[str, Any]:
        endpoint = self.config.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model,
            "temperature": self.config.temperature,
            "messages": [
                {"role": "system", "content": "你是严谨的新闻分类助手，只能输出 JSON。"},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.api_key}"}
        response = requests.post(
            endpoint, headers=headers, json=payload, timeout=self.config.timeout_seconds
        )
        response.raise_for_status()
        return response.json()

    def call_json(self, prompt: str, model: str | None = None) -> dict[str, Any]:
        if not self.config.api_key:
            raise LLMClientError("missing LLM_API_KEY")
        model_to_use = model or self.config.model

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                data = self._request(prompt=prompt, model=model_to_use)
                content = data["choices"][0]["message"]["content"]
                return _extract_json(content)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(1.2 * (attempt + 1))

        raise LLMClientError(f"llm call failed: {last_error}") from last_error

