from __future__ import annotations

import json
from typing import Any
from openai import APIStatusError, AsyncOpenAI
from app.core.config import Settings


class OpenAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model = settings.openai_llm_model or settings.openai_model
        client_kwargs: dict[str, Any] = {"api_key": settings.openai_api_key}
        if settings.openai_base_url:
            client_kwargs["base_url"] = settings.openai_base_url
        self.client = AsyncOpenAI(**client_kwargs)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if self._uses_chat_completions_api():
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            try:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
            except APIStatusError:
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
            content = response.choices[0].message.content or "{}"
            return _loads_json_object(content)

        response = await self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        if self._uses_chat_completions_api():
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
            return response.choices[0].message.content or ""

        response = await self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text

    def _uses_chat_completions_api(self) -> bool:
        api_style = self.settings.openai_api_style.strip().lower()
        if api_style in {"chat_completions", "chat-completions", "chat"}:
            return True
        return bool(self.settings.openai_base_url and "api.openai.com" not in self.settings.openai_base_url)


def _loads_json_object(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise
