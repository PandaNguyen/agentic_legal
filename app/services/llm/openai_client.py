from __future__ import annotations

import json
from typing import Any
from openai import AsyncOpenAI
from app.core.config import Settings


class OpenAIService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = AsyncOpenAI(api_key=settings.openai_api_key)

    async def embed(self, text: str) -> list[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )
        return response.data[0].embedding

    async def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = await self.client.responses.create(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            text={"format": {"type": "json_object"}},
        )
        return json.loads(response.output_text)

    async def generate_text(self, system_prompt: str, user_prompt: str) -> str:
        response = await self.client.responses.create(
            model=self.settings.openai_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.output_text
