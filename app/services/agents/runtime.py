from __future__ import annotations

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from app.services.llm.openai_client import OpenAIService


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class AgentRuntime:
    def __init__(self, openai_service: OpenAIService) -> None:
        self.openai_service = openai_service

    async def decide(
        self,
        *,
        system_prompt: str,
        user_context: dict[str, Any],
        output_schema: type[SchemaT],
        fallback: SchemaT,
    ) -> SchemaT:
        user_prompt = json.dumps(user_context, ensure_ascii=False, default=str)
        try:
            data = await self.openai_service.generate_json(system_prompt, user_prompt)
            return output_schema.model_validate(data)
        except (ValidationError, json.JSONDecodeError, TypeError, ValueError):
            return fallback

