from __future__ import annotations

from collections import defaultdict, deque
from typing import Any


class SessionRepository:
    _turns: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=10))

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._turns[session_id])

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        trace_id: str | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        self._turns[session_id].append(
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "trace_id": trace_id,
                "citations": citations or [],
            }
        )
