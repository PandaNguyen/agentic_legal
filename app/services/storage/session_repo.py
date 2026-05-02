from __future__ import annotations

from collections import deque
from typing import Any


class SessionRepository:
    def __init__(self, *, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._turns: dict[str, deque[dict[str, Any]]] = {}

    async def get_history(self, session_id: str) -> list[dict[str, Any]]:
        return list(self._turns.get(session_id, []))

    async def save_turn(
        self,
        session_id: str,
        user_message: str,
        assistant_message: str,
        *,
        trace_id: str | None = None,
        citations: list[dict[str, Any]] | None = None,
    ) -> None:
        if session_id not in self._turns:
            self._turns[session_id] = deque(maxlen=self.max_turns)
        self._turns[session_id].append(
            {
                "user_message": user_message,
                "assistant_message": assistant_message,
                "trace_id": trace_id,
                "citations": citations or [],
            }
        )

    async def clear_session(self, session_id: str) -> None:
        self._turns.pop(session_id, None)
