class SessionRepository:
    async def get_history(self, session_id: str) -> list[dict]:
        return []

    async def save_turn(self, session_id: str, user_message: str, assistant_message: str) -> None:
        return None
