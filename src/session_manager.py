"""
Session Manager Module for IntelliAssist AI.
Maintains persistent conversational context, turns, and metadata across UI reruns.
"""

import json
from typing import List, Dict, Any, Optional


class SessionManager:
    """Manages chat session history, intent logging, and session exports."""

    def __init__(self, max_history_turns: int = 12):
        self.max_history_turns = max_history_turns
        self.history: List[Dict[str, Any]] = []

    def add_interaction(
        self,
        query: str,
        answer: str,
        confidence: float,
        sources: list,
        intent: str = "General Inquiry",
        sentiment: Optional[Dict[str, Any]] = None,
        guardrail_triggered: bool = False,
        raw_context: str = ""
    ):
        """Appends a completed interaction turn to session history."""
        user_turn = {
            "role": "user",
            "content": query,
            "intent": intent,
            "sentiment": sentiment or {"label": "Neutral", "score": 0.0}
        }
        asst_turn = {
            "role": "assistant",
            "content": answer,
            "confidence": confidence,
            "sources": sources,
            "guardrail": guardrail_triggered,
            "raw_context": raw_context
        }
        self.history.extend([user_turn, asst_turn])

        # Enforce maximum turn limit (2 entries per conversational turn)
        max_entries = self.max_history_turns * 2
        if len(self.history) > max_entries:
            self.history = self.history[-max_entries:]

    def get_ui_chat_history(self) -> List[Dict[str, Any]]:
        """Returns history formatted for Streamlit UI rendering."""
        return list(self.history)

    def clear_session(self):
        """Purges active chat session history."""
        self.history = []

    def export_session_json(self) -> str:
        """Serializes current session logs to JSON string format."""
        return json.dumps({
            "total_turns": len(self.history) // 2,
            "interactions": self.history
        }, indent=2)