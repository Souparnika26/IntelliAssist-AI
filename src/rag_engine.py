"""
RAG Engine Module for IntelliAssist AI.
Supports modern Google AI Studio AQ authorization keys via direct HTTP Bearer headers.
Preserves context retrieval, guardrails, pronoun contextualization, and grounded generation.
"""

import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False


class RagEngine:
    """Core Retrieval-Augmented Generation engine aligned with app.py schema."""

    def __init__(
        self,
        vector_store: Any,
        model_name: str = "gemini-3.6-flash",
        guardrail_threshold: float = 60.0,
        confidence_threshold: Optional[float] = None
    ):
        self.vector_store = vector_store
        self.model_name = "gemini-3.6-flash"
        self.model_id = "gemini-3.6-flash"
        self.guardrail_threshold = (
            confidence_threshold if confidence_threshold is not None else guardrail_threshold
        )

        # 1. Retrieve API key from environment or Streamlit Secrets
        self.api_key = os.environ.get("GEMINI_API_KEY", "").strip()
        if not self.api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
                    self.api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            except Exception:
                pass

        # 2. SDK client fallback for legacy AIza keys only
        self.client = None
        if GENAI_AVAILABLE and self.api_key and not self.api_key.startswith("AQ."):
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

        # Explicit reference pronouns
        self.pronoun_pattern = re.compile(
            r'\b(that|this|it|the former|the latter|above)\b',
            re.IGNORECASE
        )

        # Bare elliptical follow-up tokens
        self.bare_followup_pattern = re.compile(
            r'^(give\s+(a\s+|an\s+)?(concrete\s+)?example|example|elaborate|clarify|explain\s+more|tell\s+me\s+more|more\s+details|why|why\s+so|how\s+so)\??$',
            re.IGNORECASE
        )

    def _extract_prior_subject(self, chat_history: List[Dict[str, Any]], current_query: str) -> Optional[str]:
        """Finds the most recent user turn and strips boilerplate inquiry phrases."""
        last_user_query = None
        for message in reversed(chat_history):
            role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", "")

            if role == "user" and content.strip().lower() != current_query.strip().lower():
                last_user_query = content.strip()
                break

        if not last_user_query:
            return None

        subject = re.sub(
            r'^(what is|what are|explain|define|describe|how does|can you explain)\s+',
            '',
            last_user_query,
            flags=re.IGNORECASE
        )
        return subject.rstrip('?.!').strip()

    def _contextualize_query(self, query: str, chat_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Resolves pronouns and bare elliptical phrases without corrupting standalone questions."""
        if not chat_history:
            return query

        clean_q = query.strip()
        tokens = clean_q.split()

        if len(tokens) <= 12 and self.pronoun_pattern.search(clean_q):
            subject = self._extract_prior_subject(chat_history, clean_q)
            if subject:
                return self.pronoun_pattern.sub(subject, clean_q)

        if self.bare_followup_pattern.match(clean_q):
            subject = self._extract_prior_subject(chat_history, clean_q)
            if subject:
                return f"{clean_q.rstrip('?.!')} for {subject}"

        return query

    def format_grounding_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats retrieved context chunks for LLM ingestion."""
        context_parts = []
        for chunk in chunks:
            meta = chunk.get("metadata", {})
            source = meta.get("source", "Unknown Document")
            page = meta.get("page", "?")
            text = chunk.get("content", "").strip()
            context_parts.append(f"--- Document: {source} (Page {page}) ---\n{text}")
        return "\n\n".join(context_parts)

    def generate_raw_context_view(self, chunks: List[Dict[str, Any]]) -> str:
        """Formats the context blocks for the '🔍 View Grounding Chunks' expander."""
        formatted_segments = []
        for idx, chunk in enumerate(chunks, 1):
            meta = chunk.get("metadata", {})
            src = meta.get("source", "Document")
            pg = meta.get("page", "?")
            rel = chunk.get("confidence_score")
            if rel is None:
                score = chunk.get("score", 0.0)
                rel = round(score * 100.0 if score <= 1.0 else score, 2)
            else:
                rel = round(float(rel), 2)
            content = chunk.get("content", "").strip()
            formatted_segments.append(
                f"**--- Context Segment #{idx} --- Source: {src} | Page: {pg} | Relevance: {rel}%**\n\n{content}"
            )
        return "\n\n---\n\n".join(formatted_segments)

    def _call_aq_api(self, user_content: str, system_instruction: str) -> str:
        """Executes a direct POST request using the AQ key in the Authorization header."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent"
        payload = {
            "contents": [{"parts": [{"text": user_content}]}],
            "systemInstruction": {"parts": [{"text": system_instruction}]},
            "generationConfig": {"temperature": 0.0}
        }

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "X-Goog-Api-Key": self.api_key
            },
            method="POST"
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates and "content" in candidates[0]:
                parts = candidates[0]["content"].get("parts", [])
                return "".join([p.get("text", "") for p in parts])
            return "Unable to parse response from model."

    def generate_response(
        self,
        query: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 8,
        source_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """Executes retrieval, guardrail validation, and grounded generation."""
        search_query = self._contextualize_query(query, chat_history)

        try:
            retrieved_chunks = self.vector_store.search(
                query=search_query,
                top_k=top_k,
                source_filter=source_filter
            )
        except TypeError:
            retrieved_chunks = self.vector_store.search(query=search_query, top_k=top_k)
            if source_filter:
                retrieved_chunks = [
                    c for c in retrieved_chunks
                    if c.get("metadata", {}).get("source") == source_filter
                ]

        if not retrieved_chunks:
            return {
                "answer": "I cannot provide a grounded answer. No relevant context was found in the indexed documents.",
                "confidence": 0.0,
                "guardrail_triggered": True,
                "chunks": [],
                "sources": [],
                "raw_context": ""
            }

        top_chunk = retrieved_chunks[0]
        if "confidence_score" in top_chunk:
            confidence = round(float(top_chunk["confidence_score"]), 2)
        else:
            raw_score = top_chunk.get("score", 0.0)
            confidence = round(raw_score * 100.0 if raw_score <= 1.0 else raw_score, 2)

        raw_context_view = self.generate_raw_context_view(retrieved_chunks)

        # Grounding confidence threshold check
        if confidence < self.guardrail_threshold:
            refusal_msg = (
                f"I cannot provide a grounded answer. The most relevant information in the knowledge "
                f"base only yielded a confidence of {confidence}%, which is below the required "
                f"{self.guardrail_threshold}% grounding threshold."
            )
            return {
                "answer": refusal_msg,
                "confidence": confidence,
                "guardrail_triggered": True,
                "chunks": retrieved_chunks,
                "sources": [],
                "raw_context": raw_context_view
            }

        sources_list = []
        seen_sources = set()
        for c in retrieved_chunks:
            meta = c.get("metadata", {})
            src = meta.get("source", "Unknown Document")
            pg = meta.get("page", "N/A")
            pair = (src, pg)
            if pair not in seen_sources:
                seen_sources.add(pair)
                sources_list.append({"source": src, "page": pg})

        context_str = self.format_grounding_context(retrieved_chunks)
        system_instruction = (
            "You are IntelliAssist AI, an objective, rigorous academic NLP assistant. "
            "Your answers must be strictly grounded ONLY in the provided context segments. "
            "Address specifically the exact concept requested without generic fallbacks or broad topical catalogs. "
            "Always provide exact in-line citations including the filename and page number. "
            "If a technical aspect or specific example is missing from the context for the requested concept, "
            "explicitly acknowledge that it is not present in the material. "
            "Never hallucinate external details or assumptions."
        )

        user_content = (
            f"Context Information:\n{context_str}\n\n"
            f"User Question: {search_query}\n\n"
            f"Instructions: Provide a structured, direct, focused technical answer cited directly from the context above."
        )

        # Route 1: AQ Authorization Key (Primary path)
        if self.api_key and self.api_key.startswith("AQ."):
            try:
                answer_text = self._call_aq_api(user_content, system_instruction)
                return {
                    "answer": answer_text,
                    "confidence": confidence,
                    "guardrail_triggered": False,
                    "chunks": retrieved_chunks,
                    "sources": sources_list,
                    "raw_context": raw_context_view
                }
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8")
                return {
                    "answer": f"⚠️ **Inference API Failure**: {e.code} {err_body}",
                    "confidence": confidence,
                    "guardrail_triggered": False,
                    "chunks": retrieved_chunks,
                    "sources": sources_list,
                    "raw_context": raw_context_view
                }
            except Exception as e:
                return {
                    "answer": f"⚠️ **Inference API Failure**: {str(e)}",
                    "confidence": confidence,
                    "guardrail_triggered": False,
                    "chunks": retrieved_chunks,
                    "sources": sources_list,
                    "raw_context": raw_context_view
                }

        # Route 2: Standard Client SDK Fallback
        if not self.client:
            return {
                "answer": "⚠️ **Configuration Error**: Gemini API key is missing or uninitialized.",
                "confidence": confidence,
                "guardrail_triggered": False,
                "chunks": retrieved_chunks,
                "sources": sources_list,
                "raw_context": raw_context_view
            }

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_content,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    temperature=0.0
                )
            )
            answer_text = response.text if response and response.text else "Unable to generate response from context."
            return {
                "answer": answer_text,
                "confidence": confidence,
                "guardrail_triggered": False,
                "chunks": retrieved_chunks,
                "sources": sources_list,
                "raw_context": raw_context_view
            }
        except Exception as e:
            return {
                "answer": f"⚠️ **Inference API Failure**: {str(e)}",
                "confidence": confidence,
                "guardrail_triggered": False,
                "chunks": retrieved_chunks,
                "sources": sources_list,
                "raw_context": raw_context_view
            }


RAGEngine = RagEngine