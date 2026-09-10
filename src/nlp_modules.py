"""
NLP Module for IntelliAssist AI.
Performs text analysis on user queries and document contexts:
1. Sentiment & Subjectivity Scoring (TextBlob)
2. Regex-based Query Intent Classification
3. Dynamic Multi-mode Document Summarization (Full corpus reconstruction,
   content-driven structural sections, dynamic page tracking, and transport retries)
"""

import re
import time
from typing import Any, Dict, List, Optional
from textblob import TextBlob


class NLPModule:
    """Analyzes textual tone, classifies intent, and handles summarization logic."""

    # Priority-ordered Regex Patterns: More specific intents precede general lookups
    INTENT_PATTERNS = {
        "Comparative/Analytical": [
            r"\b(compare|versus|vs\.?|difference|contrast|trade-?off|similarities)\b",
            r"\b(better|worse|advantage|disadvantage)\b"
        ],
        "Procedural/How-To": [
            r"\b(how to|steps|guide|procedure|process|implement|method|how do i|how can|how does)\b"
        ],
        "Summarization": [
            r"\b(summarize|summary|overview|outline|brief|tl;dr|recap)\b"
        ],
        "Factual/Lookup": [
            r"\b(what is|define|definition|meaning|what are|who|identify|explain|describe|which|when|where|list)\b"
        ]
    }

    @staticmethod
    def analyze_sentiment(text: str) -> Dict[str, Any]:
        """
        Calculates emotional polarity and objectivity metrics.
        Polarity: [-1.0 (Neg) to +1.0 (Pos)]
        Subjectivity: [0.0 (Fact) to 1.0 (Opinion)]
        """
        if not text or not text.strip():
            return {
                "score": 0.0,
                "label": "Neutral",
                "subjectivity_score": 0.0,
                "subjectivity_label": "Objective"
            }

        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 3)
        subjectivity = round(blob.sentiment.subjectivity, 3)

        if polarity > 0.1:
            label = "Positive"
        elif polarity < -0.1:
            label = "Negative"
        else:
            label = "Neutral"

        obj_label = "Subjective" if subjectivity >= 0.5 else "Objective"

        return {
            "score": polarity,
            "label": label,
            "subjectivity_score": subjectivity,
            "subjectivity_label": obj_label
        }

    def classify_intent(self, query: str) -> str:
        """Classifies query intent using Regex word boundaries."""
        cleaned = query.lower().strip()
        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, cleaned):
                    return intent
        return "General Inquiry"

    def generate_summary(
        self,
        rag_engine: Any,
        source_filter: Optional[str] = None,
        mode: str = "Detailed Analysis"
    ) -> str:
        """
        Retrieves all chunks across the full document by explicitly bypassing Chroma's
        default limit (limit=1000), sorts chronologically by page, and synthesizes an end-to-end summary
        with section headers derived dynamically from the document's actual subject matter.
        """
        where_clause = {"source": source_filter} if source_filter else None

        try:
            # Query with an explicit high limit to prevent ChromaDB from truncating at 10 chunks
            db_records = rag_engine.vector_store.collection.get(
                where=where_clause,
                limit=1000,
                include=["documents", "metadatas"]
            )
            raw_docs = db_records.get("documents", [])
            raw_metas = db_records.get("metadatas", [])
        except Exception:
            raw_docs, raw_metas = [], []

        # Fallback to vector search if collection.get() returns empty
        if not raw_docs:
            fallback = rag_engine.vector_store.search(
                "introduction overview core concepts methodology summary conclusion",
                top_k=30,
                source_filter=source_filter
            )
            raw_docs = [item["content"] for item in fallback]
            raw_metas = [item.get("metadata", {}) for item in fallback]

        if not raw_docs:
            return "Unable to retrieve document content for summarization."

        # Pair chunks with metadata and sort sequentially by page number, then chunk index
        paired = []
        pages_detected = set()
        for idx, (text, meta) in enumerate(zip(raw_docs, raw_metas)):
            meta_dict = meta if isinstance(meta, dict) else {}
            try:
                page_num = int(meta_dict.get("page", 0))
            except (ValueError, TypeError):
                page_num = 0

            if page_num > 0:
                pages_detected.add(page_num)

            try:
                chunk_num = int(meta_dict.get("chunk", idx))
            except (ValueError, TypeError):
                chunk_num = idx

            paired.append((page_num, chunk_num, text.strip()))

        # Linear chronological sort: start of document to end of document
        paired.sort(key=lambda x: (x[0], x[1]))
        full_text = "\n\n---\n\n".join([item[2] for item in paired if item[2]])

        # Dynamically determine document scale indicator
        chunk_count = len(paired)
        if pages_detected:
            min_page = min(pages_detected)
            max_page = max(pages_detected)
            doc_scope_desc = f"spanning Pages {min_page} to {max_page} ({chunk_count} total sections)"
        else:
            doc_scope_desc = f"spanning {chunk_count} sections"

        # Generalized mode-specific prompt instructions
        mode_lower = mode.lower()
        if "short" in mode_lower:
            style_instruction = (
                "Provide a concise executive summary in 2-3 focused paragraphs. "
                "The first paragraph should introduce the core purpose, scope, and foundational themes of the document. "
                "The subsequent paragraphs should synthesize the primary methodologies, critical findings, or technical "
                "mechanisms discussed, concluding with the practical impact or key takeaways."
            )
        elif "bullet" in mode_lower:
            style_instruction = (
                "Synthesize the document into a clean, comprehensive bullet-point list using Markdown format (*). "
                "Derive the key takeaway categories directly from the document's actual contents. "
                "Ensure coverage spans foundational definitions, primary methodologies/architectures, core analytical "
                "insights, and concluding implementations or use cases."
            )
        else:  # Detailed Analysis
            style_instruction = (
                "Provide a comprehensive technical summary using clear Markdown headers. "
                "Derive the section headers directly from the major topics and structure actually "
                "present in the document content below — do NOT impose a fixed or unrelated topic "
                "template. Organize the summary into 4-6 logical sections that reflect the document's "
                "own structure and key themes, covering foundational concepts first, then core "
                "technical details, then applications or broader context. Use clear, descriptive "
                "header titles that match the document's actual subject matter."
            )

        prompt = (
            "You are IntelliAssist AI, an expert technical document analyst.\n"
            f"Synthesize the ENTIRE document text provided below {doc_scope_desc}.\n"
            "Do NOT truncate, skip sections, or stop early. Maintain complete fidelity to the source text.\n\n"
            f"{style_instruction}\n\n"
            f"FULL DOCUMENT CONTENT:\n{full_text}\n\n"
            "SUMMARY:"
        )

        # Safely resolve model identifier across RagEngine attribute variants
        model_name = getattr(rag_engine, "model_name", getattr(rag_engine, "model_id", "gemini-2.5-flash"))

        if not getattr(rag_engine, "client", None):
            return "⚠️ Gemini client is not initialized. Please verify your GEMINI_API_KEY."

        # Resilient retry loop against transport/SSL drops
        for attempt in range(3):
            try:
                response = rag_engine.client.models.generate_content(
                    model=model_name,
                    contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    return (
                        "⚠️ **API Quota Exceeded (429 RESOURCE_EXHAUSTED)**: "
                        "The free-tier limit for this model has been reached. "
                        "Please verify your GEMINI_API_KEY or retry in a few moments."
                    )
                if attempt < 2 and ("ssl" in err_str.lower() or "eof" in err_str.lower()):
                    time.sleep(2)
                    continue
                return f"Error during summary generation: {err_str}"

        return "Summary generation timed out after multiple attempts."