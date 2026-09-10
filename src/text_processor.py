"""
Text Processor Module for IntelliAssist AI.
Handles document normalization, recursive chunking with sliding overlap,
boilerplate artifact cleansing, and semantic length validation.
"""

import re
from typing import List, Dict, Any


class TextProcessor:
    """Processes, sanitizes, and chunks extracted document content."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_char_length: int = 120
    ):
        """
        Args:
            chunk_size: Target maximum character length per chunk.
            chunk_overlap: Number of characters to overlap between contiguous chunks.
            min_char_length: Minimum character threshold required to keep a chunk.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_char_length = min_char_length

        # Regex to strip document navigation artifacts like "Reply Continue..."
        self.boilerplate_pattern = re.compile(
            r'Reply\s+["\']?Continue["\']?\s+to\s+proceed[^\n.]*',
            re.IGNORECASE
        )

    def clean_text(self, text: str) -> str:
        """Sanitizes whitespace, non-standard line breaks, and known filler patterns."""
        if not text:
            return ""

        # Remove navigational prompts
        cleaned = self.boilerplate_pattern.sub("", text)

        # Normalize line endings and collapse excessive blank spaces
        cleaned = re.sub(r"\r\n|\r", "\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

        return cleaned.strip()

    def split_text(self, text: str) -> List[str]:
        """
        Splits text into chunks using sliding window overlap, prioritizing
        natural boundary breaks (paragraphs and sentences).
        """
        text = self.clean_text(text)
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text] if len(text) >= self.min_char_length else []

        chunks = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = start + self.chunk_size

            # If this is the final slice, capture the remainder
            if end >= text_len:
                chunk = text[start:].strip()
                if len(chunk) >= self.min_char_length:
                    chunks.append(chunk)
                break

            # Try to break on paragraph, sentence, or whitespace boundary
            boundary_candidates = [
                text.rfind("\n\n", start, end),
                text.rfind("\n", start, end),
                text.rfind(". ", start, end),
                text.rfind("? ", start, end),
                text.rfind("! ", start, end),
                text.rfind(" ", start, end),
            ]

            split_pos = max(boundary_candidates)

            # If no boundary found in the range, force split at end
            if split_pos == -1 or split_pos <= start:
                split_pos = end
            else:
                # Include punctuation in current chunk if matched on sentence ending
                if text[split_pos] in {".", "?", "!"}:
                    split_pos += 1

            chunk = text[start:split_pos].strip()

            # Enforce minimum character threshold
            if len(chunk) >= self.min_char_length:
                chunks.append(chunk)

            # Shift window forward with overlap
            start = max(start + 1, split_pos - self.chunk_overlap)

        return chunks

    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Processes raw extracted documents into clean, chunked units with metadata.

        Args:
            documents: List of dicts, each with keys 'content' and 'metadata'.

        Returns:
            List of chunk dicts ready for embedding:
            [{
                "content": str,
                "metadata": {
                    "source": str,
                    "page": int/str,
                    "chunk_id": int
                }
            }]
        """
        processed_chunks: List[Dict[str, Any]] = []
        global_chunk_counter = 0

        for doc in documents:
            raw_content = doc.get("content", "")
            base_metadata = doc.get("metadata", {}).copy()

            chunks = self.split_text(raw_content)

            for chunk_str in chunks:
                chunk_meta = base_metadata.copy()
                chunk_meta["chunk_id"] = global_chunk_counter

                processed_chunks.append({
                    "content": chunk_str,
                    "metadata": chunk_meta
                })
                global_chunk_counter += 1

        return processed_chunks