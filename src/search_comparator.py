"""
Search Comparator Module for IntelliAssist AI.
Compares Dense (Vector Embedding) vs. Sparse (TF-IDF + Cosine Similarity) retrieval pipelines.
Maintains 0-100% normalized scoring, measures retrieval latencies, prevents duplicate chunk accumulation,
provides system reset capabilities, and auto-hydrates the sparse index from persistent ChromaDB storage.
"""

import time
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SearchComparator:
    """Manages sparse TF-IDF indexing and performs side-by-side search comparison."""

    def __init__(self, vector_store: Any = None):
        self.vector_store = vector_store
        self.corpus_chunks: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None

    def reset(self):
        """
        Clears all in-memory corpus chunks and fitted TF-IDF structures.
        Called by app.py when resetting the system.
        """
        self.corpus_chunks = []
        self.vectorizer = None
        self.tfidf_matrix = None

    def rebuild_from_vector_store(self, vector_store: Any = None) -> int:
        """
        Pulls all existing chunks from persistent ChromaDB storage to rebuild
        the in-memory TF-IDF index after application restarts.
        """
        store = vector_store or self.vector_store
        if not store or not hasattr(store, "collection"):
            return 0

        try:
            records = store.collection.get(include=["documents", "metadatas"])
            raw_docs = records.get("documents", [])
            raw_metas = records.get("metadatas", [])

            if not raw_docs:
                return 0

            reconstructed_chunks = [
                {"content": doc, "metadata": meta or {}}
                for doc, meta in zip(raw_docs, raw_metas)
                if doc and doc.strip()
            ]

            self.fit_sparse_index(reconstructed_chunks, accumulate=False)
            return len(reconstructed_chunks)
        except Exception:
            return 0

    def fit_sparse_index(self, chunks: List[Dict[str, Any]], accumulate: bool = True):
        """
        Fits or updates the TF-IDF sparse index with chunk documents,
        preventing duplicate chunk accumulation via composite metadata / content keys.
        """
        if not chunks:
            return

        def _get_chunk_key(chunk: Dict[str, Any]) -> str:
            meta = chunk.get("metadata", {}) or {}
            source = meta.get("source", "")
            page = meta.get("page", "")
            chunk_id = meta.get("chunk_id", meta.get("chunk", ""))

            if source or chunk_id != "":
                return f"{source}::{page}::{chunk_id}"

            content = chunk.get("content", "").strip()
            return str(hash(content))

        if accumulate:
            existing_keys = {_get_chunk_key(c) for c in self.corpus_chunks}
            new_chunks = [c for c in chunks if _get_chunk_key(c) not in existing_keys]
            self.corpus_chunks.extend(new_chunks)
        else:
            self.corpus_chunks = list(chunks)

        corpus_texts = [c.get("content", "") for c in self.corpus_chunks]
        if corpus_texts:
            self.vectorizer = TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=(1, 2)
            )
            self.tfidf_matrix = self.vectorizer.fit_transform(corpus_texts)

    def search(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Primary search method called by app.py.
        Executes TF-IDF sparse search with cosine similarity scoring,
        auto-rebuilding the index from ChromaDB if memory was cleared.
        """
        if (self.vectorizer is None or not self.corpus_chunks) and self.vector_store is not None:
            self.rebuild_from_vector_store(self.vector_store)

        if self.vectorizer is None or self.tfidf_matrix is None or not self.corpus_chunks:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        scored_docs = []
        for idx, score in enumerate(similarities):
            chunk = self.corpus_chunks[idx]
            meta = chunk.get("metadata", {}) or {}

            if source_filter and meta.get("source") != source_filter:
                continue

            conf = round(float(np.clip(score, 0.0, 1.0)) * 100, 2)

            scored_docs.append({
                "content": chunk.get("content", ""),
                "metadata": meta,
                "score": float(score),
                "confidence_score": conf
            })

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

    def sparse_search(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Alias for self.search() to maintain backward compatibility."""
        return self.search(query, top_k=top_k, source_filter=source_filter)

    def compare_retrieval(
        self,
        query: str,
        top_k: int = 3,
        source_filter: Optional[str] = None,
        vector_store: Any = None
    ) -> Dict[str, Any]:
        """
        Executes both Dense and Sparse searches, measuring latency and returning side-by-side results.
        """
        active_store = vector_store or self.vector_store

        # 1. Benchmark Sparse (TF-IDF) Retrieval
        t0 = time.perf_counter()
        sparse_results = self.search(query, top_k=top_k, source_filter=source_filter)
        sparse_time = round((time.perf_counter() - t0) * 1000, 2)

        # 2. Benchmark Dense (Embedding) Retrieval safely
        t1 = time.perf_counter()
        if active_store is not None and hasattr(active_store, "search"):
            dense_results = active_store.search(query, top_k=top_k, source_filter=source_filter)
        else:
            dense_results = []
        dense_time = round((time.perf_counter() - t1) * 1000, 2)

        return {
            "query": query,
            "sparse": {
                "results": sparse_results,
                "latency_ms": sparse_time,
                "count": len(sparse_results)
            },
            "dense": {
                "results": dense_results,
                "latency_ms": dense_time,
                "count": len(dense_results)
            }
        }