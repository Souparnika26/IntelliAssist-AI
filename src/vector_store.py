"""
Vector Store Module for IntelliAssist AI.
Manages local BERT embedding generation (BAAI/bge-small-en-v1.5),
persistent indexing in ChromaDB, and semantic search.

Calibration Logic: Uses a Logistic Sigmoid function to map raw cosine
similarity to an intuitive 0-100% confidence scale, providing a
high-contrast decision boundary for RAG guardrails.
"""

import os
import math
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


class VectorStore:
    """Manages dense embeddings and vector storage using ChromaDB."""

    COLLECTION_NAME = "intelliassist_corpus"
    DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"

    def __init__(
        self,
        persist_directory: str = "storage/chroma_db",
        model_name: str = DEFAULT_MODEL_NAME
    ):
        self.persist_directory = os.path.abspath(persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        print(f"🔄 Initializing BERT Model: {model_name}...")
        # Loads to CPU - optimized for MacBook Air M-series
        self.embedding_model = SentenceTransformer(model_name)

        print(f"🔄 Connecting to Persistent ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )

        self.collection = self.client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 64) -> int:
        """
        Computes BERT embeddings and persists them into ChromaDB in batches.
        Generates unique IDs to ensure no data collisions.
        """
        if not chunks:
            return 0

        total_chunks = len(chunks)
        print(f"📦 Neural Indexing: Processing {total_chunks} segments...")

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            texts = [c["content"] for c in batch]
            metadatas = [c["metadata"] for c in batch]

            # Professional unique IDs: filename_page_chunkid
            ids = [
                f"{c['metadata'].get('source', 'doc')}_p{c['metadata'].get('page', 1)}_c{c['metadata'].get('chunk_id', idx)}"
                for idx, c in enumerate(batch, start=i)
            ]

            # Generate normalized 384-d BERT embeddings
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,
                normalize_embeddings=True
            ).tolist()

            self.collection.upsert(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )

        print(f"✅ Success: Indexed {total_chunks} chunks into Vector DB.")
        return total_chunks

    def search(
        self,
        query: str,
        top_k: int = 4,
        source_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes semantic search with Logistic Sigmoid Calibration.
        Tuned for BGE-Small-v1.5 to provide sharp separation between noise and signal.
        """
        if not query or not query.strip():
            return []

        # 1. Encode query
        query_embedding = self.embedding_model.encode(
            [query],
            normalize_embeddings=True
        ).tolist()

        # 2. Build metadata filter for Document Scoping
        where_clause = {"source": source_filter} if source_filter else None

        # 3. Query ChromaDB
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results.get("documents") and results["documents"][0]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc_text, meta, dist in zip(docs, metas, distances):
                # --- SIGMOID CALIBRATION LOGIC ---
                # Raw Cosine Similarity = 1.0 - distance
                similarity = 1.0 - dist

                # Sigmoid parameters tuned for BGE-Small-v1.5
                # Midpoint (k): 0.72 (the point where confidence becomes 50%)
                # Steepness (s): 24.0 (controls the contrast of the S-curve)
                midpoint = 0.72
                steepness = 24.0

                # Logistic function: 1 / (1 + exp(-s * (x - k)))
                calibrated = 1.0 / (1.0 + math.exp(-steepness * (similarity - midpoint)))

                confidence_pct = round(calibrated * 100.0, 2)

                formatted_results.append({
                    "content": doc_text,
                    "metadata": meta,
                    "distance": round(dist, 4),
                    "confidence_score": confidence_pct
                })

        # Return results sorted by the new calibrated confidence
        return sorted(formatted_results, key=lambda x: x['confidence_score'], reverse=True)

    def reset_collection(self):
        """Clears the collection for a fresh index rebuild."""
        try:
            self.client.delete_collection(self.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"🧹 Vector Store Reset Successful.")
        except Exception as e:
            print(f"⚠️ Warning during reset: {e}")

    def get_collection_count(self) -> int:
        """Returns total vector count currently persisted."""
        return self.collection.count()