# IntelliAssist-AI: Enterprise Multimodal RAG Assistant

IntelliAssist-AI is a production-grade Retrieval-Augmented Generation (RAG) platform that combines dense semantic retrieval, sparse lexical search, and dynamic document synthesis to provide verifiable, hallucination-resistant answers from unstructured documents.

---

## Key Features

* **Multi-Format Ingestion:** Ingests and processes PDF, DOCX, and TXT documents into normalized text chunks.
* **Persistent Vector Storage:** Utilizes ChromaDB to store dense BGE embeddings locally with zero-loss restart recovery.
* **Dual-Pipeline Search Comparator:** Runs dense vector search alongside an in-memory, auto-hydrating TF-IDF index, displaying side-by-side relevance scores (0–100%) and latency metrics.
* **Dynamic Summarization:** Reconstructs entire document corpuses across arbitrary page lengths and structures summaries dynamically based on actual content themes.
* **NLP Intelligence Layer:** Performs real-time sentiment analysis, subjectivity scoring via TextBlob, and regex-based query intent classification.
* **Session Resilience:** Includes session state diagnostics, system reset capabilities, and exportable conversation logs.

---

## System Architecture

```text
User Input (Query / Documents)
         │
         ├── Document Processing ──► Text Chunker ──► ChromaDB (Dense)
         │                                       └──► TF-IDF Index (Sparse)
         │
         ├── NLP Analysis ──► Intent Classification & Sentiment Scoring
         │
         └── Retrieval Engine ──► Hybrid Ranker (Dense vs. Sparse)
                                      │
                                      ▼
                                Gemini API (LLM) ──► Verified Response + Citations