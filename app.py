"""
IntelliAssist AI - Production RAG & Document Intelligence Dashboard
Final Integrated Build: BGE-Small + ChromaDB + Gemini Flash + TF-IDF Analytics.

Features:
- Pinned bottom chat input
- Clean tab hierarchy with scrollable message container
- Document-scoped filtering and workspace switching
- Coreference-aware retrieval passing chat_history to resolve anaphora
- Retrieved context inspector expander displaying raw grounding chunks
"""

import os
import re
import tempfile
import streamlit as st
from dotenv import load_dotenv

from src.document_loader import DocumentLoader
from src.text_processor import TextProcessor
from src.vector_store import VectorStore
from src.rag_engine import RagEngine
from src.search_comparator import SearchComparator
from src.nlp_modules import NLPModule
from src.session_manager import SessionManager

# 1. Page Configuration & Environment
load_dotenv()
st.set_page_config(
    page_title="IntelliAssist AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Badges & Layout Styling
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-top: 5px;
    }
    .badge-positive { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .badge-negative { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .badge-neutral  { background-color: #e2e3e5; color: #383d41; border: 1px solid #d6d8db; }
    .badge-intent   { background-color: #cce5ff; color: #004085; border: 1px solid #b8daff; }
    .badge-source   { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Initializing Neural Models & Vector Store...")
def initialize_system():
    """Initializes and caches shared backend services with baseline sample documents."""
    loader = DocumentLoader()
    processor = TextProcessor(chunk_size=850, chunk_overlap=150)
    store = VectorStore(persist_directory="storage/chroma_db")

    rag = RagEngine(vector_store=store, guardrail_threshold=60.0)
    comparator = SearchComparator(vector_store=store)
    nlp = NLPModule()

    sample_dir = "data/sample_docs"
    existing_chunks = []
    if os.path.exists(sample_dir) and store.get_collection_count() == 0:
        for fname in sorted(os.listdir(sample_dir)):
            if not fname.startswith("."):
                raw = loader.process_upload(os.path.join(sample_dir, fname))
                chunks = processor.process_documents(raw)
                existing_chunks.extend(chunks)

        if existing_chunks:
            store.add_chunks(existing_chunks)
            comparator.fit_sparse_index(existing_chunks, accumulate=False)

    return loader, processor, store, rag, comparator, nlp


loader, processor, store, rag, comparator, nlp = initialize_system()

# 2. Session State Initialization
if "session_mgr" not in st.session_state:
    st.session_state.session_mgr = SessionManager(max_history_turns=12)

if "indexed_docs" not in st.session_state:
    sample_dir = "data/sample_docs"
    if os.path.exists(sample_dir):
        st.session_state.indexed_docs = [f for f in sorted(os.listdir(sample_dir)) if not f.startswith(".")]
    else:
        st.session_state.indexed_docs = []

# 3. Sidebar: Ingestion & Control
with st.sidebar:
    st.title("⚡ IntelliAssist AI")
    st.caption("v2.0 | Production Build")
    st.divider()

    st.subheader("📁 Ingest New Knowledge")
    uploaded_files = st.file_uploader(
        "Upload Technical Docs (PDF/DOCX/TXT)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Neural Indexing in progress..."):
                sample_files = [f for f in os.listdir("data/sample_docs") if not f.startswith(".")] if os.path.exists("data/sample_docs") else []
                is_currently_samples_only = all(doc in sample_files for doc in st.session_state.indexed_docs)

                # Workspace Switcher: Purge baseline files when custom documents arrive
                if is_currently_samples_only:
                    store.reset_collection()
                    comparator.reset()
                    st.session_state.session_mgr.clear_session()
                    st.session_state.indexed_docs = []

                all_new_chunks = []
                for file in uploaded_files:
                    suffix = f".{file.name.split('.')[-1]}"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        tmp.write(file.read())
                        tmp_path = tmp.name

                    raw_docs = loader.process_upload(tmp_path)
                    for d in raw_docs:
                        d["metadata"]["source"] = file.name

                    chunks = processor.process_documents(raw_docs)
                    all_new_chunks.extend(chunks)

                    if file.name not in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.append(file.name)
                    os.remove(tmp_path)

                if all_new_chunks:
                    store.add_chunks(all_new_chunks)
                    comparator.fit_sparse_index(all_new_chunks, accumulate=True)
                    st.success(f"Indexed {len(all_new_chunks)} segments.")
                    st.rerun()

    st.divider()

    # Scope Control
    st.subheader("🎯 Scope Control")
    if st.session_state.indexed_docs:
        doc_options = ["Global Intelligence (All)"] + sorted(st.session_state.indexed_docs)
        selected_scope = st.selectbox("Search Target:", doc_options)
        active_filter = None if selected_scope == "Global Intelligence (All)" else selected_scope
    else:
        st.info("No documents active.")
        active_filter = None

    st.divider()

    # Corpus Metrics & Management
    st.subheader("⚙️ System Management")
    st.metric("Corpus Size", f"{store.get_collection_count()} Vectors")

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset System", use_container_width=True, help="Purge all data"):
            store.reset_collection()
            comparator.reset()
            st.session_state.session_mgr.clear_session()
            st.session_state.indexed_docs = []
            st.rerun()
    with c2:
        export_json = st.session_state.session_mgr.export_session_json()
        st.download_button(
            "Export Log",
            data=export_json,
            file_name="intelliassist_session.json",
            mime="application/json",
            use_container_width=True
        )

# 4. Main Application Tabs
tab_chat, tab_compare, tab_summary, tab_analytics = st.tabs([
    "💬 Grounded RAG Chat",
    "🔍 Search Comparison",
    "📄 Automated Summarizer",
    "📊 System Diagnostics"
])

# --- TAB 1: GROUNDED CHAT ---
with tab_chat:
    st.header("Document Reasoning Assistant")
    st.caption("AI responses are strictly grounded in retrieved evidence with calibrated confidence.")

    chat_container = st.container()

    with chat_container:
        for msg in st.session_state.session_mgr.get_ui_chat_history():
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "user":
                    intent_val = msg.get("intent", "General Inquiry")
                    sent_val = msg.get("sentiment") or {}
                    tone_val = sent_val.get("label", "Neutral")
                    st.markdown(
                        f"<span class='metric-badge badge-intent'>Intent: {intent_val}</span>"
                        f"<span class='metric-badge badge-{tone_val.lower()}'>Tone: {tone_val}</span>",
                        unsafe_allow_html=True
                    )
                elif msg["role"] == "assistant":
                    if msg.get("guardrail"):
                        st.warning("⚠️ Hallucination Guardrail: Document confidence fell below the 60% threshold.")
                    else:
                        st.markdown(f"**Verified Confidence:** `{msg.get('confidence', 0.0)}%`")
                        if msg.get("sources"):
                            src_str = " | ".join([f"{s['source']} (P.{s.get('page', 'N/A')})" for s in msg["sources"]])
                            st.markdown(f"<span class='metric-badge badge-source'>Sources: {src_str}</span>", unsafe_allow_html=True)
                        if msg.get("raw_context"):
                            with st.expander("🔍 View Grounding Chunks"):
                                st.markdown(msg["raw_context"])

# --- TAB 2: SEARCH COMPARATOR ---
with tab_compare:
    st.header("🔍 Neural vs. Lexical Search Diagnostic")
    st.markdown(
        "Empirical side-by-side diagnostic comparing dense semantic embeddings "
        "(**BGE-Small**) against sparse term-frequency matching (**TF-IDF**)."
    )
    st.caption(
        "ℹ️ **Rank Match Agreement** measures the token-level intersection over union (IoU) "
        "between the dense and sparse chunks returned at the exact same rank position."
    )

    col_q, col_k = st.columns([4, 1])
    with col_q:
        cmp_query = st.text_input(
            "Comparison Test Query:",
            placeholder="Enter a keyword, technical acronym (e.g., 'CLIR'), or semantic paraphrase...",
            key="tab2_query_input"
        )
    with col_k:
        top_k_cmp = st.slider("Top K", min_value=1, max_value=5, value=3, step=1, key="tab2_top_k")

    run_diagnostic = st.button("Execute Dual-Search Diagnostic", type="primary", use_container_width=True)

    if run_diagnostic:
        clean_cmp_query = cmp_query.strip()
        if not clean_cmp_query:
            st.warning("⚠️ Please enter a valid query before executing the diagnostic.")
        else:
            with st.spinner("Executing dense vector retrieval and sparse lexical search..."):
                # 1. Retrieve Dense (BGE-Small) Results via store
                try:
                    dense_results = store.search(
                        query=clean_cmp_query,
                        top_k=top_k_cmp
                    )
                except Exception as e:
                    st.error(f"Dense vector search failed: {e}")
                    dense_results = []

                # 2. Retrieve Sparse (TF-IDF) Results via comparator
                try:
                    sparse_results = comparator.search_sparse(
                        query=clean_cmp_query,
                        top_k=top_k_cmp
                    )
                except AttributeError:
                    # Fallback if method is named search()
                    try:
                        sparse_results = comparator.search(query=clean_cmp_query, top_k=top_k_cmp)
                    except Exception as e:
                        st.error(f"Sparse TF-IDF search failed: {e}")
                        sparse_results = []
                except Exception as e:
                    st.error(f"Sparse TF-IDF search failed: {e}")
                    sparse_results = []

            st.markdown("---")

            # 3. Render Rank Comparison Cards
            max_ranks = max(len(dense_results), len(sparse_results))
            if max_ranks == 0:
                st.info("No matching content found across either search mechanism for this query.")
            else:
                for rank_idx in range(max_ranks):
                    rank_num = rank_idx + 1

                    dense_chunk = dense_results[rank_idx] if rank_idx < len(dense_results) else None
                    sparse_chunk = sparse_results[rank_idx] if rank_idx < len(sparse_results) else None

                    dense_text = dense_chunk.get("content", "").strip() if dense_chunk else ""
                    sparse_text = sparse_chunk.get("content", "").strip() if sparse_chunk else ""

                    # Calculate pointwise Rank Match Agreement (Token Jaccard at this rank)
                    if dense_text and sparse_text:
                        dense_tokens = set(re.findall(r'\b\w+\b', dense_text.lower()))
                        sparse_tokens = set(re.findall(r'\b\w+\b', sparse_text.lower()))
                        union = dense_tokens.union(sparse_tokens)
                        agreement_score = (len(dense_tokens.intersection(sparse_tokens)) / len(union)) * 100.0 if union else 0.0
                    else:
                        agreement_score = 0.0

                    # Relabeled Section Header
                    st.markdown(f"### Rank #{rank_num} | Rank Match Agreement: `{agreement_score:.1f}%`")

                    col_dense, col_sparse = st.columns(2)

                    # --- Dense Column ---
                    with col_dense:
                        st.markdown("#### 🔹 BGE Semantic (Dense)")
                        if dense_chunk:
                            d_meta = dense_chunk.get("metadata", {})
                            d_src = d_meta.get("source", "Unknown Document")
                            d_page = d_meta.get("page", "?")

                            raw_d_score = dense_chunk.get("confidence_score")
                            if raw_d_score is None:
                                raw_d_score = dense_chunk.get("score", 0.0)
                                d_conf = round(raw_d_score * 100.0 if raw_d_score <= 1.0 else raw_d_score, 2)
                            else:
                                d_conf = round(float(raw_d_score), 2)

                            st.markdown(f"**Confidence:** `{d_conf}%`")
                            st.caption(f"**Source:** {d_src} (Page {d_page})")
                            st.info(dense_text)
                        else:
                            st.markdown("*No dense vector result returned.*")

                    # --- Sparse Column ---
                    with col_sparse:
                        st.markdown("#### 🔸 TF-IDF Lexical (Sparse)")
                        if sparse_chunk:
                            s_meta = sparse_chunk.get("metadata", {})
                            s_src = s_meta.get("source", "Unknown Document")
                            s_page = s_meta.get("page", "?")

                            raw_s_score = sparse_chunk.get("score", 0.0)
                            s_conf = round(raw_s_score * 100.0 if raw_s_score <= 1.0 else raw_s_score, 2)

                            st.markdown(f"**TF-IDF Score:** `{s_conf}%`")
                            st.caption(f"**Source:** {s_src} (Page {s_page})")
                            st.success(sparse_text)
                        else:
                            st.markdown("*No lexical result returned.*")

                    st.markdown("---")

# --- TAB 3: SUMMARIZER ---
with tab_summary:
    st.header("Executive Summary Generator")
    if not st.session_state.indexed_docs:
        st.warning("Please index documents to enable summarization.")
    else:
        col_s1, col_s2 = st.columns([3, 2])
        with col_s1:
            target_sum = st.selectbox("Select file to summarize:", sorted(st.session_state.indexed_docs))
        with col_s2:
            summary_mode = st.radio(
                "Summary Style:",
                ["Short Overview", "Detailed Analysis", "Bullet Points"],
                horizontal=True
            )

        if st.button("Synthesize Summary", use_container_width=True, type="primary"):
            with st.spinner(f"Synthesizing {summary_mode.lower()} for {target_sum}..."):
                summary = nlp.generate_summary(rag, source_filter=target_sum, mode=summary_mode)
                st.markdown(summary)

# --- TAB 4: SYSTEM DASHBOARD ---
with tab_analytics:
    st.header("Corpus Intelligence Dashboard")
    col_a1, col_a2, col_a3, col_a4 = st.columns(4)
    col_a1.metric("Embedding Model", "BGE-Small v1.5")
    col_a2.metric("Inference Engine", "Gemini Flash")
    col_a3.metric("Guardrail Floor", "60.0% Confidence")
    col_a4.metric("Indexed Documents", len(st.session_state.indexed_docs))

    st.divider()
    if st.session_state.indexed_docs:
        st.subheader("Active Knowledge Inventory")
        for doc in sorted(st.session_state.indexed_docs):
            st.markdown(f"- ✅ `Ready` | 📄 `{doc}`")
    else:
        st.info("Corpus is empty. Upload files in the sidebar.")

# 5. Global Bottom-Pinned Chat Input
query = st.chat_input("Ask a technical question about your documents...")

if query:
    intent = nlp.classify_intent(query)
    sentiment = nlp.analyze_sentiment(query)

    chat_history = st.session_state.session_mgr.get_ui_chat_history()

    with chat_container:
        with st.chat_message("user"):
            st.markdown(query)
            st.markdown(
                f"<span class='metric-badge badge-intent'>Intent: {intent}</span>"
                f"<span class='metric-badge badge-{sentiment['label'].lower()}'>Tone: {sentiment['label']}</span>",
                unsafe_allow_html=True
            )

        with st.chat_message("assistant"):
            with st.spinner("Analyzing document context..."):
                search_filter = active_filter if active_filter != "Global Intelligence (All)" else None
                response = rag.generate_response(
                    query=query,
                    chat_history=chat_history,
                    top_k=8,
                    source_filter=search_filter
                )
                st.markdown(response["answer"])

                if response["guardrail_triggered"]:
                    st.warning("⚠️ Hallucination Guardrail: Document confidence fell below the 60% threshold.")
                else:
                    st.markdown(f"**Verified Confidence:** `{response['confidence']}%`")
                    if response["sources"]:
                        src_str = " | ".join([f"{s['source']} (P.{s.get('page', 'N/A')})" for s in response["sources"]])
                        st.markdown(f"<span class='metric-badge badge-source'>Sources: {src_str}</span>", unsafe_allow_html=True)
                    if response.get("raw_context"):
                        with st.expander("🔍 View Grounding Chunks"):
                            st.markdown(response["raw_context"])

    st.session_state.session_mgr.add_interaction(
        query=query,
        answer=response["answer"],
        confidence=response["confidence"],
        sources=response["sources"],
        intent=intent,
        sentiment=sentiment,
        guardrail_triggered=response["guardrail_triggered"],
        raw_context=response.get("raw_context", "")
    )
    st.rerun()