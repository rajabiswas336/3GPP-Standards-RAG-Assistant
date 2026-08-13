"""
Streamlit demo for the 3GPP RAG chatbot.
Run with: streamlit run app.py

Deliberately surfaces the guardrail internals (retrieved clauses,
confidence, grounding report) rather than hiding them -- for an
"anti-hallucination" project, showing your work IS the product.
"""
import streamlit as st
from rag_pipeline import GPPRagPipeline

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="3GPP Spec Assistant — RAG Chatbot",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a premium dark-themed look ────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables */
:root {
    --accent: #6C63FF;
    --accent-light: #8B83FF;
    --accent-glow: rgba(108, 99, 255, 0.15);
    --success: #00D68F;
    --warning: #FFB547;
    --danger: #FF6B6B;
    --bg-dark: #0E1117;
    --bg-card: #1A1D27;
    --bg-card-hover: #222636;
    --text-primary: #E8E8E8;
    --text-secondary: #9CA3AF;
    --border: #2D3348;
}

/* Global font */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Header area */
.main-header {
    background: linear-gradient(135deg, #1a1d27 0%, #0e1117 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -20%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}

.main-header h1 {
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, #fff 0%, var(--accent-light) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.3rem;
}

.main-header p {
    color: var(--text-secondary);
    font-size: 0.95rem;
    line-height: 1.5;
}

/* Stat badges */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
    flex-wrap: wrap;
}

.stat-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(108, 99, 255, 0.08);
    border: 1px solid rgba(108, 99, 255, 0.2);
    padding: 0.35rem 0.8rem;
    border-radius: 20px;
    font-size: 0.78rem;
    color: var(--accent-light);
    font-weight: 500;
}

/* Chat messages styling */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 0.75rem;
    transition: border-color 0.2s ease;
}

[data-testid="stChatMessage"]:hover {
    border-color: var(--accent);
}

/* Expander styling */
.streamlit-expanderHeader {
    font-weight: 500;
    font-size: 0.85rem;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141D 0%, #0E1117 100%);
}

section[data-testid="stSidebar"] .stMarkdown h2 {
    font-size: 1rem;
    font-weight: 600;
    color: var(--accent-light);
    letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Architecture card in sidebar */
.arch-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem;
    margin-bottom: 0.5rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
}

.arch-card:hover {
    transform: translateY(-1px);
    border-color: var(--accent);
}

.arch-card .label {
    color: var(--text-secondary);
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 0.2rem;
}

.arch-card .value {
    color: var(--text-primary);
    font-size: 0.85rem;
    font-weight: 500;
}

/* Guardrail pipeline visualization */
.pipeline-step {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(45, 51, 72, 0.5);
}

.pipeline-step:last-child {
    border-bottom: none;
}

.pipeline-icon {
    width: 28px;
    height: 28px;
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
}

.pipeline-text {
    font-size: 0.78rem;
    color: var(--text-secondary);
}

.pipeline-text strong {
    color: var(--text-primary);
    display: block;
    font-size: 0.82rem;
}

/* Result badges */
.result-pass {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(0, 214, 143, 0.1);
    border: 1px solid rgba(0, 214, 143, 0.3);
    color: var(--success);
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}

.result-fail {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255, 107, 107, 0.1);
    border: 1px solid rgba(255, 107, 107, 0.3);
    color: var(--danger);
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}

.result-warn {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    background: rgba(255, 181, 71, 0.1);
    border: 1px solid rgba(255, 181, 71, 0.3);
    color: var(--warning);
    padding: 0.3rem 0.7rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}

/* Chunk card */
.chunk-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
}

.chunk-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}

.chunk-citation {
    color: var(--accent-light);
    font-size: 0.82rem;
    font-weight: 600;
}

.chunk-score {
    font-size: 0.75rem;
    color: var(--text-secondary);
    background: rgba(108, 99, 255, 0.1);
    padding: 0.15rem 0.5rem;
    border-radius: 12px;
}

.chunk-text {
    color: var(--text-secondary);
    font-size: 0.8rem;
    line-height: 1.6;
    max-height: 150px;
    overflow-y: auto;
}

/* Grounding row */
.grounding-row {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    padding: 0.4rem 0;
    border-bottom: 1px solid rgba(45, 51, 72, 0.3);
    font-size: 0.8rem;
}

.grounding-row:last-child {
    border-bottom: none;
}

.grounding-score {
    flex-shrink: 0;
    width: 45px;
    text-align: center;
    font-weight: 600;
    font-size: 0.75rem;
    padding: 0.15rem 0;
    border-radius: 6px;
}

.grounding-text {
    color: var(--text-secondary);
    line-height: 1.4;
}

/* Hide default Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Smooth scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: var(--border);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: var(--accent);
}
</style>
""", unsafe_allow_html=True)


# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>📡 3GPP Standards RAG Assistant</h1>
    <p>
        Grounded Q&A over 3GPP telecom specifications with citation-level
        retrieval and multi-layer hallucination guardrails. Answers only
        from indexed specs — never from the model's own knowledge.
    </p>
    <div class="stat-row">
        <span class="stat-badge">🔒 Confidence-Gated Refusal</span>
        <span class="stat-badge">📎 Clause-Level Citations</span>
        <span class="stat-badge">🧪 Faithfulness Self-Critique</span>
        <span class="stat-badge">📊 Lexical Grounding Check</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏗️ System Architecture")

    st.markdown("""
    <div class="arch-card">
        <div class="label">Embedding Model</div>
        <div class="value">MiniLM-L6-v2 (384d)</div>
    </div>
    <div class="arch-card">
        <div class="label">Reranker</div>
        <div class="value">cross-encoder / ms-marco-MiniLM</div>
    </div>
    <div class="arch-card">
        <div class="label">Generation</div>
        <div class="value">Groq-hosted LLM · temp 0.1</div>
    </div>
    <div class="arch-card">
        <div class="label">Vector Store</div>
        <div class="value">ChromaDB (cosine space, persistent)</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 🛡️ Anti-Hallucination Pipeline")

    st.markdown("""
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(108,99,255,0.15)">1️⃣</div>
        <div class="pipeline-text">
            <strong>Clause-Aware Chunking</strong>
            Splits on 3GPP clause headers, not arbitrary windows
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(0,214,143,0.15)">2️⃣</div>
        <div class="pipeline-text">
            <strong>Cross-Encoder Reranking</strong>
            Sharpens top-12 → top-4 for precision
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(255,181,71,0.15)">3️⃣</div>
        <div class="pipeline-text">
            <strong>Confidence Gate</strong>
            Refuses answer if retrieval score too low
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(255,107,107,0.15)">4️⃣</div>
        <div class="pipeline-text">
            <strong>Grounded Generation</strong>
            Cite every claim, no outside knowledge
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(108,99,255,0.15)">5️⃣</div>
        <div class="pipeline-text">
            <strong>Lexical Grounding Check</strong>
            Per-sentence word-overlap vs context
        </div>
    </div>
    <div class="pipeline-step">
        <div class="pipeline-icon" style="background: rgba(0,214,143,0.15)">6️⃣</div>
        <div class="pipeline-text">
            <strong>LLM Faithfulness Critic</strong>
            Self-critique catches subtle drift
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("## 📚 Indexed Specs")
    st.markdown("""
    - **TS 23.501** — 5G System Architecture
    - **TS 24.501** — NAS Protocol (5G)
    - **TS 38.331** — RRC Protocol
    - **TS 23.503** — Policy & Charging
    """)

    st.markdown("---")
    st.caption("Built for Mavenir GET Assessment • RAG with near-zero hallucination")


# ── Auto-setup for Streamlit Cloud ────────────────────────────
# If the ChromaDB index doesn't exist yet, download specs and ingest them.
import os

# Pull GROQ_API_KEY from Streamlit secrets if available (for Cloud deployment)
if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
if hasattr(st, "secrets") and "GROQ_MODEL" in st.secrets:
    os.environ["GROQ_MODEL"] = st.secrets["GROQ_MODEL"]

CHROMA_DIR = os.path.join("data", "chroma_db")
if not os.path.exists(CHROMA_DIR) or not os.listdir(CHROMA_DIR):
    with st.spinner("🔄 First-time setup: downloading 3GPP specs..."):
        import download_specs
    with st.spinner("🔄 Indexing specs into vector database (this may take a few minutes)..."):
        import ingest

# ── Load pipeline ────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return GPPRagPipeline()


try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(
        f"⚠️ **Pipeline failed to load:** `{e}`\n\n"
        "Make sure you've:\n"
        "1. Run `python ingest.py` to index the specs\n"
        "2. Set `GROQ_API_KEY` environment variable"
    )
    st.stop()

# ── Chat state ───────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# Display chat history
for turn in st.session_state.history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

# ── Chat input ───────────────────────────────────────────────
query = st.chat_input("Ask about the indexed 3GPP specs...")

if query:
    st.session_state.history.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Retrieving relevant clauses and generating grounded answer..."):
            resp = pipeline.answer(query)

        st.markdown(resp.answer)

        if resp.refused:
            st.markdown(
                '<div class="result-warn">⚠️ Refused — retrieval confidence below threshold</div>',
                unsafe_allow_html=True,
            )
        else:
            # Faithfulness badge
            if resp.is_faithful:
                st.markdown(
                    '<div class="result-pass">✅ Faithfulness check passed</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-fail">⚠️ Faithfulness check flagged an issue</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("🔎 Faithfulness critic output"):
                    st.code(resp.faithfulness_verdict, language=None)

            # Retrieved context
            with st.expander(f"📎 Retrieved context — {len(resp.retrieved)} chunks"):
                for c in resp.retrieved:
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-header">
                            <span class="chunk-citation">{c.citation}</span>
                            <span class="chunk-score">score: {c.similarity:.3f}</span>
                        </div>
                        <div class="chunk-text">{c.text[:800]}{'...' if len(c.text) > 800 else ''}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # Grounding report
            with st.expander("📊 Sentence-level grounding report"):
                for s in resp.grounding_report:
                    if s["grounded"]:
                        color = "rgba(0,214,143,0.15)"
                        text_color = "#00D68F"
                    else:
                        color = "rgba(255,107,107,0.15)"
                        text_color = "#FF6B6B"

                    st.markdown(f"""
                    <div class="grounding-row">
                        <div class="grounding-score" style="background:{color}; color:{text_color}">
                            {s['overlap']:.0%}
                        </div>
                        <div class="grounding-text">{s['sentence']}</div>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.history.append({"role": "assistant", "content": resp.answer})
