"""
Central configuration for the 3GPP RAG Chatbot.
Keep every tunable in one place so the anti-hallucination knobs are
easy to point to and explain in an interview.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass(frozen=True)
class Config:
    # --- Paths ---
    RAW_DOCS_DIR: str = "data/raw"              # put downloaded 3GPP PDFs/DOCX here
    CHROMA_DIR: str = "data/chroma_db"
    COLLECTION_NAME: str = "gpp_specs"

    # --- Embedding model (same family you used on MedQuAD) ---
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # --- Reranker (cross-encoder, sharpens top-k precision) ---
    RERANK_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Chunking ---
    MAX_CHUNK_TOKENS: int = 350      # fallback split size when a clause is too long
    CHUNK_OVERLAP_TOKENS: int = 50

    # --- Retrieval ---
    TOP_K_RETRIEVE: int = 12          # candidates pulled from the vector store
    TOP_K_RERANKED: int = 4           # kept after cross-encoder reranking

    # --- Hallucination guardrails ---
    MIN_RETRIEVAL_SCORE: float = 0.35   # below this, refuse to answer (cosine sim)
    MIN_GROUNDING_OVERLAP: float = 0.30 # lexical overlap floor per answer sentence
    ENABLE_LLM_FAITHFULNESS_CHECK: bool = True  # extra Groq call to self-critique

    # --- Generation (Groq) ---
    # NOTE: Groq's available model IDs change over time -- check
    # https://console.groq.com/docs/models before running and update this.
    GROQ_MODEL: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    if GROQ_MODEL in ("qwen/qwen3-32b", "llama-3.1-70b-versatile"):
        GROQ_MODEL = "llama-3.3-70b-versatile"
    GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
    GENERATION_TEMPERATURE: float = 0.1   # low temp -> less creative drift
    MAX_ANSWER_TOKENS: int = 600

CFG = Config()
