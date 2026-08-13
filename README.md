# 3GPP Standards RAG Chatbot

> **Retrieval-Augmented Generation chatbot over 3GPP telecom specifications, engineered for minimal to near-zero hallucination.**

Built as a hands-on demonstration of grounded, citation-backed question-answering over technical standards, with a multi-layer anti-hallucination architecture that refuses to guess when it doesn't know.

---

## 🎯 Project Scope

Rather than attempting to ingest all of 3GPP (thousands of documents), this project focuses on a coherent, defensible subset — the core **5G System** specifications:

| Spec | Title | Coverage |
|------|-------|----------|
| **TS 23.501** | System Architecture for the 5G System | Network functions, SBA, network slicing |
| **TS 24.501** | NAS Protocol for 5G | Registration, authentication, mobility |
| **TS 38.331** | Radio Resource Control (RRC) | RRC states, handover, measurements |
| **TS 23.503** | Policy and Charging Control | PCF, QoS flows, policy framework |

This is a deliberate choice: a focused system I can defend clause-by-clause in an interview is stronger evidence of solution-design understanding than shallow coverage of everything.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│  data/raw/*.docx (3GPP specifications)                  │
│         │                                               │
│         ▼                                               │
│  ┌─────────────┐   Clause-aware chunking on "5.3.1.2"  │
│  │  ingest.py   │   headers + MiniLM-L6-v2 embeddings   │
│  └──────┬──────┘                                        │
│         ▼                                               │
│  ┌─────────────┐   ChromaDB (cosine space, persistent)  │
│  │ chroma_db/  │                                        │
│  └──────┬──────┘                                        │
│         ▼                                               │
│  ┌─────────────────┐  Retrieve top-12 → rerank to top-4 │
│  │ rag_pipeline.py  │  → confidence gate → grounded gen  │
│  └──────┬──────────┘                                    │
│         ▼                                               │
│  ┌──────────────────────┐  3-layer guardrail stack       │
│  │ hallucination_guard.py│                               │
│  └──────┬───────────────┘                               │
│         ▼                                               │
│  ┌──────────┐  Streamlit UI / Evaluation harness         │
│  │  app.py   │  evaluate.py                              │
│  └──────────┘                                           │
└─────────────────────────────────────────────────────────┘
```

---

## 🛡️ Anti-Hallucination Design — The Core of This Project

This is the actual point of the project. Six stacked layers, each catching a different failure mode:

| # | Layer | What It Does | Why |
|---|-------|-------------|-----|
| 1 | **Clause-Aware Chunking** | Splits documents on 3GPP's own numbered clause structure (e.g., `5.3.1.2`) | Every chunk maps to one exact, citable clause instead of an arbitrary token window |
| 2 | **Cross-Encoder Reranking** | Re-scores top-12 candidates down to the best 4 using a cross-encoder | Bi-encoder retrieval is fast but coarse; reranking fixes precision at the point that matters most |
| 3 | **Confidence-Gated Refusal** | If the best retrieval score is below threshold, the bot refuses instead of answering | The single highest-leverage fix — most RAG hallucination happens when there's no good context |
| 4 | **Strict Grounding Prompt** | System prompt instructs the model to cite every claim and forbids outside knowledge | Constrains generation even when context IS present |
| 5 | **Lexical Grounding Check** | Post-hoc, per-sentence word-overlap check against retrieved text | Cheap, fast, catches obvious unsupported additions without an extra model call |
| 6 | **LLM Faithfulness Critic** | A second Groq call fact-checks the answer against the context | Catches subtler drift (e.g., inverted conditions) that lexical overlap misses |

**None of these alone is sufficient — that's the point.** Stacking cheap, fast checks before expensive, subtle ones is a standard defense-in-depth pattern.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- A free [Groq API key](https://console.groq.com/)

### Setup

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key
# Windows:
set GROQ_API_KEY=your-key-here
# Linux/Mac:
export GROQ_API_KEY="your-key-here"

# 3. Download 3GPP specifications (automated)
python download_specs.py

# 4. Index the specs (clause-aware chunking + embedding)
python ingest.py

# 5. Launch the demo UI
streamlit run app.py

# 6. (Optional) Run the evaluation harness
python evaluate.py
```

### Verify the Groq Model

Before running, check https://console.groq.com/docs/models for the current available model ID and update `GROQ_MODEL` in [config.py](config.py) if needed — Groq's hosted model lineup changes over time.

---

## 📁 Project Structure

```
├── config.py              # All tunable parameters in one place
├── download_specs.py      # Automated 3GPP spec downloader
├── ingest.py              # Clause-aware chunking + embedding pipeline
├── rag_pipeline.py        # Core RAG: retrieve → rerank → generate → guard
├── hallucination_guard.py # 3-layer anti-hallucination guardrails
├── app.py                 # Premium Streamlit demo UI
├── evaluate.py            # 21-question evaluation harness
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── data/
│   ├── raw/               # Downloaded 3GPP spec files
│   └── chroma_db/         # Persistent vector store
└── README.md              # This file
```

---

## 🧪 Evaluation

The evaluation harness (`evaluate.py`) runs 21 test cases:

- **17 answerable questions** across all 4 indexed specs, testing retrieval accuracy, spec matching, and faithfulness
- **4 out-of-scope questions** that should be correctly refused

Metrics reported:
- **Answer attempt rate** — does it answer when it should?
- **Retrieval accuracy** — does it find the right spec?
- **Faithfulness rate** — does the answer pass self-critique?
- **Refusal accuracy** — does it refuse out-of-scope questions?

Results are saved to `eval_results.json` for submission evidence.

---

## 🔧 Design Decisions (Interview Talking Points)

### Why hand-rolled instead of LangChain?
Every step in this pipeline is something I can explain line-by-line. Framework abstractions are useful for production, but for an interview evaluation, demonstrating understanding of *what's happening at each layer* is more valuable than showing I can call `LangChain.from_llm()`.

### Why MiniLM + cross-encoder reranking?
MiniLM-L6-v2 is the same embedding model I used in my MedQuAD healthcare QA project. It's well-understood, fast, and the cross-encoder reranking step compensates for its lower-dimensional limitations by re-scoring the top candidates with a more expensive but more accurate model.

### Why clause-aware chunking?
3GPP specs are hierarchically numbered (5.3.1.2 style). Chunking on these boundaries instead of arbitrary token windows means:
1. Each chunk is semantically complete
2. Citations are precise (exact clause + title)
3. The model can't fabricate content by stitching together half-clauses

### Why confidence-gated refusal?
The single biggest source of RAG hallucination is the model filling gaps when there's no good retrieved context. By setting a minimum retrieval score threshold, the system refuses to answer rather than guess — trading recall for precision, which is the right trade-off for a hallucination-focused project.

---

## ⚠️ Known Limitations

- **Lexical grounding check is a heuristic**, not a real NLI model — it will miss paraphrase-level hallucinations. The LLM faithfulness critic is the backstop.
- **Clause-header regex assumes clean text extraction** — scanned/image-based PDFs would need OCR.
- **Cross-spec reasoning is intentionally forbidden** — the system prompt's Rule 4 prevents combining clauses to avoid fabricated inferences. This is a conscious precision-over-recall trade-off.
- **Spec coverage is limited to 4 documents** — extensible by dropping more files in `data/raw/` and re-running `python ingest.py`.

---

## 📈 Extending Scope

To add more specs:
1. Drop more PDF/DOCX files into `data/raw/`
2. Run `python ingest.py`
3. The pipeline is spec-agnostic — it auto-detects spec numbers from filenames and content

---

## 🙏 Acknowledgments

- **3GPP** for making specifications freely available
- **Groq** for fast inference API
- **Sentence-Transformers** for the embedding and reranking models
- **ChromaDB** for the lightweight vector store

---

*Built for the Mavenir Graduate Engineer Trainee (GET) assessment — August 2026*
