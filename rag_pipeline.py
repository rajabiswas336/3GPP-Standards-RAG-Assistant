"""
Core RAG pipeline tying together retrieval, reranking, grounded
generation, and the hallucination guardrails.

Kept dependency-light and framework-free on purpose: every step here
is something you should be able to explain line-by-line in a
technical interview, rather than "LangChain handles that part."
"""
from dataclasses import dataclass, field
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq

from config import CFG
import hallucination_guard as guard

SYSTEM_PROMPT = """You are a technical assistant that answers questions strictly \
using 3GPP telecom standards documentation provided as CONTEXT below.

Rules (follow all of them):
1. Answer ONLY using information present in the CONTEXT. Never use outside \
knowledge about telecom standards, even if you believe you know the answer.
2. Every factual claim you make must be immediately followed by a citation \
in the form [3GPP TS <spec> §<clause>], taken from the CONTEXT metadata.
3. If the CONTEXT does not contain enough information to answer, say so \
explicitly instead of guessing or extrapolating.
4. Do not combine information across unrelated clauses to infer something \
neither clause states on its own.
5. Be precise and technical. This is for an engineering audience.

CONTEXT:
{context}
"""


@dataclass
class RetrievedChunk:
    text: str
    citation: str
    similarity: float
    metadata: Dict = field(default_factory=dict)


@dataclass
class RAGResponse:
    answer: str
    refused: bool
    retrieved: List[RetrievedChunk]
    grounding_report: List[Dict]
    faithfulness_verdict: str = ""
    is_faithful: bool = True


class GPPRagPipeline:
    def __init__(self):
        self.embedder = SentenceTransformer(CFG.EMBED_MODEL)
        self.reranker = CrossEncoder(CFG.RERANK_MODEL)
        self.client = Groq(api_key=CFG.GROQ_API_KEY) if CFG.GROQ_API_KEY else Groq()
        db = chromadb.PersistentClient(path=CFG.CHROMA_DIR)
        self.collection = db.get_or_create_collection(CFG.COLLECTION_NAME)

    def _retrieve(self, query: str) -> List[RetrievedChunk]:
        q_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=q_emb,
            n_results=CFG.TOP_K_RETRIEVE,
        )
        if not results["documents"] or not results["documents"][0]:
            return []

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        # Chroma default distance is squared L2 or cosine distance depending on
        # config; we set cosine space in ingest.py, so similarity = 1 - distance.
        distances = results["distances"][0]
        candidates = [
            RetrievedChunk(text=d, citation=m["citation"], similarity=1 - dist, metadata=m)
            for d, m, dist in zip(docs, metas, distances)
        ]

        # Rerank with cross-encoder for better top-k precision
        pairs = [[query, c.text] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)
        for c, score in zip(candidates, rerank_scores):
            c.similarity = float(score)  # overwrite with reranker score for final ranking

        candidates.sort(key=lambda c: c.similarity, reverse=True)
        return candidates[:CFG.TOP_K_RERANKED]

    def _build_context_block(self, chunks: List[RetrievedChunk]) -> str:
        return "\n\n".join(
            f"[{c.citation}]\n{c.text}" for c in chunks
        )

    def answer(self, query: str) -> RAGResponse:
        chunks = self._retrieve(query)

        top_score = chunks[0].similarity if chunks else 0.0
        if not chunks or not guard.retrieval_confidence_ok(top_score):
            return RAGResponse(
                answer=guard.REFUSAL_MESSAGE,
                refused=True,
                retrieved=chunks,
                grounding_report=[],
            )

        context = self._build_context_block(chunks)
        completion = self.client.chat.completions.create(
            model=CFG.GROQ_MODEL,
            temperature=CFG.GENERATION_TEMPERATURE,
            max_tokens=CFG.MAX_ANSWER_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
                {"role": "user", "content": query},
            ],
        )
        answer_text = completion.choices[0].message.content.strip()

        grounding_report = guard.lexical_grounding_check(
            answer_text, [c.text for c in chunks]
        )

        is_faithful, verdict = True, ""
        if CFG.ENABLE_LLM_FAITHFULNESS_CHECK:
            is_faithful, verdict = guard.llm_faithfulness_check(
                answer_text, context, self.client
            )

        return RAGResponse(
            answer=answer_text,
            refused=False,
            retrieved=chunks,
            grounding_report=grounding_report,
            faithfulness_verdict=verdict,
            is_faithful=is_faithful,
        )


if __name__ == "__main__":
    pipeline = GPPRagPipeline()
    print("3GPP RAG chatbot ready. Type a question (or 'quit').\n")
    while True:
        q = input("> ").strip()
        if q.lower() in {"quit", "exit"}:
            break
        resp = pipeline.answer(q)
        print(f"\n{resp.answer}\n")
        if not resp.refused:
            print(f"[faithfulness check: {'PASS' if resp.is_faithful else 'FLAGGED'}]")
            for s in resp.grounding_report:
                mark = "OK" if s["grounded"] else "??"
                print(f"  [{mark} {s['overlap']}] {s['sentence'][:80]}")
        print()
