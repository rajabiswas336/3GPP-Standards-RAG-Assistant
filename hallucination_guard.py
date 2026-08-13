"""
Hallucination guardrails, kept separate from the main pipeline so each
technique can be explained and demoed independently in an interview.

Three layers:
  1. Retrieval confidence gate  -> refuse before even calling the LLM
  2. Lexical grounding scorer   -> cheap, fast, catches obvious drift
  3. LLM faithfulness critic    -> optional second pass, catches subtle drift

None of these are perfect alone; stacked, they meaningfully cut the
"confidently wrong" failure mode that plain RAG is prone to.
"""
import re
from typing import List, Dict, Tuple

from groq import Groq
from config import CFG

REFUSAL_MESSAGE = (
    "I don't have enough relevant information in the indexed 3GPP "
    "documents to answer this confidently. Try rephrasing, or this "
    "may be outside the specs currently loaded."
)

_STOPWORDS = {
    "the", "a", "an", "is", "are", "of", "to", "in", "and", "or", "for",
    "this", "that", "shall", "may", "on", "with", "as", "be", "by", "it",
    "if", "not", "when", "which", "at", "from", "will",
}


def retrieval_confidence_ok(top_score: float) -> bool:
    """Layer 1: gate on the best retrieval similarity score.
    Chroma with cosine space returns distance; caller converts to
    similarity = 1 - distance before passing in here."""
    return top_score >= CFG.MIN_RETRIEVAL_SCORE


def _tokenize(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def _sentence_split(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def lexical_grounding_check(answer: str, context_chunks: List[str]) -> List[Dict]:
    """Layer 2: for every sentence in the answer, measure word-overlap
    against the union of retrieved context. Flags sentences that look
    unsupported so they can be surfaced to the user rather than hidden.
    This is intentionally simple (no extra model call) so it's cheap
    enough to run on every response."""
    context_vocab = set()
    for chunk in context_chunks:
        context_vocab |= _tokenize(chunk)

    results = []
    for sentence in _sentence_split(answer):
        sent_vocab = _tokenize(sentence)
        if not sent_vocab:
            continue
        overlap = len(sent_vocab & context_vocab) / len(sent_vocab)
        results.append({
            "sentence": sentence,
            "overlap": round(overlap, 2),
            "grounded": overlap >= CFG.MIN_GROUNDING_OVERLAP,
        })
    return results


FAITHFULNESS_PROMPT = """You are a strict fact-checker. You will be given a CONTEXT \
extracted from 3GPP specifications and an ANSWER generated from that context.

List any claims in the ANSWER that are NOT directly supported by the CONTEXT. \
If every claim is supported, respond with exactly: SUPPORTED
Otherwise, respond with a short bullet list of the unsupported claims only. \
Do not restate supported claims.

CONTEXT:
{context}

ANSWER:
{answer}
"""


def llm_faithfulness_check(answer: str, context: str, client: Groq) -> Tuple[bool, str]:
    """Layer 3: ask the model to critique its own answer against the
    context in a separate call. Catches paraphrase-level drift that
    lexical overlap can miss (e.g. subtly inverted conditions)."""
    resp = client.chat.completions.create(
        model=CFG.GROQ_MODEL,
        temperature=0,
        max_tokens=300,
        messages=[{"role": "user", "content": FAITHFULNESS_PROMPT.format(
            context=context, answer=answer)}],
    )
    verdict = resp.choices[0].message.content.strip()
    is_faithful = verdict.upper().startswith("SUPPORTED")
    return is_faithful, verdict
