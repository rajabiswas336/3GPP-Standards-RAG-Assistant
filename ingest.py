"""
Ingestion pipeline for 3GPP specification documents.

Pipeline: raw file -> plain text -> clause-aware chunks -> embeddings -> Chroma.

Why clause-aware chunking:
3GPP specs are numbered hierarchically (e.g. "5.3.1.2 RRC Connection
Release"). Splitting on that structure -- instead of a blind token
window -- keeps each chunk semantically whole AND lets us cite an
exact clause number in every answer, which is the single biggest
lever for reducing hallucination: the model is grounded to a specific,
verifiable passage rather than a fuzzy blob of context.
"""
import os
import re
import sys
import glob
import uuid
import time
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Dict

import chromadb
from sentence_transformers import SentenceTransformer

from config import CFG

# Matches lines like "5.3.1.2 Some Clause Title" at the start of a line.
CLAUSE_HEADER_RE = re.compile(
    r"^\s*(\d+(?:\.\d+){0,5})\s+([A-Z][A-Za-z0-9 ,\-/&()']{3,120})\s*$",
    re.MULTILINE,
)

# 3GPP spec numbers look like "38.331" or "TS 24.501" in filenames/headers.
SPEC_NUMBER_RE = re.compile(r"(?:TS|TR)?\s?(\d{2}\.\d{3})", re.IGNORECASE)

# XML namespace for DOCX (Office Open XML)
WORD_NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'


def _log(msg: str):
    """Print with immediate flush so output is visible in real time."""
    print(msg, flush=True)


def extract_text_docx_fast(filepath: str) -> str:
    """Extract text from a DOCX using direct XML parsing (zip + ElementTree).
    
    This is MUCH faster than python-docx for large 3GPP specs because:
    - python-docx loads the full DOM including styles, formatting, etc.
    - This just extracts the raw text from <w:t> tags in document.xml
    """
    text_parts = []
    with zipfile.ZipFile(filepath, 'r') as z:
        # Main document body
        if 'word/document.xml' in z.namelist():
            with z.open('word/document.xml') as f:
                tree = ET.parse(f)
                root = tree.getroot()
                
                # Find all paragraph elements
                for para in root.iter(f'{WORD_NS}p'):
                    para_texts = []
                    for t_elem in para.iter(f'{WORD_NS}t'):
                        if t_elem.text:
                            para_texts.append(t_elem.text)
                    if para_texts:
                        text_parts.append(''.join(para_texts))
    
    return '\n'.join(text_parts)


def extract_text(filepath: str) -> str:
    """Extract raw text from a PDF or DOCX 3GPP document."""
    if filepath.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(filepath)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filepath.lower().endswith((".docx", ".doc")):
        return extract_text_docx_fast(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")


def guess_spec_number(filepath: str, text: str) -> str:
    """Best-effort extraction of the spec number (e.g. '38.331') for citations."""
    m = SPEC_NUMBER_RE.search(os.path.basename(filepath))
    if m:
        return m.group(1)
    m = SPEC_NUMBER_RE.search(text[:2000])
    return m.group(1) if m else "UNKNOWN"


def split_by_clause(text: str) -> List[Dict]:
    """Split document text into (clause_id, clause_title, body) segments."""
    matches = list(CLAUSE_HEADER_RE.finditer(text))
    if not matches:
        # No detectable clause structure -- fall back to one giant blob,
        # the token-level splitter downstream will chop it up.
        return [{"clause_id": "N/A", "clause_title": "N/A", "body": text}]

    segments = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if len(body) < 20:  # skip near-empty headers (e.g. table of contents lines)
            continue
        segments.append({
            "clause_id": m.group(1),
            "clause_title": m.group(2).strip(),
            "body": body,
        })
    return segments


def split_long_body(body: str, max_tokens: int, overlap_tokens: int) -> List[str]:
    """Word-count-based fallback splitter for clauses that are still too long
    (e.g. clauses containing large tables). ~1 token ~ 0.75 words, approximated
    here as words for simplicity -- good enough for chunk sizing, not exact."""
    words = body.split()
    if len(words) <= max_tokens:
        return [body]

    chunks, start = [], 0
    step = max_tokens - overlap_tokens
    while start < len(words):
        chunk_words = words[start:start + max_tokens]
        chunks.append(" ".join(chunk_words))
        start += step
    return chunks


def build_chunks(filepath: str) -> List[Dict]:
    _log(f"  Extracting text...")
    t = time.time()
    text = extract_text(filepath)
    _log(f"  Text extracted in {time.time()-t:.1f}s ({len(text):,} chars)")

    spec_number = guess_spec_number(filepath, text)
    _log(f"  Spec number: {spec_number}")

    _log(f"  Splitting by clause...")
    t = time.time()
    clauses = split_by_clause(text)
    _log(f"  Found {len(clauses)} clauses in {time.time()-t:.1f}s")

    chunks = []
    for clause in clauses:
        pieces = split_long_body(clause["body"], CFG.MAX_CHUNK_TOKENS, CFG.CHUNK_OVERLAP_TOKENS)
        for idx, piece in enumerate(pieces):
            part_suffix = f" (part {idx + 1}/{len(pieces)})" if len(pieces) > 1 else ""
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": piece,
                "metadata": {
                    "spec_number": spec_number,
                    "clause_id": clause["clause_id"],
                    "clause_title": clause["clause_title"] + part_suffix,
                    "source_file": os.path.basename(filepath),
                    "citation": f"3GPP TS {spec_number} §{clause['clause_id']} "
                                 f"({clause['clause_title']})",
                },
            })
    return chunks


def ingest_all():
    files = glob.glob(os.path.join(CFG.RAW_DOCS_DIR, "*.pdf")) + \
            glob.glob(os.path.join(CFG.RAW_DOCS_DIR, "*.docx"))
    if not files:
        _log(f"No documents found in {CFG.RAW_DOCS_DIR}/. "
              f"Download specs there first (see README).")
        return

    _log(f"Found {len(files)} document(s). Loading embedding model...")
    t = time.time()
    embedder = SentenceTransformer(CFG.EMBED_MODEL)
    _log(f"Model loaded in {time.time()-t:.1f}s")

    client = chromadb.PersistentClient(path=CFG.CHROMA_DIR)
    collection = client.get_or_create_collection(
        CFG.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    total_chunks = 0
    for filepath in files:
        _log(f"\nProcessing {filepath} ...")
        file_start = time.time()
        chunks = build_chunks(filepath)
        if not chunks:
            _log(f"  WARNING: no chunks extracted from {filepath}")
            continue

        _log(f"  Embedding {len(chunks)} chunks...")
        t = time.time()
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=True, batch_size=64).tolist()
        _log(f"  Embedded in {time.time()-t:.1f}s")

        _log(f"  Storing in ChromaDB...")
        t = time.time()
        # ChromaDB has a batch size limit, so insert in batches
        BATCH_SIZE = 500
        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            batch_emb = embeddings[i:i + BATCH_SIZE]
            collection.add(
                ids=[c["id"] for c in batch],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
                embeddings=batch_emb,
            )
        _log(f"  Stored in {time.time()-t:.1f}s")

        total_chunks += len(chunks)
        _log(f"  -> {len(chunks)} chunks stored ({time.time()-file_start:.1f}s total for this file)")

    _log(f"\nDone. {total_chunks} total chunks indexed into '{CFG.COLLECTION_NAME}' "
          f"at {CFG.CHROMA_DIR}")


if __name__ == "__main__":
    ingest_all()
