"""
RAG Pipeline — full production implementation:

  1. PARENT-CHILD CHUNKING
     Parent chunks (512 tokens) store full context.
     Child chunks (128 tokens) are embedded and retrieved.
     When a child is retrieved, its parent is returned for the LLM.

  2. HYBRID RETRIEVAL
     Dense:  BGE embeddings + FAISS cosine similarity
     Sparse: BM25Okapi keyword matching
     Fusion: Reciprocal Rank Fusion (RRF)

  3. CROSS-ENCODER RERANKING (via Groq)
     Uses LLM to score each candidate chunk 1-5 for relevance.
     Keeps top-k after reranking.

  4. GROUNDED ANSWER GENERATION
     Strict prompt: answer only from context.
"""
from __future__ import annotations
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from docx import Document as DocxDoc
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from pypdf import PdfReader
from rank_bm25 import BM25Okapi

from app.core.config import get_settings
from app.monitoring.telemetry import get_logger, track_llm_call

logger = get_logger(__name__)

GROUNDED_PROMPT = """You are an educational analytics assistant for Tamil Nadu DMU.
Answer using ONLY the context passages below. Cite specific numbers or facts.
If the answer is not present, say: "Not found in the available documents."

Context:
{context}

Question: {question}

Answer:"""

RERANK_PROMPT = """Rate how relevant this passage is for answering the question.
Question: {question}
Passage: {passage}
Respond with a single integer 1-5 (5=highly relevant, 1=irrelevant). Number only:"""


class ParentChildChunker:
    """
    Splits text into parent chunks (large, for context)
    and child chunks (small, for retrieval).
    Each child stores its parent_id for lookup.
    """

    def __init__(self, parent_size: int = 512, child_size: int = 128, overlap: int = 20):
        self.parent_size = parent_size
        self.child_size  = child_size
        self.overlap     = overlap

    def chunk(self, text: str) -> tuple[dict[int, str], list[dict]]:
        """
        Returns:
            parents: {parent_id: text}
            children: [{"id": int, "text": str, "parent_id": int}]
        """
        parents: dict[int, str] = {}
        children: list[dict]    = []

        # Split into parent chunks
        p_i, p_id = 0, 0
        while p_i < len(text):
            p_text = text[p_i: p_i + self.parent_size].strip()
            if len(p_text) > 40:
                parents[p_id] = p_text
            p_i += self.parent_size - self.overlap
            p_id += 1

        # Split each parent into child chunks
        child_id = 0
        for pid, ptext in parents.items():
            c_i = 0
            while c_i < len(ptext):
                c_text = ptext[c_i: c_i + self.child_size].strip()
                if len(c_text) > 30:
                    children.append({"id": child_id, "text": c_text, "parent_id": pid})
                    child_id += 1
                c_i += self.child_size - self.overlap

        return parents, children


class RAGPipeline:
    """Full production RAG pipeline."""

    def __init__(self):
        cfg = get_settings()
        self.cfg   = cfg
        self._llm  = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        self._emb  = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)
        self._chunker = ParentChildChunker(
            parent_size=cfg.PARENT_CHUNK_SIZE,
            child_size=cfg.CHILD_CHUNK_SIZE,
            overlap=cfg.CHUNK_OVERLAP,
        )
        self._parents:   dict[int, str] = {}
        self._children:  list[dict]     = []
        self._bm25:      Optional[BM25Okapi]     = None
        self._faiss_idx: Optional[faiss.Index]   = None
        self._child_embs: Optional[np.ndarray]   = None

        # Cache paths
        store = Path(cfg.VECTOR_STORE_DIR)
        store.mkdir(parents=True, exist_ok=True)
        self._cache_path = store / "rag_store.pkl"

    # ── Index management ──────────────────────────────────────────────────────

    def build_index(self, data_dir: Optional[str] = None) -> int:
        """Load all PDF/DOCX files, build parent-child index, persist to disk."""
        if self._cache_path.exists():
            return self._load_cache()

        data_dir = Path(data_dir or self.cfg.DATA_DIR)
        docs = self._load_documents(data_dir)
        combined = "\n\n".join(docs)

        self._parents, self._children = self._chunker.chunk(combined)
        if not self._children:
            logger.warning("rag_no_children_built")
            return 0

        child_texts = [c["text"] for c in self._children]

        # BM25
        self._bm25 = BM25Okapi([t.lower().split() for t in child_texts])

        # FAISS
        embs = self._emb.embed_documents(child_texts)
        self._child_embs = np.array(embs, dtype=np.float32)
        faiss.normalize_L2(self._child_embs)
        self._faiss_idx = faiss.IndexFlatIP(self._child_embs.shape[1])
        self._faiss_idx.add(self._child_embs)

        self._save_cache()
        logger.info("rag_index_built",
                    parents=len(self._parents), children=len(self._children))
        return len(self._children)

    def _load_cache(self) -> int:
        with open(self._cache_path, "rb") as f:
            data = pickle.load(f)
        self._parents    = data["parents"]
        self._children   = data["children"]
        self._child_embs = data["child_embs"]
        self._bm25 = BM25Okapi([c["text"].lower().split() for c in self._children])
        self._faiss_idx = faiss.IndexFlatIP(self._child_embs.shape[1])
        faiss.normalize_L2(self._child_embs)
        self._faiss_idx.add(self._child_embs)
        logger.info("rag_index_from_cache", children=len(self._children))
        return len(self._children)

    def _save_cache(self):
        with open(self._cache_path, "wb") as f:
            pickle.dump({
                "parents":    self._parents,
                "children":   self._children,
                "child_embs": self._child_embs,
            }, f)

    def _load_documents(self, data_dir: Path) -> list[str]:
        docs = []
        for fpath in data_dir.iterdir():
            try:
                if fpath.suffix == ".pdf":
                    text = "".join(
                        p.extract_text() or "" for p in PdfReader(str(fpath)).pages
                    )
                    docs.append(text)
                elif fpath.suffix == ".docx":
                    doc = DocxDoc(str(fpath))
                    docs.append("\n".join(p.text for p in doc.paragraphs if p.text.strip()))
            except Exception as e:
                logger.warning("doc_load_error", path=str(fpath), error=str(e))
        if not docs:
            docs.append("DMU Analytics: Educational performance data for Nilgiris district, Tamil Nadu.")
        return docs

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[str]:
        """
        1. Retrieve child chunks via FAISS + BM25 + RRF
        2. Rerank with Groq cross-encoder
        3. Return parent chunks for the top-k results
        """
        if self._faiss_idx is None:
            self.build_index()

        cfg   = self.cfg
        k_ret = cfg.TOP_K_RETRIEVAL
        k_fin = cfg.TOP_K_RERANK

        child_texts = [c["text"] for c in self._children]

        # Dense retrieval
        q_emb = np.array(self._emb.embed_documents([query]), dtype=np.float32)
        faiss.normalize_L2(q_emb)
        _, faiss_ids = self._faiss_idx.search(q_emb, k_ret)

        # Sparse retrieval
        bm25_scores = self._bm25.get_scores(query.lower().split())
        bm25_ids    = np.argsort(bm25_scores)[::-1][:k_ret]

        # RRF fusion
        fused: dict[int, float] = {}
        for rank, idx in enumerate(faiss_ids[0]):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1 / (61 + rank)
        for rank, idx in enumerate(bm25_ids):
            fused[int(idx)] = fused.get(int(idx), 0.0) + 1 / (61 + rank)

        top_child_ids = sorted(fused, key=fused.get, reverse=True)[:k_ret]

        # Cross-encoder reranking via Groq
        candidates = [(i, self._children[i]["text"]) for i in top_child_ids if i < len(self._children)]
        reranked   = self._rerank(query, candidates)[:k_fin]

        # Fetch parent chunks (deduplicated)
        seen_parents: set[int] = set()
        result_contexts: list[str] = []
        for child_id, _ in reranked:
            parent_id = self._children[child_id]["parent_id"]
            if parent_id not in seen_parents:
                seen_parents.add(parent_id)
                result_contexts.append(self._parents[parent_id])

        return result_contexts

    def _rerank(self, query: str, candidates: list[tuple[int, str]]) -> list[tuple[int, float]]:
        """Score each candidate chunk 1-5 using Groq LLM as cross-encoder."""
        scored: list[tuple[int, float]] = []
        for child_id, text in candidates:
            try:
                prompt = RERANK_PROMPT.format(question=query, passage=text[:400])
                resp   = self._llm.invoke(prompt)
                score  = float(resp.content.strip()[0])
                time.sleep(0.3)  # Groq rate limit
            except Exception:
                score = 3.0
            scored.append((child_id, score))
        return sorted(scored, key=lambda x: x[1], reverse=True)

    # ── Answer generation ─────────────────────────────────────────────────────

    def answer(self, query: str) -> tuple[str, list[str]]:
        """Return (answer_text, list_of_context_chunks)."""
        contexts = self.retrieve(query)
        context_str = "\n\n---\n\n".join(contexts)[: self.cfg.PARENT_CHUNK_SIZE * self.cfg.TOP_K_RERANK]
        prompt = GROUNDED_PROMPT.format(context=context_str, question=query)
        t0 = time.perf_counter()
        resp = self._llm.invoke(prompt)
        usage = track_llm_call(resp, self.cfg.LLM_MODEL, t0)
        logger.info("rag_answered", query=query[:60], tokens=usage.total_tokens,
                    cost_usd=usage.cost_usd, latency_ms=usage.latency_ms)
        return resp.content.strip(), contexts
