"""
core/rag_engine.py
==================
LlamaIndex RAG pipeline — the knowledge retrieval engine for Agent B.

Startup Preloading Strategy:
  - The `build_rag_engine()` function is called ONCE in FastAPI's lifespan.
  - Both local models (Embedding + Reranker) are loaded into memory at startup.
  - All subsequent requests use the cached engine — NO runtime model loading.

Pipeline:
  1. MarkdownNodeParser  → Structural chunking by '##' headings
  2. SentenceSplitter    → Fine-grained chunking (with configurable overlap)
  3. VectorIndexRetriever (dense)  + BM25Retriever (sparse) → Hybrid search
  4. QueryFusionRetriever          → Reciprocal Rank Fusion merge
  5. SentenceTransformerRerank     → Cross-encoder precision re-rank
"""

import logging
import os
from pathlib import Path
from typing import Optional

from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex,
    Settings,
)
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike as LlamaOpenAI
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever

from core.config import get_settings

logger = logging.getLogger(__name__)


class RAGEngine:
    """
    Encapsulates the entire LlamaIndex retrieval pipeline.
    Instantiated once at startup; thread-safe for concurrent reads.
    """

    def __init__(
        self,
        query_engine: RetrieverQueryEngine,
        chunk_count: int,
    ):
        self._query_engine = query_engine
        self.chunk_count = chunk_count
        logger.info(f"RAGEngine ready — total indexed chunks: {chunk_count}")

    def query(self, question: str) -> str:
        """
        Execute the full hybrid-rerank retrieval pipeline.
        Returns the synthesized answer as a string.

        Raises:
            ValueError: If question is empty after stripping.
            RuntimeError: If LlamaIndex pipeline fails internally.
        """
        question = question.strip()
        if not question:
            raise ValueError("Query question must not be empty.")

        try:
            response = self._query_engine.query(question)
            return str(response)
        except Exception as e:
            logger.error(f"RAG query failed for question='{question}': {e}", exc_info=True)
            raise RuntimeError(f"Knowledge base retrieval failed: {e}") from e


def _two_pass_chunk(doc_path: str, chunk_size: int, chunk_overlap: int):
    """
    Two-pass document chunking strategy:
      Pass 1: MarkdownNodeParser splits by '##' section headings.
              This preserves each medical paper section as an atomic unit.
      Pass 2: SentenceSplitter further breaks oversized sections.
              Overlap ensures context continuity at boundaries.
    """
    doc_path_obj = Path(doc_path)
    if not doc_path_obj.exists():
        raise FileNotFoundError(
            f"Knowledge base document not found at: {doc_path_obj.resolve()}"
        )

    logger.info(f"Loading document: {doc_path_obj.resolve()}")
    documents = SimpleDirectoryReader(
        input_files=[str(doc_path_obj)]
    ).load_data()
    logger.info(f"Loaded {len(documents)} document(s).")

    # Pass 1: Structural split on Markdown headings
    md_parser = MarkdownNodeParser()
    structural_nodes = md_parser.get_nodes_from_documents(documents)
    logger.info(f"Pass 1 (MarkdownNodeParser): {len(structural_nodes)} structural nodes.")

    # Pass 2: Fine-grained sentence splitting with overlap
    # SentenceSplitter implements TransformComponent.__call__(nodes) — use it as callable
    sentence_splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    fine_nodes = sentence_splitter(structural_nodes)
    logger.info(f"Pass 2 (SentenceSplitter): {len(fine_nodes)} fine nodes.")

    return fine_nodes


def build_rag_engine() -> RAGEngine:
    """
    Factory — builds and warms up the full RAG pipeline.
    Called exactly ONCE during application startup (FastAPI lifespan).

    Returns:
        RAGEngine: a ready-to-query engine instance.

    Raises:
        FileNotFoundError: If the knowledge base doc is missing.
        RuntimeError: If any model or index step fails.
    """
    settings = get_settings()

    logger.info("=== RAG Engine startup: loading local models ===")

    # ── CUDA availability check ───────────────────────────────────────────────
    import torch
    if settings.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Config specifies DEVICE='{settings.device}' but CUDA is not available. "
                "Check your NVIDIA driver, CUDA toolkit, and torch CUDA installation. "
                "Set DEVICE=cpu in .env to run on CPU instead."
            )
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # ── Configure LlamaIndex global settings ──────────────────────────────────
    # Embedding model: local HuggingFace weights, runs on GPU (settings.device)
    logger.info(f"Loading Embedding model from: {settings.embed_model_name} on {settings.device}")
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=settings.embed_model_name,
        cache_folder=None,
        device=settings.device,   # ← CUDA
    )

    # LLM for synthesis (deepseek-chat via OpenAI-compatible endpoint)
    # Using OpenAILike instead of OpenAI to bypass model name whitelist validation
    Settings.llm = LlamaOpenAI(
        model=settings.deepseek_model,
        api_base=settings.deepseek_base_url,
        api_key=settings.deepseek_api_key,
        is_chat_model=True,
        context_window=65536,   # deepseek-chat supports 64K context
        temperature=0,
    )
    Settings.chunk_size = settings.chunk_size
    Settings.chunk_overlap = settings.chunk_overlap

    # ── Two-pass chunking ─────────────────────────────────────────────────────
    nodes = _two_pass_chunk(
        doc_path=settings.knowledge_base_doc,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    # ── Build vector index (in-memory) ────────────────────────────────────────
    logger.info("Building VectorStoreIndex (in-memory)...")
    index = VectorStoreIndex(nodes)
    logger.info("VectorStoreIndex built.")

    # ── Dense retriever ───────────────────────────────────────────────────────
    dense_retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=settings.rag_hybrid_top_k,
    )

    # ── Sparse retriever (BM25) ───────────────────────────────────────────────
    bm25_retriever = BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=settings.rag_hybrid_top_k,
    )

    # ── Hybrid fusion via Reciprocal Rank Fusion (RRF) ────────────────────────
    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=settings.rag_hybrid_top_k,
        num_queries=1,           # do not generate sub-queries, use original query
        mode="reciprocal_rerank",
        use_async=False,
    )

    # ── Cross-encoder Reranker (loaded from local path, runs on GPU) ──────────
    logger.info(f"Loading Reranker model from: {settings.rerank_model_name} on {settings.device}")
    reranker = SentenceTransformerRerank(
        model=settings.rerank_model_name,
        top_n=settings.rag_rerank_top_n,
        device=settings.device,   # ← CUDA
    )

    # ── Assemble final query engine ───────────────────────────────────────────
    query_engine = RetrieverQueryEngine.from_args(
        retriever=fusion_retriever,
        node_postprocessors=[reranker],
    )

    logger.info("=== RAG Engine startup complete ===")
    return RAGEngine(query_engine=query_engine, chunk_count=len(nodes))
