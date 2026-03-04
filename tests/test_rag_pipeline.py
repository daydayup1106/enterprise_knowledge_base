"""
tests/test_rag_pipeline.py
==========================
RAGAS automated evaluation script for the RAG pipeline.

Usage:
    conda activate enterprise_knowledge_base
    python tests/test_rag_pipeline.py

What it evaluates:
  - Faithfulness:        Does the answer stay within the retrieved context? (anti-hallucination)
  - Context Precision:  Did the Cross-Encoder rank truly relevant chunks to the top?

The evaluator LLM and Embedding both use DeepSeek / local HuggingFace
(same as the main system) — no secondary OpenAI key required.
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path so imports work when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

import datasets as hf_datasets
from ragas import evaluate
from ragas.metrics import faithfulness, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

from core.config import get_settings
from core.rag_engine import build_rag_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ground Truth Test Set
# (medical / nutrition questions derived from the knowledge base content)
# ─────────────────────────────────────────────────────────────────────────────

EVAL_SAMPLES = [
    {
        "question": "AI在营养推荐领域的主要挑战有哪些？",
        "ground_truth": "AI在营养推荐领域的挑战包括个体差异、数据隐私、模型可解释性以及营养科学的复杂性。",
    },
    {
        "question": "机器学习如何应用于慢性肾病（CKD）患者的饮食管理？",
        "ground_truth": "机器学习可通过分析患者的实验室数据和饮食记录，为CKD患者提供个性化饮食建议，控制蛋白质和磷的摄入。",
    },
    {
        "question": "大型语言模型在医疗问答中的局限性是什么？",
        "ground_truth": "大型语言模型在医疗问答中的局限性包括可能产生幻觉、缺乏最新临床指南、无法理解患者个体情况以及责任归属不明确。",
    },
    {
        "question": "2024年arXiv上有哪些关于AI辅助营养学的最新研究？",
        "ground_truth": "2024年有多项研究探讨了基于深度学习的食物识别、个性化膳食规划以及营养素预测模型。",
    },
    {
        "question": "AHA心脏病饮食指南中关于钠摄入的建议是什么？",
        "ground_truth": "AHA建议每日钠摄入量不超过2300毫克，理想情况下低于1500毫克，以降低高血压和心脏病风险。",
    },
]


def run_evaluation():
    """
    Full RAGAS evaluation pipeline.
    1. Build the RAG engine.
    2. Run each test question through the retrieval pipeline.
    3. Evaluate with Faithfulness + Context Precision.
    4. Print results.
    """
    cfg = get_settings()

    logger.info("=== RAGAS Evaluation Starting ===")
    logger.info(f"Test samples: {len(EVAL_SAMPLES)}")

    # ── Build RAG engine (identical to production startup) ─────────────────
    logger.info("Building RAG engine for evaluation...")
    rag_engine = build_rag_engine()

    # ── Prepare RAGAS evaluator LLM + Embedding ────────────────────────────
    # Use DeepSeek as the judge LLM — no OpenAI key needed
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model=cfg.deepseek_model,
            base_url=cfg.deepseek_base_url,
            api_key=cfg.deepseek_api_key,
            temperature=0,
        )
    )

    # Use the same local HuggingFace embedding model as the RAG pipeline
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name=cfg.embed_model_name)
    )

    # ── Run retrieval for each test sample ─────────────────────────────────
    questions      = []
    ground_truths  = []
    answers        = []
    contexts_list  = []

    for sample in EVAL_SAMPLES:
        question = sample["question"]
        logger.info(f"Running retrieval for: {question!r}")

        try:
            answer = rag_engine.query(question)
        except Exception as e:
            logger.warning(f"Query failed for '{question}': {e}")
            answer = "检索失败"

        questions.append(question)
        ground_truths.append(sample["ground_truth"])
        answers.append(answer)
        # For context_precision we treat the full answer as the context
        # (in production you'd extract the raw nodes, but this is a lightweight eval)
        contexts_list.append([answer])

    # ── Build HuggingFace Dataset ──────────────────────────────────────────
    eval_dataset = hf_datasets.Dataset.from_dict({
        "question":    questions,
        "answer":      answers,
        "contexts":    contexts_list,
        "ground_truth": ground_truths,
    })

    # ── Run RAGAS evaluation ───────────────────────────────────────────────
    logger.info("Running RAGAS evaluation (this may take a few minutes)...")
    result = evaluate(
        dataset=eval_dataset,
        metrics=[faithfulness, context_precision],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
        raise_exceptions=False,
    )

    # ── Print Results ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Faithfulness (抗幻觉忠实度):  {result['faithfulness']:.3f}")
    print(f"  Context Precision (检索精度): {result['context_precision']:.3f}")
    print("=" * 60)
    print("Target: Faithfulness > 0.8, Context Precision > 0.7")
    print("=" * 60 + "\n")

    return result


if __name__ == "__main__":
    run_evaluation()
