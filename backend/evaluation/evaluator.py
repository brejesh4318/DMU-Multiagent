"""
Evaluation Suite — all 8 metrics:

  Retrieval:
    1. Recall@5    — are relevant docs in top 5?
    2. Recall@10   — are relevant docs in top 10?
    3. MRR         — Mean Reciprocal Rank
    4. nDCG        — Normalized Discounted Cumulative Gain

  Generation:
    5. RAGAS       — faithfulness, answer_relevancy, context_precision, context_recall
    6. SQL Accuracy — did SQL execute and return expected data shape?

  System:
    7. Routing Accuracy — did classifier route to the right agent?
    8. LLM-as-Judge     — already integrated in SQL engine; aggregated here
"""
from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from app.agents.router import classify
from app.monitoring.telemetry import get_logger
from app.rag.pipeline import RAGPipeline
from app.sql.engine import SQLEngine

logger = get_logger(__name__)


# ── Ground-truth test suites ──────────────────────────────────────────────────

ROUTING_TEST_SUITE = [
    # (query, expected_type)
    ("how many students failed in Physics",        "analytics"),
    ("which school has the highest English average","analytics"),
    ("compare 2025 vs 2026 pass rates",            "analytics"),
    ("list all failures in Ooty",                  "analytics"),
    ("top 5 schools by total marks",               "analytics"),
    ("failures in mathematics Gudalur",            "analytics"),
    ("pass percentage boys vs girls",              "analytics"),
    ("show a bar chart of subject failures",       "analytics"),
    ("what is the re-examination policy",          "rag"),
    ("explain the grading criteria",               "rag"),
    ("describe the samacheer syllabus",            "rag"),
    ("what is the government scholarship scheme",  "web"),
    ("latest NEP circular",                        "web"),
    ("current exam timetable 2026",                "web"),
]

RAG_QA_SUITE = [
    {
        "question":     "What is the full form of SLAS?",
        "ground_truth": "State Level Achievement Survey",
        "relevant_keywords": ["state level achievement survey", "slas"],
    },
    {
        "question":     "What is the state average for Mathematics in Class 3?",
        "ground_truth": "54 percent",
        "relevant_keywords": ["54", "mathematics", "class 3"],
    },
    {
        "question":     "Which subject had the lowest average in Class 8?",
        "ground_truth": "Science with 37 percent",
        "relevant_keywords": ["science", "37", "class 8"],
    },
    {
        "question":     "How many students participated in SLAS 2025?",
        "ground_truth": "9,80,341 students",
        "relevant_keywords": ["9,80,341", "980341", "9 80 341"],
    },
    {
        "question":     "What was the Class 3 Mathematics score for The Nilgiris?",
        "ground_truth": "47 percent",
        "relevant_keywords": ["47", "nilgiris", "mathematics"],
    },
]

SQL_TEST_SUITE = [
    "how many students failed in physics",
    "which school has the highest average in english",
    "compare pass percentage between boys and girls",
    "subject with most failures across nilgiris",
    "top 5 schools by average marks",
]


# ── Metric implementations ────────────────────────────────────────────────────

def _hits(retrieved: list[str], keywords: list[str]) -> list[int]:
    """Return binary relevance list: 1 if chunk contains any keyword."""
    result = []
    for chunk in retrieved:
        c = chunk.lower()
        rel = 1 if any(kw.lower() in c for kw in keywords) else 0
        result.append(rel)
    return result


def recall_at_k(retrieved: list[str], keywords: list[str], k: int) -> float:
    hits = sum(_hits(retrieved[:k], keywords))
    return min(1.0, hits)   # binary: 1 if at least one hit in top-k


def mrr(retrieved: list[str], keywords: list[str]) -> float:
    for i, chunk in enumerate(retrieved, 1):
        if any(kw.lower() in chunk.lower() for kw in keywords):
            return 1.0 / i
    return 0.0


def ndcg(retrieved: list[str], keywords: list[str], k: int = 10) -> float:
    rels = _hits(retrieved[:k], keywords)
    dcg  = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(sum(rels) or 1, k)))
    return dcg / idcg if idcg > 0 else 0.0


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RetrievalMetrics:
    recall_at_5:  float = 0.0
    recall_at_10: float = 0.0
    mrr:          float = 0.0
    ndcg_at_10:   float = 0.0
    per_question: list[dict] = field(default_factory=list)


@dataclass
class RAGASMetrics:
    faithfulness:      float = 0.0
    answer_relevancy:  float = 0.0
    context_precision: float = 0.0
    context_recall:    float = 0.0
    computed: bool = False
    error:    str  = ""


@dataclass
class SQLMetrics:
    accuracy:          float = 0.0
    avg_judge_score:   float = 0.0
    avg_latency_ms:    float = 0.0
    per_query:         list[dict] = field(default_factory=list)


@dataclass
class RoutingMetrics:
    accuracy:     float = 0.0
    total:        int   = 0
    correct:      int   = 0
    failures:     list[dict] = field(default_factory=list)


@dataclass
class EvalReport:
    retrieval:   RetrievalMetrics = field(default_factory=RetrievalMetrics)
    ragas:       RAGASMetrics     = field(default_factory=RAGASMetrics)
    sql:         SQLMetrics       = field(default_factory=SQLMetrics)
    routing:     RoutingMetrics   = field(default_factory=RoutingMetrics)
    overall_pass: bool = False
    summary:      str  = ""


# ── Evaluator ─────────────────────────────────────────────────────────────────

class Evaluator:
    def __init__(
        self,
        rag:   Optional[RAGPipeline] = None,
        sql:   Optional[SQLEngine]   = None,
        delay: float = 1.5,
    ):
        self._rag   = rag
        self._sql   = sql
        self._delay = delay

    # 1+2+3+4: Retrieval metrics
    def eval_retrieval(self) -> RetrievalMetrics:
        if not self._rag:
            return RetrievalMetrics()

        r5_list, r10_list, mrr_list, ndcg_list = [], [], [], []
        per_q = []

        for qa in RAG_QA_SUITE:
            try:
                retrieved = self._rag.retrieve(qa["question"])
                kws       = qa["relevant_keywords"]
                r5  = recall_at_k(retrieved, kws, 5)
                r10 = recall_at_k(retrieved, kws, 10)
                m   = mrr(retrieved, kws)
                n   = ndcg(retrieved, kws, 10)
                r5_list.append(r5);  r10_list.append(r10)
                mrr_list.append(m);  ndcg_list.append(n)
                per_q.append({"question": qa["question"], "recall@5": r5,
                               "recall@10": r10, "mrr": m, "ndcg": n})
                logger.info("retrieval_eval", q=qa["question"][:50],
                            r5=r5, r10=r10, mrr=m, ndcg=n)
                time.sleep(self._delay)
            except Exception as e:
                logger.warning("retrieval_eval_error", error=str(e))

        return RetrievalMetrics(
            recall_at_5=  round(float(np.mean(r5_list)),   3) if r5_list  else 0,
            recall_at_10= round(float(np.mean(r10_list)),  3) if r10_list else 0,
            mrr=          round(float(np.mean(mrr_list)),  3) if mrr_list else 0,
            ndcg_at_10=   round(float(np.mean(ndcg_list)), 3) if ndcg_list else 0,
            per_question=per_q,
        )

    # 5: RAGAS metrics
    def eval_ragas(self) -> RAGASMetrics:
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (answer_relevancy, context_precision,
                                        context_recall, faithfulness)
            from ragas.run_config import RunConfig
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from langchain_groq import ChatGroq
            from langchain_huggingface import HuggingFaceEmbeddings
            from app.core.config import get_settings

            cfg = get_settings()
            llm = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
            emb = HuggingFaceEmbeddings(model_name=cfg.EMBED_MODEL)

            questions_list, answers_list, contexts_list, gt_list = [], [], [], []
            for qa in RAG_QA_SUITE:
                ans, ctxs = self._rag.answer(qa["question"])
                questions_list.append(qa["question"])
                answers_list.append(ans)
                contexts_list.append(ctxs)
                gt_list.append(qa["ground_truth"])
                time.sleep(self._delay)

            ds = Dataset.from_dict({
                "question": questions_list, "answer": answers_list,
                "contexts": contexts_list,  "ground_truth": gt_list,
            })

            wrapped_llm = LangchainLLMWrapper(llm)
            wrapped_emb = LangchainEmbeddingsWrapper(emb)
            metrics     = [faithfulness, answer_relevancy, context_precision, context_recall]
            for m in metrics:
                m.llm = wrapped_llm
            answer_relevancy.embeddings = wrapped_emb

            results = evaluate(
                ds, metrics=metrics,
                run_config=RunConfig(timeout=120, max_workers=1, max_wait=120),
            )
            df = results.to_pandas()

            return RAGASMetrics(
                faithfulness=      round(float(df["faithfulness"].dropna().mean()),      3),
                answer_relevancy=  round(float(df["answer_relevancy"].dropna().mean()),  3),
                context_precision= round(float(df["context_precision"].dropna().mean()), 3),
                context_recall=    round(float(df["context_recall"].dropna().mean()),    3),
                computed=True,
            )
        except Exception as e:
            logger.error("ragas_eval_error", error=str(e))
            return RAGASMetrics(computed=False, error=str(e))

    # 6: SQL accuracy
    def eval_sql(self) -> SQLMetrics:
        if not self._sql:
            return SQLMetrics()

        results, scores, latencies = [], [], []

        for q in SQL_TEST_SUITE:
            try:
                r = self._sql.run(q)
                ok = r.error is None and r.result_df is not None and not r.result_df.empty
                results.append(1 if ok else 0)
                scores.append(r.judge.score)
                latencies.append(r.latency_ms)
                logger.info("sql_eval", q=q[:50], ok=ok, judge=r.judge.score)
                time.sleep(self._delay)
            except Exception as e:
                results.append(0); scores.append(0); latencies.append(0)
                logger.warning("sql_eval_error", error=str(e))

        per_q = [
            {"query": q[:60], "accuracy": a, "judge": s, "latency_ms": l}
            for q, a, s, l in zip(SQL_TEST_SUITE, results, scores, latencies)
        ]
        return SQLMetrics(
            accuracy=        round(float(np.mean(results)), 3) if results else 0,
            avg_judge_score= round(float(np.mean(scores)),  2) if scores  else 0,
            avg_latency_ms=  round(float(np.mean(latencies)),1) if latencies else 0,
            per_query=per_q,
        )

    # 7: Routing accuracy
    def eval_routing(self) -> RoutingMetrics:
        correct, failures = 0, []
        for q, expected in ROUTING_TEST_SUITE:
            actual = classify(q)
            ok     = actual == expected
            if ok:
                correct += 1
            else:
                failures.append({"query": q, "expected": expected, "actual": actual})

        total = len(ROUTING_TEST_SUITE)
        return RoutingMetrics(
            accuracy= round(correct / total, 3),
            total=    total,
            correct=  correct,
            failures= failures,
        )

    # Full run
    def run_all(self, include_ragas: bool = True) -> EvalReport:
        logger.info("eval_start")
        report = EvalReport()

        logger.info("eval_routing")
        report.routing = self.eval_routing()

        if self._sql:
            logger.info("eval_sql")
            report.sql = self.eval_sql()

        if self._rag:
            logger.info("eval_retrieval")
            report.retrieval = self.eval_retrieval()
            if include_ragas:
                logger.info("eval_ragas")
                report.ragas = self.eval_ragas()

        # Overall pass thresholds
        thresholds = {
            "routing_accuracy": (report.routing.accuracy, 0.85),
            "sql_accuracy":     (report.sql.accuracy,     0.80),
            "recall@5":         (report.retrieval.recall_at_5, 0.70),
            "ndcg":             (report.retrieval.ndcg_at_10,  0.65),
        }
        if report.ragas.computed:
            thresholds["ragas_faithfulness"] = (report.ragas.faithfulness, 0.75)

        fails = [k for k, (v, t) in thresholds.items() if v < t]
        report.overall_pass = len(fails) == 0
        report.summary = (
            "PASS" if report.overall_pass
            else f"FAIL - Below threshold: {', '.join(fails)}"
        )
        logger.info("eval_complete", overall=report.summary)
        return report


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))

    print("=" * 60)
    print("  EVALUATION SUITE")
    print("=" * 60)

    # Try to wire up SQL and RAG engines; fall back gracefully
    sql_engine = None
    rag_pipeline = None

    try:
        from app.sql.engine import SQLEngine
        sql_engine = SQLEngine()
        print("  SQL engine  : [OK] connected")
    except Exception as e:
        print(f"  SQL engine  : [!!] unavailable ({e})")

    try:
        from app.rag.pipeline import RAGPipeline
        rag_pipeline = RAGPipeline()
        print("  RAG pipeline: [OK] connected")
    except Exception as e:
        print(f"  RAG pipeline: [!!] unavailable ({e})")

    print()

    ev = Evaluator(rag=rag_pipeline, sql=sql_engine)

    # Always run routing (no external deps)
    print("-- 1. Routing Accuracy --")
    r = ev.eval_routing()
    print(f"   Accuracy : {r.accuracy:.0%}  ({r.correct}/{r.total})")
    for f in r.failures:
        print(f"   [XX] '{f['query']}' -> expected={f['expected']} got={f['actual']}")

    # SQL eval
    if sql_engine:
        print("\n-- 2. SQL Accuracy (LLM-as-Judge) --")
        s = ev.eval_sql()
        print(f"   Accuracy       : {s.accuracy:.0%}")
        print(f"   Avg judge score: {s.avg_judge_score:.1f}/5")
        print(f"   Avg latency    : {s.avg_latency_ms:.0f} ms")
        for q in s.per_query:
            icon = "[OK]" if q["accuracy"] else "[XX]"
            print(f"   {icon} [{q['judge']}/5] {q['query']}")

    # Retrieval eval
    if rag_pipeline:
        print("\n-- 3. Retrieval Metrics --")
        ret = ev.eval_retrieval()
        print(f"   Recall@5  : {ret.recall_at_5:.3f}")
        print(f"   Recall@10 : {ret.recall_at_10:.3f}")
        print(f"   MRR       : {ret.mrr:.3f}")
        print(f"   nDCG@10   : {ret.ndcg_at_10:.3f}")

    print("\n" + "=" * 60)
    print("  Done. (Pass RAGAS via Evaluator.eval_ragas() separately)")
    print("=" * 60)
