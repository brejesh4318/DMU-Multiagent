#!/usr/bin/env python3
"""
Usage:
  python evaluate.py              # full suite
  python evaluate.py --no-ragas  # skip RAGAS (faster)
  python evaluate.py --no-sql    # skip SQL eval
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.monitoring.telemetry import configure_logging
from app.evaluation.evaluator import Evaluator
from app.rag.pipeline import RAGPipeline
from app.sql.engine import SQLEngine

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-ragas", action="store_true")
    parser.add_argument("--no-sql",   action="store_true")
    parser.add_argument("--output",   default="eval_report.json")
    args = parser.parse_args()

    configure_logging("WARNING")

    print("=" * 60)
    print("  DMU Analytics — Evaluation Suite")
    print("=" * 60)

    rag = RAGPipeline()
    rag.build_index()
    sql = None if args.no_sql else SQLEngine()

    ev     = Evaluator(rag=rag, sql=sql)
    report = ev.run_all(include_ragas=not args.no_ragas)

    # ── Routing ───────────────────────────────────────────────────────────
    r = report.routing
    print(f"\n[1] ROUTING ACCURACY:  {r.accuracy:.0%}  ({r.correct}/{r.total})")
    for f in r.failures:
        print(f"    ❌ '{f['query'][:50]}' → got={f['actual']} expected={f['expected']}")

    # ── SQL ───────────────────────────────────────────────────────────────
    s = report.sql
    print(f"\n[2] SQL ACCURACY:      {s.accuracy:.0%}")
    print(f"    Avg judge score:   {s.avg_judge_score:.1f}/5")
    print(f"    Avg latency:       {s.avg_latency_ms:.0f}ms")
    for pq in s.per_query:
        icon = "✅" if pq["accuracy"] else "❌"
        print(f"    {icon} [{pq['judge']}/5] {pq['query'][:55]}")

    # ── Retrieval ─────────────────────────────────────────────────────────
    ret = report.retrieval
    print(f"\n[3] RETRIEVAL METRICS:")
    print(f"    Recall@5:   {ret.recall_at_5:.3f}   (target ≥ 0.70)")
    print(f"    Recall@10:  {ret.recall_at_10:.3f}   (target ≥ 0.80)")
    print(f"    MRR:        {ret.mrr:.3f}   (target ≥ 0.60)")
    print(f"    nDCG@10:    {ret.ndcg_at_10:.3f}   (target ≥ 0.65)")

    # ── RAGAS ─────────────────────────────────────────────────────────────
    rg = report.ragas
    if rg.computed:
        print(f"\n[4] RAGAS METRICS:")
        print(f"    Faithfulness:      {rg.faithfulness:.3f}   (target ≥ 0.80)")
        print(f"    Answer Relevancy:  {rg.answer_relevancy:.3f}   (target ≥ 0.75)")
        print(f"    Context Precision: {rg.context_precision:.3f}   (target ≥ 0.70)")
        print(f"    Context Recall:    {rg.context_recall:.3f}   (target ≥ 0.65)")
    elif rg.error:
        print(f"\n[4] RAGAS: skipped — {rg.error}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    icon = " PASS" if report.overall_pass else " FAIL"
    print(f"  {icon} — {report.summary}")
    print("=" * 60)

    # Save JSON
    with open(args.output, "w") as f:
        json.dump({
            "overall_pass": report.overall_pass,
            "summary":      report.summary,
            "routing":      {"accuracy": r.accuracy, "failures": r.failures},
            "sql":          {"accuracy": s.accuracy, "avg_judge": s.avg_judge_score},
            "retrieval":    {"recall5": ret.recall_at_5, "recall10": ret.recall_at_10,
                             "mrr": ret.mrr, "ndcg": ret.ndcg_at_10},
            "ragas":        {"faithfulness": rg.faithfulness,
                             "answer_relevancy": rg.answer_relevancy,
                             "context_precision": rg.context_precision,
                             "context_recall": rg.context_recall,
                             "computed": rg.computed},
        }, f, indent=2)
    print(f"\n  Report saved → {args.output}\n")

    sys.exit(0 if report.overall_pass else 1)

if __name__ == "__main__":
    main()
