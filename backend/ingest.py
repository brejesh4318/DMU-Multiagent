#!/usr/bin/env python3
"""
Usage:
  python ingest.py                   # use DATA_DIR from .env
  python ingest.py --data-dir ./data # override
  python ingest.py --rebuild-rag     # also rebuild FAISS index
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.config import get_settings
from app.ingestion.excel_loader import ingest_all_excel
from app.monitoring.telemetry import configure_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default=None)
    parser.add_argument("--rebuild-rag", action="store_true")
    args = parser.parse_args()

    configure_logging("INFO")
    print("=" * 55)
    print("  DMU Analytics — Data Ingestion")
    print("=" * 55)

    counts = ingest_all_excel(args.data_dir)
    print(f"\n✅ marks table:    {counts['marks_rows']:,} rows")
    print(f"✅ students table: {counts['student_rows']:,} rows")
    print(f"✅ Files loaded:   {counts['files']}")

    if args.rebuild_rag:
        from pathlib import Path
        cfg = get_settings()
        cache = Path(cfg.VECTOR_STORE_DIR) / "rag_store.pkl"
        cache.unlink(missing_ok=True)
        from app.rag.pipeline import RAGPipeline
        rag = RAGPipeline()
        n = rag.build_index(args.data_dir)
        print(f"✅ RAG index:      {n:,} child chunks")

    print("\n✅ Done.\n")

if __name__ == "__main__":
    main()
