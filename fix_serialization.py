"""
Run this script from your dmu folder:
    python fix_serialization.py

It patches graph.py and main.py in-place to fix the DataFrame serialization error.
"""
import os, sys
from pathlib import Path

# Auto-find the backend folder
candidates = [
    Path("backend"),
    Path("dmu/backend"),
    Path("../backend"),
]
backend = None
for c in candidates:
    if (c / "app" / "agents" / "graph.py").exists():
        backend = c
        break

if not backend:
    print("ERROR: Could not find backend/app/agents/graph.py")
    print("Run this script from your dmu/ folder")
    sys.exit(1)

print(f"Found backend at: {backend.resolve()}")

# ═══════════════════════════════════════════════════════
# FIX 1: graph.py — convert DataFrame before storing
# ═══════════════════════════════════════════════════════
graph_path = backend / "app" / "agents" / "graph.py"
graph_src  = graph_path.read_text(encoding="utf-8")

# Check if already fixed
if "result_records" in graph_src and "astype(object)" in graph_src:
    print("graph.py: already fixed")
else:
    # Pattern 1: old return dict still using result_df
    if '"result_df":' in graph_src:
        graph_src = graph_src.replace(
            '"result_df":    result.result_df,',
            '"result_records": _df_to_records(result.result_df),  # JSON safe'
        )
        print("graph.py: patched result_df in return dict")

    # Pattern 2: add helper function after imports if not present
    if "_df_to_records" not in graph_src:
        helper = '''
import math as _math

def _df_to_records(df):
    """Convert DataFrame to JSON-safe list of dicts. Handles NaN/numpy types."""
    if df is None:
        return None
    try:
        import numpy as np
        safe = df.astype(object).where(df.notna(), other=None)
        records = safe.to_dict(orient="records")
        result = []
        for row in records:
            clean = {}
            for k, v in row.items():
                if v is None:
                    clean[str(k)] = None
                elif isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                    clean[str(k)] = None
                elif hasattr(v, "item"):  # numpy scalar
                    try:
                        item = v.item()
                        clean[str(k)] = None if (isinstance(item, float) and (_math.isnan(item) or _math.isinf(item))) else item
                    except Exception:
                        clean[str(k)] = str(v)
                else:
                    clean[str(k)] = v
            result.append(clean)
        return result
    except Exception as e:
        return []

'''
        # Insert after the last import line
        lines = graph_src.splitlines()
        last_import = 0
        for i, line in enumerate(lines):
            if line.startswith("from ") or line.startswith("import "):
                last_import = i
        lines.insert(last_import + 1, helper)
        graph_src = "\n".join(lines)
        print("graph.py: added _df_to_records helper")

    graph_path.write_text(graph_src, encoding="utf-8")

# ═══════════════════════════════════════════════════════
# FIX 2: main.py — add _safe() and wrap all state access
# ═══════════════════════════════════════════════════════
main_path = backend / "app" / "api" / "main.py"
main_src  = main_path.read_text(encoding="utf-8")

if "def _safe(" in main_src:
    print("main.py: _safe already present")
else:
    safe_func = '''
import math as _math

def _safe(obj):
    """Recursively convert any object to JSON-serializable Python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, float):
        if _math.isnan(obj) or _math.isinf(obj):
            return None
        return obj
    if isinstance(obj, int):
        return obj
    type_name = type(obj).__name__
    module    = type(obj).__module__ or ""
    if module.startswith("numpy"):
        try:
            v = obj.item()
            return None if (isinstance(v, float) and (_math.isnan(v) or _math.isinf(v))) else v
        except Exception:
            return str(obj)
    if type_name == "DataFrame":
        try:
            safe = obj.astype(object).where(obj.notna(), other=None)
            records = safe.to_dict(orient="records")
            return [{str(k): _safe(v) for k, v in row.items()} for row in records]
        except Exception:
            return []
    if type_name == "Series":
        try:
            return [_safe(v) for v in obj.tolist()]
        except Exception:
            return []
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    try:
        import dataclasses
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return _safe(dataclasses.asdict(obj))
    except Exception:
        pass
    return str(obj)

'''
    # Insert before the first route definition
    insert_point = main_src.find("@asynccontextmanager")
    if insert_point == -1:
        insert_point = main_src.find("app = ")
    main_src = main_src[:insert_point] + safe_func + main_src[insert_point:]
    print("main.py: added _safe() function")

# Now make sure the query endpoint wraps state in _safe
if "_safe(state.get" not in main_src:
    # Find the line that reads sql_result from state and wrap it
    old_sql_r = 'sql_r = state.get("sql_result", {})'
    new_sql_r = 'sql_r = _safe(state.get("sql_result", {}))'
    if old_sql_r in main_src:
        main_src = main_src.replace(old_sql_r, new_sql_r)
        print("main.py: wrapped sql_result in _safe()")
    else:
        # More aggressive: find the return QueryResponse block and wrap everything
        old_chart = 'chart=state.get("chart_output") or None,'
        new_chart = 'chart=_safe(state.get("chart_output")) or None,'
        if old_chart in main_src:
            main_src = main_src.replace(old_chart, new_chart)
            print("main.py: wrapped chart_output in _safe()")

        old_recs = 'recommendations=state.get("recommendations", []),'
        new_recs = 'recommendations=_safe(state.get("recommendations", [])),'
        if old_recs in main_src:
            main_src = main_src.replace(old_recs, new_recs)
            print("main.py: wrapped recommendations in _safe()")

main_path.write_text(main_src, encoding="utf-8")

# ═══════════════════════════════════════════════════════
# VERIFY: parse both files
# ═══════════════════════════════════════════════════════
import ast
errors = []
for path in [graph_path, main_path]:
    try:
        ast.parse(path.read_text(encoding="utf-8"))
        print(f"Syntax OK: {path}")
    except SyntaxError as e:
        errors.append(f"SYNTAX ERROR in {path}: {e}")

if errors:
    for e in errors:
        print(e)
    sys.exit(1)

print()
print("=" * 50)
print("  DONE. Restart your backend:")
print("  uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000")
print("=" * 50)
