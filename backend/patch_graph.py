"""
Run from your backend folder:
    cd backend
    python patch_graph.py
"""
import re, ast, sys
from pathlib import Path

path = Path("app/agents/graph.py")
if not path.exists():
    print("ERROR: Run this from your backend/ folder")
    sys.exit(1)

src = path.read_text(encoding="utf-8")

# ── Remove every trace of _df_to_records ─────────────────────────────────────

# 1. Remove the helper function definition if present
src = re.sub(
    r'\ndef _df_to_records\(df\):.*?(?=\ndef |\nclass |\n# ──)',
    '\n',
    src,
    flags=re.DOTALL
)

# 2. Replace any call to _df_to_records(...) in the return dict
src = re.sub(
    r'"result_df"\s*:\s*_df_to_records\([^)]+\)[^,\n]*',
    '',
    src
)
src = re.sub(
    r'"result_records"\s*:\s*_df_to_records\([^)]+\)[^,\n]*,?',
    '',
    src
)

# 3. Remove stray import math as _math if it was injected
src = re.sub(r'^import math as _math\n', '', src, flags=re.MULTILINE)

# ── Rebuild analytics_agent return dict cleanly ───────────────────────────────
# Find the return { ... } block inside analytics_agent and replace it entirely

OLD_RETURN = re.compile(
    r'(    return \{[^}]*"sql_result"\s*:\s*\{[^}]*\}[^}]*\})',
    re.DOTALL
)

NEW_RETURN = '''\
    # ── Convert DataFrame → JSON-safe records ─────────────────────────────
    result_records = None
    result_columns = []
    if result.result_df is not None and not result.result_df.empty:
        try:
            import math as _m
            safe = result.result_df.astype(object).where(
                result.result_df.notna(), other=None
            )
            raw_records = safe.to_dict(orient="records")
            result_records = []
            for row in raw_records:
                clean = {}
                for k, v in row.items():
                    if v is None:
                        clean[str(k)] = None
                    elif isinstance(v, float) and (_m.isnan(v) or _m.isinf(v)):
                        clean[str(k)] = None
                    elif hasattr(v, "item"):
                        try:
                            iv = v.item()
                            clean[str(k)] = None if (isinstance(iv, float) and (_m.isnan(iv) or _m.isinf(iv))) else iv
                        except Exception:
                            clean[str(k)] = str(v)
                    else:
                        clean[str(k)] = v
                result_records.append(clean)
            result_columns = list(safe.columns)
        except Exception as exc:
            result_records = []
            result_columns = []

    return {
        "messages": [AIMessage(content=f"[ANALYTICS]\\n{result.answer}")],
        "sql_result": {
            "answer":         result.answer,
            "sql":            result.sql,
            "result_records": result_records,
            "result_columns": result_columns,
            "judge_score":    int(result.judge.score) if result.judge.score is not None else None,
            "judge_issues":   list(result.judge.issues or []),
            "attempts":       int(result.attempts),
            "latency_ms":     float(result.latency_ms),
            "tokens":         int(result.token_usage.total_tokens),
            "cost_usd":       float(result.token_usage.cost_usd),
            "error":          str(result.error) if result.error else None,
        },
        "recommendations": recs,
    }'''

# Find and replace the return block
match = OLD_RETURN.search(src)
if match:
    src = src[:match.start()] + NEW_RETURN + src[match.end():]
    print("Replaced return block")
else:
    # Fallback: find "return {" after get_metrics_store().record and replace to end of function
    idx = src.find('get_metrics_store().record(m)')
    if idx != -1:
        end_of_func = src.find('\ndef ', idx)
        block_start = src.find('\n    return {', idx)
        if block_start != -1 and (end_of_func == -1 or block_start < end_of_func):
            # Find matching closing brace
            depth = 0
            pos = block_start
            for i, ch in enumerate(src[block_start:]):
                if ch == '{': depth += 1
                if ch == '}':
                    depth -= 1
                    if depth == 0:
                        pos = block_start + i + 1
                        break
            src = src[:block_start] + '\n' + NEW_RETURN + src[pos:]
            print("Replaced return block (fallback)")
        else:
            print("WARNING: Could not locate return block — manual fix needed")
    else:
        print("WARNING: Could not locate return block — manual fix needed")

# ── Also fix viz_agent if it still uses result_df ────────────────────────────
if 'state.get("sql_result", {}).get("result_df")' in src:
    src = src.replace(
        'state.get("sql_result", {}).get("result_df")',
        '(lambda r: __import__("pandas").DataFrame(r) if r else None)(state.get("sql_result", {}).get("result_records"))'
    )
    print("Fixed viz_agent result_df reference")

# ── Verify syntax ─────────────────────────────────────────────────────────────
try:
    ast.parse(src)
    print("Syntax OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    print("Saving anyway — check the file manually")

path.write_text(src, encoding="utf-8")
print(f"Written: {path}")

# ── Quick sanity checks ───────────────────────────────────────────────────────
assert '_df_to_records' not in src, "FAIL: _df_to_records still present"
assert 'result_records' in src,     "FAIL: result_records missing"
assert 'astype(object)' in src,     "FAIL: astype fix missing"
print("All checks passed")
print()
print("Now restart uvicorn:")
print("  uvicorn app.api.main:app --reload --host 0.0.0.0 --port 8000")
