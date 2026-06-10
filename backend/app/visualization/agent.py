"""
Visualization Agent — converts DuckDB DataFrames to Plotly charts.
Auto-selects chart type based on query keywords and data shape.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_groq import ChatGroq
from app.core.config import get_settings
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)

_VALID_TYPES = {"line", "bar", "pie", "heatmap"}


@dataclass
class ChartOutput:
    chart_json: dict[str, Any]
    chart_type: str
    title: str
    html: str = ""
    error: Optional[str] = None


def _pick_type(df: pd.DataFrame, query: str) -> str:
    """Deterministic fallback used when the LLM planner is unavailable/invalid."""
    q = query.lower()
    if any(w in q for w in ["trend", "over time", "year", "2025", "2026"]):
        return "line"
    if any(w in q for w in ["heatmap", "matrix", "across"]):
        return "heatmap"
    if len(df) <= 3:
        return "pie"
    return "bar"


def _llm_plan(df: pd.DataFrame, query: str) -> Optional[dict[str, Any]]:
    """
    Ask the VIZ_MODEL (70B) to choose chart type + axis columns from the query
    and the actual data shape. Returns a validated dict {chart_type, x, y, y2}
    or None on any failure (caller falls back to the rule-based picker).
    """
    cfg = get_settings()
    cols     = list(df.columns)
    dtypes   = {c: ("numeric" if pd.api.types.is_numeric_dtype(df[c]) else "categorical") for c in cols}
    sample   = df.head(5).to_string(index=False)

    prompt = (
        "You are a data-visualization planner. Choose the best Plotly chart for the "
        "user's question and data. Respond with ONLY a JSON object, no prose.\n\n"
        f'Question: "{query}"\n'
        f"Columns and types: {json.dumps(dtypes)}\n"
        f"Row count: {len(df)}\n"
        f"Sample rows:\n{sample}\n\n"
        "JSON schema:\n"
        '{"chart_type": "line|bar|pie|heatmap", "x": "<column>", "y": "<numeric column>", '
        '"y2": "<second categorical column, only for heatmap, else null>"}\n'
        "Rules: pie only for <=4 rows with one category + one numeric; line for trends "
        "over an ordered column (e.g. year); heatmap needs two categorical columns + one numeric; "
        "otherwise bar. x/y/y2 must be exact column names from the list above."
    )

    try:
        llm  = ChatGroq(model=cfg.VIZ_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        t0   = time.perf_counter()
        resp = llm.invoke(prompt)
        time.sleep(0.3)  # Groq rate-limit courtesy
        raw  = (resp.content or "").strip()
        # Tolerate ```json fences / surrounding text.
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return None
        plan = json.loads(raw[start : end + 1])

        ctype = str(plan.get("chart_type", "")).lower()
        if ctype not in _VALID_TYPES:
            return None
        # Validate referenced columns actually exist.
        for key in ("x", "y", "y2"):
            v = plan.get(key)
            if v is not None and v not in cols:
                if key == "y2":
                    plan["y2"] = None
                else:
                    return None
        logger.info("viz_llm_plan", chart_type=ctype, latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return plan
    except Exception as e:
        logger.warning("viz_llm_plan_failed", error=str(e))
        return None


def render(df: Optional[pd.DataFrame], query: str, title: Optional[str] = None) -> ChartOutput:
    if df is None or df.empty:
        return ChartOutput(chart_json={}, chart_type="none",
                           title="No data", error="Empty result")

    t = title or query[:80]
    cols     = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]

    # LLM-driven chart selection (70B), with deterministic fallback.
    plan  = _llm_plan(df, query)
    if plan:
        ctype  = plan["chart_type"]
        x_col  = plan.get("x")
        y_col  = plan.get("y")
        y2_col = plan.get("y2")
    else:
        ctype  = _pick_type(df, query)
        x_col = y_col = y2_col = None

    # Sensible defaults if the planner omitted axes (or we fell back to rules).
    x_col  = x_col  or (cat_cols[0] if cat_cols else cols[0])
    y_col  = y_col  or (num_cols[0] if num_cols else (cols[1] if len(cols) > 1 else cols[0]))
    y2_col = y2_col or (cat_cols[1] if len(cat_cols) >= 2 else None)

    try:
        if ctype == "line" and num_cols:
            fig = px.line(df, x=x_col, y=y_col, title=t, markers=True)
        elif ctype == "pie" and num_cols and cat_cols:
            fig = px.pie(df, values=y_col if y_col in num_cols else num_cols[0],
                         names=x_col if x_col in cat_cols else cat_cols[0], title=t)
        elif ctype == "heatmap" and y2_col and num_cols:
            pivot = df.pivot_table(index=x_col, columns=y2_col,
                                   values=y_col if y_col in num_cols else num_cols[0])
            fig = px.imshow(pivot, title=t, color_continuous_scale="Blues")
        else:
            ctype = "bar"
            x = x_col
            y = y_col if y_col in num_cols else (num_cols[0] if num_cols else y_col)
            fig = px.bar(df, x=x, y=y, title=t, color=y,
                         color_continuous_scale="Blues", text=y)
            fig.update_traces(textposition="outside")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"family": "Inter, sans-serif"},
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
        chart_json = json.loads(fig.to_json())
        html       = fig.to_html(full_html=False, include_plotlyjs="cdn")
        logger.info("chart_rendered", type=ctype, rows=len(df))
        return ChartOutput(chart_json=chart_json, chart_type=ctype, title=t, html=html)

    except Exception as e:
        logger.error("chart_error", error=str(e))
        return ChartOutput(chart_json={}, chart_type="error", title=t, error=str(e))
