"""
Visualization Agent — converts DuckDB DataFrames to Plotly charts.
Auto-selects chart type based on query keywords and data shape.
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)


@dataclass
class ChartOutput:
    chart_json: dict[str, Any]
    chart_type: str
    title: str
    html: str = ""
    error: Optional[str] = None


def _pick_type(df: pd.DataFrame, query: str) -> str:
    q = query.lower()
    if any(w in q for w in ["trend", "over time", "year", "2025", "2026"]):
        return "line"
    if any(w in q for w in ["heatmap", "matrix", "across"]):
        return "heatmap"
    if len(df) <= 3:
        return "pie"
    return "bar"


def render(df: Optional[pd.DataFrame], query: str, title: Optional[str] = None) -> ChartOutput:
    if df is None or df.empty:
        return ChartOutput(chart_json={}, chart_type="none",
                           title="No data", error="Empty result")

    t = title or query[:80]
    cols     = list(df.columns)
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    ctype    = _pick_type(df, query)

    try:
        if ctype == "line" and num_cols:
            fig = px.line(df, x=cat_cols[0] if cat_cols else cols[0],
                          y=num_cols[0], title=t, markers=True)
        elif ctype == "pie" and num_cols and cat_cols:
            fig = px.pie(df, values=num_cols[0], names=cat_cols[0], title=t)
        elif ctype == "heatmap" and len(cat_cols) >= 2 and num_cols:
            pivot = df.pivot_table(index=cat_cols[0], columns=cat_cols[1], values=num_cols[0])
            fig = px.imshow(pivot, title=t, color_continuous_scale="Blues")
        else:
            ctype = "bar"
            x = cat_cols[0] if cat_cols else cols[0]
            y = num_cols[0] if num_cols else cols[1] if len(cols) > 1 else cols[0]
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
