"""
LangGraph multi-agent orchestration — LLM-driven supervisor.

Architecture:
  The supervisor is a real LLM agent that reasons over the user query and
  decides which tool to call:
    run_analytics  → analytics_agent  (SQL / DuckDB)
    run_rag        → rag_agent        (policy / syllabus documents)
    run_web        → web_agent        (live internet via Tavily)
    run_viz        → viz_agent        (chart from last SQL result)

  Flow (LangGraph ToolNode pattern):
    supervisor → tools node → supervisor → ... → END

State carries full context across nodes.
LangSmith traces every step automatically via LANGCHAIN_TRACING_V2.
"""
from __future__ import annotations
import json
import math
import operator
import time
import uuid
from typing import Annotated, Optional, Sequence

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from tavily import TavilyClient
from typing_extensions import TypedDict
from enum import Enum
from functools import lru_cache
from app.core.config import get_settings
from app.monitoring.telemetry import (
    RequestMetrics, TokenUsage, get_logger,
    get_metrics_store, track_llm_call,
)
from app.agents.router import classify as route_classify, wants_viz
from app.rag.pipeline import RAGPipeline
from app.sql.engine import SQLEngine
from app.visualization.agent import render

logger = get_logger(__name__)

# ── Singletons ────────────────────────────────────────────────────────────────
_sql_engine:   Optional[SQLEngine]   = None
_rag_pipeline: Optional[RAGPipeline] = None


def _sql() -> SQLEngine:
    global _sql_engine
    if not _sql_engine:
        _sql_engine = SQLEngine()
    return _sql_engine


def _rag() -> RAGPipeline:
    global _rag_pipeline
    if not _rag_pipeline:
        _rag_pipeline = RAGPipeline()
        _rag_pipeline.build_index()
    return _rag_pipeline


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:        Annotated[Sequence[BaseMessage], operator.add]
    request_id:      str
    query_type:      str          # set after first tool call
    sql_result:      dict
    rag_result:      dict
    web_result:      dict
    chart_output:    dict
    recommendations: list
    metrics:         dict


def _query(state: AgentState) -> str:
    # Latest human turn, not the first — otherwise accumulated checkpointer
    # history would pin the supervisor to the very first question forever.
    return next(
        (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        "",
    )


# ── Tools ──────────────────────────────────────────────────────────────────────
# These exist ONLY as schema carriers for `.bind_tools(TOOLS)` — the supervisor
# LLM reads their docstrings to choose a tool. Their bodies never run: the graph
# executes tool calls through `tools_node_wrapper` (not the prebuilt ToolNode),
# which contains the real analytics/RAG/web/viz logic.

@tool
def run_analytics(question: str) -> str:
    """
    Query the Nilgiris student marks database using SQL.
    Use this for any question about counts, averages, pass/fail rates,
    rankings, comparisons, subject performance, school performance,
    gender breakdown, year comparisons, or any numerical data from the DB.
    Returns a plain-text answer and the SQL used.It only contains details of students from nilgiris only.
    """


@tool
def run_rag(question: str) -> str:
    """
    Search the policy, syllabus, curriculum, and survey documents (RAG).

    Use this for questions about:

    SLAS 2025 SURVEY DATA (State Level Achievement Survey, Tamil Nadu):
    - State-level or district-level average scores for Class 3, 5, or 8
    - Subject-wise performance: Tamil, English, Mathematics, Science, EVS,
      Social Science across classes
    - Which districts scored above/below state average
    - Gender-wise performance (boys vs girls scores)
    - Government vs Government-Aided school performance comparison
    - Participation statistics (how many students, schools were assessed)
    - Question-level / learning-outcome-level analysis (Bloom's taxonomy domains)
    - Ennum Ezhuththum scheme impact on learning outcomes
    - SLAS methodology: sampling logic, OMR scanning, Field Investigators,
      Bloom's taxonomy-based assessment design
    - Nilgiris district scores specifically:
        Class 3 — Tamil 62, English 70, Math 47, EVS 73
        Class 5 — Tamil 72, English 51, Math 49, EVS 50
        Class 8 — Tamil 45, English 37, Math 31, Science 31, Social Science 46
    - Recommendations and policy conclusions from the SLAS report

    CURRICULUM & POLICY DOCUMENTS:
    - Samacheer Kalvi syllabus, curriculum framework, textbook structure
    - Examination policy, grading criteria, promotion rules
    - Samagra Shiksha, SCERT programmes, district resource teams
    - Any factual question needing document knowledge, not live DB numbers

    DO NOT use for: counts, averages, pass rates, or rankings from the Nilgiris
    student marks database — use run_analytics for those instead.

    Returns a grounded answer with source context from indexed documents.
    """


@tool
def run_web(question: str) -> str:
    """
    Search the live internet via Tavily.
    Use this ONLY for questions about:
    - Latest government orders, circulars, recent notifications
    - Current exam dates, timetables, news announcements
    - National Education Policy updates, recent scheme launches
    - Anything requiring up-to-date external information not in the DB or documents
    Returns a synthesized answer from web results.
    """


@tool
def run_viz(question: str) -> str:
    """
    Generate a chart or visualization from the last SQL query result.
    Use this ONLY when the user explicitly asks for a chart, graph, plot,
    bar chart, trend, heatmap, or any visual representation.
    Must be called AFTER run_analytics has already retrieved data.
    """


TOOLS = [run_analytics, run_rag, run_web, run_viz]

class Intent(str, Enum):
    CONVERSATIONAL = "conversational"
    ANALYTICS      = "analytics"
    RAG            = "rag"
    WEB            = "web"
    VIZ            = "viz"
    AMBIGUOUS      = "ambiguous"
 
 
CLASSIFIER_SYSTEM = """\
You are a query router for an educational analytics platform.
 
Step 1: Identify what the user is asking for.
Step 2: Check if it could fit multiple categories.
Step 3: Assign an intent and confidence.
 
Intent categories:
  conversational — greetings, thanks, chit-chat, unrelated small talk
  analytics      — student marks, pass rates, school rankings, subject scores, comparisons,
                   counts, averages from the Nilgiris student DB. This includes requests to
                   "show", "list", "top N", "compare" when referring to data or numbers.
  rag            — SLAS survey, syllabus, policy, state-level document knowledge
  web            — needs live internet: recent orders, current news
  viz            — ONLY when user explicitly uses words: chart, graph, plot, bar chart,
                   pie chart, heatmap, visualize, visualization, trend graph, line graph.
                   "Show me a bar chart of X" → viz. "Show top 5 schools" → analytics.
  ambiguous      — could fit 2+ categories equally well AND none of the above rules apply
 
Disambiguation rules (apply these first):
  - "top N ...", "best schools", "average scores", "pass rate", "failures" → analytics
  - "show X" where X is data/rankings/numbers → analytics (not viz)
  - "show a chart/graph/plot of X" → viz
  - "SLAS", "state average", "Tamil Nadu average", "Class 3/5/8" → rag
  - "latest news", "recent order", "current policy" → web
 
Confidence guide:
  0.95+ → only one category fits, clear disambiguation rule matches
  0.80  → one clear winner but another category is plausible
  0.65  → two categories roughly tied, leaning one way
  <0.65 → genuinely unclear → use "ambiguous"
 
Reply with ONLY this JSON (no extra text):
{"reasoning": "<one sentence>", "intent": "<value>", "confidence": <float>}
"""
 
CONFIDENCE_THRESHOLD = 0.65
 
 
@lru_cache(maxsize=256)
def classify_intent(query: str) -> tuple[str, float]:
    """
    Classifies query using a small fast model before touching the full supervisor.
    Cached so identical repeated queries cost nothing.
    """
    cfg = get_settings()
    classifier_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.0,
        max_tokens=80,
        api_key=cfg.GROQ_API_KEY,
    )
    try:
        response = classifier_llm.invoke([
            SystemMessage(content=CLASSIFIER_SYSTEM),
            HumanMessage(content=query),
        ])
        raw = response.content.strip()
        raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        parsed = json.loads(raw)
        intent     = parsed.get("intent", "ambiguous")
        confidence = float(parsed.get("confidence", 0.5))
    except Exception:
        intent     = "ambiguous"
        confidence = 0.0
 
    return intent, confidence
 
 
def get_conversational_reply(query: str, history: Optional[Sequence[BaseMessage]] = None) -> str:
    """Tiny model for friendly replies — no tools, no RAG, no cost.

    Prior turns are threaded in so the assistant can recall earlier context
    (e.g. a name the user mentioned) across a conversation.
    """
    cfg = get_settings()
    small_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        max_tokens=80,
        api_key=cfg.GROQ_API_KEY,
    )
    system = (
        "You are a friendly assistant for the Nilgiris DMU educational platform. "
        "Reply warmly and briefly (1-2 sentences). "
        "If relevant, mention you can help with student performance data or SLAS results."
    )
    # Keep only the plain conversational turns (drop tool-call/ToolMessage noise)
    # and trim to the most recent few to stay within the tiny model's budget.
    prior = [
        m for m in (history or [])
        if isinstance(m, (HumanMessage, AIMessage))
        and not getattr(m, "tool_calls", None)
        and (m.content or "").strip()
    ][-6:]
    try:
        resp = small_llm.invoke(
            [SystemMessage(content=system)] + prior + [HumanMessage(content=query)]
        )
        return resp.content.strip()
    except Exception:
        return "Hey! Ask me anything about Nilgiris student performance or Tamil Nadu's Education performance ."

# ── Supervisor system prompt ──────────────────────────────────────────────────

SUPERVISOR_SYSTEM = """\
You are the supervisor agent for the Nilgiris District Monitoring Unit (DMU) \
educational analytics platform.

Your ONLY job is to decide which specialist tool to call based on the user's question.
Think step by step, then call the most appropriate tool.

TOOL SELECTION RULES:

run_analytics → Nilgiris student marks DATABASE queries. Use for:
  - Counts, averages, pass/fail rates from the local school database
  - School-level or student-level rankings and comparisons
  - Subject performance, gender breakdown, year-over-year trends
  - Any question answered by SQL over live student records
  - Run analytics for students details from Nilgiris district only

run_rag → DOCUMENT knowledge. Use for:
  - SLAS 2025 survey: state/district average scores, Class 3/5/8 results,
    subject-wise performance, gender gaps, Govt vs Aided comparison,
    districts above/below state average, learning outcome (LO) analysis,
    Bloom's taxonomy domains, Ennum Ezhuththum impact, SLAS methodology
  - Nilgiris-specific SLAS scores (Class 3 Math 47, Class 8 Math 31, etc.)
  - Samacheer Kalvi syllabus, curriculum, grading policy, promotion rules
  - SCERT, Samagra Shiksha programmes, district resource teams

run_web → LIVE internet only. Use for:
  - Recent government orders, circulars, new notifications
  - Current exam timetables, latest news announcements
  - National Education Policy updates, new scheme launches
  - Latest news

run_viz → CHART generation only. Use ONLY when:
  - User explicitly asks for a chart, graph, plot, or visual
  - AND run_analytics was already called earlier in this conversation

DISAMBIGUATION — when the query could be either analytics or RAG:
  - "average marks in Nilgiris schools" → run_analytics (from DB)
  - "how does Nilgiris compare to Tamil Nadu state average" → run_rag (SLAS doc)
  - "pass percentage this year" → run_analytics (DB)
  - "SLAS survey results for Class 8" → run_rag (document)
  - "which districts performed best in Tamil Nadu" → run_rag (SLAS doc)
  - "which schools in Nilgiris performed best" → run_analytics (DB)

IMPORTANT:
- Call EXACTLY ONE tool per turn.
- answer directly if it is conversational or greeting.
- Do NOT call run_viz unless run_analytics was already called first.
- For hybrid questions (e.g. "how does Nilgiris rank vs Tamil Nadu?"),
  call run_analytics first, then run_rag on the next turn.
"""



# ── Nodes ─────────────────────────────────────────────────────────────────────

# The API extracts the final answer by looking for one of these tags in an
# AIMessage. The supervisor stamps the right tag onto the final reply.
_ANSWER_TAG = {"analytics": "[ANALYTICS]", "rag": "[RAG]", "web": "[WEB]", "viz": "[ANALYTICS]"}


def _final_answer_message(state: AgentState) -> AIMessage:
    """Build the tagged final AIMessage from whatever result a tool produced."""
    if (state.get("sql_result") or {}).get("answer"):
        return AIMessage(content=f"[ANALYTICS] {state['sql_result']['answer']}")
    if (state.get("rag_result") or {}).get("answer"):
        return AIMessage(content=f"[RAG] {state['rag_result']['answer']}")
    if (state.get("web_result") or {}).get("answer"):
        return AIMessage(content=f"[WEB] {state['web_result']['answer']}")
    # Tool failed before storing a structured result — surface its message.
    last_tool = next(
        (m for m in reversed(state["messages"]) if isinstance(m, ToolMessage)), None
    )
    tag = _ANSWER_TAG.get(state.get("query_type", ""), "[ANALYTICS]")
    body = last_tool.content if last_tool else "I couldn't produce an answer for that."
    return AIMessage(content=f"{tag} {body}")


def supervisor_node(state: AgentState) -> dict:
    """
    Two-layer supervisor:
      Layer 1 — classify_intent()  (~80ms, llama-3.1-8b-instant)
                Handles clear cases without touching the 70b model.
      Layer 2 — Full supervisor LLM (only for ambiguous cases)
                Your existing logic, unchanged.
    """
    query = _query(state)

    # ── Post-tool: a specialist already ran this turn ─────────────────────────
    # Decide whether a follow-up hop is needed (analytics → viz) or we're done.
    # Without this the supervisor re-routes the same query forever until the
    # graph recursion limit throws.
    if state.get("messages") and isinstance(state["messages"][-1], ToolMessage):
        if wants_viz(query) and state.get("sql_result") and not state.get("chart_output"):
            logger.info("second_hop", route="viz")
            return {"messages": [AIMessage(
                content="",
                tool_calls=[{"name": "run_viz", "args": {"question": query}, "id": "hop_viz"}],
            )]}
        logger.info("supervisor_finish", query_type=state.get("query_type"))
        return {"messages": [_final_answer_message(state)]}

    # ── Layer 1: fast classifier ──────────────────────────────────────────────
    intent, confidence = classify_intent(query)
    logger.info("classifier_result", intent=intent, confidence=confidence, query=query[:60])
 
    if confidence >= CONFIDENCE_THRESHOLD:
 
        # Conversational → reply directly, no tool call, hits END immediately
        if intent == "conversational":
            # All prior turns; the current query is the trailing HumanMessage,
            # so drop it here and let get_conversational_reply re-append it.
            reply = get_conversational_reply(query, history=list(state["messages"])[:-1])
            logger.info("fast_path", route="conversational")
            return {"messages": [AIMessage(content=reply)]}
 
        # Clear tool intent → synthetic tool call, skip the supervisor LLM entirely.
        # Note "viz" maps to run_analytics, NOT run_viz: a chart query like
        # "bar chart of school Math averages" is also a data question, so we fetch
        # the data first. The post-tool second-hop logic above (wants_viz + sql_result)
        # then adds the chart automatically. The classifier never emits a direct
        # run_viz call — that avoids the "no data to visualize" dead-end.
        tool_map = {
            "analytics": "run_analytics",
            "rag":       "run_rag",
            "web":       "run_web",
            "viz":       "run_analytics",
        }
        if intent in tool_map:
            logger.info("fast_path", route=intent, tool=tool_map[intent])
            synthetic_ai = AIMessage(
                content="",
                tool_calls=[{
                    "name": tool_map[intent],
                    "args": {"question": query},
                    "id":   f"fast_{intent}",
                }],
            )
            return {"messages": [synthetic_ai]}
 
    # ── Layer 2: full supervisor LLM (ambiguous or low confidence) ────────────
    logger.info("slow_path", reason="ambiguous_or_low_confidence", confidence=confidence)
 
    cfg = get_settings()
    llm = ChatGroq(
        model=cfg.LLM_MODEL,
        temperature=0.0,
        api_key=cfg.GROQ_API_KEY,
    ).bind_tools(TOOLS)

    messages = [SystemMessage(content=SUPERVISOR_SYSTEM)] + list(state["messages"])
    try:
        response = llm.invoke(messages)
        # A usable tool call → route through it.
        if response.tool_calls:
            logger.info(
                "supervisor_decision",
                tool_calls=[tc["name"] for tc in response.tool_calls],
            )
            return {"messages": [response]}
        # No tool call but the model answered directly → pass the reply through.
        if (response.content or "").strip():
            return {"messages": [response]}
        raise ValueError("supervisor returned neither a tool call nor content")
    except Exception as e:
        # Weak models sometimes emit malformed tool calls (e.g. {"sql": ...}
        # instead of {"question": ...}), which Groq rejects with a 400
        # tool_use_failed. Fall back to the deterministic regex router, which
        # can't crash and always produces a schema-valid {"question": ...} call.
        logger.warning("supervisor_llm_failed", error=str(e))

    route = "viz" if (wants_viz(query) and state.get("sql_result")) else route_classify(query)
    tool_map = {
        "analytics": "run_analytics",
        "rag":       "run_rag",
        "web":       "run_web",
        "viz":       "run_viz",
    }
    logger.info("supervisor_fallback", route=route)
    synthetic_ai = AIMessage(
        content="",
        tool_calls=[{
            "name": tool_map.get(route, "run_rag"),
            "args": {"question": query},
            "id":   f"fallback_{route}",
        }],
    )
    return {"messages": [synthetic_ai]}


def tools_node_wrapper(state: AgentState) -> dict:
    """
    Executes the tool calls made by the supervisor, then enriches AgentState
    with the structured results (sql_result, rag_result, etc.).
    """
    last_ai: AIMessage = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if not last_ai or not last_ai.tool_calls:
        return {}

    updates: dict = {}
    tool_messages = []

    for tc in last_ai.tool_calls:
        name = tc["name"]
        args = tc["args"]
        tid  = tc.get("id", str(uuid.uuid4())[:8])
        q    = args.get("question", _query(state))

        t0 = time.perf_counter()

        try:
            if name == "run_analytics":
                result = _sql().run(q)
                latency = round((time.perf_counter() - t0) * 1000, 1)

                # Build recommendations
                recs = []
                if result.result_df is not None and not result.result_df.empty:
                    cfg = get_settings()
                    rec_llm = ChatGroq(model=cfg.LLM_MODEL, temperature=0.1, api_key=cfg.GROQ_API_KEY)
                    try:
                        rec_prompt = (
                            f"Based on this educational data for Nilgiris district:\n"
                            f"{result.result_df.head(10).to_string(index=False)}\n\n"
                            f"Original question: {q}\n\n"
                            "Give exactly 2 short, actionable recommendations. "
                            "One line each, numbered list only:"
                        )
                        rec_resp = rec_llm.invoke(rec_prompt)
                        recs = [
                            {"rank": i + 1, "text": line.lstrip("0123456789.) ").strip()}
                            for i, line in enumerate(rec_resp.content.strip().splitlines())
                            if line.strip()
                        ][:2]
                    except Exception:
                        pass

                # Serialize DataFrame
                result_records, result_columns = [], []
                if result.result_df is not None and not result.result_df.empty:
                    try:
                        safe = result.result_df.astype(object).where(result.result_df.notna(), other=None)
                        raw = safe.to_dict(orient="records")
                        for row in raw:
                            clean = {}
                            for k, v in row.items():
                                if v is None:
                                    clean[str(k)] = None
                                elif isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                                    clean[str(k)] = None
                                elif hasattr(v, "item"):
                                    try:
                                        iv = v.item()
                                        clean[str(k)] = None if (isinstance(iv, float) and (math.isnan(iv) or math.isinf(iv))) else iv
                                    except Exception:
                                        clean[str(k)] = str(v)
                                else:
                                    clean[str(k)] = v
                            result_records.append(clean)
                        result_columns = list(safe.columns)
                    except Exception:
                        pass

                # Record metrics
                m = RequestMetrics(
                    request_id=state.get("request_id", str(uuid.uuid4())[:8]),
                    query=q, query_type="analytics",
                    agent_path=["supervisor", "analytics_agent"],
                    token_usage=result.token_usage,
                    judge_score=result.judge.score,
                    sql_executed=result.error is None,
                    sql_error=result.error,
                    latency_ms=latency,
                )
                get_metrics_store().record(m)

                updates["sql_result"] = {
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
                }
                updates["recommendations"] = recs
                updates["query_type"] = "analytics"
                tool_result = result.answer

            elif name == "run_rag":
                answer, contexts = _rag().answer(q)
                latency = round((time.perf_counter() - t0) * 1000, 1)

                m = RequestMetrics(
                    request_id=state.get("request_id", ""),
                    query=q, query_type="rag",
                    agent_path=["supervisor", "rag_agent"],
                    latency_ms=latency,
                )
                get_metrics_store().record(m)

                updates["rag_result"] = {"answer": answer, "contexts": contexts}
                updates["query_type"] = "rag"
                tool_result = answer

            elif name == "run_web":
                cfg = get_settings()
                try:
                    client = TavilyClient(api_key=cfg.TAVILY_API_KEY)
                    results = client.search(query=q, max_results=3, include_answer=True)
                    snippets = [
                        f"{r.get('title', '')}\n{r.get('content', '')[:400]}"
                        for r in results.get("results", [])
                    ]
                    tavily_ans = results.get("answer", "")
                    context = "\n\n".join(snippets)[:2000]
                    llm = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
                    synth = (
                        f"Answer using ONLY the web results below.\n\n"
                        f"Summary: {tavily_ans}\n\nResults:\n{context}\n\nQuestion: {q}\nAnswer:"
                    )
                    resp = llm.invoke(synth)
                    answer = resp.content.strip()
                except Exception as e:
                    answer = f"Web search failed: {e}"
                    logger.error("web_tool_error", error=str(e))

                latency = round((time.perf_counter() - t0) * 1000, 1)
                m = RequestMetrics(
                    request_id=state.get("request_id", ""),
                    query=q, query_type="web",
                    agent_path=["supervisor", "web_agent"],
                    latency_ms=latency,
                )
                get_metrics_store().record(m)

                updates["web_result"] = {"answer": answer}
                updates["query_type"] = "web"
                tool_result = answer

            elif name == "run_viz":
                sql_res = state.get("sql_result", {})
                records = sql_res.get("result_records")
                import pandas as pd
                df = pd.DataFrame(records) if records else None
                out = render(df, q)
                updates["chart_output"] = {
                    "chart_json": out.chart_json,
                    "chart_type": out.chart_type,
                    "title":      out.title,
                    "html":       out.html,
                    "error":      out.error,
                }
                tool_result = f"Chart rendered: {out.title} ({out.chart_type})"

            else:
                tool_result = f"Unknown tool: {name}"

        except Exception as e:
            logger.error("tool_execution_error", tool=name, error=str(e))
            tool_result = f"Tool {name} failed: {e}"

        tool_messages.append(
            ToolMessage(content=str(tool_result), tool_call_id=tid, name=name)
        )

    updates["messages"] = tool_messages
    return updates


def _should_continue(state: AgentState) -> str:
    """
    After supervisor runs: if it made tool calls → execute them.
    If the last message is a plain AIMessage with no tool calls → END.
    """
    last_ai = next(
        (m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
        None,
    )
    if last_ai and getattr(last_ai, "tool_calls", None):
        return "tools"
    return END


# ── Graph assembly ────────────────────────────────────────────────────────────

def build_graph():
    g = StateGraph(AgentState)

    g.add_node("supervisor", supervisor_node)
    g.add_node("tools",      tools_node_wrapper)

    g.set_entry_point("supervisor")

    g.add_conditional_edges(
        "supervisor",
        _should_continue,
        {"tools": "tools", END: END},
    )

    # After tools run, always go back to supervisor so it can
    # decide if another tool call is needed (e.g. analytics → viz)
    g.add_edge("tools", "supervisor")

    return g.compile(checkpointer=MemorySaver())


_graph = None


def get_graph():
    global _graph
    if not _graph:
        _graph = build_graph()
    return _graph
