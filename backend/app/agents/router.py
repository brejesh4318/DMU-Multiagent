"""
Query router — classifies queries into analytics / rag / web.

Routing strategy (in order):
  1. _RAG regex   — explicit doc/policy keywords → rag  (fast, no LLM)
  2. _WEB regex   — external/current-events keywords → web  (fast, no LLM)
  3. _ANALYTICS regex:
       a. also matches _DOC_PHRASING? → call LLM to disambiguate  (~0.5s)
       b. clean analytics query?      → analytics  (fast, no LLM)
  4. Default      → rag
"""
from __future__ import annotations
import re
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)

QueryType = str  # "analytics" | "rag" | "web"

_ANALYTICS = re.compile(
    r"\b(how many|count|total|average|avg|sum|highest|lowest|top|bottom|"
    r"percentage|percent|fail|pass|rank|compare|mark|score|student|school|"
    r"subject|district|physics|chemistry|math|english|biology|commerce|"
    r"ooty|gudalur|coonoor|kotagiri|nilgiris|kunjappanai|masinagudi|"
    r"2025|2026|year|gender|boys?|girls?|management|tribal|govt|"
    r"chart|plot|graph|bar|trend|distribution|heatmap)\b",
    re.IGNORECASE,
)

# Checked BEFORE _WEB so known doc keywords don't fall to web search
_RAG = re.compile(
    r"\b(re.?examination|reexam|grading\s+criteria|grading\s+policy|"
    r"samacheer|syllabus|curriculum|textbook|lesson|chapter|"
    r"evaluation\s+scheme|marking\s+scheme|examination\s+policy|"
    r"pass\s+criteria|promotion\s+rule|attendance\s+rule|"
    r"scholarship\s+eligibility|slas|achievement\s+survey)\b",
    re.IGNORECASE,
)

_WEB = re.compile(
    r"\b(government\s+order|go\s+order|circular|scholarship\s+scheme|"
    r"national\s+education|nep|exam\s+date|timetable|"
    r"latest|recent|news|notification|announcement|website|contact|"
    r"how\s+does|where\s+canslas|state\s+level\s+achievement\s+survey|learning\s+outcomes?|state\s+report\s+card|district\s+report\s+card|samagra\s+shiksha|scert|emis|omr|resource\s+custody\s+center|field\s+investigator|district\s+resource\s+team|district\s+level\s+committee|block\s+level\s+committee|stratified\s+random\s+sampling|blooms?\s+taxonomy|t\d{3}|e\d{3}|m\d{3})\b",
    re.IGNORECASE,
)

# Signals that an analytics-looking query is actually about document content.
# Example: "what does the report say about school performance?" contains
# "school" (_ANALYTICS) but is clearly asking for document text.
_DOC_PHRASING = re.compile(
    r"\b(what\s+does\s+the\s+report|according\s+to|as\s+per|report\s+says?|"
    r"document\s+says?|policy\s+on|describe|explain|overview|summary|"
    r"definition\s+of|meaning\s+of|what\s+is\s+the\s+purpose|"
    r"performance\s+report|slas\s+report|what\s+does\s+slas)\b",
    re.IGNORECASE,
)

_VIZ = re.compile(
    r"\b(chart|plot|graph|visuali[zs]e?|bar|pie|line|heatmap|histogram)\b",
    re.IGNORECASE,
)

_FALLBACK_PROMPT = """\
You are a query router for an education analytics platform.
Classify the query into exactly one category:

  analytics — needs SQL from a student marks database (counts, averages, pass/fail rates)
  rag       — needs text from policy/curriculum/report documents(Contains data of whole tamil nadu)
  web       — needs current external information from the internet

Query: "{query}"

Respond with a single word only: analytics, rag, or web."""


def _llm_classify(query: str) -> QueryType:
    """LLM fallback for ambiguous queries. Called only when needed."""
    try:
        from langchain_groq import ChatGroq
        from app.core.config import get_settings
        cfg = get_settings()
        llm = ChatGroq(
            model=cfg.LLM_MODEL,
            temperature=0.0,
            api_key=cfg.GROQ_API_KEY,
            max_tokens=5,
        )
        resp = llm.invoke(_FALLBACK_PROMPT.format(query=query)).content.strip().lower()
        if "analytics" in resp:
            return "analytics"
        if "web" in resp:
            return "web"
        return "rag"
    except Exception as e:
        logger.warning("llm_router_fallback_error", error=str(e))
        return "rag"   # safe default on failure


def classify(query: str) -> QueryType:
    q = query.strip()

    # 1. Explicit RAG keywords — fast path, no LLM
    if _RAG.search(q):
        logger.debug("routed_rag_regex", query=q[:60])
        return "rag"

    # 2. Web / external — fast path, no LLM
    if _WEB.search(q):
        logger.debug("routed_web_regex", query=q[:60])
        return "web"

    # 3. Analytics keyword hit — but is it phrased like a document question?
    if _ANALYTICS.search(q):
        if _DOC_PHRASING.search(q):
            # e.g. "what does the report say about school performance?"
            # Regex alone can't decide → ask the LLM (~0.5 s overhead)
            result = _llm_classify(q)
            logger.debug("routed_llm_fallback", query=q[:60], result=result)
            return result
        # Clean analytics query — fast path
        logger.debug("routed_analytics_regex", query=q[:60])
        return "analytics"

    # 4. Nothing matched — default to RAG (factual but not in DB)
    logger.debug("routed_rag_default", query=q[:60])
    return "rag"


def matched_route(query: str) -> QueryType | None:
    """Regex-only routing with NO default. Returns 'rag'/'web'/'analytics' when a
    keyword clearly matches, else None (the query is unroutable → the caller can
    ask the human to clarify instead of guessing). Mirrors classify()'s priority
    order but skips the LLM disambiguation and the rag fallback.
    """
    q = query.strip()
    if _RAG.search(q):
        return "rag"
    if _WEB.search(q):
        return "web"
    if _ANALYTICS.search(q):
        return "analytics"
    return None


def wants_viz(query: str) -> bool:
    return bool(_VIZ.search(query))
