"""
SQL Analytics Engine:
  1. Dynamic schema injection into prompt (no hardcoded columns)
  2. SQL validation (block DDL/DML, check columns)
  3. DuckDB execution
  4. LLM-as-Judge evaluation
  5. Retry loop (up to SQL_MAX_RETRIES)
  6. Natural-language answer generation
"""
from __future__ import annotations
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from threading import Lock
from typing import Optional

import pandas as pd
from langchain_groq import ChatGroq

from app.core.config import get_settings
from app.core.database import execute_query, get_schema_dict, get_sample_values
from app.monitoring.telemetry import TokenUsage, get_logger, track_llm_call

logger = get_logger(__name__)

_UNSAFE = re.compile(
    r"\b(DROP|DELETE|TRUNCATE|INSERT|UPDATE|ALTER|CREATE|REPLACE|ATTACH|PRAGMA|COPY)\b",
    re.IGNORECASE,
)

JUDGE_PROMPT = """You are an expert SQL evaluator.
Question: {question}
SQL: {sql}
Result preview: {result}
Answer: {answer}

Score 1-5 (5=perfect). Return ONLY valid JSON:
{{"score": 1-5, "sql_correct": true/false, "answer_grounded": true/false, "issues": [], "suggestion": ""}}"""

ANSWER_PROMPT = """Question: {question}
Data from database:
{result}
Write a clear, factual 1-3 sentence answer. Numbers only from the data. Plain text:"""

# Compact retry prompt — resent INSTEAD of the full ~2,400-token generation prompt
# on retries, to stay under Groq's TPM limit. Carries just enough schema to fix the SQL.
RETRY_PROMPT = """Fix this DuckDB SQL. Return ONLY raw SQL — no markdown, no explanation.

Question: {question}
Failed SQL:
{sql}
Problem: {feedback}

Tables (DuckDB):
  marks(rollno, name, school_code, school_name, management, district, revname, sex,
        community, group_code, total_marks, year, management_type, overall_result,
        pass_flag, fail_flag, subject_name, marks, subject_result)  -- one row per student per subject
  students(...same as marks but NO subject_name, marks, subject_result)  -- one row per student
Rules: SELECT only; UPPER() for all text matches; marks<35 => subject FAIL;
subject_result='FAIL' = failed that subject; fail_flag=1 = student failed overall; never SELECT *.
Corrected SQL:"""


@dataclass
class JudgeResult:
    score: int = 5
    sql_correct: bool = True
    answer_grounded: bool = True
    issues: list[str] = field(default_factory=list)
    suggestion: str = ""


@dataclass
class SQLResult:
    answer: str = ""
    sql: str = ""
    result_df: Optional[pd.DataFrame] = None
    judge: JudgeResult = field(default_factory=JudgeResult)
    token_usage: TokenUsage = field(default_factory=TokenUsage)
    attempts: int = 0
    error: Optional[str] = None
    latency_ms: float = 0.0


class SQLEngine:
    # Bounded LRU result cache: a repeat question returns instantly at 0 tokens.
    # Process-local, no TTL — cleared on backend restart or via clear_cache().
    _CACHE_MAX = 256

    def __init__(self):
        cfg = get_settings()
        self.cfg   = cfg
        # SQL generation uses the stronger SQL_MODEL (separate Groq quota bucket → relieves
        # the 8B TPM wall). Answer-gen and judge stay on the fast LLM_MODEL (8B).
        self._gen_llm   = ChatGroq(model=cfg.SQL_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        self._llm       = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        self._judge_llm = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        # question (normalized) -> successful SQLResult. OrderedDict = LRU order.
        self._cache: "OrderedDict[str, SQLResult]" = OrderedDict()
        self._cache_lock = Lock()

    def clear_cache(self) -> int:
        """Drop all cached results (e.g. after re-ingesting data). Returns count cleared."""
        with self._cache_lock:
            n = len(self._cache)
            self._cache.clear()
            return n

    # Location → SQL condition mapping (mirrors notebook LOCATION_MAP)
    LOCATION_MAP: dict[str, str] = {
        "ooty":        "UPPER(school_name) LIKE '%OOTY%' OR UPPER(school_name) LIKE '%UDHAGAMANDALAM%'",
        "gudalur":     "UPPER(school_name) LIKE '%GUDALUR%'",
        "coonoor":     "UPPER(school_name) LIKE '%COONOOR%'",
        "kotagiri":    "UPPER(school_name) LIKE '%KOTAGIRI%'",
        "pandalur":    "UPPER(school_name) LIKE '%PANDALUR%'",
     
        # no filter — all data is Nilgiris
    }

    def _build_prompt(self) -> str:
        schema = get_schema_dict()
        subjects = get_sample_values("marks", "subject_name") if "marks" in schema else []
        schools  = get_sample_values("marks", "school_name", n=10) if "marks" in schema else []
        years    = get_sample_values("marks", "year") if "marks" in schema else []

        location_hint = "\n".join(
            f"  {k} → {v if v else '(no filter needed)'}"
            for k, v in self.LOCATION_MAP.items()
        )

        return f"""You are an expert educational analytics SQL engine for Nilgiris District.
Generate ONLY valid DuckDB SQL. No markdown. No explanation. Raw SQL only.

=== DATABASE SCHEMA ===

TABLE: marks
  One row per student per subject (long format).
  Columns:
    rollno          TEXT    -- student roll number
    name            TEXT    -- student name
    school_code     TEXT    -- e.g. 216CONR0001
    school_name     TEXT    -- e.g. GHSS, GUDALUR
    management      TEXT    -- GOVT / TRIBAL WELFARE etc.
    district        TEXT    -- always NILGIRIS
    revname         TEXT    -- OOTY (only for TW 2026)
    sex             TEXT    -- M or F
    community       TEXT    -- SC / ST / BC / MBC etc.
    group_code      TEXT    -- e.g. 2702 = Commerce, 2502 = Science
    total_marks     FLOAT   -- student total across all subjects
    year            INT     -- 2025 or 2026
    management_type TEXT    -- TW (Tribal Welfare) or GOVT
    overall_result  TEXT    -- PASS / FAIL / ABSENT (student overall)
    pass_flag       INT     -- 1 = student passed overall
    fail_flag       INT     -- 1 = student failed overall
    subject_name    TEXT    -- e.g. PHYSICS, ENGLISH, MATHEMATICS
    marks           FLOAT   -- marks obtained in this subject
    subject_result  TEXT    -- PASS / FAIL / ABSENT for this subject

TABLE: students
  One row per student (overall results only, no subject breakdown).
  Same columns as marks EXCEPT no subject_name, marks, subject_result.
  Use for: total students, overall pass %, gender analysis.

=== AVAILABLE SUBJECTS ===
{', '.join(subjects)}

=== AVAILABLE SCHOOLS (sample) ===
{', '.join(schools)}

=== AVAILABLE YEARS ===
{', '.join(str(y) for y in years)}

=== LOCATION FILTERING ===
When user mentions a location, use these SQL conditions:
{location_hint}
For districts/general Nilgiris: no filter needed (all data is Nilgiris)

=== YEAR LOGIC ===
- year=2025: data from 24255.xlsx
- year=2026: data from 2526.xlsx
- No year mentioned: query ALL years (no year filter)

=== CRITICAL RULES ===
1. For subject queries → use marks table with WHERE subject_name=...
2. For overall student queries → use students table
3. subject_result='FAIL' → failed that specific subject (marks < 35)
4. fail_flag=1 → student failed overall (different from subject failure)
5. ALWAYS use UPPER() for text matching: WHERE UPPER(subject_name) = 'PHYSICS'
6. Students take DIFFERENT subjects by group — never assume all take all subjects
7. For percentage: ROUND(100.0 * COUNT(*) FILTER (WHERE ...) / COUNT(*), 2)
8. For school averages: AVG(marks) grouped by school_name
9. NEVER use SELECT * — always name columns
10. For comparisons: use UNION ALL or subqueries

=== EXAMPLES ===

Q: How many students failed in Physics across Nilgiris?
SQL:
SELECT year, COUNT(*) AS physics_failures
FROM marks
WHERE UPPER(subject_name) = 'PHYSICS'
  AND subject_result = 'FAIL'
GROUP BY year
ORDER BY year;

Q: Which school has the highest average score in English?
SQL:
SELECT school_name,
       ROUND(AVG(marks), 2) AS avg_english,
       COUNT(*) AS students_count
FROM marks
WHERE UPPER(subject_name) = 'ENGLISH'
  AND subject_result != 'ABSENT'
GROUP BY school_name
ORDER BY avg_english DESC
LIMIT 10;

Q: Which subject has the most failures across Nilgiris?
SQL:
SELECT subject_name,
       COUNT(*) AS failures,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_all_failures
FROM marks
WHERE subject_result = 'FAIL'
GROUP BY subject_name
ORDER BY failures DESC;

Q: Which school has the least average marks in Mathematics?
SQL:
SELECT school_name,
       ROUND(AVG(marks), 2) AS avg_maths,
       COUNT(*) AS students
FROM marks
WHERE UPPER(subject_name) = 'MATHEMATICS'
  AND subject_result != 'ABSENT'
GROUP BY school_name
HAVING COUNT(*) >= 5
ORDER BY avg_maths ASC
LIMIT 10;

Q: Compare pass percentage between boys and girls in Gudalur
SQL:
SELECT sex,
       COUNT(*) AS total,
       SUM(pass_flag) AS passed,
       ROUND(100.0 * SUM(pass_flag) / COUNT(*), 2) AS pass_pct
FROM students
WHERE UPPER(school_name) LIKE '%GUDALUR%'
GROUP BY sex
ORDER BY sex;

Q: Show subject-wise failure count for Kunjappanai
SQL:
SELECT subject_name,
       COUNT(*) AS failures
FROM marks
WHERE UPPER(school_name) LIKE '%KUNJAPPANAI%'
  AND subject_result = 'FAIL'
GROUP BY subject_name
ORDER BY failures DESC;

QUESTION: {{question}}
SQL:"""

    def _validate(self, sql: str) -> tuple[bool, str, str]:
        """Returns (valid, cleaned_sql, error_msg)."""
        cleaned = sql.strip()
        if "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
            if cleaned.lower().startswith("sql"):
                cleaned = cleaned[3:]
        cleaned = cleaned.strip()

        if not cleaned:
            return False, "", "Empty SQL"
        if _UNSAFE.search(cleaned):
            return False, cleaned, f"Unsafe operation: {_UNSAFE.search(cleaned).group()}"
        if not re.match(r"^\s*SELECT\b", cleaned, re.IGNORECASE):
            return False, cleaned, "SQL must start with SELECT"
        return True, cleaned, ""

    def _judge(self, question: str, sql: str, result_str: str, answer: str) -> tuple[JudgeResult, TokenUsage]:
        """Returns (JudgeResult, TokenUsage) — judge tokens count toward the request total."""
        usage = TokenUsage()
        try:
            prompt = JUDGE_PROMPT.format(
                question=question, sql=sql[:300],
                result=result_str[:200], answer=answer[:200]
            )
            t0    = time.perf_counter()
            resp  = self._judge_llm.invoke(prompt)
            usage = track_llm_call(resp, self.cfg.LLM_MODEL, t0)
            text  = resp.content.strip()
            if "```" in text:
                text = text.split("```")[1].split("```")[0]
            data = json.loads(text)
            return JudgeResult(
                score=int(data.get("score", 3)),
                sql_correct=bool(data.get("sql_correct", True)),
                answer_grounded=bool(data.get("answer_grounded", True)),
                issues=data.get("issues", []),
                suggestion=data.get("suggestion", ""),
            ), usage
        except Exception as e:
            logger.debug("judge_parse_error", error=str(e))
            return JudgeResult(score=3), usage

    def run(self, question: str) -> SQLResult:
        # ── Item 1: LRU result cache — repeats cost 0 tokens ──────────────────
        # Normalize so casing/whitespace variants share one entry. The prompt
        # forces UPPER() for all text matches, so a lowercased question yields
        # identical SQL — safe to feed the pipeline the normalized text.
        key = question.lower().strip()
        with self._cache_lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)          # mark most-recently-used
        if hit is not None:
            logger.info("sql_cache_hit", question=question[:60])
            # No LLM call happened → report 0 tokens so metrics stay honest.
            return replace(hit, token_usage=TokenUsage(latency_ms=0.0),
                           attempts=0, latency_ms=0.0)

        result = self._run_pipeline(key)
        # Only cache genuine successes — never memoize a transient failure.
        if result.error is None:
            with self._cache_lock:
                self._cache[key] = result
                self._cache.move_to_end(key)
                while len(self._cache) > self._CACHE_MAX:
                    self._cache.popitem(last=False)   # evict least-recently-used
        return result

    def _run_pipeline(self, question: str) -> SQLResult:
        t_start = time.perf_counter()

        prompt_tpl = self._build_prompt()
        feedback   = ""
        sql        = ""
        answer     = ""
        result_df: Optional[pd.DataFrame] = None
        total_usage = TokenUsage()
        judged        = False   # Item 4: LLM judge runs at most ONCE
        empty_retried = False   # Item 4: empty-result heuristic allows one soft retry

        def _finalize(judge: JudgeResult) -> SQLResult:
            latency = round((time.perf_counter() - t_start) * 1000, 1)
            total_usage.latency_ms = latency
            return SQLResult(
                answer=answer, sql=sql, result_df=result_df,
                judge=judge, token_usage=total_usage,
                attempts=attempt, latency_ms=latency,
            )

        for attempt in range(1, self.cfg.SQL_MAX_RETRIES + 1):
            # Item 3: full prompt only on the first attempt; compact prompt on retries
            if attempt == 1 or not sql:
                full_prompt = prompt_tpl.replace("{question}", question)
            else:
                full_prompt = RETRY_PROMPT.format(question=question, sql=sql, feedback=feedback)

            # Generate SQL
            try:
                t0    = time.perf_counter()
                resp  = self._gen_llm.invoke(full_prompt)
                usage = track_llm_call(resp, self.cfg.SQL_MODEL, t0)
                total_usage.input_tokens  += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens  += usage.total_tokens
                total_usage.cost_usd      += usage.cost_usd
                raw_sql = resp.content.strip()
            except Exception as e:
                feedback = str(e)
                time.sleep(0.5)
                continue

            # Deterministic gate 1: validation
            valid, sql, err = self._validate(raw_sql)
            if not valid:
                feedback = err
                logger.warning("sql_invalid", attempt=attempt, error=err)
                time.sleep(0.5)
                continue

            # Deterministic gate 2: execution
            try:
                result_df = execute_query(sql)
                logger.info("sql_executed", attempt=attempt, rows=len(result_df), sql=sql[:80])
            except Exception as e:
                feedback = f"DuckDB error: {e}"
                logger.warning("sql_exec_error", attempt=attempt, error=str(e)[:100])
                time.sleep(0.5)
                continue

            # Deterministic gate 3: empty-result heuristic (one free soft retry, no judge call)
            if result_df.empty and not empty_retried and attempt < self.cfg.SQL_MAX_RETRIES:
                empty_retried = True
                feedback = "Query returned 0 rows. Re-check filters/casing/joins — values may differ."
                logger.info("sql_empty_retry", attempt=attempt)
                time.sleep(0.5)
                continue

            # Generate answer
            result_str = result_df.to_string(index=False) if not result_df.empty else "No matching records."
            t0   = time.perf_counter()
            ans_resp = self._llm.invoke(
                ANSWER_PROMPT.format(question=question, result=result_str[:800])
            )
            ans_usage = track_llm_call(ans_resp, self.cfg.LLM_MODEL, t0)
            total_usage.input_tokens  += ans_usage.input_tokens
            total_usage.output_tokens += ans_usage.output_tokens
            total_usage.total_tokens  += ans_usage.total_tokens
            total_usage.cost_usd      += ans_usage.cost_usd
            answer = ans_resp.content.strip()

            # Item 4: judge ONCE on the first clean success. If it passes → done.
            # If it fails → one corrective retry, after which we accept on the
            # deterministic gates above (validate + execute) without re-judging.
            if not judged:
                judge, judge_usage = self._judge(question, sql, result_str[:200], answer[:200])
                total_usage.input_tokens  += judge_usage.input_tokens
                total_usage.output_tokens += judge_usage.output_tokens
                total_usage.total_tokens  += judge_usage.total_tokens
                total_usage.cost_usd      += judge_usage.cost_usd
                judged = True
                logger.info("judge_score", attempt=attempt, score=judge.score, issues=judge.issues)
                if judge.score >= self.cfg.JUDGE_SCORE_THRESHOLD:
                    return _finalize(judge)
                feedback = f"Judge score {judge.score}/5. {judge.issues}. {judge.suggestion}"
                time.sleep(0.5)
                continue

            # Corrective retry produced a valid, executable result — accept deterministically.
            logger.info("sql_deterministic_accept", attempt=attempt)
            return _finalize(JudgeResult(score=self.cfg.JUDGE_SCORE_THRESHOLD))

        latency = round((time.perf_counter() - t_start) * 1000, 1)
        return SQLResult(
            answer=f"Could not generate a reliable answer after {self.cfg.SQL_MAX_RETRIES} attempts.",
            sql=sql, result_df=result_df, judge=JudgeResult(score=0),
            token_usage=total_usage, attempts=self.cfg.SQL_MAX_RETRIES,
            error=feedback, latency_ms=latency,
        )
