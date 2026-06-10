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
from dataclasses import dataclass, field
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
    def __init__(self):
        cfg = get_settings()
        self.cfg   = cfg
        self._llm  = ChatGroq(model=cfg.LLM_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)
        self._judge_llm = ChatGroq(model=cfg.JUDGE_MODEL, temperature=0.0, api_key=cfg.GROQ_API_KEY)

    def _build_prompt(self) -> str:
        schema = get_schema_dict()
        subjects = get_sample_values("marks", "subject_name") if "marks" in schema else []
        schools  = get_sample_values("marks", "school_name", n=10) if "marks" in schema else []
        years    = get_sample_values("marks", "year") if "marks" in schema else []

        schema_str = "\n".join(
            f"TABLE {t}:\n  " + "\n  ".join(cols)
            for t, cols in schema.items()
        )
        return f"""You are a DuckDB SQL expert for Tamil Nadu educational analytics.
Generate ONLY valid DuckDB SQL. No markdown. No explanation. Raw SQL only.

SCHEMA:
{schema_str}

KNOWN VALUES:
subjects: {', '.join(subjects[:15])}
schools: {', '.join(schools)}
years: {', '.join(years)}

RULES:
1. subject queries → marks table with UPPER(subject_name) = '...'
2. overall student queries → students table
3. subject_result='FAIL' → failed that subject (marks < 35)
4. fail_flag=1 → student failed overall
5. ALWAYS use UPPER() for text matching
6. Pass%: ROUND(100.0*SUM(pass_flag)/COUNT(*),2)
7. NEVER SELECT * — name every column
8. School stats: HAVING COUNT(*) >= 5

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

    def _judge(self, question: str, sql: str, result_str: str, answer: str) -> JudgeResult:
        try:
            prompt = JUDGE_PROMPT.format(
                question=question, sql=sql[:300],
                result=result_str[:200], answer=answer[:200]
            )
            resp = self._judge_llm.invoke(prompt).content.strip()
            if "```" in resp:
                resp = resp.split("```")[1].split("```")[0]
            data = json.loads(resp)
            return JudgeResult(
                score=int(data.get("score", 3)),
                sql_correct=bool(data.get("sql_correct", True)),
                answer_grounded=bool(data.get("answer_grounded", True)),
                issues=data.get("issues", []),
                suggestion=data.get("suggestion", ""),
            )
        except Exception as e:
            logger.debug("judge_parse_error", error=str(e))
            return JudgeResult(score=3)

    def run(self, question: str) -> SQLResult:
        t_start   = time.perf_counter()
        prompt_tpl = self._build_prompt()
        feedback   = ""
        sql        = ""
        result_df: Optional[pd.DataFrame] = None
        total_usage = TokenUsage()

        for attempt in range(1, self.cfg.SQL_MAX_RETRIES + 1):
            extra = f"\nPREVIOUS ATTEMPT FAILED: {feedback}\nFix the SQL.\n" if feedback else ""
            full_prompt = prompt_tpl.replace("{question}", question) + extra

            # Generate SQL
            try:
                t0    = time.perf_counter()
                resp  = self._llm.invoke(full_prompt)
                usage = track_llm_call(resp, self.cfg.LLM_MODEL, t0)
                total_usage.input_tokens  += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                total_usage.total_tokens  += usage.total_tokens
                total_usage.cost_usd      += usage.cost_usd
                raw_sql = resp.content.strip()
            except Exception as e:
                feedback = str(e)
                time.sleep(0.5)
                continue

            valid, sql, err = self._validate(raw_sql)
            if not valid:
                feedback = err
                logger.warning("sql_invalid", attempt=attempt, error=err)
                time.sleep(0.5)
                continue

            # Execute
            try:
                result_df = execute_query(sql)
                logger.info("sql_executed", attempt=attempt, rows=len(result_df), sql=sql[:80])
            except Exception as e:
                feedback = f"DuckDB error: {e}"
                logger.warning("sql_exec_error", attempt=attempt, error=str(e)[:100])
                time.sleep(0.5)
                continue

            # Generate answer
            result_str = result_df.to_string(index=False) if not result_df.empty else "No matching records."
            t0   = time.perf_counter()
            ans_resp = self._llm.invoke(
                ANSWER_PROMPT.format(question=question, result=result_str[:800])
            )
            ans_usage = track_llm_call(ans_resp, self.cfg.LLM_MODEL, t0)
            total_usage.cost_usd += ans_usage.cost_usd
            total_usage.total_tokens += ans_usage.total_tokens
            answer = ans_resp.content.strip()

            # Judge
            judge = self._judge(question, sql, result_str[:200], answer[:200])
            logger.info("judge_score", attempt=attempt, score=judge.score, issues=judge.issues)

            if judge.score >= self.cfg.JUDGE_SCORE_THRESHOLD:
                latency = round((time.perf_counter() - t_start) * 1000, 1)
                total_usage.latency_ms = latency
                return SQLResult(
                    answer=answer, sql=sql, result_df=result_df,
                    judge=judge, token_usage=total_usage,
                    attempts=attempt, latency_ms=latency,
                )

            feedback = f"Judge score {judge.score}/5. {judge.issues}. {judge.suggestion}"
            time.sleep(0.5)

        latency = round((time.perf_counter() - t_start) * 1000, 1)
        return SQLResult(
            answer=f"Could not generate a reliable answer after {self.cfg.SQL_MAX_RETRIES} attempts.",
            sql=sql, result_df=result_df, judge=JudgeResult(score=0),
            token_usage=total_usage, attempts=self.cfg.SQL_MAX_RETRIES,
            error=feedback, latency_ms=latency,
        )
