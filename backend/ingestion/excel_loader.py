"""
Excel ingestion pipeline.

Design principles:
  1. AUTO-DETECT: scans DATA_DIR for any .xlsx files, reads ALL sheets.
     No hardcoded filenames or sheet names — adding a new file just works.
  2. LONG FORMAT: every row = one student × one subject.
  3. YEAR inference: from filename (looks for 4 digits) or falls back to sheet metadata.
  4. MANAGEMENT TYPE: inferred from sheet name keywords (TW/GOVT/AIDED).
  5. SCHEMA: fixed long-format schema regardless of source file structure.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional
import pandas as pd
from app.core.config import get_settings
from app.core.database import execute_ddl, get_connection
from app.monitoring.telemetry import get_logger

logger = get_logger(__name__)

# Subject column pairs: (description_col, mark_col)
SUBJECT_SLOTS = [(f"S{i}_DESC", f"MARK0{i}") for i in range(1, 7)]

LONG_FORMAT_COLUMNS = [
    "rollno", "name", "school_code", "school_name", "management",
    "district", "revname", "sex", "community", "group_code",
    "total_marks", "year", "management_type", "source_file", "source_sheet",
    "overall_result", "pass_flag", "fail_flag",
    "subject_name", "marks", "subject_result",
]


def _infer_year(filename: str) -> Optional[int]:
    """Extract year from filename like '2526' -> 2026, '24255' -> 2025."""
    digits = re.findall(r"\d{4,}", Path(filename).stem)
    for d in digits:
        if d.startswith("25"):
            return 2026
        if d.startswith("24"):
            return 2025
    return None


def _infer_management(sheet_name: str) -> str:
    s = sheet_name.upper()
    if "TW" in s or "TRIBAL" in s:
        return "TW"
    if "GOVT" in s or "GOV" in s:
        return "GOVT"
    if "AIDED" in s:
        return "AIDED"
    return "UNKNOWN"


def _normalize_col(df: pd.DataFrame, *candidates: str) -> Optional[str]:
    """Return the first candidate column name that exists in df (case-insensitive)."""
    cols_upper = {c.upper(): c for c in df.columns}
    for cand in candidates:
        found = cols_upper.get(cand.upper())
        if found:
            return found
    return None


def _to_long(
    df: pd.DataFrame,
    year: int,
    management_type: str,
    source_file: str,
    source_sheet: str,
) -> pd.DataFrame:
    """Convert wide-format DataFrame to long format."""
    rows: list[dict] = []

    # Map flexible column names to canonical names
    col = lambda *c: _normalize_col(df, *c)

    rollno_col    = col("ROLLNO", "ROLL_NO", "ROLL NO", "REGNO")
    name_col      = col("NAME", "STUDENT_NAME", "STU_NAME")
    schl_col      = col("SCHL", "SCHOOL_CODE", "SCH_CODE", "SCHOOL CODE")
    schname_col   = col("SCH_NAME", "SCHOOL_NAME", "SCHNAME")
    mgmt_col      = col("MANAGEMENT", "MGMT")
    dist_col      = col("DISTNAME", "DISTRICT", "DIST_NAME")
    revname_col   = col("REVNAME", "REV_NAME")
    sex_col       = col("SEX", "GENDER")
    com_col       = col("COM", "COMMUNITY", "CASTE")
    grp_col       = col("GROUP", "GRP", "GROUP_CODE")
    total_col     = col("TOTAL", "TOTAL_MARKS", "GRAND_TOTAL")
    pass_col      = col("PASS", "RESULT", "PASS_FAIL")

    for _, row in df.iterrows():
        raw_pass = str(row.get(pass_col, "")).strip().upper() if pass_col else ""
        is_pass  = raw_pass == "P"
        is_fail  = raw_pass == "F" or (year == 2026 and raw_pass not in ("P",))

        base = {
            "rollno":          str(row.get(rollno_col, "")).strip() if rollno_col else "",
            "name":            str(row.get(name_col, "")).strip() if name_col else "",
            "school_code":     str(row.get(schl_col, "")).strip() if schl_col else "",
            "school_name":     str(row.get(schname_col, "")).strip() if schname_col else "",
            "management":      str(row.get(mgmt_col, "")).strip() if mgmt_col else "",
            "district":        str(row.get(dist_col, "NILGIRIS")).strip() if dist_col else "NILGIRIS",
            "revname":         str(row.get(revname_col, "")).strip() if revname_col else "",
            "sex":             str(row.get(sex_col, "")).strip() if sex_col else "",
            "community":       str(row.get(com_col, "")).strip() if com_col else "",
            "group_code":      str(row.get(grp_col, "")).strip() if grp_col else "",
            "total_marks":     row.get(total_col, None) if total_col else None,
            "year":            year,
            "management_type": management_type,
            "source_file":     source_file,
            "source_sheet":    source_sheet,
            "overall_result":  "PASS" if is_pass else ("FAIL" if is_fail else "ABSENT"),
            "pass_flag":       1 if is_pass else 0,
            "fail_flag":       1 if is_fail else 0,
        }

        for desc_col, mark_col in SUBJECT_SLOTS:
            actual_desc = _normalize_col(df, desc_col)
            actual_mark = _normalize_col(df, mark_col)
            if not actual_desc or not actual_mark:
                continue
            subj = str(row.get(actual_desc, "")).strip().upper()
            if not subj or subj in ("NAN", "NONE", ""):
                continue
            try:
                m_val = row.get(actual_mark)
                m = float(str(m_val).replace("XXX", "").strip())
                if pd.isna(m):
                    sub_result = "ABSENT"
                elif m < 35:
                    sub_result = "FAIL"
                else:
                    sub_result = "PASS"
            except (ValueError, TypeError):
                m = None
                sub_result = "ABSENT"

            rows.append({**base, "subject_name": subj, "marks": m, "subject_result": sub_result})

    return pd.DataFrame(rows, columns=LONG_FORMAT_COLUMNS) if rows else pd.DataFrame()


def ingest_all_excel(data_dir: Optional[str] = None) -> dict[str, int]:
    """
    Auto-scan DATA_DIR for .xlsx files, read ALL sheets, normalise, write to DuckDB.
    Works with any number of files and any sheet names.
    """
    cfg = get_settings()
    data_dir = Path(data_dir or cfg.DATA_DIR)
    con = get_connection()
    all_long: list[pd.DataFrame] = []

    xlsx_files = list(data_dir.glob("*.xlsx"))
    if not xlsx_files:
        raise RuntimeError(f"No .xlsx files found in {data_dir}")

    logger.info("excel_scan", found=len(xlsx_files), files=[f.name for f in xlsx_files])

    for fpath in xlsx_files:
        year = _infer_year(fpath.name)
        try:
            xl = pd.ExcelFile(fpath)
        except Exception as e:
            logger.warning("excel_open_failed", file=fpath.name, error=str(e))
            continue

        for sheet in xl.sheet_names:
            mgmt = _infer_management(sheet)
            inferred_year = year or 2025
            try:
                df = pd.read_excel(fpath, sheet_name=sheet)
                if df.empty:
                    continue
                long = _to_long(df, inferred_year, mgmt, fpath.name, sheet)
                if not long.empty:
                    all_long.append(long)
                    logger.info("sheet_loaded",
                                file=fpath.name, sheet=sheet,
                                rows=len(df), long_rows=len(long))
            except Exception as e:
                logger.warning("sheet_failed", file=fpath.name, sheet=sheet, error=str(e))

    if not all_long:
        raise RuntimeError("No data was successfully loaded from any Excel sheet.")

    marks_df   = pd.concat(all_long, ignore_index=True)
    students_df = marks_df.drop_duplicates("rollno").drop(
        columns=["subject_name", "marks", "subject_result"], errors="ignore"
    )

    con.register("_marks_tmp",    marks_df)
    con.register("_students_tmp", students_df)
    con.execute("CREATE OR REPLACE TABLE marks    AS SELECT * FROM _marks_tmp")
    con.execute("CREATE OR REPLACE TABLE students AS SELECT * FROM _students_tmp")

    m_count = con.execute("SELECT COUNT(*) FROM marks").fetchone()[0]
    s_count = con.execute("SELECT COUNT(*) FROM students").fetchone()[0]

    logger.info("ingest_done", marks_rows=m_count, student_rows=s_count,
                files=len(xlsx_files), sheets=len(all_long))
    return {"marks_rows": m_count, "student_rows": s_count, "files": len(xlsx_files)}
