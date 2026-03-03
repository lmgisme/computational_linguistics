"""
build_dataframe.py
------------------
Module 3 of 3: DataFrame construction and summary statistics.

Responsibilities:
  - Wire together fetch_asjp and normalize_ipa into a single clean DataFrame.
  - Compute per-language coverage statistics.
  - Generate the Phase 1 summary report.
  - Write outputs to disk.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from config import OUTPUT_DIR, OUTPUT_CSV, REPORT_TXT, ASJP_CONCEPTS
from fetch_asjp import fetch_turkic_data
from normalize_ipa import process_form

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Build the raw records list
# ─────────────────────────────────────────────────────────────────────────────

def build_records(turkic_doculects: dict[str, list[dict]]) -> list[dict]:
    """
    Iterate over each matched doculect and each concept, run the normalization
    pipeline on every raw ASJP form, and collect a flat list of record dicts.

    When multiple doculects matched the same canonical language name, we use
    the first match only (as warned by extract_turkic_doculects).
    """
    records: list[dict] = []

    for lang_name, matches in turkic_doculects.items():
        if not matches:
            logger.warning(f"Skipping '{lang_name}': no doculect found in ASJP data.")
            # Still emit placeholder rows so the DataFrame is complete
            for concept in ASJP_CONCEPTS:
                records.append({
                    "language":   lang_name,
                    "gloss":      concept,
                    "form":       None,
                    "ipa_form":   None,
                    "ipa_tokens": [],
                    "anomaly":    "MISSING_DOCULECT",
                })
            continue

        # Use the first matching doculect (most common case: exactly one match)
        doculect = matches[0]
        forms_dict = doculect.get("forms", {})

        for concept in ASJP_CONCEPTS:
            raw_form = forms_dict.get(concept, "")
            record = process_form(lang_name, concept, raw_form)
            records.append(record)

    return records


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Assemble the DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def assemble_dataframe(records: list[dict]) -> pd.DataFrame:
    """
    Convert the flat records list to a pandas DataFrame with consistent dtypes.

    Columns:
        language    – canonical language name (str)
        gloss       – Swadesh concept label   (str)
        form        – raw ASJP orthographic form (str, NaN if missing)
        ipa_form    – IPA string after conversion (str, NaN if missing)
        ipa_tokens  – list of IPA phone tokens (object column; [] if empty)
        anomaly     – anomaly flag string (str, NaN if clean)
    """
    df = pd.DataFrame(records)

    # Ensure correct column order
    col_order = ["language", "gloss", "form", "ipa_form", "ipa_tokens", "anomaly"]
    df = df[col_order]

    # Coerce missing strings to NaN (pandas convention)
    for col in ("form", "ipa_form", "anomaly"):
        df[col] = df[col].replace("", pd.NA)

    # ipa_tokens: replace None with empty list for consistency
    df["ipa_tokens"] = df["ipa_tokens"].apply(lambda x: x if isinstance(x, list) else [])

    logger.info(f"DataFrame assembled: {len(df):,} rows × {len(df.columns)} columns.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Summary statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_summary(df: pd.DataFrame) -> dict:
    """
    Compute per-language and global coverage statistics.

    Returns a nested dict:
        {
          'global': { 'total_rows': int, 'total_missing': int, 'total_anomalies': int },
          'by_language': {
              lang_name: {
                  'total_concepts':   int,
                  'present':          int,
                  'missing':          int,
                  'coverage_pct':     float,
                  'anomaly_count':    int,
                  'missing_glosses':  list[str],
                  'anomalous_glosses': list[str],
              },
              ...
          }
        }
    """
    stats: dict = {"global": {}, "by_language": {}}
    n_concepts = len(ASJP_CONCEPTS)

    for lang in df["language"].unique():
        sub = df[df["language"] == lang]

        # A form is "missing" if form is NaN or flagged as absent/empty/no-doculect
        missing_mask = (
            sub["form"].isna()
            | sub["anomaly"].isin(["ASJP_ABSENT_MARKER", "EMPTY_RAW_FORM", "MISSING_DOCULECT"])
        )
        present_mask  = ~missing_mask
        anomaly_mask  = sub["anomaly"].notna() & ~missing_mask  # anomalies in otherwise present forms

        missing_glosses   = sub.loc[missing_mask,  "gloss"].tolist()
        anomalous_glosses = sub.loc[anomaly_mask,  "gloss"].tolist()

        n_present  = int(present_mask.sum())
        n_missing  = int(missing_mask.sum())
        n_anomaly  = int(anomaly_mask.sum())
        coverage   = round(100.0 * n_present / n_concepts, 1)

        stats["by_language"][lang] = {
            "total_concepts":    n_concepts,
            "present":           n_present,
            "missing":           n_missing,
            "coverage_pct":      coverage,
            "anomaly_count":     n_anomaly,
            "missing_glosses":   missing_glosses,
            "anomalous_glosses": anomalous_glosses,
        }

    # Global aggregates
    total_missing   = sum(s["missing"]       for s in stats["by_language"].values())
    total_anomalies = sum(s["anomaly_count"] for s in stats["by_language"].values())
    stats["global"] = {
        "total_rows":       len(df),
        "total_missing":    total_missing,
        "total_anomalies":  total_anomalies,
    }

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Report generation
# ─────────────────────────────────────────────────────────────────────────────

def format_report(summary: dict) -> str:
    """
    Produce a human-readable plain-text Phase 1 summary report.
    """
    lines = [
        "=" * 70,
        "  PHASE 1 SUMMARY REPORT — Turkic Substrate Detection Project",
        "  Data source: ASJP v19 (Automated Similarity Judgment Program)",
        "=" * 70,
        "",
        f"  Total rows in DataFrame : {summary['global']['total_rows']:>6,}",
        f"  Total missing concepts  : {summary['global']['total_missing']:>6,}",
        f"  Total anomaly flags     : {summary['global']['total_anomalies']:>6,}",
        "",
        "-" * 70,
        f"  {'LANGUAGE':<16} {'PRESENT':>7}  {'MISSING':>7}  {'COV%':>6}  {'ANOMALIES':>9}",
        "-" * 70,
    ]

    for lang, s in summary["by_language"].items():
        lines.append(
            f"  {lang:<16} {s['present']:>7}  {s['missing']:>7}  "
            f"{s['coverage_pct']:>5.1f}%  {s['anomaly_count']:>9}"
        )

    lines += ["", "-" * 70, "  MISSING CONCEPTS BY LANGUAGE", "-" * 70]
    for lang, s in summary["by_language"].items():
        if s["missing_glosses"]:
            glosses = ", ".join(s["missing_glosses"])
            lines.append(f"  {lang}: {glosses}")
        else:
            lines.append(f"  {lang}: (none — full coverage)")

    lines += ["", "-" * 70, "  ANOMALOUS FORMS BY LANGUAGE", "-" * 70]
    for lang, s in summary["by_language"].items():
        if s["anomalous_glosses"]:
            glosses = ", ".join(s["anomalous_glosses"])
            lines.append(f"  {lang}: {glosses}")
        else:
            lines.append(f"  {lang}: (none flagged)")

    lines += [
        "",
        "=" * 70,
        "  END OF PHASE 1 REPORT",
        "=" * 70,
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# 5.  Write outputs to disk
# ─────────────────────────────────────────────────────────────────────────────

def save_outputs(df: pd.DataFrame, report_text: str) -> None:
    """
    Persist the DataFrame to CSV and the report to a text file.
    """
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path    = out_dir / OUTPUT_CSV
    report_path = out_dir / REPORT_TXT

    # CSV — serialize ipa_tokens list as a space-joined string for readability
    df_export = df.copy()
    df_export["ipa_tokens"] = df_export["ipa_tokens"].apply(
        lambda toks: " ".join(toks) if toks else ""
    )
    df_export.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logger.info(f"DataFrame saved to: {csv_path}")

    report_path.write_text(report_text, encoding="utf-8")
    logger.info(f"Report saved to:    {report_path}")
