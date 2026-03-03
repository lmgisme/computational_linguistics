"""
main.py
-------
Phase 1 entry point: Turkic substrate detection — data ingestion & cleaning.

Usage:
    python main.py [--force-download] [--no-normalize]

Flags:
    --force-download   : Re-fetch ASJP data even if a local cache exists.
    --no-normalize     : Skip IPA normalization; keep raw ASJP forms.

Pipeline stages:
    1. Fetch or load the ASJP v19 flat file.
    2. Parse all ~7,000 doculects.
    3. Extract the 9 target Turkic language doculects.
    4. For each (language, concept) pair: normalize ASJP → IPA and tokenize.
    5. Assemble into a pandas DataFrame.
    6. Compute summary statistics and write the Phase 1 report.
    7. Save outputs to /output/.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure output dir exists BEFORE the FileHandler tries to open the log file
Path("output").mkdir(exist_ok=True)

# ── Configure logging ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("output") / "phase1_run.log", mode="w", encoding="utf-8"),
    ],
)

logger = logging.getLogger("main")

# ── Project imports ───────────────────────────────────────────────────────────
from fetch_asjp     import fetch_turkic_data
from fetch_lexibank  import build_merged_dataset, report_merged, LEXIBANK_CSV, MERGED_CSV, LEXIBANK_REPORT
from build_dataframe import build_records, assemble_dataframe, compute_summary, format_report, save_outputs
from config         import NORMALIZE_TO_IPA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 1: Turkic Swadesh list ingestion and IPA normalization."
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download ASJP data even if a local cache exists.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip ASJP→IPA normalization; store raw forms in ipa_form column.",
    )
    parser.add_argument(
        "--lexibank",
        action="store_true",
        help="Also run Lexibank CLDF ingestion and produce merged publication-quality dataset.",
    )
    parser.add_argument(
        "--lexibank-only",
        action="store_true",
        help="Skip ASJP entirely; run Lexibank ingestion only.",
    )
    return parser.parse_args()


def run_pipeline(force_download: bool = False, skip_normalize: bool = False) -> None:
    """
    Execute the full Phase 1 pipeline and return the resulting DataFrame.
    """
    logger.info("=" * 60)
    logger.info("  PHASE 1  —  Turkic substrate detection")
    logger.info("  Stage: Data ingestion & IPA normalization")
    logger.info("=" * 60)

    # ── Stage 1-3: Fetch + parse + filter (all handled in fetch_asjp) ─────────
    logger.info("[1/5] Fetching ASJP data (API → Zenodo → local fallback)...")
    turkic = fetch_turkic_data(force_download=force_download)

    matched   = sum(1 for v in turkic.values() if v)
    unmatched = sum(1 for v in turkic.values() if not v)
    logger.info(f"      Matched: {matched}/9  |  Unmatched: {unmatched}/9")

    if unmatched > 0:
        logger.warning(
            "Some target languages have no ASJP match. They will appear in the "
            "DataFrame with MISSING_DOCULECT flags."
        )

    # Placeholders to keep step numbering readable in logs
    logger.info("[2/5] Parse complete (handled within fetch stage).")
    logger.info("[3/5] Extraction complete (handled within fetch stage).")

    # ── Stage 4: Build normalized records ────────────────────────────────────
    normalize_flag = NORMALIZE_TO_IPA and not skip_normalize
    logger.info(
        f"[4/5] Building records (IPA normalization: {'ON' if normalize_flag else 'OFF'})..."
    )
    records = build_records(turkic)
    logger.info(f"      Total records generated: {len(records):,}")

    # ── Stage 5: Assemble DataFrame + statistics ──────────────────────────────
    logger.info("[5/5] Assembling DataFrame and computing statistics...")
    df      = assemble_dataframe(records)
    summary = compute_summary(df)
    report  = format_report(summary)

    # Print report to stdout
    print("\n" + report + "\n")

    # Save to disk
    save_outputs(df, report)

    logger.info("Phase 1 (ASJP) complete. See /output/ for ASJP results.")
    return df


def run_lexibank_pipeline(
    asjp_csv_path: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Run the Lexibank CLDF ingestion pipeline.
    Optionally gap-fills from the ASJP Phase 1 CSV.
    """
    from pathlib import Path as _Path
    import pandas as _pd

    logger.info("=" * 60)
    logger.info("  PHASE 1b — Lexibank CLDF ingestion")
    logger.info("  Publication-quality data pipeline")
    logger.info("=" * 60)

    merged = build_merged_dataset(
        asjp_csv_path=asjp_csv_path,
        force_download=force_download,
    )

    report = report_merged(merged)
    print("\n" + report + "\n")

    # Save outputs
    out_dir = _Path("output")
    out_dir.mkdir(exist_ok=True)

    # Serialize ipa_tokens list for CSV
    merged_export = merged.copy()
    merged_export["ipa_tokens"] = merged_export["ipa_tokens"].apply(
        lambda t: " ".join(t) if isinstance(t, list) else (t or "")
    )
    merged_export.to_csv(MERGED_CSV, index=False, encoding="utf-8-sig")
    LEXIBANK_REPORT.write_text(report, encoding="utf-8")

    logger.info(f"Merged dataset saved to: {MERGED_CSV}")
    logger.info(f"Lexibank report saved to: {LEXIBANK_REPORT}")
    logger.info("Lexibank pipeline complete.")
    return merged


if __name__ == "__main__":
    from pathlib import Path as _Path
    args = parse_args()

    asjp_csv = None
    if not args.lexibank_only:
        run_pipeline(
            force_download=args.force_download,
            skip_normalize=args.no_normalize,
        )
        asjp_csv = _Path("output") / "turkic_swadesh_phase1.csv"

    if args.lexibank or args.lexibank_only:
        run_lexibank_pipeline(
            asjp_csv_path=asjp_csv,
            force_download=args.force_download,
        )
