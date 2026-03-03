"""
Phase 3 - Step 4: Substrate Report Generator
=============================================
Produces the final Phase 3 report:
  - Anomaly counts by language
  - Cluster profiles with phonological characterization
  - Top 30 most statistically irregular words with cross-linguistic forms
  - Interpretation notes for substrate candidates
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "output"
SCORES_CSV    = BASE / "regularity_scores.csv"
ANOMALIES_CSV = BASE / "anomalies.csv"
CLUSTERS_CSV  = BASE / "substrate_clusters.csv"
PROFILES_CSV  = BASE / "cluster_profiles.csv"
HYBRID_CSV    = BASE / "hybrid_cognates.csv"
REPORT_TXT    = BASE / "phase3_report.txt"

DIVIDER  = "=" * 72
SUBDIV   = "-" * 72

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_bool_col(series):
    """Normalize a column that may be bool, 'True'/'False' strings, or 0/1."""
    return series.astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)

def cross_lang_forms(hybrid_df, gloss):
    rows = hybrid_df[hybrid_df["gloss"] == gloss]
    return {r["language"]: r["form"] for _, r in rows.iterrows()}

# ---------------------------------------------------------------------------
# Main report
# ---------------------------------------------------------------------------
def main():
    scores_df = pd.read_csv(SCORES_CSV)
    scores_df["adjusted_score"]      = pd.to_numeric(scores_df["adjusted_score"], errors="coerce")
    scores_df["yakut_artifact_flag"] = load_bool_col(scores_df["yakut_artifact_flag"])
    scores_df["lexstat_penalized"]   = load_bool_col(scores_df["lexstat_penalized"])

    anomalies_df = pd.read_csv(ANOMALIES_CSV)
    anomalies_df["adjusted_score"] = pd.to_numeric(anomalies_df["adjusted_score"], errors="coerce")

    hybrid_df = pd.read_csv(HYBRID_CSV, dtype=str).fillna("")

    try:
        clusters_df = pd.read_csv(CLUSTERS_CSV)
        clusters_df["adjusted_score"] = pd.to_numeric(clusters_df["adjusted_score"], errors="coerce")
        has_clusters = True
    except FileNotFoundError:
        clusters_df = anomalies_df.copy()
        clusters_df["cluster"] = "1"
        has_clusters = False

    try:
        profiles_df = pd.read_csv(PROFILES_CSV, dtype=str)
        has_profiles = True
    except FileNotFoundError:
        has_profiles = False

    lines = []
    lines.append(DIVIDER)
    lines.append("PHASE 3 REPORT: SUBSTRATE ANOMALY DETECTION")
    lines.append("Turkic Computational Historical Linguistics Project")
    lines.append(DIVIDER)
    lines.append("")

    # ------------------------------------------------------------------
    # Section 1: Score Distribution and Threshold
    # ------------------------------------------------------------------
    lines.append("1. REGULARITY SCORE DISTRIBUTION")
    lines.append(SUBDIV)
    lines.append(f"  Total words scored:          {len(scores_df)}")
    lines.append(f"  Mean adjusted score:         {scores_df['adjusted_score'].mean():.4f}")
    lines.append(f"  Median adjusted score:       {scores_df['adjusted_score'].median():.4f}")
    lines.append(f"  Std dev:                     {scores_df['adjusted_score'].std():.4f}")
    lines.append(f"  Min score:                   {scores_df['adjusted_score'].min():.4f}")
    lines.append(f"  Max score:                   {scores_df['adjusted_score'].max():.4f}")
    lines.append(f"  LexStat-penalized words:     {scores_df['lexstat_penalized'].sum()}")
    lines.append(f"  Yakut artifact flags:        {scores_df['yakut_artifact_flag'].sum()}")
    lines.append("")

    cal = scores_df.loc[~scores_df["yakut_artifact_flag"], "adjusted_score"].dropna()
    threshold = cal.mean() - 1.5 * cal.std()
    lines.append(f"  Anomaly threshold (mean - 1.5σ): {threshold:.4f}")
    lines.append(f"  [Yakut artifact rows excluded from threshold calibration]")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 2: Anomaly Counts by Language
    # ------------------------------------------------------------------
    lines.append("2. ANOMALY COUNTS BY LANGUAGE")
    lines.append(SUBDIV)
    anomaly_lang = anomalies_df["language"].value_counts()
    total_lang   = scores_df["language"].value_counts()
    lines.append(f"  {'Language':<16} {'Anomalies':>10} {'Total':>8} {'Rate':>8}")
    lines.append(f"  {'-'*16} {'-'*10} {'-'*8} {'-'*8}")
    for lang in sorted(total_lang.index):
        n_anom  = anomaly_lang.get(lang, 0)
        n_total = total_lang.get(lang, 0)
        rate    = n_anom / n_total if n_total > 0 else 0
        lines.append(f"  {lang:<16} {n_anom:>10} {n_total:>8} {rate:>7.1%}")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 3: Loan Source Breakdown
    # ------------------------------------------------------------------
    lines.append("3. LOAN SOURCE CLASSIFICATION OF ANOMALIES")
    lines.append(SUBDIV)
    if len(anomalies_df) > 0:
        loan_counts = anomalies_df["loan_source"].value_counts()
        for source, count in loan_counts.items():
            pct = 100 * count / len(anomalies_df)
            lines.append(f"  {source:<25} {count:>5}  ({pct:.1f}%)")
    else:
        lines.append("  No anomalies detected.")
    lines.append("")
    lines.append("  [Note: Arabic/Persian/Russian flags are heuristic phonological screens.")
    lines.append("   Manual verification against Starostin et al. (2003) required for")
    lines.append("   final substrate determination. Mongolic_candidate flagging is based")
    lines.append("   on uvular fricative presence and requires cross-check with Mongolic")
    lines.append("   Swadesh lists in Phase 4.]")
    lines.append("")

    # ------------------------------------------------------------------
    # Section 4: Cluster Profiles
    # ------------------------------------------------------------------
    lines.append("4. UNKNOWN-SOURCE ANOMALY CLUSTERS")
    lines.append(SUBDIV)
    if has_clusters and has_profiles:
        for _, row in profiles_df.iterrows():
            lines.append(f"  Cluster {row['cluster']}  (n={row['size']}, mean score={row['mean_score']})")
            lines.append(f"    Languages:    {row['languages']}")
            lines.append(f"    Top glosses:  {row['top_glosses']}")
            lines.append(f"    Sample forms: {row['sample_forms']}")
            lines.append(f"    Loan flags:   {row['loan_sources']}")
            lines.append("")
    else:
        lines.append("  [Cluster profiles not available - run phonological_clusterer.py first]")
        lines.append("")

    # ------------------------------------------------------------------
    # Section 5: Top 30 Most Irregular Words
    # ------------------------------------------------------------------
    lines.append("5. TOP 30 MOST STATISTICALLY IRREGULAR WORDS")
    lines.append(SUBDIV)
    lines.append("  (unknown-source anomalies, sorted by adjusted_score ascending)")
    lines.append("")

    if has_clusters:
        unknown_df = clusters_df.copy()
        if "loan_source" in unknown_df.columns:
            unknown_df = unknown_df[~unknown_df["loan_source"].isin({"Arabic", "Persian", "Russian"})]
    else:
        unknown_df = anomalies_df[~anomalies_df["loan_source"].isin({"Arabic", "Persian", "Russian"})].copy()

    if len(unknown_df) == 0:
        lines.append("  No unknown-source anomalies found.")
        lines.append("  This likely means the threshold excluded all words or the dataset is")
        lines.append("  too small. Consider reducing to mean - 1.0σ in anomaly_detector.py.")
    else:
        unknown_df["adjusted_score"] = pd.to_numeric(unknown_df["adjusted_score"], errors="coerce")
        top30 = unknown_df.nsmallest(min(30, len(unknown_df)), "adjusted_score")

        lines.append(f"  {'#':<3} {'Lang':<14} {'Gloss':<12} {'Score':>7}  {'Form':<20}  Cross-linguistic forms")
        lines.append(f"  {'-'*3} {'-'*14} {'-'*12} {'-'*7}  {'-'*20}  {'-'*40}")

        for rank, (_, row) in enumerate(top30.iterrows(), 1):
            gloss   = row["gloss"]
            clf     = cross_lang_forms(hybrid_df, gloss)
            clf_str = "  |  ".join(
                f"{l}: {f}" for l, f in sorted(clf.items()) if l != row["language"]
            )
            score = float(row["adjusted_score"])
            lines.append(
                f"  {rank:<3} {row['language']:<14} {gloss:<12} {score:>7.3f}  "
                f"{row['form']:<20}  {clf_str}"
            )
    lines.append("")

    # ------------------------------------------------------------------
    # Section 6: Interpretation
    # ------------------------------------------------------------------
    lines.append("6. INTERPRETATION NOTES")
    lines.append(SUBDIV)
    lines.append("""
  Threshold and model limitations:
  - The regularity scorer uses a token-level lookup against the Phase 2
    correspondence model. Because Phase 2 used a 40-item Swadesh list,
    coverage is limited: phonemes that appear rarely in the list will
    score at the LOG_PROB_FLOOR regardless of their true regularity.
    This biases anomaly detection toward rare phonemes and short words.

  - Words in small cognate groups (n=2 pairs) have noisier correspondence
    probabilities and may be over-flagged. Treat clusters with mean cogid
    size < 3 with caution.

  - LexStat-sourced cognate IDs received a 0.30 penalty. These entries
    have less Savelyev grounding and may include false cognates that
    inflate anomaly rates.

  Filters applied:
  - Yakut s->e artifact (Phase 2 confirmed alignment artifact): flagged
    but NOT included in anomaly threshold calibration or clustering.
  - Oghuz k->q/ɢ alternation: scored as regular (log P = 0) since this
    is the regular Oghuz uvular shift, not substrate.

  Substrate candidates (Phase 4 targets):
  - Words in the "unknown" loan_source category with adjusted_score
    below threshold and NOT in core Swadesh glosses are primary Phase 4
    candidates for Mongolic/Tungusic comparison.
  - Cluster phonological profiles should be compared against Mongolic
    and Tungusic Swadesh lists (via Lexibank) in Phase 4.
  - The Turkmen horn=sah (Phase 2 flag) is a Persian loan and should
    be excluded from substrate analysis.
  - Uyghur bark=qovzaq (Phase 2 flag, missing Savelyev coverage) remains
    a primary substrate candidate pending Mongolic comparison.
""")

    report_text = "\n".join(lines)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(report_text)
    print(f"\nReport written to: {REPORT_TXT}")
    return report_text


if __name__ == "__main__":
    main()
