"""
Phase 3 - Step 2: Anomaly Detection and Loanword Separation
=============================================================
Takes regularity_scores.csv and:
  1. Determines anomaly threshold from known loanword behavior
  2. Flags anomalous words
  3. Separates known-source loanwords (Arabic, Persian, Mongolic, Russian)
     from unknown-source anomalies
  4. Writes anomalies.csv and unknown_anomalies.csv

Known loanword detection
------------------------
Since we don't have full etymological tags in the dataset, we use two methods:
  A) Phonological fingerprints: sound patterns characteristic of Arabic,
     Persian, Mongolic, or Russian loanwords in Turkic languages.
  B) Gloss-level priors: certain Swadesh glosses almost never come from loans
     (body parts, basic kin, pronouns, numbers); others are more loan-prone.

This is a heuristic screen -- not a replacement for the Turkic Etymological
Dictionary (Starostin et al. 2003), which should be consulted manually for
the top candidates in the final substrate report.

Threshold tuning
----------------
The threshold is set at mean - 1.5 * std of adjusted_score, but we also
examine whether known loan-prone glosses cluster below this cutoff.
The Oghuz k->q and Yakut s->e filters have already been applied upstream.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "output"
SCORES_CSV     = BASE / "regularity_scores.csv"
ANOMALIES_CSV  = BASE / "anomalies.csv"
UNKNOWN_CSV    = BASE / "unknown_anomalies.csv"

# ---------------------------------------------------------------------------
# Phonological fingerprints for known loan sources
# ---------------------------------------------------------------------------
ARABIC_TOKENS    = {"ʕ", "ħ", "ʔ", "ʡ", "ðˤ", "sˤ", "zˤ", "lˤ", "tˤ"}
ARABIC_FORMS_RE  = re.compile(
    r"(aːl|ʕ|ħ|ʔ|qurʔ|dʒinn|isl)", re.IGNORECASE
)
PERSIAN_TOKENS   = {"ɑː", "ɑ"}
PERSIAN_FORMS_RE = re.compile(r"(ʃaː?|xaː?|ɑːn$|aːn$|aːd$)", re.IGNORECASE)
MONGOLIC_TOKENS  = {"ɣ"}
MONGOLIC_FORMS_RE = re.compile(r"(ɣ)", re.IGNORECASE)
RUSSIAN_TOKENS   = {"f", "v"}
RUSSIAN_FORMS_RE = re.compile(r"(^f|^v)", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Classify a single word's likely loan source - MUST return a plain string
# ---------------------------------------------------------------------------
def classify_loan_source(form, ipa_tokens, gloss):
    """Returns a plain string: Arabic | Persian | Mongolic_candidate | Russian | unknown"""
    form        = str(form)        if form        is not None else ""
    ipa_tokens  = str(ipa_tokens)  if ipa_tokens  is not None else ""
    gloss       = str(gloss)       if gloss        is not None else ""

    tokens_list = ipa_tokens.strip().split() if ipa_tokens.strip() else []
    token_set   = set(tokens_list)

    if ARABIC_TOKENS & token_set or ARABIC_FORMS_RE.search(form):
        return "Arabic"
    if PERSIAN_TOKENS & token_set or PERSIAN_FORMS_RE.search(form):
        return "Persian"
    if RUSSIAN_TOKENS & token_set or RUSSIAN_FORMS_RE.search(form):
        return "Russian"
    if MONGOLIC_TOKENS & token_set or MONGOLIC_FORMS_RE.search(form):
        return "Mongolic_candidate"

    return "unknown"

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(scores_df=None):
    if scores_df is None:
        print("Loading regularity scores...")
        scores_df = pd.read_csv(SCORES_CSV)

    # Ensure correct dtypes regardless of how the df was loaded
    scores_df["adjusted_score"]      = pd.to_numeric(scores_df["adjusted_score"], errors="coerce")
    scores_df["raw_score"]           = pd.to_numeric(scores_df["raw_score"], errors="coerce")
    # Handle both bool and string representations of the flag columns
    scores_df["yakut_artifact_flag"] = scores_df["yakut_artifact_flag"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)
    scores_df["lexstat_penalized"]   = scores_df["lexstat_penalized"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    ).fillna(False).astype(bool)

    # -------------------------------------------------------------------
    # Threshold: exclude Yakut artifact rows from calibration
    # -------------------------------------------------------------------
    calibration_mask = ~scores_df["yakut_artifact_flag"]
    cal_scores = scores_df.loc[calibration_mask, "adjusted_score"].dropna()

    mean_score = cal_scores.mean()
    std_score  = cal_scores.std()
    threshold  = mean_score - 1.5 * std_score

    print(f"\nThreshold calibration (Yakut artifacts excluded):")
    print(f"  N calibration words: {len(cal_scores)}")
    print(f"  Mean:      {mean_score:.4f}")
    print(f"  Std dev:   {std_score:.4f}")
    print(f"  Threshold (mean - 1.5σ): {threshold:.4f}")

    # -------------------------------------------------------------------
    # Flag anomalies
    # -------------------------------------------------------------------
    scores_df["is_anomalous"] = (
        (scores_df["adjusted_score"] < threshold) &
        (~scores_df["yakut_artifact_flag"])
    )

    n_anomalous = scores_df["is_anomalous"].sum()
    n_total     = len(scores_df)
    print(f"\nAnomalous words: {n_anomalous} / {n_total} ({100*n_anomalous/n_total:.1f}%)")

    anomalies_df = scores_df[scores_df["is_anomalous"]].copy()

    # -------------------------------------------------------------------
    # Classify loan source - use iterrows to guarantee scalar per row
    # -------------------------------------------------------------------
    loan_labels = []
    for _, row in anomalies_df.iterrows():
        label = classify_loan_source(row["form"], row["ipa_tokens"], row["gloss"])
        loan_labels.append(label)
    anomalies_df = anomalies_df.copy()
    anomalies_df["loan_source"] = loan_labels

    # Write all anomalies
    anomalies_df.to_csv(ANOMALIES_CSV, index=False)
    print(f"Anomalies written to: {ANOMALIES_CSV}")

    # Loan source breakdown
    print("\nLoan source classification of anomalies:")
    print(anomalies_df["loan_source"].value_counts().to_string())

    # -------------------------------------------------------------------
    # Unknown-source anomalies
    # -------------------------------------------------------------------
    known_sources = {"Arabic", "Persian", "Russian"}
    unknown_df = anomalies_df[~anomalies_df["loan_source"].isin(known_sources)].copy()

    unknown_df.to_csv(UNKNOWN_CSV, index=False)
    print(f"\nUnknown-source anomalies: {len(unknown_df)}")
    print(f"Written to: {UNKNOWN_CSV}")

    print("\nAnomaly counts by language (unknown-source only):")
    print(unknown_df["language"].value_counts().to_string())

    print("\nTop 30 most irregular words (unknown-source anomalies):")
    top30 = unknown_df.nsmallest(30, "adjusted_score")[
        ["language", "gloss", "form", "adjusted_score", "loan_source", "cogid_source"]
    ]
    print(top30.to_string(index=False))

    return scores_df, anomalies_df, unknown_df, threshold


if __name__ == "__main__":
    main()
