"""
Infra 6A: Per-Branch Threshold Calibration (v2 FINAL)
======================================================
Replaces the single family-wide anomaly threshold with per-branch baselines
and a two-condition gate.

This is the production version used as input for Phase 6.

Architecture:
  Branch groups:
    A_oghuz    -- Turkish, Azerbaijani (calibration)
                  Turkmen: thin NLex coverage, thin_lang_flag set
    A_kipchak  -- Kazakh, Uzbek (calibration)
    B_yakut    -- Yakut
    C_chuvash  -- Chuvash

  Two-condition gate:
    (1) Score < group_mean - 1.5*sigma  (within own branch group)
    (2) Score also < A_oghuz threshold on the Oghuz phoneme index,
        confirming phoneme sequence is not predictable from central Turkic.
        For Chuvash: also requires >= 1 non-Bulgar token.

Known noise sources (handled downstream in Phase 6 Resolution):
  - A_kipchak: Kazakh ʒ/ʁ/ɾ tokens score low due to model sparsity even
    though these are native Kipchak phonemes. These words will be filtered
    by the cross-language clustering requirement (Phase 6) and proto-form
    comparison (Phase 6 Resolution).
  - A_oghuz: similar sparsity noise for tʃ/dʒ/ɾ in the NLex-expanded model.
  - Known false positive on record: Turkmen jaɣ 'grease' (PT *yāg, debunked
    in Phase 5). Retained in candidates.csv for completeness; exclude from
    Phase 6 cluster analysis.
  - Token-level vs position-level scoring limitation documented in
    project_overview section 5.2. Accepted constraint for this phase.

Inputs:  output/regularity_expanded.csv
         output/prob_model_expanded.json
         output/anomalies_expanded.csv
Outputs: output/infra6a_thresholds.csv
         output/infra6a_candidates.csv
         output/infra6a_artifacts_cleared.csv
         output/infra6a_report.txt
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE = Path("output")
SCORES_CSV      = BASE / "regularity_expanded.csv"
PROB_MODEL_JSON = BASE / "prob_model_expanded.json"
TASK3_ANOM_CSV  = BASE / "anomalies_expanded.csv"
THRESH_CSV      = BASE / "infra6a_thresholds.csv"
CANDIDATES_CSV  = BASE / "infra6a_candidates.csv"
CLEARED_CSV     = BASE / "infra6a_artifacts_cleared.csv"
REPORT_TXT      = BASE / "infra6a_report.txt"

LOG_FLOOR = -6.0
SIGMA     = 1.5

OGHUZ   = {"Turkish", "Azerbaijani", "Turkmen"}
KIPCHAK = {"Kazakh", "Uzbek"}
YAKUT   = {"Yakut"}
CHUVASH = {"Chuvash"}
THIN    = {"Kyrgyz", "Uyghur", "Turkmen"}

OGHUZ_CALIB   = {"Turkish", "Azerbaijani"}
KIPCHAK_CALIB = {"Kazakh", "Uzbek"}

OGHUZ_UVULAR = {"k", "q", "ɢ", "g"}

BULGAR_PHONEMES = {"ɘ", "tɕ", "ɕ", "ʋ", "ʂ", "r", "p"}


def group_of(lang):
    if lang in OGHUZ:    return "A_oghuz"
    if lang in KIPCHAK:  return "A_kipchak"
    if lang in YAKUT:    return "B_yakut"
    if lang in CHUVASH:  return "C_chuvash"
    return "unknown"


def load_scores():
    df = pd.read_csv(SCORES_CSV, dtype=str).fillna("")
    df["adjusted_score"]      = pd.to_numeric(df["adjusted_score"], errors="coerce")
    df["raw_score"]           = pd.to_numeric(df["raw_score"], errors="coerce")
    df["yakut_artifact_flag"] = (
        df["yakut_artifact_flag"].str.lower()
        .map({"true": True, "false": False}).fillna(False)
    )
    df["thin_lang_flag"] = (
        df["thin_lang_flag"].str.lower()
        .map({"true": True, "false": False}).fillna(False)
    )
    df["group"] = df["language"].apply(group_of)
    print(f"[load] {len(df)} rows from {SCORES_CSV}")
    return df


def load_prob_model():
    with open(PROB_MODEL_JSON, "r", encoding="utf-8") as f:
        model = json.load(f)
    print(f"[load] {len(model)} correspondence contexts")
    return model


def load_task3_anomalies():
    df = pd.read_csv(TASK3_ANOM_CSV, dtype=str).fillna("")
    df["adjusted_score"] = pd.to_numeric(df["adjusted_score"], errors="coerce")
    print(f"[load] {len(df)} Task 3 anomalies")
    return df


def build_oghuz_index(prob_model):
    accum = defaultdict(list)
    for key, reflex_dist in prob_model.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        lang_a, lang_b, phoneme_a = parts
        if lang_a not in OGHUZ or lang_b not in OGHUZ:
            continue
        top_p = max(reflex_dist.values())
        accum[phoneme_a].append(math.log(max(top_p, 1e-6)))
    index = {ph: sum(v) / len(v) for ph, v in accum.items()}
    print(f"[oghuz index] {len(index)} phonemes indexed from Oghuz pairs")
    return index


def compute_thresholds(df):
    thresholds = {}

    def _thresh(mask, label, calib_langs):
        scores = df.loc[
            mask &
            (~df["yakut_artifact_flag"]) &
            (df["adjusted_score"] > LOG_FLOOR),
            "adjusted_score"
        ].dropna()
        mean, std = scores.mean(), scores.std()
        thresholds[label] = {
            "mean": mean, "std": std,
            "threshold": mean - SIGMA * std,
            "n": len(scores),
            "calibration_langs": calib_langs,
        }

    _thresh(df["language"].isin(OGHUZ_CALIB),   "A_oghuz",   sorted(OGHUZ_CALIB))
    _thresh(df["language"].isin(KIPCHAK_CALIB),  "A_kipchak", sorted(KIPCHAK_CALIB))
    _thresh(df["language"].isin(YAKUT),          "B_yakut",   ["Yakut"])
    _thresh(df["language"].isin(CHUVASH),        "C_chuvash", ["Chuvash"])

    print("\n[thresholds]")
    for grp, p in thresholds.items():
        print(f"  {grp:12s}: N={p['n']:5d}, mean={p['mean']:.4f}, "
              f"std={p['std']:.4f}, threshold={p['threshold']:.4f}")
    return thresholds


def get_tokens(tok_str):
    t = str(tok_str).strip()
    if not t:
        return []
    t = re.sub(r'^\?\s*', '', t)
    t = re.sub(r'\(.*?\)', '', t).strip()
    return t.split() if t else []


def apply_condition1(df, thresholds):
    flags = []
    for _, row in df.iterrows():
        grp = row["group"]
        if grp not in thresholds or row["yakut_artifact_flag"]:
            flags.append(False)
            continue
        flags.append(float(row["adjusted_score"]) < thresholds[grp]["threshold"])
    df = df.copy()
    df["cond1_anomalous"] = flags
    return df


def oghuz_score(lang, tokens, oghuz_index):
    if not tokens:
        return LOG_FLOOR
    log_probs = []
    for tok in tokens:
        if tok in {"-", "+", "?", ""}:
            continue
        if lang in OGHUZ and tok in OGHUZ_UVULAR:
            log_probs.append(0.0)
            continue
        log_probs.append(oghuz_index.get(tok, LOG_FLOOR))
    return sum(log_probs) / len(log_probs) if log_probs else LOG_FLOOR


def has_non_bulgar_token(tokens):
    for tok in tokens:
        if tok in {"-", "+", "?", ""}:
            continue
        if tok not in BULGAR_PHONEMES:
            return True
    return False


def apply_condition2(df, oghuz_index, threshold_oghuz):
    c2_score = []
    c2_pass  = []

    for _, row in df.iterrows():
        grp = row["group"]

        if not row["cond1_anomalous"]:
            c2_score.append(float(row["adjusted_score"]))
            c2_pass.append(False)
            continue

        if grp in ("A_oghuz", "A_kipchak"):
            c2_score.append(float(row["adjusted_score"]))
            c2_pass.append(True)
            continue

        tokens = get_tokens(str(row.get("ipa_tokens", "")))
        cs = oghuz_score(row["language"], tokens, oghuz_index)
        c2_score.append(cs)

        passes = cs < threshold_oghuz
        if grp == "C_chuvash":
            passes = passes and has_non_bulgar_token(tokens)

        c2_pass.append(passes)

    df = df.copy()
    df["central_score"] = c2_score
    df["cond2_pass"]    = c2_pass
    return df


def classify_loan(row):
    toks = str(row.get("ipa_tokens", "")).lower()
    if any(c in toks for c in ["ʕ", "ħ", "ʔ"]):
        return "persian_arabic_candidate"
    if "ɣ" in toks:
        return "mongolic_candidate"
    return "unknown"


def main():
    print("\n" + "=" * 70)
    print("INFRA 6A v2-FINAL: PER-BRANCH THRESHOLD CALIBRATION")
    print("=" * 70 + "\n")

    scores_df   = load_scores()
    prob_model  = load_prob_model()
    task3_anom  = load_task3_anomalies()
    oghuz_index = build_oghuz_index(prob_model)
    thresholds  = compute_thresholds(scores_df)

    thresh_rows = []
    for grp, p in thresholds.items():
        thresh_rows.append({
            "group":             grp,
            "calibration_langs": ", ".join(p["calibration_langs"]),
            "n_calibration":     p["n"],
            "mean":              round(p["mean"], 4),
            "std":               round(p["std"], 4),
            "threshold":         round(p["threshold"], 4),
        })
    pd.DataFrame(thresh_rows).to_csv(THRESH_CSV, index=False)
    print(f"\n[output] Thresholds -> {THRESH_CSV}")

    df   = apply_condition1(scores_df, thresholds)
    n_c1 = int(df["cond1_anomalous"].sum())
    print(f"\n[condition 1] {n_c1} flagged")
    for grp in ["A_oghuz", "A_kipchak", "B_yakut", "C_chuvash"]:
        n = int(df[df["group"] == grp]["cond1_anomalous"].sum())
        print(f"  {grp}: {n}")

    threshold_oghuz = thresholds["A_oghuz"]["threshold"]
    df = apply_condition2(df, oghuz_index, threshold_oghuz)

    candidates = df[df["cond1_anomalous"] & df["cond2_pass"]].copy()
    candidates["loan_class"] = candidates.apply(classify_loan, axis=1)

    print(f"\n[candidates] {len(candidates)} pass both conditions")
    print("[candidates] By group:")
    for grp in ["A_oghuz", "A_kipchak", "B_yakut", "C_chuvash"]:
        n = len(candidates[candidates["group"] == grp])
        print(f"  {grp}: {n}")
    print("\n[candidates] By loan class:")
    print(candidates["loan_class"].value_counts().to_string())
    print("\n[candidates] By language:")
    for lang, sub in candidates.groupby("language"):
        flag = " [thin-lang]" if lang in THIN else ""
        print(f"  {lang}: {len(sub)}{flag}")

    out_cols = [
        "language", "gloss", "form", "ipa_tokens",
        "group", "cogid_source",
        "adjusted_score", "central_score",
        "cond1_anomalous", "cond2_pass",
        "loan_class", "thin_lang_flag",
    ]
    candidates[out_cols].sort_values(["group", "adjusted_score"]).to_csv(
        CANDIDATES_CSV, index=False
    )
    print(f"\n[output] Candidates -> {CANDIDATES_CSV}")

    task3_keys   = set(zip(task3_anom["language"].str.strip(),
                           task3_anom["gloss"].str.strip()))
    infra6a_keys = set(zip(candidates["language"].str.strip(),
                           candidates["gloss"].str.strip()))
    cleared         = task3_keys - infra6a_keys
    survived        = task3_keys & infra6a_keys
    new_only        = infra6a_keys - task3_keys
    cleared_yakut   = {(l, g) for l, g in cleared if l == "Yakut"}
    cleared_chuvash = {(l, g) for l, g in cleared if l == "Chuvash"}
    cleared_central = cleared - cleared_yakut - cleared_chuvash

    print(f"\n[comparison] Task 3 anomalies:     {len(task3_keys)}")
    print(f"[comparison] Infra 6A candidates:  {len(infra6a_keys)}")
    print(f"[comparison] Cleared (artifacts):  {len(cleared)}")
    print(f"  Yakut cleared:                   {len(cleared_yakut)}")
    print(f"  Chuvash cleared:                 {len(cleared_chuvash)}")
    print(f"  Central cleared:                 {len(cleared_central)}")
    print(f"[comparison] Survived into 6A:     {len(survived)}")
    print(f"[comparison] New in 6A only:       {len(new_only)}")

    cleared_rows = []
    for lang, gloss in sorted(cleared):
        row = task3_anom[(task3_anom["language"] == lang) &
                         (task3_anom["gloss"] == gloss)]
        score = float(row["adjusted_score"].values[0]) if len(row) > 0 else None
        if lang == "Yakut":
            reason = "Yakut within-group normal -- inherited Siberian divergence"
        elif lang == "Chuvash":
            reason = "Chuvash within-group normal or Bulgar phonology"
        else:
            reason = "Central Turkic: passes Infra 6A two-condition gate"
        cleared_rows.append({"language": lang, "gloss": gloss,
                              "task3_score": score, "clearance_reason": reason})
    pd.DataFrame(cleared_rows).to_csv(CLEARED_CSV, index=False)
    print(f"\n[output] Cleared artifacts -> {CLEARED_CSV}")

    dist_lines = [
        "\nSCORE DISTRIBUTION BY BRANCH GROUP:",
        f"{'Group':<14} {'N':>6} {'Mean':>8} {'Std':>8} "
        f"{'Threshold':>10} {'Cond1':>8} {'Final':>7}",
        "-" * 65,
    ]
    for grp in ["A_oghuz", "A_kipchak", "B_yakut", "C_chuvash"]:
        grp_df = df[df["group"] == grp]
        p      = thresholds[grp]
        c1_n   = int(grp_df["cond1_anomalous"].sum())
        c2_n   = len(candidates[candidates["group"] == grp])
        dist_lines.append(
            f"{grp:<14} {len(grp_df):>6} {p['mean']:>8.4f} {p['std']:>8.4f} "
            f"{p['threshold']:>10.4f} {c1_n:>8} {c2_n:>7}"
        )
    print("\n".join(dist_lines))

    report = [
        "INFRA 6A v2-FINAL: PER-BRANCH THRESHOLD CALIBRATION",
        "=" * 70,
        "",
        "ARCHITECTURE",
        "------------",
        "Branch groups:",
        "  A_oghuz   -- Turkish, Azerbaijani (calibration); Turkmen: thin NLex",
        "  A_kipchak -- Kazakh, Uzbek",
        "  B_yakut   -- Yakut",
        "  C_chuvash -- Chuvash",
        "",
        "Two-condition gate:",
        "  (1) Score < group_mean - 1.5*sigma",
        "  (2) Score < A_oghuz threshold on Oghuz phoneme index",
        "      For Chuvash: also requires >= 1 non-Bulgar token.",
        "",
        "Known noise: A_kipchak and A_oghuz candidates include some words",
        "where anomalous scores reflect model sparsity for native phonemes",
        "(ʒ/ʁ/ɾ in Kazakh; tʃ/dʒ/ɾ in Turkish/Azerbaijani) rather than",
        "genuine irregularity. These are filtered downstream by:",
        "  (a) cross-language clustering requirement in Phase 6",
        "  (b) proto-form comparison in Phase 6 Resolution",
        "Known false positive on record: Turkmen jaɣ 'grease' (PT *yāg,",
        "debunked Phase 5). Retained in candidates.csv; exclude from Phase 6.",
        "",
        "THRESHOLD PARAMETERS",
        "--------------------",
    ]
    for grp, p in thresholds.items():
        report.append(
            f"  {grp}: mean={p['mean']:.4f}, std={p['std']:.4f}, "
            f"threshold={p['threshold']:.4f}  (N={p['n']}, "
            f"calib={', '.join(p['calibration_langs'])})"
        )
    report += [
        "",
        "RESULTS",
        "-------",
        f"Task 3 anomaly count:          {len(task3_keys)}",
        f"Infra 6A candidate count:      {len(infra6a_keys)}",
        f"  Cleared as artifacts:        {len(cleared)}",
        f"    Yakut cleared:             {len(cleared_yakut)}",
        f"    Chuvash cleared:           {len(cleared_chuvash)}",
        f"    Central cleared:           {len(cleared_central)}",
        f"  Survived from Task 3:        {len(survived)}",
        f"  New in Infra 6A only:        {len(new_only)}",
        "",
        "\n".join(dist_lines),
        "",
        "STATUS: Infra 6A complete. Candidates ready for Phase 6.",
        "Phase 6 cross-language clustering will reduce the candidate list",
        "substantially before proto-form comparison is applied.",
        "",
        "OUTPUT FILES:",
        f"  {THRESH_CSV}",
        f"  {CANDIDATES_CSV}",
        f"  {CLEARED_CSV}",
        f"  {REPORT_TXT}",
    ]

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"\n[output] Report -> {REPORT_TXT}")
    print("\n[done] Infra 6A complete.")


if __name__ == "__main__":
    main()
