"""
Phase 5 Task 3: Rebuild correspondence model and anomaly detector on expanded dataset
======================================================================================
Inputs:
  output/northeuralex_merged.csv     -- 6,966 rows, 9 languages, ~1000 concepts
  output/hybrid_cognates.csv         -- original 353-row Phase 3 input (for comparison)
  output/regularity_scores.csv       -- Phase 3 scores (for comparison)
  output/unknown_anomalies.csv       -- Phase 3 anomaly candidates (for comparison)

Outputs:
  output/lingpy_input_expanded.tsv   -- LexStat input
  output/cognate_sets_expanded.tsv   -- LexStat cognate assignments
  output/correspondence_expanded.csv -- pairwise correspondence table
  output/prob_model_expanded.json    -- rebuilt probabilistic model
  output/regularity_expanded.csv     -- regularity scores for all 6,966 words
  output/anomalies_expanded.csv      -- flagged anomalies (unknown source)
  output/task3_report.txt            -- comparison against Phase 3 results

Known issues carried forward:
  - LingPy 2.6.13: runs=0 required (integer join bug in get_scorer)
  - Input must be TSV file, not dict
  - Kyrgyz/Turkmen/Uyghur have ~38-40 items only; excluded from threshold calibration
  - Oghuz k->q/g uvular shift is regular, not substrate signal
  - Yakut s->e patterns are alignment artifacts

Run from project root with venv311 active:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python phase5_substrate\\task3_expand_model.py

Estimated runtime: 5-15 minutes depending on machine (LexStat on ~6000 rows).
"""

import json
import math
import re
import warnings
import csv
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

try:
    from lingpy import LexStat, Alignments
    LINGPY_OK = True
except ImportError:
    LINGPY_OK = False
    print("[ERROR] LingPy not available. Activate venv311.")
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE   = Path("output")
BASE.mkdir(exist_ok=True)

MERGED_CSV      = BASE / "northeuralex_merged.csv"
HYBRID_CSV      = BASE / "hybrid_cognates.csv"
P3_SCORES_CSV   = BASE / "regularity_scores.csv"
P3_ANOMALY_CSV  = BASE / "unknown_anomalies.csv"

TSV_INPUT       = BASE / "lingpy_input_expanded.tsv"
COGSETS_TSV     = BASE / "cognate_sets_expanded"      # LingPy appends .tsv
CORR_CSV        = BASE / "correspondence_expanded.csv"
PROB_MODEL_JSON = BASE / "prob_model_expanded.json"
SCORES_CSV      = BASE / "regularity_expanded.csv"
ANOMALIES_CSV   = BASE / "anomalies_expanded.csv"
REPORT_TXT      = BASE / "task3_report.txt"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEXSTAT_THRESHOLD  = 0.55   # slightly more permissive than Phase 3's 0.50
                             # because dataset is much larger and more diverse
LEXSTAT_PENALTY    = 0.30
LOG_PROB_FLOOR     = -6.0
ANOMALY_SIGMA      = 1.5    # threshold = mean - 1.5*sigma

# Languages with full NorthEuraLex coverage — use for threshold calibration only
FULL_COVERAGE_LANGS = {"Turkish", "Azerbaijani", "Uzbek", "Kazakh", "Yakut", "Chuvash"}
THIN_LANGS          = {"Kyrgyz", "Turkmen", "Uyghur"}   # ~38-40 items only

OGHUZ_LANGS         = {"Turkish", "Azerbaijani", "Turkmen"}
OGHUZ_UVULAR        = {"k", "q", "ɢ", "g"}
YAKUT_ARTIFACT      = {"e"}

# ---------------------------------------------------------------------------
# Step 1: Load merged dataset
# ---------------------------------------------------------------------------
def load_merged():
    df = pd.read_csv(MERGED_CSV, dtype=str).fillna("")
    print(f"[load] {len(df)} rows | {df['language'].nunique()} languages "
          f"| {df['gloss'].nunique()} concepts")
    for lang, grp in df.groupby("language"):
        print(f"       {lang}: {len(grp)} entries")
    return df

# ---------------------------------------------------------------------------
# Step 2: Write LingPy TSV input
# ---------------------------------------------------------------------------
def write_lingpy_tsv(df: pd.DataFrame) -> str:
    rows = []
    idx = 1
    skipped = 0
    for _, row in df.iterrows():
        tok_str = str(row.get("ipa_tokens", "")).strip()
        if not tok_str:
            skipped += 1
            continue
        # Clean: remove parentheticals, leading ?
        tok_str = re.sub(r'^\?\s*', '', tok_str)
        tok_str = re.sub(r'\(.*?\)', '', tok_str).strip()
        tokens  = tok_str.split()
        if not tokens:
            skipped += 1
            continue
        rows.append({
            "ID":       str(idx),
            "DOCULECT": row["language"],
            "CONCEPT":  row["gloss"],
            "TOKENS":   " ".join(tokens),
            "IPA":      str(row.get("form", "")),
        })
        idx += 1

    tsv_df = pd.DataFrame(rows, columns=["ID", "DOCULECT", "CONCEPT", "TOKENS", "IPA"])
    tsv_df.to_csv(TSV_INPUT, sep="\t", index=False)
    print(f"[tsv] {len(rows)} entries written ({skipped} skipped, no tokens)")
    return str(TSV_INPUT)

# ---------------------------------------------------------------------------
# Step 3: LexStat cognate detection
# ---------------------------------------------------------------------------
def run_lexstat(tsv_path: str):
    print(f"\n[LexStat] Loading wordlist from {tsv_path} ...")
    print(f"[LexStat] threshold={LEXSTAT_THRESHOLD}, runs=0 (SCA fallback)")
    lex = LexStat(tsv_path)
    lex.get_scorer(runs=0)
    print("[LexStat] Clustering ...")
    lex.cluster(method="lexstat", threshold=LEXSTAT_THRESHOLD, ref="cogid")
    cog_ids = set(lex[i, "cogid"] for i in lex)
    print(f"[LexStat] {len(cog_ids)} cognate sets identified")
    lex.output("tsv", filename=str(COGSETS_TSV), ignore="all", prettify=False)
    print(f"[LexStat] Output written to {COGSETS_TSV}.tsv")
    return lex

# ---------------------------------------------------------------------------
# Step 4: SCA alignment and correspondence extraction
# ---------------------------------------------------------------------------
def run_alignment_and_extract(lex):
    print("\n[SCA] Running multiple alignment ...")
    alm = Alignments(lex, ref="cogid")
    alm.align(method="progressive")
    print("[SCA] Done.")

    records = []
    for cogset_id, msa_block in alm.msa["cogid"].items():
        taxa    = msa_block["taxa"]
        aligns  = msa_block["alignment"]
        concept = msa_block.get("concept", "?")
        if len(taxa) < 2:
            continue
        n_cols = max(len(a) for a in aligns)
        for col_idx in range(n_cols):
            col = {
                lang: (alignment[col_idx] if col_idx < len(alignment) else "-")
                for lang, alignment in zip(taxa, aligns)
            }
            langs = list(col.keys())
            for i in range(len(langs)):
                for j in range(i + 1, len(langs)):
                    la, lb = langs[i], langs[j]
                    pa, pb = col[la], col[lb]
                    if pa == "-" and pb == "-":
                        continue
                    records.append({
                        "lang_a": la, "lang_b": lb,
                        "phoneme_a": pa, "phoneme_b": pb,
                        "gloss": concept,
                        "cogset_id": cogset_id,
                        "position": col_idx,
                    })

    corr_df = pd.DataFrame(records)
    print(f"[correspondences] {len(corr_df)} pairwise phoneme tokens extracted")
    corr_df.to_csv(CORR_CSV, index=False)
    print(f"[correspondences] saved to {CORR_CSV}")
    return corr_df

# ---------------------------------------------------------------------------
# Step 5: Build probabilistic model
# ---------------------------------------------------------------------------
def build_prob_model(corr_df: pd.DataFrame) -> dict:
    counts = defaultdict(lambda: defaultdict(int))
    for _, row in corr_df.iterrows():
        la, lb = row["lang_a"], row["lang_b"]
        pa, pb = row["phoneme_a"], row["phoneme_b"]
        counts[(la, lb, pa)][pb] += 1
        counts[(lb, la, pb)][pa] += 1

    prob_model = {}
    for (la, lb, pa), cnt_dict in counts.items():
        total = sum(cnt_dict.values())
        prob_model[f"{la}|{lb}|{pa}"] = {
            ph: round(n / total, 4)
            for ph, n in sorted(cnt_dict.items(), key=lambda x: -x[1])
        }
    print(f"[prob model] {len(prob_model)} contexts modelled")
    with open(PROB_MODEL_JSON, "w", encoding="utf-8") as f:
        json.dump(prob_model, f, ensure_ascii=False, indent=2)
    print(f"[prob model] saved to {PROB_MODEL_JSON}")
    return prob_model

# ---------------------------------------------------------------------------
# Step 6: Build token score index
# ---------------------------------------------------------------------------
def build_token_index(prob_model: dict) -> dict:
    accum = defaultdict(list)
    for key, reflex_dist in prob_model.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        lang_a, lang_b, phoneme_a = parts
        top_p = max(reflex_dist.values())
        accum[(lang_a, phoneme_a)].append(math.log(max(top_p, 1e-6)))
    index = {k: sum(v) / len(v) for k, v in accum.items()}
    print(f"[token index] {len(index)} (language, phoneme) pairs indexed")
    return index

# ---------------------------------------------------------------------------
# Step 7: Score all words
# ---------------------------------------------------------------------------
def get_tokens(tok_str: str) -> list:
    t = str(tok_str).strip()
    if not t:
        return []
    t = re.sub(r'^\?\s*', '', t)
    t = re.sub(r'\(.*?\)', '', t).strip()
    return t.split() if t else []

def score_word(lang: str, tokens: list, token_index: dict) -> tuple:
    if not tokens:
        return (LOG_PROB_FLOOR, 0, 0)
    log_probs = []
    n_lookups = 0
    for token in tokens:
        if token in {"-", "+", "?", ""}:
            continue
        # Oghuz uvular regular shift: assign max score
        if lang in OGHUZ_LANGS and token in OGHUZ_UVULAR:
            log_probs.append(0.0)
            n_lookups += 1
            continue
        key = (lang, token)
        if key in token_index:
            log_probs.append(token_index[key])
        else:
            log_probs.append(LOG_PROB_FLOOR)
        n_lookups += 1
    if not log_probs:
        return (LOG_PROB_FLOOR, len(tokens), 0)
    return (sum(log_probs) / len(log_probs), len(tokens), n_lookups)

def is_yakut_artifact(lang: str, tokens: list) -> bool:
    if lang != "Yakut" or not tokens:
        return False
    return tokens.count("e") / len(tokens) >= 0.4

def score_all(df: pd.DataFrame, token_index: dict) -> pd.DataFrame:
    print("\n[scoring] Computing regularity scores ...")
    results = []
    for _, row in df.iterrows():
        lang         = row["language"]
        tokens       = get_tokens(row.get("ipa_tokens", ""))
        yakut_flag   = is_yakut_artifact(lang, tokens)
        raw, n_tok, n_look = score_word(lang, tokens, token_index)
        is_lexstat   = str(row.get("cogid_source", "")).lower() == "lexstat"
        adjusted     = raw - (LEXSTAT_PENALTY if is_lexstat else 0.0)
        results.append({
            "language":            lang,
            "gloss":               row.get("gloss", ""),
            "form":                row.get("form", ""),
            "ipa_tokens":          row.get("ipa_tokens", ""),
            "cogid_source":        row.get("cogid_source", ""),
            "raw_score":           round(raw, 4),
            "adjusted_score":      round(adjusted, 4),
            "lexstat_penalized":   is_lexstat,
            "yakut_artifact_flag": yakut_flag,
            "thin_lang_flag":      lang in THIN_LANGS,
            "n_tokens":            n_tok,
            "n_lookups":           n_look,
        })
    out = pd.DataFrame(results)
    out.to_csv(SCORES_CSV, index=False)
    print(f"[scoring] Scores written to {SCORES_CSV}")
    return out

# ---------------------------------------------------------------------------
# Step 8: Anomaly detection and clustering
# ---------------------------------------------------------------------------
def detect_anomalies(scores_df: pd.DataFrame) -> pd.DataFrame:
    print("\n[anomaly] Computing threshold ...")

    # Calibrate on full-coverage languages only, exclude Yakut artifacts
    calib = scores_df[
        (scores_df["language"].isin(FULL_COVERAGE_LANGS)) &
        (~scores_df["yakut_artifact_flag"]) &
        (scores_df["adjusted_score"] > LOG_PROB_FLOOR)
    ]["adjusted_score"]

    mean_score = calib.mean()
    std_score  = calib.std()
    threshold  = mean_score - ANOMALY_SIGMA * std_score

    print(f"[anomaly] Calibration set: {len(calib)} words (full-coverage, non-artifact)")
    print(f"[anomaly] Mean={mean_score:.4f}, Std={std_score:.4f}, Threshold={threshold:.4f}")

    # Flag anomalies across all languages
    anomalies = scores_df[scores_df["adjusted_score"] < threshold].copy()
    anomalies["threshold"] = round(threshold, 4)

    # Heuristic loan classification
    def classify(row):
        form = str(row.get("form", "")).lower()
        toks = str(row.get("ipa_tokens", "")).lower()
        # Persian/Arabic markers
        if any(c in toks for c in ["ʃ", "χ", "ʁ", "ʔ"]):
            return "persian_arabic_candidate"
        # Mongolic markers: uvular fricative
        if "ɣ" in toks:
            return "mongolic_candidate"
        # Russian loans: often have consonant clusters unusual in Turkic
        if row["language"] in {"Kazakh", "Kyrgyz", "Uzbek"} and len(toks.split()) > 6:
            return "possible_russian"
        return "unknown"

    anomalies["loan_class"] = anomalies.apply(classify, axis=1)
    anomalies["thin_lang_note"] = anomalies["thin_lang_flag"].apply(
        lambda x: "LOW_CONFIDENCE_thin_lang" if x else ""
    )

    # Save unknown-source anomalies (primary substrate candidates)
    unknown = anomalies[anomalies["loan_class"] == "unknown"].copy()
    unknown.to_csv(ANOMALIES_CSV, index=False)
    print(f"[anomaly] {len(anomalies)} total anomalies flagged")
    print(f"[anomaly] {len(unknown)} unknown-source anomalies saved to {ANOMALIES_CSV}")

    # Breakdown by language
    print("\n[anomaly] Breakdown by language:")
    for lang, grp in anomalies.groupby("language"):
        thin = " [thin-lang, low confidence]" if lang in THIN_LANGS else ""
        print(f"       {lang}: {len(grp)} anomalies{thin}")

    return anomalies, threshold, mean_score, std_score

# ---------------------------------------------------------------------------
# Step 9: Compare against Phase 3 results
# ---------------------------------------------------------------------------
def compare_with_phase3(anomalies: pd.DataFrame) -> str:
    try:
        p3 = pd.read_csv(P3_ANOMALY_CSV, dtype=str).fillna("")
        p3_keys = set(zip(p3["language"].str.strip(), p3["gloss"].str.strip()))
    except Exception as e:
        return f"[comparison] Could not load Phase 3 anomalies: {e}"

    new_keys = set(zip(
        anomalies["language"].str.strip(),
        anomalies["gloss"].str.strip()
    ))

    survived  = p3_keys & new_keys
    dropped   = p3_keys - new_keys
    new_finds = new_keys - p3_keys

    lines = [
        "COMPARISON WITH PHASE 3 ANOMALY LIST",
        "=" * 50,
        f"Phase 3 unknown anomalies:        {len(p3_keys)}",
        f"Task 3 total anomalies (unknown): {len(new_keys)}",
        f"",
        f"Survived into Task 3:  {len(survived)}",
        f"Dropped (Phase 3 artifact or now regular): {len(dropped)}",
        f"New anomalies (NorthEuraLex expansion): {len(new_finds)}",
        "",
        "SURVIVED (high confidence — present in both Phase 3 and Task 3):",
    ]
    for lang, gloss in sorted(survived):
        lines.append(f"  {lang}: {gloss}")

    lines += ["", "DROPPED (were anomalous on 40-item data, now regular):"]
    for lang, gloss in sorted(dropped):
        lines.append(f"  {lang}: {gloss}")

    lines += ["", f"NEW ANOMALIES FROM EXPANDED DATASET (first 30):"]
    for lang, gloss in sorted(list(new_finds))[:30]:
        lines.append(f"  {lang}: {gloss}")
    if len(new_finds) > 30:
        lines.append(f"  ... and {len(new_finds) - 30} more (see anomalies_expanded.csv)")

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("\n" + "=" * 70)
    print("PHASE 5 TASK 3: EXPANDED MODEL AND ANOMALY DETECTION")
    print("=" * 70 + "\n")

    # Load
    df = load_merged()

    # LexStat
    tsv_path = write_lingpy_tsv(df)
    lex      = run_lexstat(tsv_path)

    # Alignment + correspondences
    corr_df = run_alignment_and_extract(lex)

    # Prob model
    prob_model  = build_prob_model(corr_df)
    token_index = build_token_index(prob_model)

    # Score
    scores_df = score_all(df, token_index)

    # Anomaly detection
    anomalies, threshold, mean_s, std_s = detect_anomalies(scores_df)

    # Comparison with Phase 3
    comparison = compare_with_phase3(
        anomalies[anomalies["loan_class"] == "unknown"]
    )
    print("\n" + comparison)

    # Score distribution by language
    dist_lines = ["\nSCORE DISTRIBUTION BY LANGUAGE (full-coverage only):"]
    dist_lines.append(f"{'Language':<14} {'N':>5} {'Mean':>7} {'Std':>7} "
                      f"{'Min':>7} {'Anomalies':>10}")
    dist_lines.append("-" * 55)
    for lang, grp in scores_df.groupby("language"):
        anom_n = len(anomalies[anomalies["language"] == lang])
        dist_lines.append(
            f"{lang:<14} {len(grp):>5} {grp['adjusted_score'].mean():>7.3f} "
            f"{grp['adjusted_score'].std():>7.3f} "
            f"{grp['adjusted_score'].min():>7.3f} {anom_n:>10}"
        )
    print("\n".join(dist_lines))

    # Write report
    report = "\n".join([
        "PHASE 5 TASK 3 REPORT",
        "=" * 70,
        f"Date: 2026-03-04",
        f"Input: {MERGED_CSV} ({len(df)} rows)",
        f"LexStat threshold: {LEXSTAT_THRESHOLD}",
        f"Anomaly threshold: mean - {ANOMALY_SIGMA}*sigma = {threshold:.4f}",
        f"  Calibration mean: {mean_s:.4f}, std: {std_s:.4f}",
        f"  Calibrated on: {', '.join(sorted(FULL_COVERAGE_LANGS))}",
        f"  Excluded from calibration: {', '.join(sorted(THIN_LANGS))} (thin coverage)",
        "",
        f"Total anomalies flagged: {len(anomalies)}",
        f"Unknown-source anomalies: {len(anomalies[anomalies['loan_class']=='unknown'])}",
        "",
        "\n".join(dist_lines),
        "",
        comparison,
        "",
        "OUTPUT FILES:",
        f"  {TSV_INPUT}",
        f"  {COGSETS_TSV}.tsv",
        f"  {CORR_CSV}",
        f"  {PROB_MODEL_JSON}",
        f"  {SCORES_CSV}",
        f"  {ANOMALIES_CSV}",
    ])

    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n[done] Report written to {REPORT_TXT}")

if __name__ == "__main__":
    main()
