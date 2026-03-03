"""
Phase 2: Cognate Detection, Sound Correspondence Extraction, and Validation
Computational Historical Linguistics — Turkic Substrate Detection

Inputs:  output/turkic_merged_phase1.csv
Outputs: output/cognate_sets.tsv
         output/alignments.html
         output/correspondence_table.csv
         output/prob_model.json
         output/flagged_correspondences.csv
         output/phase2_report.txt

Dependencies:
    pip install lingpy pandas numpy scipy
"""

import json
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import numpy as np

try:
    from lingpy import LexStat, Alignments
    LINGPY_AVAILABLE = True
except ImportError:
    LINGPY_AVAILABLE = False
    print("[WARNING] LingPy not found. Install with: pip install lingpy")

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

# LexStat threshold. Default for most families is 0.55; we use 0.50 for Turkic
# because the family is shallow (~2100 BP) and we want conservative, high-confidence
# cognate sets going into Phase 3 anomaly detection.
LEXSTAT_THRESHOLD = 0.50

LANGUAGES = [
    "Turkish", "Uzbek", "Kazakh", "Kyrgyz", "Uyghur",
    "Yakut", "Chuvash", "Azerbaijani", "Turkmen"
]

# Documented Proto-Turkic sound laws for validation.
# Format: (proto_phoneme_as_appears_in_data, target_language, expected_reflex)
# Sources: Johanson 1998, Clauson 1972, Savelyev & Robbeets 2020.
#
# Note: the savelyevturkic data uses pre-tokenized IPA segments, so the
# "proto phoneme" here is whatever the non-Turkish/non-Yakut cognates show —
# effectively the conservative reflex that maps to the known change.
PROTO_TURKIC_LAWS = [
    ("d", "Turkish",    "y"),   # *d- > y- (Oghuz)
    ("d", "Yakut",      "s"),   # *d- > s- (Siberian/Sakha)
    ("d", "Chuvash",    "ś"),   # *d- > ś- (Bulgar branch)
    ("d", "Uzbek",      "j"),   # *d- > j- (Qarluq)
    ("d", "Kazakh",     "ʒ"),   # *d- > ʒ- (Kipchak)
    ("ŋ", "Turkish",    "n"),   # *ŋ- > n- (Oghuz word-initial)
    ("ŋ", "Yakut",      "ŋ"),   # preserved in Yakut
    ("b", "Chuvash",    "p"),   # *b > p (Chuvash devoicing, archaic)
    ("z", "Chuvash",    "r"),   # *z > r (Chuvash rhotacism, the diagnostic Bulgar feature)
]

FRONT_VOWELS = {"e", "i", "ö", "ü", "æ", "ɛ", "ɪ", "y", "œ"}
BACK_VOWELS  = {"a", "ɯ", "o", "u", "ɑ", "ʌ", "ɔ"}


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Load Phase 1 data
# ─────────────────────────────────────────────────────────────────────────────

def _parse_token_col(val):
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("["):
            import ast
            try:
                return ast.literal_eval(s)
            except Exception:
                pass
        return s.split()
    return []


def load_phase1(csv_path: str = "output/turkic_merged_phase1.csv") -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"language", "gloss", "form", "ipa_tokens"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Phase 1 CSV missing columns: {missing}")
    df["ipa_tokens"] = df["ipa_tokens"].apply(_parse_token_col)
    print(f"[load] {len(df)} rows | {df['language'].nunique()} languages "
          f"| {df['gloss'].nunique()} glosses")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LexStat cognate detection
# ─────────────────────────────────────────────────────────────────────────────

def build_lingpy_wordlist(df: pd.DataFrame) -> str:
    """Write a TSV file in LingPy format and return the path."""
    tsv_path = "output/lingpy_input.tsv"
    rows = []
    idx = 1
    for _, row in df.iterrows():
        toks = row["ipa_tokens"]
        if not toks or (isinstance(toks, float) and np.isnan(toks)):
            continue
        if isinstance(toks, list) and len(toks) == 0:
            continue
        tok = toks if isinstance(toks, list) else toks.split()
        tok_str = " ".join(tok)
        rows.append({
            "ID": str(idx),
            "DOCULECT": row["language"],
            "CONCEPT":  row["gloss"],
            "TOKENS":   tok_str,
            "IPA":      row["form"],
        })
        idx += 1

    tsv_df = pd.DataFrame(rows)
    tsv_df.to_csv(tsv_path, sep="\t", index=False)
    print(f"[wordlist] {idx-1} entries written to {tsv_path}")
    return tsv_path


def run_lexstat(tsv_path: str, threshold: float = LEXSTAT_THRESHOLD):
    print(f"[LexStat] Loading wordlist and running SCA-based clustering ...")
    lex = LexStat(tsv_path)
    # get_scorer with runs>0 triggers the broken permutation code in 2.6.13.
    # Using runs=0 skips permutation testing and falls back to the SCA
    # sound-class model, which is valid for a closely related family like Turkic.
    lex.get_scorer(runs=0)
    print(f"[LexStat] Clustering at threshold={threshold} ...")
    lex.cluster(method="lexstat", threshold=threshold, ref="cogid")
    cog_ids = set(lex[i, "cogid"] for i in lex)
    print(f"[LexStat] {len(cog_ids)} cognate sets identified")
    return lex

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — SCA alignment and correspondence extraction
# ─────────────────────────────────────────────────────────────────────────────

def run_sca_alignment(lex):
    print("[SCA] Running multiple alignment ...")
    alm = Alignments(lex, ref="cogid")
    alm.align(method="progressive")
    print("[SCA] Done.")
    return alm


def extract_correspondence_pairs(alm) -> pd.DataFrame:
    records = []
    for cogset_id, msa_block in alm.msa["cogid"].items():
        taxa   = msa_block["taxa"]
        aligns = msa_block["alignment"]
        concept = msa_block.get("concept", "?")
        if len(taxa) < 2:
            continue
        n_cols = max(len(a) for a in aligns)
        for col_idx in range(n_cols):
            col = {
                lang: (alignment[col_idx] if col_idx < len(alignment) else "-")
                for lang, alignment in zip(taxa, aligns)
            }
            lang_list = list(col.keys())
            for i in range(len(lang_list)):
                for j in range(i + 1, len(lang_list)):
                    la, lb = lang_list[i], lang_list[j]
                    pa, pb = col[la], col[lb]
                    if pa == "-" and pb == "-":
                        continue
                    records.append({
                        "lang_a":    la, "lang_b":    lb,
                        "phoneme_a": pa, "phoneme_b": pb,
                        "gloss":     concept,
                        "cogset_id": cogset_id,
                        "position":  col_idx,
                    })
    df = pd.DataFrame(records)
    print(f"[correspondences] {len(df)} pairwise phoneme tokens extracted")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FALLBACK: Manual initial-phoneme alignment (no LingPy)
# ─────────────────────────────────────────────────────────────────────────────

def manual_correspondence_from_phase1(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts INITIAL PHONEME correspondences by treating co-glossed Swadesh
    entries as cognates. Valid approximation for the Turkic core vocabulary
    given shallow divergence; do not use this for phase 3 scoring.
    """
    records = []
    for gloss in df["gloss"].unique():
        sub = df[df["gloss"] == gloss]
        if len(sub) < 2:
            continue
        initials = {}
        for _, row in sub.iterrows():
            toks = row["ipa_tokens"]
            if isinstance(toks, list) and toks:
                initials[row["language"]] = toks[0]
            elif isinstance(toks, str) and toks.strip():
                first = toks.strip().split()[0]
                if first:
                    initials[row["language"]] = first
        langs = list(initials.keys())
        for i in range(len(langs)):
            for j in range(i + 1, len(langs)):
                la, lb = langs[i], langs[j]
                records.append({
                    "lang_a":    la,       "lang_b":    lb,
                    "phoneme_a": initials[la], "phoneme_b": initials[lb],
                    "gloss":     gloss,
                    "cogset_id": f"sw_{gloss}",
                    "position":  0,
                })
    out = pd.DataFrame(records)
    print(f"[manual fallback] {len(out)} initial-phoneme pairs  "
          "** install LingPy for full alignment **")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Probabilistic model
# ─────────────────────────────────────────────────────────────────────────────

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
    return prob_model


def save_prob_model(prob_model: dict, path: str = "output/prob_model.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prob_model, f, ensure_ascii=False, indent=2)
    print(f"[prob model] saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Sound law validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_sound_laws(corr_df: pd.DataFrame, prob_model: dict) -> str:
    lines = [
        "=" * 70,
        "SOUND LAW VALIDATION",
        "=" * 70,
        "",
        "[PASS] law is top-ranked reflex in learned model",
        "[WEAK] law present but outranked by another reflex",
        "[MISS] law absent from extracted correspondences",
        "",
    ]

    for proto, lang, expected in PROTO_TURKIC_LAWS:
        candidates = []
        for key, dist in prob_model.items():
            parts = key.split("|")
            if len(parts) != 3:
                continue
            la, lb, pa = parts
            if pa == proto and lb == lang:
                candidates.append((la, lb, pa, dist))
            if pa == proto and la == lang:
                candidates.append((la, lb, pa, dist))

        if not candidates:
            lines.append(
                f"[MISS] *{proto} > {expected} in {lang}: "
                f"no '{proto}' contexts found for {lang} in data"
            )
            continue

        for la, lb, pa, dist in candidates:
            ranked = sorted(dist.items(), key=lambda x: -x[1])
            top_ph, top_p = ranked[0]
            partner = lb if la == lang else la

            exp_rank = next(
                (r for r, (ph, _) in enumerate(ranked, 1) if ph == expected),
                None
            )
            exp_prob = dict(ranked).get(expected)

            if top_ph == expected:
                lines.append(
                    f"[PASS] *{proto} > {expected} in {lang} "
                    f"(vs {partner}): P={top_p:.3f}, rank 1/{len(ranked)}"
                )
            elif exp_rank:
                lines.append(
                    f"[WEAK] *{proto} > {expected} in {lang} "
                    f"(vs {partner}): rank {exp_rank}/{len(ranked)}, "
                    f"P={exp_prob:.3f}; top='{top_ph}' P={top_p:.3f}"
                )
            else:
                lines.append(
                    f"[MISS] *{proto} > {expected} in {lang} "
                    f"(vs {partner}): '{expected}' absent; "
                    f"top='{top_ph}' P={top_p:.3f}"
                )

    return "\n".join(lines)


def check_vowel_harmony(df: pd.DataFrame) -> str:
    lines = [
        f"{'Language':<14} {'Total':>6} {'Harmonic':>9} "
        f"{'Violators':>10} {'Viol%':>7}",
        "-" * 50,
    ]
    flags = []
    expected_high = {"Chuvash", "Uzbek"}

    for lang in LANGUAGES:
        sub = df[df["language"] == lang]
        total = violations = harmonic = 0
        for _, row in sub.iterrows():
            toks = row["ipa_tokens"]
            if not isinstance(toks, list) or not toks:
                continue
            f = sum(1 for t in toks if t in FRONT_VOWELS)
            b = sum(1 for t in toks if t in BACK_VOWELS)
            total += 1
            if f + b < 2:
                continue
            if f > 0 and b > 0:
                violations += 1
            else:
                harmonic += 1

        vp = 100 * violations / max(total, 1)
        note = ""
        if vp > 35 and lang not in expected_high:
            note = "  <-- UNEXPECTED"
            flags.append(lang)
        lines.append(
            f"{lang:<14} {total:>6} {harmonic:>9} {violations:>10} {vp:>6.1f}%{note}"
        )

    lines.append("")
    if flags:
        lines.append(f"[NOTE] Unexpected violations: {', '.join(flags)} → flag for Phase 3")
    else:
        lines.append("[NOTE] Violation rates within expected ranges.")
        lines.append("       Chuvash/Uzbek elevation is expected (Persian/Arabic loans; "
                     "harmony erosion in Bulgar).")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Correspondence table and flagging
# ─────────────────────────────────────────────────────────────────────────────

def build_correspondence_table(corr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (la, lb, pa), grp in corr_df.groupby(["lang_a", "lang_b", "phoneme_a"]):
        counts = grp["phoneme_b"].value_counts()
        total  = counts.sum()
        probs  = counts / total
        entropy = float(-np.sum(probs * np.log2(probs + 1e-12)))
        rows.append({
            "lang_a":       la,
            "lang_b":       lb,
            "phoneme_a":    pa,
            "top_reflex":   counts.index[0],
            "top_prob":     round(float(probs.iloc[0]), 3),
            "total_count":  int(total),
            "entropy":      round(entropy, 3),
            "all_reflexes": ", ".join(f"{ph}:{n}" for ph, n in counts.items()),
        })
    return pd.DataFrame(rows).sort_values(
        ["lang_a", "lang_b", "phoneme_a"]
    ).reset_index(drop=True)


def flag_unexpected_patterns(tbl: pd.DataFrame) -> pd.DataFrame:
    # entropy > 1.5 bits AND ≥ 3 examples: genuinely irregular
    return tbl[
        (tbl["entropy"] > 1.5) & (tbl["total_count"] >= 3)
    ].copy().assign(
        flag_reason=lambda r: "H=" + r["entropy"].astype(str) +
                              " bits — irregular, review for substrate"
    )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def run_phase2(csv_path: str = "output/turkic_merged_phase1.csv"):
    print("\n" + "=" * 70)
    print("PHASE 2: COGNATE DETECTION & SOUND CORRESPONDENCE EXTRACTION")
    print("=" * 70 + "\n")

    df = load_phase1(csv_path)

    # ── Cognate detection ──────────────────────────────────────────────────
    if LINGPY_AVAILABLE:
        tsv_path = build_lingpy_wordlist(df)
        try:
            lex = run_lexstat(tsv_path)
            lex.output("tsv", filename="output/cognate_sets",
                       ignore="all", prettify=False)
            alm = run_sca_alignment(lex)
            alm.output("html", filename="output/alignments")
            corr_df = extract_correspondence_pairs(alm)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[WARNING] LexStat/SCA failed ({e}) — using manual fallback")
    else:
        corr_df = manual_correspondence_from_phase1(df)

    # ── Correspondence table ───────────────────────────────────────────────
    corr_tbl = build_correspondence_table(corr_df)
    corr_tbl.to_csv("output/correspondence_table.csv", index=False, encoding="utf-8")

    # ── Probabilistic model ───────────────────────────────────────────────
    prob_model = build_prob_model(corr_df)
    save_prob_model(prob_model)

    # ── Validation ────────────────────────────────────────────────────────
    val_report  = validate_sound_laws(corr_df, prob_model)
    harm_report = check_vowel_harmony(df)

    # ── Flagging ──────────────────────────────────────────────────────────
    flagged = flag_unexpected_patterns(corr_tbl)
    flagged.to_csv("output/flagged_correspondences.csv", index=False, encoding="utf-8")

    # ── Report ────────────────────────────────────────────────────────────
    top30 = corr_tbl.nlargest(30, "total_count")
    top30_lines = [
        f"{'lang_a':<12} {'lang_b':<12} {'phon_a':>7} {'top':>7} "
        f"{'P':>6} {'n':>5} {'H':>7}",
        "-" * 62,
    ]
    for _, r in top30.iterrows():
        top30_lines.append(
            f"{r['lang_a']:<12} {r['lang_b']:<12} {r['phoneme_a']:>7} "
            f"{r['top_reflex']:>7} {r['top_prob']:>6.3f} "
            f"{r['total_count']:>5} {r['entropy']:>7.3f}"
        )

    report = "\n".join([
        "=" * 70, "PHASE 2 REPORT", "=" * 70, "",
        f"Correspondence pairs: {len(corr_df)}",
        f"Unique contexts:      {len(corr_tbl)}",
        f"High-entropy flags:   {len(flagged)}",
        f"LexStat threshold:    {LEXSTAT_THRESHOLD}",
        f"LingPy used:          {LINGPY_AVAILABLE}",
        "",
        "FILES:", "  output/cognate_sets.tsv",
        "  output/alignments.html",
        "  output/correspondence_table.csv",
        "  output/prob_model.json",
        "  output/flagged_correspondences.csv",
        "", "-" * 70, "TOP 30 CORRESPONDENCES BY FREQUENCY", "-" * 70, "",
        *top30_lines,
        "", "", val_report,
        "", "-" * 70, "VOWEL HARMONY", "-" * 70, "", harm_report,
    ])

    if len(flagged):
        flag_block = flagged[
            ["lang_a","lang_b","phoneme_a","top_reflex",
             "top_prob","total_count","entropy","all_reflexes"]
        ].to_string(index=False)
        report += (
            f"\n\n{'-'*70}\n"
            "HIGH-ENTROPY FLAGS (Phase 3 substrate candidates)\n"
            f"{'-'*70}\n\n{flag_block}"
        )

    with open("output/phase2_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print("\n[DONE] output/phase2_report.txt written")


if __name__ == "__main__":
    import sys
    run_phase2(sys.argv[1] if len(sys.argv) > 1 else "output/turkic_merged_phase1.csv")