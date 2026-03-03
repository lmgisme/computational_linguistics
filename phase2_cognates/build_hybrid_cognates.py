"""
build_hybrid_cognates.py
========================
Builds a hybrid cognate table for Phase 3 anomaly detection.

Priority:
    1. Savelyev & Robbeets (2020) expert cognate judgments — ground truth
    2. LexStat Phase 2 output — fills gaps
    3. 'none' — unassigned singleton, automatic anomaly candidate in Phase 3

Output:
    output/hybrid_cognates.csv
    output/hybrid_summary.txt
"""

import pandas as pd
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────────
PHASE1_CSV  = "phase1_ingestion/output/turkic_merged_phase1.csv"
COGNATE_TSV = "output/cognate_sets.tsv"
SV_COGNATES = "phase1_ingestion/output/lexibank_cache/savelyevturkic/cognates.csv"
OUT_CSV     = "output/hybrid_cognates.csv"
OUT_SUMMARY = "output/hybrid_summary.txt"

LEXSTAT_OFFSET = 100_000  # prevents ID collision between Savelyev and LexStat sets

# ── Language mapping ──────────────────────────────────────────────────────
SV_TO_OURS = {
    "Azeri":    "Azerbaijani",
    "Chuvash":  "Chuvash",
    "Kazakh":   "Kazakh",
    "Kirghiz":  "Kyrgyz",
    "Turkish":  "Turkish",
    "Turkmen":  "Turkmen",
    "Uighur":   "Uyghur",
    "Uzbek":    "Uzbek",
    "Yakut":    "Yakut",
}
OUR_LANGS = set(SV_TO_OURS.values())

# ── Swadesh gloss -> Savelyev param ID (the PARAM segment in Form_ID) ─────
# Form_ID format confirmed: "Doculect-PARAM-formnum"  e.g. "Azeri-2_1plpronoun-1"
# PARAM = "2_1plpronoun"   extracted by pattern_B: r'^[A-Za-z]+-(\d+_[^-]+)-'
GLOSS_TO_SV_PARAM = {
    "I":        "3_1sg",
    "we":       "2_1plpronoun",
    "you_2sg":  "5_2sgpronoun",
    "bark":     "15_barkn",
    "big":      "17_big",
    "bird":     "18_birdn",
    "blood":    "22_bloodn",
    "bone":     "24_bonen",
    "dog":      "53_dogn",
    "ear":      "57_earn",
    "egg":      "60_eggn",
    "eye":      "61_eyen",
    "feather":  "67_feathern",
    "fish":     "73_fishn",
    "flesh":    "130_meatn",    # Concepticon MEAT (634)
    "grease":   "64_fatn",      # Concepticon FAT (323)
    "hair":     "94_hairn",
    "head":     "96_headn",
    "horn":     "104_hornn",
    "leaf":     "117_leafn",
    "long":     "124_long",
    "louse":    "127_lousen",
    "man":      "128_mann",
    "many":     "129_many",
    "not":      "145_not",
    "one":      "147_one",
    "person":   "150_personn",
    "root":     "164_rootn",
    "seed":     "174_seedn",
    "skin":     "182_skinhiden",
    "small":    "185_small",
    "tail":     "207_tailn",
    "that":     "211_that",
    "this":     "218_this",
    "tree":     "227_treen",
    "two":      "230_two",
    "what":     "238_what",
    "who":      "243_who",
    "woman":    "248_womann",
    # 'all' has no Savelyev equivalent — LexStat only
}
PARAM_TO_GLOSS = {v: k for k, v in GLOSS_TO_SV_PARAM.items()}


# ── Loaders ───────────────────────────────────────────────────────────────
def load_phase1(path):
    df = pd.read_csv(path)
    df = df[["language", "gloss", "form", "ipa_tokens"]].copy()
    df = df[df["language"].isin(OUR_LANGS)]
    print(f"[Phase1]   {len(df)} rows | {df['language'].nunique()} langs "
          f"| {df['gloss'].nunique()} glosses")
    return df


def load_lexstat(path):
    df = pd.read_csv(path, sep="\t")
    df = df[["DOCULECT", "CONCEPT", "IPA", "COGID"]].copy()
    df.columns = ["language", "gloss", "form_ls", "lexstat_cogid"]
    df = df[df["language"].isin(OUR_LANGS)]
    df["lexstat_cogid_offset"] = df["lexstat_cogid"] + LEXSTAT_OFFSET
    print(f"[LexStat]  {len(df)} rows | {df['lexstat_cogid'].nunique()} cognate sets")
    return df


def load_savelyev(path):
    sv = pd.read_csv(path)

    # Form_ID: "Doculect-PARAM-formnum"  (3 dash-separated parts)
    # Doculect extraction: everything before the first "-N_"
    sv["doculect"] = sv["Form_ID"].str.extract(r"^([A-Za-z]+)-\d+_")

    # PARAM extraction: confirmed working pattern from diagnostic
    # r'^[A-Za-z]+-(\d+_[^-]+)-'  captures e.g. "2_1plpronoun", "1_fingernailn"
    sv["sv_param"] = sv["Form_ID"].str.extract(r"^[A-Za-z]+-(\d+_[^-]+)-")

    sv["our_lang"] = sv["doculect"].map(SV_TO_OURS)
    sv = sv[sv["our_lang"].notna() & sv["sv_param"].notna()].copy()

    sv["savelyev_cogid"] = pd.to_numeric(sv["Cognateset_ID"], errors="coerce")
    sv = sv.dropna(subset=["savelyev_cogid"])
    sv["savelyev_cogid"] = sv["savelyev_cogid"].astype(int)

    sv["gloss"] = sv["sv_param"].map(PARAM_TO_GLOSS)
    sv = sv[sv["gloss"].notna()].copy()

    sv = sv[["our_lang", "gloss", "Form", "savelyev_cogid"]].copy()
    sv.columns = ["language", "gloss", "form_sv", "savelyev_cogid"]

    print(f"[Savelyev] {len(sv)} rows matched | "
          f"{sv['gloss'].nunique()} glosses | "
          f"{sv['savelyev_cogid'].nunique()} cognate sets")

    present = sorted(sv["gloss"].unique())
    missing = sorted(set(GLOSS_TO_SV_PARAM.keys()) - set(present))
    print(f"           Glosses covered ({len(present)}): {present}")
    if missing:
        print(f"           Glosses missing  ({len(missing)}): {missing}")
    return sv


# ── Merge ─────────────────────────────────────────────────────────────────
def build_hybrid(phase1, lexstat, savelyev):
    merged = phase1.copy()

    ls_key = lexstat[["language", "gloss", "lexstat_cogid", "lexstat_cogid_offset"]]
    merged = merged.merge(ls_key, on=["language", "gloss"], how="left")

    # One Savelyev row per (language, gloss) — take first if doublets exist
    sv_key = (savelyev[["language", "gloss", "savelyev_cogid"]]
              .drop_duplicates(subset=["language", "gloss"], keep="first"))
    merged = merged.merge(sv_key, on=["language", "gloss"], how="left")

    def assign(row):
        if pd.notna(row["savelyev_cogid"]):
            return int(row["savelyev_cogid"]), "savelyev"
        elif pd.notna(row["lexstat_cogid"]):
            return int(row["lexstat_cogid_offset"]), "lexstat"
        return np.nan, "none"

    res = merged.apply(assign, axis=1, result_type="expand")
    merged["cogid"] = res[0]
    merged["cogid_source"] = res[1]

    # Conflict flag: concept has both sources and they disagreed in comparison
    try:
        cmp = pd.read_csv("output/cognate_comparison_detail.csv")
        conflict_glosses = set(
            cmp[cmp["verdict"].isin(["DISAGREE_LS_ONLY", "DISAGREE_SV_ONLY"])]["concept"]
        )
        merged["conflict"] = (
            merged["gloss"].isin(conflict_glosses)
            & merged["savelyev_cogid"].notna()
            & merged["lexstat_cogid"].notna()
        )
    except FileNotFoundError:
        merged["conflict"] = False

    merged = merged.drop(columns=["lexstat_cogid_offset"], errors="ignore")
    return merged[[
        "language", "gloss", "form", "ipa_tokens",
        "cogid", "cogid_source",
        "savelyev_cogid", "lexstat_cogid", "conflict"
    ]]


# ── Summary report ────────────────────────────────────────────────────────
def write_summary(hybrid):
    lines = [
        "=" * 70,
        "HYBRID COGNATE TABLE — PHASE 2 COMPLETE",
        "=" * 70,
        "",
        f"Total entries : {len(hybrid)}",
        f"Languages     : {hybrid['language'].nunique()}",
        f"Glosses       : {hybrid['gloss'].nunique()}",
        "",
        "SOURCE BREAKDOWN",
        "-" * 40,
    ]
    for src, count in hybrid["cogid_source"].value_counts().items():
        lines.append(f"  {src:<12}: {count:>4}  ({count/len(hybrid):.1%})")

    lines += ["", f"{'GLOSS':<14} {'SV':>4} {'LS':>4} {'CONFLICT':>9}  {'SETS':>5}  SOURCE",
              "-" * 55]
    for gloss in sorted(hybrid["gloss"].unique()):
        g = hybrid[hybrid["gloss"] == gloss]
        sv   = int(g["savelyev_cogid"].notna().sum())
        ls   = int(g["lexstat_cogid"].notna().sum())
        cf   = int(g["conflict"].sum())
        sets = int(g.dropna(subset=["cogid"])["cogid"].nunique())
        src  = "savelyev" if sv > 0 else ("lexstat" if ls > 0 else "none")
        lines.append(f"  {gloss:<14} {sv:>4} {ls:>4} {cf:>9}  {sets:>5}  {src}")

    lines += [
        "",
        "PHASE 3 NOTES",
        "-" * 40,
        "  savelyev  -> expert ground truth (Savelyev & Robbeets 2020)",
        "  lexstat   -> automated SCA clustering, lower confidence",
        "  none      -> unassigned; automatic anomaly candidate in Phase 3",
        "  conflict  -> both sources present and disagreed; Savelyev wins",
        "",
        "Concepts where LexStat is sole source should be weighted lower",
        "in Phase 3 regularity scoring.",
    ]

    report = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    print(f"\n[DONE] {OUT_SUMMARY}")


# ── Main ──────────────────────────────────────────────────────────────────
def run():
    print("\n" + "=" * 70)
    print("BUILDING HYBRID COGNATE TABLE")
    print("=" * 70 + "\n")

    phase1   = load_phase1(PHASE1_CSV)
    lexstat  = load_lexstat(COGNATE_TSV)
    savelyev = load_savelyev(SV_COGNATES)
    hybrid   = build_hybrid(phase1, lexstat, savelyev)

    hybrid.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\n[DONE] {OUT_CSV}  ({len(hybrid)} rows)")
    write_summary(hybrid)


if __name__ == "__main__":
    run()
