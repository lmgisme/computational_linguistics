"""
Phase 2 supplementary: Compare LexStat cognate sets against Savelyev & Robbeets (2020)
manual expert cognate judgments.

Inputs:
    output/cognate_sets.tsv                                    -- LexStat output
    phase1_ingestion/output/lexibank_cache/savelyevturkic/cognates.csv  -- expert judgments

Output:
    output/cognate_comparison_report.txt

The Savelyev dataset covers a much wider set of Turkic languages than our 9.
We restrict comparison to the 9 languages in our study using name-matching.
"""

import pandas as pd
from pathlib import Path

# ── Language name mapping ──────────────────────────────────────────────────
# Savelyev doculect names -> our Phase 2 DOCULECT names
SAVELYEV_TO_OURS = {
    "Azeri":    "Azerbaijani",
    "Kazakh":   "Kazakh",
    "Kirghiz":  "Kyrgyz",
    "Turkish":  "Turkish",
    "Turkmen":  "Turkmen",
    "Uyghur":   "Uyghur",   # Savelyev may use "Uyghur" or "EasternUyghur"
    "Uzbek":    "Uzbek",
    "Yakut":    "Yakut",
    "Chuvash":  "Chuvash",
    # Alternate spellings that appear in the dataset
    "EasternUyghur": "Uyghur",
    "Uighur":        "Uyghur",
}

OUR_LANGS = set(SAVELYEV_TO_OURS.values())

# ── Load data ─────────────────────────────────────────────────────────────

def load_lexstat(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    # Keep only columns we need
    df = df[["DOCULECT", "CONCEPT", "IPA", "COGID"]].copy()
    df.columns = ["language", "concept", "form", "lexstat_cogid"]
    df = df[df["language"].isin(OUR_LANGS)]
    print(f"[LexStat]   {len(df)} entries across {df['language'].nunique()} languages")
    return df


def load_savelyev(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Extract doculect name from Form_ID: "Azeri-1_fingernailn-1" -> "Azeri"
    df["savelyev_lang"] = df["Form_ID"].str.extract(r"^([A-Za-z]+)-\d+_")
    df["our_lang"] = df["savelyev_lang"].map(SAVELYEV_TO_OURS)
    df = df[df["our_lang"].notna()].copy()

    # Extract concept from Form_ID: "Azeri-1_fingernailn-1" -> "fingernail"
    # The concept is the part after the first underscore, before the trailing -N
    df["concept_raw"] = df["Form_ID"].str.extract(r"_([a-z]+)n?-\d+$")

    # Keep relevant columns
    df = df[["our_lang", "concept_raw", "Form", "Cognateset_ID", "Root"]].copy()
    df.columns = ["language", "concept_raw", "form", "savelyev_cogid", "root"]

    print(f"[Savelyev]  {len(df)} entries across {df['language'].nunique()} languages "
          f"after filtering to our 9")
    return df


# ── Concept alignment ──────────────────────────────────────────────────────
# Our 40-item Swadesh concepts -> substrings likely to appear in Savelyev concept_raw
CONCEPT_MAP = {
    "I":        ["fingernail"],   # placeholder — 'I' (pronoun) is tricky, skip
    "all":      ["all"],
    "bark":     ["bark"],
    "big":      ["big", "large"],
    "bird":     ["bird"],
    "blood":    ["blood"],
    "bone":     ["bone"],
    "dog":      ["dog"],
    "ear":      ["ear"],
    "egg":      ["egg"],
    "eye":      ["eye"],
    "feather":  ["feather", "plume"],
    "fish":     ["fish"],
    "flesh":    ["flesh", "meat"],
    "grease":   ["grease", "fat"],
    "hair":     ["hair"],
    "head":     ["head"],
    "horn":     ["horn"],
    "leaf":     ["leaf"],
    "long":     ["long"],
    "louse":    ["louse"],
    "man":      ["man"],
    "many":     ["many"],
    "not":      ["not", "negat"],
    "one":      ["one"],
    "person":   ["person"],
    "root":     ["root"],
    "seed":     ["seed"],
    "skin":     ["skin"],
    "small":    ["small", "little"],
    "tail":     ["tail"],
    "that":     ["that"],
    "this":     ["this"],
    "tree":     ["tree"],
    "two":      ["two"],
    "we":       ["we"],
    "what":     ["what"],
    "who":      ["who"],
    "woman":    ["woman"],
    "you_2sg":  ["you", "thou"],
}


def match_concepts(savelyev_df: pd.DataFrame) -> pd.DataFrame:
    """Map Savelyev concept_raw strings to our 40-item Swadesh concepts."""
    rows = []
    for our_concept, keywords in CONCEPT_MAP.items():
        for kw in keywords:
            matches = savelyev_df[
                savelyev_df["concept_raw"].str.contains(kw, case=False, na=False)
            ].copy()
            matches["our_concept"] = our_concept
            rows.append(matches)
    return pd.concat(rows, ignore_index=True).drop_duplicates(
        subset=["language", "our_concept", "savelyev_cogid"]
    )


# ── Comparison logic ───────────────────────────────────────────────────────

def compare_cognate_sets(lexstat_df: pd.DataFrame,
                         savelyev_matched: pd.DataFrame) -> dict:
    """
    For each concept, compare LexStat groupings against Savelyev groupings.

    We ask: for each pair of languages (A, B) that share a concept entry,
    do LexStat and Savelyev agree on whether their forms are cognate?

    Agreement types:
        AGREE_COGNATE    -- both say cognate
        AGREE_NON        -- both say non-cognate
        DISAGREE_LS_ONLY -- LexStat lumps them, Savelyev splits
        DISAGREE_SV_ONLY -- Savelyev lumps them, LexStat splits
    """
    results = []

    concepts = set(lexstat_df["concept"].unique()) & set(
        savelyev_matched["our_concept"].unique()
    )

    for concept in sorted(concepts):
        ls_sub = lexstat_df[lexstat_df["concept"] == concept]
        sv_sub = savelyev_matched[savelyev_matched["our_concept"] == concept]

        langs_ls = set(ls_sub["language"].unique())
        langs_sv = set(sv_sub["language"].unique())
        shared_langs = langs_ls & langs_sv

        if len(shared_langs) < 2:
            continue

        # Build lookup: language -> cogid
        ls_cog = dict(zip(ls_sub["language"], ls_sub["lexstat_cogid"]))
        sv_cog = dict(zip(sv_sub["language"], sv_sub["savelyev_cogid"]))

        lang_list = sorted(shared_langs)
        for i in range(len(lang_list)):
            for j in range(i + 1, len(lang_list)):
                la, lb = lang_list[i], lang_list[j]
                ls_same = (ls_cog.get(la) == ls_cog.get(lb))
                sv_same = (sv_cog.get(la) == sv_cog.get(lb))

                if ls_same and sv_same:
                    verdict = "AGREE_COGNATE"
                elif not ls_same and not sv_same:
                    verdict = "AGREE_NON"
                elif ls_same and not sv_same:
                    verdict = "DISAGREE_LS_ONLY"
                else:
                    verdict = "DISAGREE_SV_ONLY"

                results.append({
                    "concept":  concept,
                    "lang_a":   la,
                    "lang_b":   lb,
                    "ls_cogid_a":  ls_cog.get(la),
                    "ls_cogid_b":  ls_cog.get(lb),
                    "sv_cogid_a":  sv_cog.get(la),
                    "sv_cogid_b":  sv_cog.get(lb),
                    "verdict":  verdict,
                })

    return pd.DataFrame(results)


# ── Report ────────────────────────────────────────────────────────────────

def write_report(cmp_df: pd.DataFrame):
    total = len(cmp_df)
    counts = cmp_df["verdict"].value_counts()

    agree_cog  = counts.get("AGREE_COGNATE",    0)
    agree_non  = counts.get("AGREE_NON",         0)
    dis_ls     = counts.get("DISAGREE_LS_ONLY",  0)
    dis_sv     = counts.get("DISAGREE_SV_ONLY",  0)

    agree_total = agree_cog + agree_non
    disagree_total = dis_ls + dis_sv
    agreement_rate = agree_total / max(total, 1)

    lines = [
        "=" * 70,
        "COGNATE SET COMPARISON: LexStat vs Savelyev & Robbeets (2020)",
        "=" * 70,
        "",
        f"Total language-pair-concept comparisons: {total}",
        f"Agreement:    {agree_total} ({agreement_rate:.1%})",
        f"  Agree cognate:     {agree_cog}",
        f"  Agree non-cognate: {agree_non}",
        f"Disagreement: {disagree_total} ({1-agreement_rate:.1%})",
        f"  LexStat lumps / Savelyev splits: {dis_ls}",
        f"  Savelyev lumps / LexStat splits: {dis_sv}",
        "",
        "-" * 70,
        "INTERPRETATION",
        "-" * 70,
        "",
        "DISAGREE_LS_ONLY (LexStat too permissive): LexStat groups forms that",
        "Savelyev expert judgment treats as non-cognate. Threshold may be too",
        "high, or these are genuine distant cognates the paper conservatively",
        "excluded. These are Phase 3 false-negative risks.",
        "",
        "DISAGREE_SV_ONLY (LexStat too conservative): Savelyev groups forms",
        "that LexStat splits. Threshold may be too low for highly divergent",
        "members (Chuvash, Yakut). Consider raising threshold to 0.55 for",
        "a second run and comparing. These are Phase 3 false-positive risks.",
        "",
    ]

    # Per-concept breakdown
    lines += ["-" * 70, "PER-CONCEPT AGREEMENT", "-" * 70, ""]
    concept_summary = cmp_df.groupby("concept")["verdict"].value_counts().unstack(fill_value=0)
    for col in ["AGREE_COGNATE", "AGREE_NON", "DISAGREE_LS_ONLY", "DISAGREE_SV_ONLY"]:
        if col not in concept_summary.columns:
            concept_summary[col] = 0
    concept_summary["total"] = concept_summary.sum(axis=1)
    concept_summary["agree_rate"] = (
        (concept_summary["AGREE_COGNATE"] + concept_summary["AGREE_NON"])
        / concept_summary["total"]
    ).round(2)
    concept_summary = concept_summary.sort_values("agree_rate")

    lines.append(f"{'concept':<14} {'agree%':>7} {'agr_cog':>8} {'agr_non':>8} "
                 f"{'ls_only':>8} {'sv_only':>8}")
    lines.append("-" * 56)
    for concept, row in concept_summary.iterrows():
        lines.append(
            f"{concept:<14} {row['agree_rate']:>7.0%} "
            f"{row['AGREE_COGNATE']:>8} {row['AGREE_NON']:>8} "
            f"{row['DISAGREE_LS_ONLY']:>8} {row['DISAGREE_SV_ONLY']:>8}"
        )

    # Flag the worst disagreements for Phase 3
    lines += ["", "-" * 70,
              "HIGH-PRIORITY DISAGREEMENTS (Phase 3 attention list)",
              "-" * 70, ""]

    bad = cmp_df[cmp_df["verdict"].isin(["DISAGREE_LS_ONLY", "DISAGREE_SV_ONLY"])]
    bad = bad.sort_values(["verdict", "concept"])
    if len(bad) == 0:
        lines.append("None — full agreement.")
    else:
        lines.append(bad[["concept", "lang_a", "lang_b", "verdict"]].to_string(index=False))

    report = "\n".join(lines)
    out_path = "output/cognate_comparison_report.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + report)
    print(f"\n[DONE] Report saved -> {out_path}")
    cmp_df.to_csv("output/cognate_comparison_detail.csv", index=False, encoding="utf-8")
    print("[DONE] Detail saved  -> output/cognate_comparison_detail.csv")


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    lexstat_path   = "output/cognate_sets.tsv"
    savelyev_path  = "phase1_ingestion/output/lexibank_cache/savelyevturkic/cognates.csv"

    print("\n" + "=" * 70)
    print("COGNATE COMPARISON: LexStat vs Savelyev Expert Judgments")
    print("=" * 70 + "\n")

    ls_df  = load_lexstat(lexstat_path)
    sv_df  = load_savelyev(savelyev_path)
    sv_matched = match_concepts(sv_df)

    print(f"[Concept match] {sv_matched['our_concept'].nunique()} concepts matched "
          f"in Savelyev data")

    cmp_df = compare_cognate_sets(ls_df, sv_matched)
    print(f"[Comparison] {len(cmp_df)} language-pair-concept pairs evaluated")

    write_report(cmp_df)


if __name__ == "__main__":
    run()
