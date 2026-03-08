"""
Phase 7 — Pseudo-Substrate Sensitivity Test (Option A)

Purpose: Estimate the pipeline's sensitivity to foreign (non-Turkic) lexical
material by checking what fraction of independently known loans the anomaly
detector flagged. This provides a credibility bound on the null substrate
result: if the pipeline catches X% of known foreign items, then the claim
"no substrate detected" is credible only to the extent that a substrate
signal would be at least as detectable as a known-source loan.

Three test components:
  1. Curated known-loan inventory (manually compiled from all phases)
  2. Automated phonological loan screen (structural markers of Arabic/Persian)
  3. Cross-reference: which known loans were flagged vs missed?

Run from project root:
  venv311\\Scripts\\python.exe phase7_sensitivity\\sensitivity_test.py

Output: output/sensitivity_report.txt, output/sensitivity_scores.csv
"""

import csv
import os

# ── paths ──────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

REG_FILE = os.path.join(OUTPUT, "regularity_expanded.csv")
THRESH_FILE = os.path.join(OUTPUT, "infra6a_thresholds.csv")
CANDIDATES_FILE = os.path.join(OUTPUT, "infra6a_candidates.csv")

OUT_REPORT = os.path.join(OUTPUT, "sensitivity_report.txt")
OUT_CSV = os.path.join(OUTPUT, "sensitivity_scores.csv")

# ── constants ──────────────────────────────────────────────────────────────
FAMILY_WIDE_THRESHOLD = -1.1851  # Phase 5 Task 3

LANG_TO_BRANCH = {
    "Turkish": "A_oghuz",
    "Azerbaijani": "A_oghuz",
    "Turkmen": "A_oghuz",
    "Kazakh": "A_kipchak",
    "Uzbek": "A_kipchak",
    "Kyrgyz": "A_kipchak",
    "Uyghur": "A_kipchak",
    "Yakut": "B_yakut",
    "Chuvash": "C_chuvash",
}

# ── KNOWN LOAN INVENTORY ──────────────────────────────────────────────────
# (language, gloss_pattern, source_language, evidence, category)
#   A = independently identified (NOT found by pipeline first) — gold standard
#   B = pipeline-flagged, then attributed during triage — partially circular

# Alternate glosses: some concepts have different labels in different sources
GLOSS_ALTERNATES = {
    "flesh": ["flesh", "meat"],
    "tree": ["tree"],
    "woman": ["woman"],
    "egg": ["egg"],
    "clever": ["clever", "smart", "intelligent"],
    "flower": ["flower"],
    "fat": ["fat", "grease"],
}

KNOWN_LOANS = [
    # ── Category A: Independently identified (Infra 6B) ──
    ("Uzbek", "woman", "Persian", "xatun > xatin; Infra 6B", "A"),
    ("Uzbek", "tree", "Persian", "daraxt; Infra 6B", "A"),
    ("Uzbek", "egg", "Persian", "toxm > tuxum; Infra 6B", "A"),
    ("Uzbek", "flesh", "Persian", "gusht > gosht; Infra 6B", "A"),

    # ── Category B: Pipeline-flagged, attributed during triage/resolution ──
    ("Azerbaijani", "and", "Arabic/Persian", "wa/va; Phase 6 triage", "B"),
    ("Turkish", "and", "Arabic/Persian", "wa/va; Phase 6 triage", "B"),
    ("Azerbaijani", "at that time", "Persian", "vaqt; Phase 6 triage", "B"),
    ("Azerbaijani", "border", "Arabic", "hudud; Phase 6 triage", "B"),
    ("Azerbaijani", "boss", "French", "chef via Ottoman; Phase 6 triage", "B"),
    ("Turkish", "boss", "French", "chef via Ottoman; Phase 6 triage", "B"),
    ("Azerbaijani", "cross", "Armenian", "xac; Phase 6 triage", "B"),
    ("Turkish", "cross", "Armenian", "hac; Phase 6 triage", "B"),
    ("Azerbaijani", "stove", "Russian", "pec; Phase 6 triage", "B"),
    ("Uzbek", "stove", "Russian", "pecka; Phase 6 triage", "B"),
    ("Turkish", "still", "Arabic", "hala; Phase 6 triage", "B"),
    ("Kazakh", "wish", "Arabic", "muddaa; Phase 5/6", "B"),
    ("Kazakh", "flower", "Persian", "gul; Phase 5 Task 3", "B"),
    ("Turkish", "clever", "Arabic", "dhaki > zeki; Phase 5 Task 3", "B"),
]


def load_branch_thresholds():
    thresholds = {}
    with open(THRESH_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            thresholds[row["group"]] = float(row["threshold"])
    return thresholds


def load_regularity_scores():
    rows = []
    with open(REG_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["adjusted_score"] = float(row["adjusted_score"])
            row["tokens_list"] = row["ipa_tokens"].split()
            rows.append(row)
    return rows


def load_infra6a_candidates():
    candidates = set()
    with open(CANDIDATES_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            candidates.add((row["language"], row["gloss"].lower()))
    return candidates


# ── Automated phonological loan markers ────────────────────────────────────
def has_geminate(tokens):
    """Geminate consonants (e.g. dd, tt, ss) — not native to Turkic."""
    return any("ː" in t and t[0] not in "aeiouɯœøyɛɪʊɔʌəɘ" for t in tokens)

def has_loan_cluster(tokens):
    """Consonant clusters typical of Persian/Arabic loans."""
    token_str = " ".join(tokens)
    return any(p in token_str for p in ["x t", "ʃ t", "s t", "f t", "ħ", "ʕ"])

def has_initial_h(tokens):
    """Word-initial h- (lost in Proto-Turkic; present in loans)."""
    return bool(tokens) and tokens[0] in ("h", "ħ")


def run_sensitivity_test():
    print("Loading data...")
    scores = load_regularity_scores()
    branch_thresholds = load_branch_thresholds()
    candidates_6a = load_infra6a_candidates()

    print(f"  {len(scores)} regularity scores")
    print(f"  {len(candidates_6a)} Infra 6A candidates")
    print(f"  {len(KNOWN_LOANS)} known loan entries")

    # ── Match known loans against regularity scores ────────────────────────
    matched = []
    unmatched = []

    for kl_lang, kl_gloss, source, evidence, cat in KNOWN_LOANS:
        found = False
        kl_g = kl_gloss.lower()
        # Pass 1: exact match. Pass 2: substring (for compound glosses).
        for exact in (True, False):
            if found:
                break
            for row in scores:
                rg = row["gloss"].lower()
                hit = (rg == kl_g) if exact else (kl_g in rg)
                if row["language"] == kl_lang and hit:
                    branch = LANG_TO_BRANCH.get(kl_lang, "unknown")
                    bt = branch_thresholds.get(branch, FAMILY_WIDE_THRESHOLD)
                    matched.append({
                        "language": kl_lang,
                        "gloss": row["gloss"],
                        "form": row["form"],
                        "source": source,
                        "evidence": evidence,
                        "category": cat,
                        "score": row["adjusted_score"],
                        "branch": branch,
                        "branch_threshold": bt,
                        "flagged_family": row["adjusted_score"] < FAMILY_WIDE_THRESHOLD,
                        "flagged_branch": row["adjusted_score"] < bt,
                        "in_6a": (kl_lang, row["gloss"].lower()) in candidates_6a,
                    })
                    found = True
                    break
        if not found:
            unmatched.append((kl_lang, kl_gloss, source, evidence, cat))

    # ── Automated loan screen ──────────────────────────────────────────────
    auto_loans = []
    for row in scores:
        tokens = row["tokens_list"]
        markers = []
        if has_geminate(tokens):
            markers.append("geminate")
        if has_loan_cluster(tokens):
            markers.append("loan_cluster")
        if has_initial_h(tokens):
            markers.append("initial_h")
        if markers:
            branch = LANG_TO_BRANCH.get(row["language"], "unknown")
            bt = branch_thresholds.get(branch, FAMILY_WIDE_THRESHOLD)
            auto_loans.append({
                "language": row["language"],
                "gloss": row["gloss"],
                "form": row["form"],
                "score": row["adjusted_score"],
                "markers": "|".join(markers),
                "flagged_branch": row["adjusted_score"] < bt,
                "in_6a": (row["language"], row["gloss"].lower()) in candidates_6a,
            })

    # ── Compute metrics ────────────────────────────────────────────────────
    cat_a = [m for m in matched if m["category"] == "A"]
    cat_b = [m for m in matched if m["category"] == "B"]

    def stats(items, label):
        n = len(items)
        if n == 0:
            return f"  {label}: no items"
        nf_fam = sum(1 for m in items if m["flagged_family"])
        nf_br = sum(1 for m in items if m["flagged_branch"])
        n6a = sum(1 for m in items if m["in_6a"])
        return (f"  {label} (N={n}):\n"
                f"    Family-wide threshold: {nf_fam}/{n} = {nf_fam/n*100:.1f}%\n"
                f"    Per-branch threshold:  {nf_br}/{n} = {nf_br/n*100:.1f}%\n"
                f"    In Infra 6A list:      {n6a}/{n} = {n6a/n*100:.1f}%")

    auto_missed = [a for a in auto_loans if not a["flagged_branch"]]

    # ── Build report ───────────────────────────────────────────────────────
    L = []
    L.append("=" * 72)
    L.append("PHASE 7 — PSEUDO-SUBSTRATE SENSITIVITY TEST")
    L.append("=" * 72)
    L.append("")
    L.append("PURPOSE: Estimate pipeline sensitivity to foreign lexical material.")
    L.append("If the pipeline catches X% of known loans, the null substrate result")
    L.append("is credible to the extent that substrate items would be at least as")
    L.append("phonologically distinct as known-source loans.")
    L.append("")

    L.append("-" * 72)
    L.append("1. KNOWN-LOAN INVENTORY")
    L.append("-" * 72)
    L.append(f"Catalogued: {len(KNOWN_LOANS)} entries")
    L.append(f"  Category A (independently identified): {len([k for k in KNOWN_LOANS if k[4]=='A'])}")
    L.append(f"  Category B (pipeline-discovered):      {len([k for k in KNOWN_LOANS if k[4]=='B'])}")
    L.append(f"Matched in dataset: {len(matched)}")
    L.append(f"Unmatched: {len(unmatched)}")
    for ul in unmatched:
        L.append(f"  NOT FOUND: {ul[0]} '{ul[1]}' ({ul[2]})")
    L.append("")

    L.append("-" * 72)
    L.append("2. SENSITIVITY — CATEGORY A (GOLD STANDARD)")
    L.append("-" * 72)
    L.append("Items identified as loans INDEPENDENTLY of the pipeline.")
    L.append("Detection rate = true pipeline sensitivity estimate.")
    L.append("")
    L.append(stats(cat_a, "Category A"))
    L.append("")
    L.append("Detail:")
    for m in cat_a:
        tag = "FLAGGED" if m["flagged_branch"] else "MISSED"
        L.append(f"  [{tag}] {m['language']} '{m['gloss']}' = {m['form']}  "
                 f"score={m['score']:.4f}  thresh={m['branch_threshold']:.4f}  "
                 f"src={m['source']}")
    L.append("")

    L.append("-" * 72)
    L.append("3. SENSITIVITY — CATEGORY B (PIPELINE-DISCOVERED)")
    L.append("-" * 72)
    L.append("Items found because the pipeline flagged them, then attributed.")
    L.append("Partially circular but confirms pipeline catches this material.")
    L.append("")
    L.append(stats(cat_b, "Category B"))
    L.append("")
    L.append("Detail:")
    for m in cat_b:
        tag = "FLAGGED" if m["flagged_branch"] else "MISSED"
        L.append(f"  [{tag}] {m['language']} '{m['gloss']}' = {m['form']}  "
                 f"score={m['score']:.4f}  thresh={m['branch_threshold']:.4f}  "
                 f"src={m['source']}")
    L.append("")

    L.append("-" * 72)
    L.append("4. COMBINED SENSITIVITY")
    L.append("-" * 72)
    L.append(stats(matched, "All known loans"))
    L.append("")

    L.append("-" * 72)
    L.append("5. AUTOMATED PHONOLOGICAL LOAN SCREEN")
    L.append("-" * 72)
    L.append("Scanned all items for structural loan markers:")
    L.append("  - Geminate consonants (not native to Turkic)")
    L.append("  - Coda clusters typical of Persian/Arabic (-xt, -st, -ft)")
    L.append("  - Word-initial h- (lost in Proto-Turkic; present in loans)")
    L.append("  - Pharyngeals (Arabic diagnostic)")
    L.append("")
    L.append(f"Items with >=1 loan marker: {len(auto_loans)}")
    if auto_loans:
        nf = sum(1 for a in auto_loans if a["flagged_branch"])
        L.append(f"  Flagged (per-branch): {nf}/{len(auto_loans)} = {nf/len(auto_loans)*100:.1f}%")
        L.append(f"  Not flagged:          {len(auto_missed)}/{len(auto_loans)} = {len(auto_missed)/len(auto_loans)*100:.1f}%")
    L.append("")
    L.append(f"Potential false negatives (loan markers but not flagged): {len(auto_missed)}")
    if auto_missed:
        L.append("Sample (first 25):")
        for a in auto_missed[:25]:
            L.append(f"  {a['language']} '{a['gloss']}' = {a['form']}  "
                     f"score={a['score']:.4f}  markers={a['markers']}")
    L.append("")

    L.append("-" * 72)
    L.append("6. INTERPRETATION")
    L.append("-" * 72)
    L.append("")
    if cat_a:
        rate_a = sum(1 for m in cat_a if m["flagged_branch"]) / len(cat_a) * 100
        L.append(f"Gold-standard sensitivity (Category A, per-branch): {rate_a:.1f}%")
        L.append("")
        if rate_a >= 75:
            L.append("The pipeline detects the majority of independently known foreign")
            L.append("material. The null substrate result is credible: if substrate items")
            L.append("existed with phonological distinctness comparable to Persian/Arabic")
            L.append("loans, the pipeline would likely have flagged them.")
        elif rate_a >= 50:
            L.append("The pipeline detects a moderate fraction of known foreign material.")
            L.append("The null substrate result should be interpreted with caution: the")
            L.append("pipeline may miss substrate items phonologically similar to native")
            L.append("Turkic vocabulary.")
        else:
            L.append("The pipeline misses a substantial fraction of known foreign material.")
            L.append("The null substrate result is NOT strong evidence for absence of")
            L.append("substrate. The pipeline may lack sufficient sensitivity.")
    L.append("")

    L.append("-" * 72)
    L.append("7. CAVEATS")
    L.append("-" * 72)
    L.append("- Category A has only 4 items (Uzbek Persian loans). Small N limits")
    L.append("  the precision of the sensitivity estimate.")
    L.append("- All Category A items are from one language (Uzbek) and one source")
    L.append("  (Persian). Sensitivity may differ for other host/source combinations.")
    L.append("- The automated loan screen identifies structural markers, not confirmed")
    L.append("  loans. Some flagged items may be native Turkic with loan-like features.")
    L.append("- Substrate items may violate Turkic correspondences in different ways")
    L.append("  than known-source loans. A substrate from an unknown language might")
    L.append("  not share the phonological profile of Persian/Arabic borrowing.")
    L.append("- Persian loans in Uzbek are among the most phonologically distinct")
    L.append("  foreign material in the dataset (different consonant inventories,")
    L.append("  clusters, vowel patterns). Substrate items from a typologically")
    L.append("  closer source would be harder to detect.")
    L.append("")

    report = "\n".join(L)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport: {OUT_REPORT}")

    # ── CSV output ─────────────────────────────────────────────────────────
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[
            "language", "gloss", "form", "source", "evidence", "category",
            "score", "branch", "branch_threshold",
            "flagged_family", "flagged_branch", "in_6a"
        ])
        w.writeheader()
        for m in matched:
            w.writerow(m)
    print(f"CSV:    {OUT_CSV}")

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SENSITIVITY SUMMARY")
    print("=" * 60)
    if cat_a:
        r = sum(1 for m in cat_a if m["flagged_branch"]) / len(cat_a) * 100
        print(f"Category A (gold standard):       {r:.0f}% ({sum(1 for m in cat_a if m['flagged_branch'])}/{len(cat_a)})")
    if cat_b:
        r = sum(1 for m in cat_b if m["flagged_branch"]) / len(cat_b) * 100
        print(f"Category B (pipeline-discovered): {r:.0f}% ({sum(1 for m in cat_b if m['flagged_branch'])}/{len(cat_b)})")
    r_all = sum(1 for m in matched if m["flagged_branch"]) / len(matched) * 100
    print(f"Combined:                         {r_all:.0f}% ({sum(1 for m in matched if m['flagged_branch'])}/{len(matched)})")
    if auto_loans:
        print(f"Auto loan screen:                 {len(auto_loans)} items with markers, {len(auto_missed)} not flagged")
    print("=" * 60)


if __name__ == "__main__":
    run_sensitivity_test()
