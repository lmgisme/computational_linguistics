"""
Phase 6: Cultural Vocabulary Substrate Detection
=================================================
Uses the Infra 6A per-branch thresholds and candidate list to:
  1. Assign semantic domains to all NorthEuraLex concepts
  2. Compute anomaly rates per domain vs. each branch group baseline
  3. Apply cross-language clustering filter (same gloss, >=2 languages anomalous)
  4. Report domain-level rates + surviving multi-language clusters for Phase 6 Resolution

Inputs:
  output/infra6a_candidates.csv      -- 340 candidates from Infra 6A
  output/infra6a_thresholds.csv      -- group thresholds for reference
  output/northeuralex_merged.csv     -- full expanded dataset (6,966 rows)
  output/regularity_expanded.csv     -- regularity scores for all rows
  phase1_ingestion/output/lexibank_cache/northeuralex/parameters.csv
                                     -- NorthEuraLex concept list for domain mapping

Outputs:
  output/phase6_domain_rates.csv        -- anomaly rate per domain per group
  output/phase6_domain_summary.txt      -- plain-text domain rate report
  output/phase6_candidates_annotated.csv -- infra6a candidates with domain tags
  output/phase6_clusters.csv            -- multi-language clusters (>=2 langs, same gloss)
  output/phase6_report.txt              -- full narrative report

Run:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python phase6_cultural\\phase6_cultural_detection.py
"""

import csv
import os
from collections import defaultdict

# ─── paths ───────────────────────────────────────────────────────────────────
BASE    = r"C:\Users\lmgisme\Desktop\computational_linguistics"
OUTPUT  = os.path.join(BASE, "output")
NEL_PAR = os.path.join(BASE, "phase1_ingestion", "output",
                        "lexibank_cache", "northeuralex", "parameters.csv")

CAND_CSV    = os.path.join(OUTPUT, "infra6a_candidates.csv")
THRESH_CSV  = os.path.join(OUTPUT, "infra6a_thresholds.csv")
MERGED_CSV  = os.path.join(OUTPUT, "northeuralex_merged.csv")
REG_CSV     = os.path.join(OUTPUT, "regularity_expanded.csv")

OUT_RATES   = os.path.join(OUTPUT, "phase6_domain_rates.csv")
OUT_SUMMARY = os.path.join(OUTPUT, "phase6_domain_summary.txt")
OUT_ANNOT   = os.path.join(OUTPUT, "phase6_candidates_annotated.csv")
OUT_CLUST   = os.path.join(OUTPUT, "phase6_clusters.csv")
OUT_REPORT  = os.path.join(OUTPUT, "phase6_report.txt")

# ─── DOMAIN TAXONOMY ─────────────────────────────────────────────────────────
# Built from the NorthEuraLex parameters.csv concept IDs.
# The ID format is {N}_{gloss}. Ranges below are derived from the
# NorthEuraLex Dellert et al. 2020 paper's semantic field structure,
# mapped onto the actual concept ID numbering in the CLDF export.
# Domains of primary substrate interest are flagged with HIGH_INTEREST.
#
# Note: NorthEuraLex CLDF does not ship a domain column.
# This taxonomy is constructed manually from concept content.

DOMAIN_DEFS = [
    # (domain_label, priority, set_of_concept_name_keywords_or_id_ranges)
    # We use concept NAME substring matching (lowercase) as the primary method.
    # A concept is assigned the FIRST matching domain.
    ("body_parts",          "LOW",  {
        "eye","ear","nose","mouth","tooth","tongue","lip","cheek","face",
        "forehead","hair","moustache","beard","chin","jaw","throat","neck",
        "nape","head","back","belly","navel","bosom","breast","shoulder",
        "arm","elbow","hand","palm","finger","fingernail","nail","toe",
        "foot","heel","knee","thigh","leg","body","skin","blood","vein",
        "sinew","bone","brain","heart","stomach","liver","breath","hunger",
        "tear","flavour","odour","sleep","dream",
    }),
    ("natural_world",       "LOW",  {
        "sky","sun","moon","star","air","wind","wave","water","stone",
        "ground","earth","dust","smoke","spark","fire","light","shadow",
        "weather","fog","cloud","rain","snow","ice","frost","chill","heat",
        "hoarfrost","rainbow","thunder","current","drop","foam","dirt",
        "lake","swamp","moor","meadow","forest","hill","elevation","mountain",
        "summit","cave","slope","source","brook","river","shore","coast",
        "land","sea","cove","bay","island",
    }),
    ("flora",               "MEDIUM", {
        "flower","grass","root","tree","trunk","bark","limb","twig","leaf",
        "birch","pine","willow","fir","mushroom","onion","seed","grain",
        "hay","peel","husk","berry","apple",
    }),
    ("fauna",               "HIGH",   {
        "animal","flock","cow","bull","horse","sheep","pig","dog","cat",
        "bear","squirrel","elk","fox","hare","mouse","wolf","bird","swarm",
        "chicken","cock","goose","eagle","duck","owl","crane","crow",
        "cuckoo","swan","fish","perch","pike","snake","worm","spider",
        "ant","louse","gnat","fly","butterfly","horn","feather","fur",
        "wing","claw","paw","tail","egg","nest","lair",
    }),
    ("herding_pastoralism", "HIGH",   {
        "pasture","herd","milk","feed","tie up","tether","drive",
        "flock","wool","leather","trap","noose","track","leash",
        "whip","yoke",
    }),
    ("metallurgy_materials","HIGH",   {
        "iron","gold","silver","coal","glass","clay","sand","ash",
        "coal","metal","forge","smelt","cast",
    }),
    ("agriculture",         "HIGH",   {
        "grain","seed","hay","corn","bread","mush","harvest","sow",
        "reap","plough","field","crop","barn","mill","grind","thresh",
        "ripen","gather","pick",
    }),
    ("food_drink",          "MEDIUM", {
        "meal","food","dish","meat","corn","mush","bread","slice","fat",
        "butter","oil","salt","soup","honey","milk","tea","beer","wine",
        "cook","boil","bake","fry","stir","eat","drink","swallow",
    }),
    ("kinship",             "HIGH",   {
        "family","grandfather","grandmother","parents","father","mother",
        "son","daughter","brother","sister","uncle","aunt","husband","wife",
        "child","boy","girl","man","woman","kin","clan",
    }),
    ("social_political",    "MEDIUM", {
        "people","nation","world","country","state","king","power","border",
        "war","enemy","violence","fight","bow","arrow","gun","worker",
        "boss","master","doctor","teacher","guest","friend","companion",
        "village","town","road","path","way","bridge",
    }),
    ("material_culture",    "MEDIUM", {
        "boat","oar","sleigh","ski","campfire","load","house","home","stove",
        "floor","table","chair","cradle","bed","shelf","box","window","door",
        "gate","fence","roof","ladder","broom","spade","shovel","fork",
        "spoon","knife","nail","net","hook","handle","lock","pouch","bundle",
        "bag","bucket","cover","lid","dishes","sack","cup","pot","kettle",
        "needle","thread","button","knot","paint","clothes","shirt","collar",
        "sleeve","trousers","belt","cap","shoe","boot","ring","ribbon",
        "comb","mirror","strap","string","leash","blanket","pillow","scarf",
        "towel","cloth","wool","leather",
    }),
    ("cognitive_abstract",  "LOW",    {
        "joy","laughter","happiness","grief","wish","desire","spirit",
        "thought","memory","mind","meaning","reason","truth","talk",
        "fairy tale","story","message","news","name","puzzle","speech",
        "language","voice","word","sign","call","noise","sound","tone",
        "song","calm","game","work","help","company","gift","matter",
        "count","sort","piece","part","half",
    }),
    ("spatial_temporal",    "LOW",    {
        "circle","line","stroke","gap","distance","area","space","place",
        "side","middle","item","thing","fringe","edge","corner","tip","end",
        "hole","angle","pattern","size","length","height","weight","amount",
        "heap","row","north","south","west","east","day","morning","noon",
        "evening","night","week","month","year","spring","summer","autumn",
        "winter","time","age",
    }),
    ("verbs_actions",       "LOW",    set()),   # catch-all for verbs — see below
    ("other",               "LOW",    set()),   # true catch-all
]

# Verbs: any concept with POS tag V in the NorthEuraLex_Gloss column (::V suffix)
# will be assigned domain "verbs_actions" if not already matched above.

def build_domain_map(params_path: str) -> dict:
    """
    Returns {concept_name_lowercase: domain_label}
    Built from NorthEuraLex parameters.csv.
    """
    domain_map = {}
    verb_names = set()

    with open(params_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row["Name"].strip().lower()
            nel_gloss = row.get("NorthEuralex_Gloss", "")
            is_verb = nel_gloss.endswith("::V")

            if is_verb:
                verb_names.add(name)

            # Try each domain in order (first match wins)
            matched = False
            for domain_label, priority, keywords in DOMAIN_DEFS:
                if not keywords:
                    continue
                # Check if any keyword appears in the concept name
                for kw in keywords:
                    if kw in name:
                        domain_map[name] = domain_label
                        matched = True
                        break
                if matched:
                    break

            if not matched:
                if is_verb:
                    domain_map[name] = "verbs_actions"
                else:
                    domain_map[name] = "other"

    # Assign verbs_actions to unmatched verbs that weren't caught above
    for name in verb_names:
        if name not in domain_map:
            domain_map[name] = "verbs_actions"

    return domain_map


def load_csv(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_thresholds(path: str) -> dict:
    thresholds = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            thresholds[row["group"]] = {
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "threshold": float(row["threshold"]),
                "calibration_langs": row["calibration_langs"],
            }
    return thresholds


# Language -> branch group assignment
LANG_GROUP = {
    "Turkish":      "A_oghuz",
    "Azerbaijani":  "A_oghuz",
    "Turkmen":      "A_oghuz",   # thin coverage — excluded from calibration
    "Kazakh":       "A_kipchak",
    "Uzbek":        "A_kipchak",
    "Kyrgyz":       "A_kipchak", # thin coverage
    "Uyghur":       "A_kipchak", # thin coverage
    "Yakut":        "B_yakut",
    "Chuvash":      "C_chuvash",
}

THIN_LANGS = {"Kyrgyz", "Turkmen", "Uyghur"}

# Known false positive — exclude from Phase 6 cluster analysis
KNOWN_FALSE_POSITIVES = {
    ("Turkmen", "grease"),
    ("Turkmen", "jaɣ"),
}


def assign_domain(gloss: str, domain_map: dict) -> str:
    g = gloss.strip().lower()
    # Direct lookup
    if g in domain_map:
        return domain_map[g]
    # Partial match: check if the gloss contains any mapped concept name
    for concept, domain in domain_map.items():
        if concept in g or g in concept:
            return domain
    return "other"


def main():
    print("Phase 6: Cultural Vocabulary Substrate Detection")
    print("=" * 60)

    # ── Load data ─────────────────────────────────────────────────
    print("\nLoading data...")
    domain_map   = build_domain_map(NEL_PAR)
    candidates   = load_csv(CAND_CSV)
    thresholds   = load_thresholds(THRESH_CSV)
    merged       = load_csv(MERGED_CSV)
    reg_rows     = load_csv(REG_CSV) if os.path.exists(REG_CSV) else []

    print(f"  Candidates:       {len(candidates)}")
    print(f"  Merged rows:      {len(merged)}")
    print(f"  Reg score rows:   {len(reg_rows)}")
    print(f"  Domain map size:  {len(domain_map)} concepts")

    # ── Assign domains to candidates ──────────────────────────────
    print("\nAssigning domains to candidates...")
    for row in candidates:
        row["domain"] = assign_domain(row["gloss"], domain_map)

    # ── Remove known false positives ──────────────────────────────
    pre_filter = len(candidates)
    candidates = [
        r for r in candidates
        if (r["language"], r["gloss"]) not in KNOWN_FALSE_POSITIVES
    ]
    print(f"  Removed {pre_filter - len(candidates)} known false positive(s) (Turkmen jaɣ).")

    # ── Assign domains to all merged rows for denominator counts ──
    for row in merged:
        row["domain"]     = assign_domain(row["gloss"], domain_map)
        row["lang_group"] = LANG_GROUP.get(row["language"], "unknown")

    # ── Build denominator: total rows per (group, domain) ─────────
    # Only count rows from calibration languages for the denominator
    # (thin-coverage languages are excluded from rate calculation)
    CALIB_LANGS = {lang for lang, g in LANG_GROUP.items()
                   if lang not in THIN_LANGS}

    denom = defaultdict(int)    # (group, domain) -> n_total
    for row in merged:
        lang = row["language"]
        if lang not in CALIB_LANGS:
            continue
        group  = LANG_GROUP.get(lang, "unknown")
        domain = row["domain"]
        denom[(group, domain)] += 1

    # ── Build numerator: candidates per (group, domain) ───────────
    numer = defaultdict(list)   # (group, domain) -> [candidate rows]
    for row in candidates:
        lang   = row["language"]
        if lang in THIN_LANGS:
            continue  # thin-lang candidates not used for rate calculation
        group  = LANG_GROUP.get(lang, "unknown")
        domain = row["domain"]
        numer[(group, domain)].append(row)

    # ── Compute anomaly rate per (group, domain) ──────────────────
    print("\nComputing domain anomaly rates...")
    rate_rows = []
    all_domains = sorted(set(domain for _, domain in denom.keys()))
    groups      = sorted(thresholds.keys())

    for domain in all_domains:
        for group in groups:
            n_total = denom.get((group, domain), 0)
            n_anom  = len(numer.get((group, domain), []))
            rate    = n_anom / n_total if n_total > 0 else None
            rate_rows.append({
                "domain":    domain,
                "group":     group,
                "n_total":   n_total,
                "n_anomalous": n_anom,
                "rate":      f"{rate:.4f}" if rate is not None else "N/A",
                "rate_float": rate if rate is not None else -1,
            })

    # Sort by descending rate for readability
    rate_rows.sort(key=lambda r: r["rate_float"], reverse=True)

    # ── Cross-language clustering filter ──────────────────────────
    # A cluster is a gloss that is anomalous in >=2 languages (within
    # calibration languages). Thin-language candidates are included for
    # CROSS-REFERENCING only (they can contribute to a cluster but cannot
    # form one alone).
    print("\nApplying cross-language clustering filter...")

    # Collect all candidates (including thin-lang for cross-ref)
    gloss_to_langs = defaultdict(set)   # gloss -> set of languages anomalous
    gloss_to_rows  = defaultdict(list)  # gloss -> list of candidate rows

    for row in candidates:
        g = row["gloss"].strip().lower()
        gloss_to_langs[g].add(row["language"])
        gloss_to_rows[g].append(row)

    # A cluster survives if >=2 *distinct* languages show the same gloss
    # as anomalous; at least one of those languages must be a calibration lang.
    clusters = []
    for gloss, langs in gloss_to_langs.items():
        calib_langs_in_cluster = langs - THIN_LANGS
        if len(langs) >= 2 and len(calib_langs_in_cluster) >= 1:
            cluster_rows = gloss_to_rows[gloss]
            clusters.append({
                "gloss":       gloss,
                "n_languages": len(langs),
                "languages":   "; ".join(sorted(langs)),
                "domain":      cluster_rows[0]["domain"],
                "rows":        cluster_rows,
            })

    clusters.sort(key=lambda c: (-c["n_languages"], c["gloss"]))
    print(f"  Multi-language clusters: {len(clusters)}")

    # ── Domain priority assessment ─────────────────────────────────
    # Map domain to HIGH_INTEREST flag
    domain_priority = {label: priority for label, priority, _ in DOMAIN_DEFS}

    # ── Write outputs ─────────────────────────────────────────────

    # 1. Domain rates CSV
    rate_out_cols = ["domain","group","n_total","n_anomalous","rate"]
    with open(OUT_RATES, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rate_out_cols, extrasaction="ignore")
        writer.writeheader()
        for r in rate_rows:
            writer.writerow(r)
    print(f"\nWrote: {OUT_RATES}")

    # 2. Candidates annotated with domain
    cand_cols = list(candidates[0].keys()) if candidates else []
    with open(OUT_ANNOT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cand_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(candidates)
    print(f"Wrote: {OUT_ANNOT}")

    # 3. Clusters CSV
    cluster_out_rows = []
    for cl in clusters:
        for r in cl["rows"]:
            cluster_out_rows.append({
                "gloss":         cl["gloss"],
                "n_languages":   cl["n_languages"],
                "all_languages": cl["languages"],
                "domain":        cl["domain"],
                "language":      r["language"],
                "form":          r["form"],
                "ipa_tokens":    r["ipa_tokens"],
                "group":         r["group"],
                "adjusted_score": r["adjusted_score"],
                "loan_class":    r["loan_class"],
                "thin_lang_flag": r.get("thin_lang_flag", ""),
            })
    cluster_cols = ["gloss","n_languages","all_languages","domain","language",
                    "form","ipa_tokens","group","adjusted_score","loan_class",
                    "thin_lang_flag"]
    with open(OUT_CLUST, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cluster_cols)
        writer.writeheader()
        writer.writerows(cluster_out_rows)
    print(f"Wrote: {OUT_CLUST}")

    # ── Build domain summary text ──────────────────────────────────
    lines = []
    lines.append("PHASE 6 DOMAIN ANOMALY RATE SUMMARY")
    lines.append("=" * 60)
    lines.append("")
    lines.append("Anomaly rates: n_anomalous / n_total per branch group.")
    lines.append("Only calibration languages used for rate computation.")
    lines.append("Thin-coverage languages (Kyrgyz, Turkmen, Uyghur) excluded.")
    lines.append("")

    # High-interest domains first
    HIGH_DOMAINS = [d for d, p, _ in DOMAIN_DEFS if p == "HIGH"]
    lines.append("── HIGH INTEREST DOMAINS (primary substrate candidates) ──")
    lines.append("")
    for domain in HIGH_DOMAINS:
        rows_for_domain = [r for r in rate_rows if r["domain"] == domain]
        lines.append(f"  Domain: {domain}")
        for r in rows_for_domain:
            if r["n_total"] > 0:
                lines.append(f"    {r['group']:15s}  {r['n_anomalous']:3d}/{r['n_total']:4d}  ({r['rate']})")
        lines.append("")

    lines.append("── MEDIUM INTEREST DOMAINS ──")
    lines.append("")
    MED_DOMAINS = [d for d, p, _ in DOMAIN_DEFS if p == "MEDIUM"]
    for domain in MED_DOMAINS:
        rows_for_domain = [r for r in rate_rows if r["domain"] == domain]
        lines.append(f"  Domain: {domain}")
        for r in rows_for_domain:
            if r["n_total"] > 0:
                lines.append(f"    {r['group']:15s}  {r['n_anomalous']:3d}/{r['n_total']:4d}  ({r['rate']})")
        lines.append("")

    lines.append("── LOW INTEREST DOMAINS (reference baseline) ──")
    lines.append("")
    LOW_DOMAINS = [d for d, p, _ in DOMAIN_DEFS if p == "LOW"]
    for domain in LOW_DOMAINS:
        rows_for_domain = [r for r in rate_rows if r["domain"] == domain]
        lines.append(f"  Domain: {domain}")
        for r in rows_for_domain:
            if r["n_total"] > 0:
                lines.append(f"    {r['group']:15s}  {r['n_anomalous']:3d}/{r['n_total']:4d}  ({r['rate']})")
        lines.append("")

    summary_text = "\n".join(lines)
    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Wrote: {OUT_SUMMARY}")
    print("")
    print(summary_text)

    # ── Build full narrative report ────────────────────────────────
    report_lines = []
    report_lines.append("PHASE 6: CULTURAL VOCABULARY SUBSTRATE DETECTION")
    report_lines.append("Report generated from infra6a_candidates.csv + northeuralex_merged.csv")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Total infra6a candidates entering Phase 6: {pre_filter}")
    report_lines.append(f"After known false positive removal:        {len(candidates)}")
    report_lines.append(f"Multi-language clusters (>=2 langs):       {len(clusters)}")
    report_lines.append("")
    report_lines.append("MULTI-LANGUAGE CLUSTERS")
    report_lines.append("-" * 50)
    report_lines.append("Cross-language clustering filter: a gloss must appear anomalous in")
    report_lines.append(">=2 distinct languages (at least 1 calibration language) to survive.")
    report_lines.append("")

    if not clusters:
        report_lines.append("  No multi-language clusters found. All candidates are single-language flags.")
        report_lines.append("  This result is consistent with the Phase 5 null: no substrate signal")
        report_lines.append("  in basic vocabulary. Phase 6 Resolution is not triggered.")
    else:
        for cl in clusters:
            priority = domain_priority.get(cl["domain"], "LOW")
            report_lines.append(f"  GLOSS: {cl['gloss']}")
            report_lines.append(f"    Languages ({cl['n_languages']}): {cl['languages']}")
            report_lines.append(f"    Domain: {cl['domain']}  [Priority: {priority}]")
            for r in cl["rows"]:
                report_lines.append(
                    f"      {r['language']:15s}  {r['form']:20s}  score={r['adjusted_score']:>8s}"
                    f"  source={r['cogid_source']:12s}  loan={r['loan_class']}"
                )
            report_lines.append("")

    report_lines.append("")
    report_lines.append("DOMAIN ANOMALY RATES — FULL TABLE")
    report_lines.append("-" * 50)
    report_lines.append(f"{'Domain':<25} {'Group':<15} {'Anomalous':>10} {'Total':>6} {'Rate':>8}")
    report_lines.append("-" * 70)
    for r in rate_rows:
        if r["n_total"] == 0:
            continue
        priority = domain_priority.get(r["domain"], "LOW")
        flag = " ***" if priority == "HIGH" and float(r["rate_float"]) > 0.05 else ""
        report_lines.append(
            f"{r['domain']:<25} {r['group']:<15} {r['n_anomalous']:>10} {r['n_total']:>6} {r['rate']:>8}{flag}"
        )

    report_lines.append("")
    report_lines.append("NOTE ON A_KIPCHAK NOISE")
    report_lines.append("-" * 50)
    report_lines.append("A_kipchak and A_oghuz candidates include words where anomalous scores")
    report_lines.append("reflect model sparsity for native phonemes (ʒ/ʁ/ɾ in Kazakh;")
    report_lines.append("tʃ/dʒ/ɾ in Turkish/Azerbaijani) rather than genuine irregularity.")
    report_lines.append("These are substantially filtered by the cross-language clustering")
    report_lines.append("requirement. Single-language A_kipchak flags are treated as noise")
    report_lines.append("unless corroborated by a second language in the same gloss.")
    report_lines.append("")
    report_lines.append("STATUS: Phase 6 complete.")
    report_lines.append("Next step: Phase 6 Resolution — proto-form comparison for any")
    report_lines.append("multi-language clusters surviving this filter.")

    report_text = "\n".join(report_lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nWrote full report: {OUT_REPORT}")

    # ── Console summary ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PHASE 6 COMPLETE")
    print(f"  Multi-language clusters surviving filter: {len(clusters)}")
    if clusters:
        print("\n  Clusters:")
        for cl in clusters:
            priority = domain_priority.get(cl["domain"], "LOW")
            print(f"    [{priority}] {cl['gloss']} — {cl['n_languages']} langs: {cl['languages']}")
    else:
        print("  No multi-language clusters. Extended null result.")
    print("")
    print("Outputs written to output/:")
    print("  phase6_domain_rates.csv")
    print("  phase6_domain_summary.txt")
    print("  phase6_candidates_annotated.csv")
    print("  phase6_clusters.csv")
    print("  phase6_report.txt")


if __name__ == "__main__":
    main()
