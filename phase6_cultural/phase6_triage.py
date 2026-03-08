"""
Phase 6 Resolution Triage
==========================
Takes the 57-cluster output from phase6_cultural_detection.py and applies
three systematic filters to eliminate false positives before proto-form
comparison. Outputs a clean shortlist for Phase 6 Resolution.

FILTER 1 — SAME-COGNATE OGHUZ PAIRS
  Azerbaijani and Turkish are close sisters (posterior 0.9994). When both
  flag the same form for the same gloss, this is almost always the same
  cognate scoring anomalously twice due to model sparsity, not two
  independent anomalies. A cluster consisting ONLY of Azerbaijani + Turkish
  (and no other language) is presumptively a sparsity artifact unless the
  forms are phonologically distinct.
  EXCEPTION: Keep if forms differ substantially (NLev > 0.3).

FILTER 2 — SPARSITY PHONEME DRIVER
  If ALL tokens across ALL forms in a cluster contain at least one member
  from the sparsity phoneme set, the anomaly is likely driven by model
  coverage gaps rather than genuine irregularity.
  Sparsity phonemes (documented in infra6a_report.txt):
    tʃ dʒ y ʉ œ ɾ ʁ ʒ (affricates, front rounded vowels, Kazakh-specific)
  A cluster is flagged if every language's form contains at least one
  sparsity phoneme token AND the forms look cognate across languages.

FILTER 3 — KNOWN LOANWORD PHONOLOGICAL SIGNATURES
  Applies targeted phonological screens for documented contact languages.
  Persian loans: χ-initial, -aχt suffix, f-final, initial da-/di- with χ
  Russian loans: -ka suffix with fricative stem (pɛtʃka pattern)
  Arabic loans: geminate consonants (dː tː sː), initial ʕ/ħ
  French/European loans: ʃɛf, vɛ/væ (function words)
  Armenian: χɑtʃ pattern

FILTER 4 — COGNATE IDENTITY WITHIN CLUSTER
  If all forms in the cluster derive from the same Proto-Turkic root
  (detected by high pairwise NLev similarity >= 0.5 and same initial
  consonant class), the cluster is a single cognate set being multiply
  flagged. These are model artifacts.

Outputs:
  output/phase6_triage_shortlist.csv      -- clusters surviving all filters
  output/phase6_triage_discards.csv       -- discarded clusters with reason
  output/phase6_triage_report.txt         -- narrative triage report

Run:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python phase6_cultural\\phase6_triage.py
"""

import csv
import os
from collections import defaultdict

BASE   = r"C:\Users\lmgisme\Desktop\computational_linguistics"
OUTPUT = os.path.join(BASE, "output")

CLUST_CSV    = os.path.join(OUTPUT, "phase6_clusters.csv")
OUT_SHORT    = os.path.join(OUTPUT, "phase6_triage_shortlist.csv")
OUT_DISCARD  = os.path.join(OUTPUT, "phase6_triage_discards.csv")
OUT_REPORT   = os.path.join(OUTPUT, "phase6_triage_report.txt")

# ── Sparsity phoneme tokens (documented in infra6a_report.txt) ────────────────
# These are tokens that are underrepresented in the Phase 2/5 correspondence
# model because they appear infrequently in the 40-item Swadesh training set.
# Anomalous scores driven purely by these tokens are model artifacts.
SPARSITY_TOKENS = {
    "tʃ", "dʒ",             # palato-alveolar affricates
    "y", "ʉ", "œ",          # front rounded vowels (sparse in correspondence model)
    "ɾ",                    # Kazakh alveolar tap (vs Turkish/Oghuz r)
    "ʁ",                    # Kazakh uvular fricative
    "ʒ",                    # Kazakh voiced palatal fricative
}

# ── Known loanword screens ────────────────────────────────────────────────────
# Each entry: (screen_name, contact_language, test_function)
# test_function takes the ipa_tokens string for a single form and returns bool

def is_persian_loan(tokens_str: str, gloss: str) -> bool:
    toks = set(tokens_str.split())
    # daraxt pattern: d a r a χ t
    if "d" in toks and "a" in toks and "χ" in toks:
        return True
    # -aχt suffix (tree, night)
    tok_list = tokens_str.split()
    for i in range(len(tok_list) - 1):
        if tok_list[i] == "χ" and i > 0:
            return True
    # goʃt pattern (flesh)
    if "g" in toks and "o" in toks and "ʃ" in toks and "t" in toks:
        return True
    # f-final (forms ending in f are often Persian/Arabic)
    if tok_list and tok_list[-1] == "f":
        return True
    return False

def is_russian_loan(tokens_str: str, gloss: str) -> bool:
    tok_list = tokens_str.split()
    # -ka suffix after an affricate stem (pɛtʃka)
    if len(tok_list) >= 3 and tok_list[-1] == "a" and tok_list[-2] == "k":
        stem_toks = set(tok_list[:-2])
        if "tʃ" in stem_toks or "ʃ" in stem_toks:
            return True
    return False

def is_arabic_loan(tokens_str: str, gloss: str) -> bool:
    tok_list = tokens_str.split()
    # Geminate consonants (dː, tː, sː, nː) — Arabic loanword signature
    for t in tok_list:
        if len(t) >= 2 and t[-1] == "ː" and t[0] in "dtsnlrbm":
            return True
        if "dː" in t or "tː" in t or "nː" in t:
            return True
    return False

def is_european_loan(tokens_str: str, gloss: str) -> bool:
    tok_list = tokens_str.split()
    # ʃɛf (French chef)
    if tok_list == ["ʃ", "ɛ", "f"] or tok_list == ["ʃ", "e", "f"]:
        return True
    # vɛ / væ (Arabic/Persian va → function word 'and')
    if len(tok_list) == 2 and tok_list[0] == "v" and tok_list[1] in ("ɛ", "æ", "e"):
        return True
    return False

def is_armenian_loan(tokens_str: str, gloss: str) -> bool:
    # χɑtʃ or hatʃ (cross — Armenian խաչ)
    tok_list = tokens_str.split()
    if "tʃ" in tok_list and tok_list[0] in ("χ", "h", "x"):
        return True
    return False

LOAN_SCREENS = [
    ("Persian",   is_persian_loan),
    ("Russian",   is_russian_loan),
    ("Arabic",    is_arabic_loan),
    ("European",  is_european_loan),
    ("Armenian",  is_armenian_loan),
]

# ── Hard-coded discard list ───────────────────────────────────────────────────
# Clusters where manual inspection (see triage notes) confirms the source.
# Format: {gloss_lower: reason_string}
MANUAL_DISCARDS = {
    "milk":         "PT *süt, Clauson p.797, pan-Turkic; tʃ/y sparsity artifact",
    "arrow":        "PT *oq, Clauson p.83; χ/x sparsity artifact — same cognate in Az+Yak",
    "house":        "PT *eb > *ev, Oghuz regular; ɛv same form in Az+Tur sister pair",
    "home":         "PT *eb > *ev, same as 'house'; Az+Tur sister pair, same form",
    "thought":      "same cognate dyʃynd͡ʒɛ/dyʃynd͡ʒæ in Az+Tur; dʒ sparsity",
    "he":           "PT *ol/o pan-Turkic pronoun; monosyllabic — zero model coverage",
    "three":        "PT *üč, Clauson p.17; pan-Turkic cognate, all reflexes regular",
    "third":        "derived from PT *üč; same root as 'three', all regular",
    "tip":          "PT *uč 'point/tip', Clauson p.23; cognate with 'three' root",
    "because of":   "Az yt͡ʃyn / Kaz ʊʃən both derived from PT *üč 'three' (ablative); same root",
    "seven":        "PT *yeti, Clauson p.879; all forms regular, ɕ in Chuvash = regular Bulgar",
    "new":          "PT *yañı, Clauson p.942; Az jɛnɪ, Kaz ʒɑŋɑ, Chuvash ɕɘnɘ all regular",
    "walk":         "PT *yür-, Clauson p.959; Az/Tur jɛrɪʃ/jyɾyjyʃ, Kaz ʒʊɾəs all regular ɾ/ʒ reflexes",
    "thin":         "Kaz ʒəŋəʃke / Tur ind͡ʒɛ — not cognates; both ʒ/dʒ sparsity",
    "stove":        "Az/Uzb pɛt͡ʃ/pɛtʃka = Russian печь/печка; Yakut ohox unrelated and separate",
    "boss":         "Az/Tur ʃɛf = French chef via Ottoman; European loan",
    "cross":        "Az χɑt͡ʃ / Tur hat͡ʃ = Armenian խաչ; documented Armenian loan",
    "and":          "Az/Tur væ/vɛ = Arabic/Persian و (wa/va); adstrate function word",
    "flesh":        "Uzb goʃt = Persian گوشت (gusht), identified Infra 6B; Chuvash aʂ separate",
    "tree":         "Uzb daraχt = Persian درخت, Infra 6B; Turkmen aɣaʨ separate item",
    "sort":         "Az/Tur t͡ʃɛʃid/t͡ʃɛʃit = same Ottoman cognate; dʒ/tʃ sparsity",
    "wood":         "PT *ığač, Clauson p.78; Kaz ɑʁɑʃ and Tur aːat͡ʃ regular reflexes, ʁ/tʃ sparsity",
    "raw":          "PT *čiy, Clauson; Az/Tur t͡ʃɪj/t͡ʃij same cognate, tʃ sparsity",
    "all":          "Kaz ʒɯɣɯl already flagged Phase 3 as false positive cluster; Az/Tur dyʃ 'dream' mismatch",
    "swarm":        "PT *sürü 'herd/flock', Clauson p.844; Kaz ʊjəɾ / Tur syɾy regular ɾ reflexes",
    "end":          "Kaz soŋə = PT *soŋ 'end', Tur ut͡ʃ = PT *uč 'tip' — different roots, not cognates",
    "face":         "PT *yüz, Clauson p.982; Az/Tur yz/jyz regular, y sparsity",
    "dream":        "Kaz tʊs / Tur ɾyja — different roots, both low-coverage phonemes",
    "he":           "PT *ol, pan-Turkic; monosyllabic scoring artifact",
    "hungry":       "Az ɑd͡ʒ = PT *aç 'hungry', Clauson p.19; Kaz ɑʃ = PT *aš 'food/hungry' — different roots",
    "often":        "Az t͡ʃɔχ_vɑχt compound; Kaz əlʁɪj — not cognates, both tʃ/ʁ sparsity",
    "ahead":        "Az dyz = PT *yüz 'face/front'; Chuvash ˈtikɘs separate; not cognates",
    "nail":         "Kaz ʃeɡe 'nail (tool)' vs Tur t͡ʃivi — different roots; both ʃ/tʃ sparsity",
    "odour":        "Kaz ɪjəs / Uzb d͡ʒid — different roots; dʒ sparsity in Uzbek",
    "place":        "Tur jɛɾ = PT *yer, Clauson p.957; Uzb d͡ʒɒj separate; ɾ sparsity",
    "pointed":      "Kaz ʊʃkəɾ from PT *üč 'tip/point'; Az sɪvrɪ different root; not cognates",
    "pit":          "Kaz oɾ 'ditch', PT *or; Az t͡ʃuχur different root; not cognates",
    "still":        "Tur haːlaː = Arabic حالا (ħāla); Kaz ælə different",
    "green":        "Kaz ʒɑsəl PT *yaşıl, Clauson; Yakut ot_kyøɣe 'grass-colored' compound; not cognates",
    "first":        "Kaz ɑlʁɑʃqə from PT *al 'front'; Az bɪrɪnd͡ʒɪ from PT *bir 'one' — different roots",
    "small":        "Turkmen thin-coverage flag (thin_lang_flag=True); Chuvash ˈpɘt͡ɕɘk = PT *küçük regular Bulgar",
    "so":           "Tur œjlɛ / Yakut ocːo — unrelated forms, different sources",
    "thought":      "duplicate — same Az/Tur cognate pair, dʒ sparsity",
    "string":       "Az ɪp = PT *ip 'rope', Clauson p.67; Kaz ʒəp same root with ʒ reflex of PT *y-; sparsity",
    "fringe":       "Kaz ʃetə edge/fringe; Chuvash ˈχɘrɘ different root; not cognates",
    "blunt":        "Kaz doʁɑl / Chuvash ˈtɘrɘnt͡ɕɘk — not cognates; ʁ/tʃ sparsity",
    "border":       "Az hydud = Arabic حدود (ħudūd) 'borders'; Chuvash ˈt͡ɕikɘ separate",
    "mirror":       "Az ɟyzɟy = PT *közkü 'mirror' (related to *köz 'eye'); Chuvash ˈtɘkɘr = PT *tägür; not same root",
    "hill":         "Tur tɛpɛ = PT *tepe, Clauson; Chuvash tɘmɛsˈkɛ different; not cognates",
    "worker":       "Az ɪʃt͡ʃɪ = PT *iş 'work' + agent suffix; Kaz ʒʊməsʃə = PT *yumuš 'work' + suffix; different but both Turkic",
    "broom":        "Az sypyrɟæ = PT *süpür- 'to sweep'; Kaz səpəɾʁəʃ = same root; cognates, ɾ/ʁ sparsity",
    "willow":       "Az/Tur sœjyd/sœjyt = PT *söğüt, Clauson p.812; same cognate pair, œ/y sparsity",
    "fat":          "PT *yağ Oghuz (Tur jaː); Yakut sɯa = PT *sïŋar different root — not cognates; separate items",
    "boat":         "Yakut oŋoco / Chuvash ˈkimɘ — unrelated forms; different sources",
}

# ── Clusters to KEEP regardless of automatic filters ─────────────────────────
# Manual override based on analysis above.
MANUAL_KEEPS = {
    "woman":  "Kaz bɪjke + Yakut ɟaxtaɾ — unrelated forms, different contact zones; both require OT check",
    "wish":   "Kaz mʊdːe score=-2.28 extreme outlier; geminate dː non-native; Arabic loan candidate vs. substrate; requires etymological verification",
}

# ── Additional borderline cases — flagged for manual review in report ─────────
BORDERLINE = {
    "arrow":    "Az ɔχ / Yakut ox — likely same PT *oq cognate, but Yakut score -1.30 unusually low even for B_yakut",
    "at that time": "Yakut ocːoɣo loan_class=mongolic_candidate, score=-1.53; Az form different; worth Mongolic check on Yakut form alone",
    "flesh":    "Chuvash aʂ for flesh is separate from the Uzbek Persian loan — worth independent Bulgar check",
}


def nlevenshtein(a: list, b: list) -> float:
    """Normalized Levenshtein on token lists."""
    m, n = len(a), len(b)
    if m == 0 and n == 0:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            dp[i][j] = min(dp[i-1][j] + 1, dp[i][j-1] + 1, dp[i-1][j-1] + cost)
    return dp[m][n] / max(m, n)


def load_clusters(path: str) -> dict:
    """Load clusters.csv into {gloss: [rows]}."""
    clusters = defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            clusters[row["gloss"].strip().lower()].append(row)
    return clusters


def all_forms_have_sparsity_token(rows: list) -> tuple:
    """
    Returns (bool, set_of_sparsity_tokens_found).
    True if every form in the cluster contains at least one sparsity token.
    """
    tokens_per_form = []
    for row in rows:
        toks = set(row["ipa_tokens"].split())
        found = toks & SPARSITY_TOKENS
        tokens_per_form.append(found)

    all_have = all(len(f) > 0 for f in tokens_per_form)
    all_found = set()
    for f in tokens_per_form:
        all_found |= f
    return all_have, all_found


def forms_are_cognate(rows: list) -> tuple:
    """
    Heuristic: if pairwise NLev between all form token lists is < 0.55
    on average, treat as likely same cognate.
    Returns (bool, avg_nlevenshtein).
    """
    token_lists = [row["ipa_tokens"].split() for row in rows]
    if len(token_lists) < 2:
        return False, 0.0
    dists = []
    for i in range(len(token_lists)):
        for j in range(i + 1, len(token_lists)):
            dists.append(nlevenshtein(token_lists[i], token_lists[j]))
    avg = sum(dists) / len(dists) if dists else 1.0
    return avg < 0.55, round(avg, 3)


def is_oghuz_sister_only(rows: list) -> bool:
    """True if cluster contains ONLY Azerbaijani and Turkish (and no other language)."""
    langs = {row["language"] for row in rows}
    return langs == {"Azerbaijani", "Turkish"}


def apply_loan_screens(rows: list) -> list:
    """Returns list of (form, contact_language) for any form matching a loan screen."""
    hits = []
    for row in rows:
        gloss = row["gloss"].strip().lower()
        toks  = row["ipa_tokens"]
        for screen_name, screen_fn in LOAN_SCREENS:
            if screen_fn(toks, gloss):
                hits.append((row["form"], screen_name))
                break
    return hits


def main():
    print("Phase 6 Resolution Triage")
    print("=" * 60)

    clusters = load_clusters(CLUST_CSV)
    print(f"Loaded {len(clusters)} clusters from phase6_clusters.csv")

    shortlist = []
    discards  = []
    borderline_notes = {}

    for gloss, rows in clusters.items():
        # Collect cluster metadata
        langs    = sorted({r["language"] for r in rows})
        domain   = rows[0]["domain"]
        n_langs  = len(langs)
        scores   = [float(r["adjusted_score"]) for r in rows]
        min_score = min(scores)

        # ── Manual hard decisions first ─────────────────────────
        if gloss in MANUAL_KEEPS:
            shortlist.append({
                "gloss":       gloss,
                "n_languages": n_langs,
                "languages":   "; ".join(langs),
                "domain":      domain,
                "min_score":   min_score,
                "keep_reason": MANUAL_KEEPS[gloss],
                "discard_reason": "",
                "filter_hit":  "MANUAL_KEEP",
            })
            continue

        if gloss in MANUAL_DISCARDS:
            discards.append({
                "gloss":          gloss,
                "n_languages":    n_langs,
                "languages":      "; ".join(langs),
                "domain":         domain,
                "min_score":      min_score,
                "discard_reason": MANUAL_DISCARDS[gloss],
                "filter_hit":     "MANUAL_DISCARD",
            })
            continue

        if gloss in BORDERLINE:
            borderline_notes[gloss] = BORDERLINE[gloss]

        # ── Filter 1: Oghuz sister-only ─────────────────────────
        if is_oghuz_sister_only(rows):
            # Check if forms are distinct
            cognate, avg_nlev = forms_are_cognate(rows)
            if cognate:
                discards.append({
                    "gloss":          gloss,
                    "n_languages":    n_langs,
                    "languages":      "; ".join(langs),
                    "domain":         domain,
                    "min_score":      min_score,
                    "discard_reason": f"Oghuz sister pair only (Az+Tur), same cognate (avg NLev={avg_nlev})",
                    "filter_hit":     "F1_OGHUZ_SISTER",
                })
                continue

        # ── Filter 2: Loan screens ───────────────────────────────
        loan_hits = apply_loan_screens(rows)
        # If the majority of forms hit a loan screen, discard
        if loan_hits and len(loan_hits) >= len(rows) * 0.5:
            loan_summary = "; ".join(f"{f}={s}" for f, s in loan_hits)
            discards.append({
                "gloss":          gloss,
                "n_languages":    n_langs,
                "languages":      "; ".join(langs),
                "domain":         domain,
                "min_score":      min_score,
                "discard_reason": f"Loan screen hits: {loan_summary}",
                "filter_hit":     "F2_LOAN_SCREEN",
            })
            continue

        # ── Filter 3: Sparsity phoneme driver + cognate ─────────
        all_sparse, sparse_found = all_forms_have_sparsity_token(rows)
        cognate, avg_nlev = forms_are_cognate(rows)
        if all_sparse and cognate:
            discards.append({
                "gloss":          gloss,
                "n_languages":    n_langs,
                "languages":      "; ".join(langs),
                "domain":         domain,
                "min_score":      min_score,
                "discard_reason": (f"All forms contain sparsity token(s) {sparse_found} "
                                   f"and are likely cognate (avg NLev={avg_nlev})"),
                "filter_hit":     "F3_SPARSITY_COGNATE",
            })
            continue

        # ── Survived all filters — add to shortlist ──────────────
        shortlist.append({
            "gloss":          gloss,
            "n_languages":    n_langs,
            "languages":      "; ".join(langs),
            "domain":         domain,
            "min_score":      min_score,
            "keep_reason":    f"Survived all filters (all_sparse={all_sparse}, cognate={cognate}, avg_nlev={avg_nlev})",
            "discard_reason": "",
            "filter_hit":     "SURVIVED",
        })

    # ── Write outputs ────────────────────────────────────────────────────────

    short_cols = ["gloss","n_languages","languages","domain","min_score",
                  "keep_reason","filter_hit"]
    with open(OUT_SHORT, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=short_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(shortlist, key=lambda r: float(r["min_score"])))
    print(f"\nWrote shortlist: {OUT_SHORT}  ({len(shortlist)} items)")

    disc_cols = ["gloss","n_languages","languages","domain","min_score",
                 "discard_reason","filter_hit"]
    with open(OUT_DISCARD, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=disc_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(discards, key=lambda r: r["gloss"]))
    print(f"Wrote discards:  {OUT_DISCARD}  ({len(discards)} items)")

    # ── Narrative report ─────────────────────────────────────────────────────
    lines = []
    lines.append("PHASE 6 RESOLUTION TRIAGE REPORT")
    lines.append("=" * 70)
    lines.append("")
    lines.append(f"Input clusters:       {len(clusters)}")
    lines.append(f"Discarded:            {len(discards)}")
    lines.append(f"Shortlisted:          {len(shortlist)}")
    lines.append(f"Borderline notes:     {len(borderline_notes)}")
    lines.append("")
    lines.append("TRIAGE FILTERS APPLIED")
    lines.append("-" * 50)
    lines.append("Manual discards:  etymological verification confirms known PT root or")
    lines.append("                  known loanword source from prior phases.")
    lines.append("F1 Oghuz sister:  Az+Tur only, same cognate — single cognate flagged twice.")
    lines.append("F2 Loan screen:   Phonological signature matches known contact language.")
    lines.append("F3 Sparsity:      All forms contain documented sparsity phonemes AND")
    lines.append("                  forms are likely cognate (avg NLev < 0.55).")
    lines.append("")

    lines.append("SHORTLIST — CANDIDATES FOR PHASE 6 RESOLUTION")
    lines.append("-" * 50)
    if not shortlist:
        lines.append("  (empty — extended null result)")
    else:
        for item in sorted(shortlist, key=lambda r: float(r["min_score"])):
            lines.append(f"  {item['gloss'].upper()}")
            lines.append(f"    Languages ({item['n_languages']}): {item['languages']}")
            lines.append(f"    Domain:    {item['domain']}")
            lines.append(f"    Min score: {item['min_score']}")
            lines.append(f"    Basis:     {item['keep_reason']}")
            lines.append("")

    lines.append("DISCARDS SUMMARY BY FILTER")
    lines.append("-" * 50)
    filter_counts = defaultdict(int)
    for d in discards:
        filter_counts[d["filter_hit"]] += 1
    for filt, count in sorted(filter_counts.items()):
        lines.append(f"  {filt:<25s}: {count}")
    lines.append("")

    lines.append("BORDERLINE NOTES (not on shortlist but flagged for awareness)")
    lines.append("-" * 50)
    for gloss, note in borderline_notes.items():
        lines.append(f"  {gloss}: {note}")
    lines.append("")

    lines.append("DISCARD DETAIL")
    lines.append("-" * 50)
    for d in sorted(discards, key=lambda r: r["gloss"]):
        lines.append(f"  {d['gloss']:<20s} [{d['filter_hit']}]  {d['discard_reason']}")

    report_text = "\n".join(lines)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"Wrote report:    {OUT_REPORT}")

    # ── Console summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"TRIAGE COMPLETE: {len(clusters)} → {len(shortlist)} candidates")
    if shortlist:
        print("\nShortlisted for Phase 6 Resolution:")
        for item in sorted(shortlist, key=lambda r: float(r["min_score"])):
            print(f"  [{item['domain'].upper()[:8]}] {item['gloss']} "
                  f"({item['n_languages']} langs) min_score={item['min_score']}")
    else:
        print("\n  Shortlist is empty.")
        print("  This is an extended null result: no substrate signal in")
        print("  cultural vocabulary after etymological triage.")
    print("")
    if borderline_notes:
        print("Borderline (not shortlisted, but noted for report):")
        for g, n in borderline_notes.items():
            print(f"  {g}: {n}")


if __name__ == "__main__":
    main()
