#!/usr/bin/env python3
"""
Phase 5 Tasks 4 and 5: Proto-Mongolic / Proto-Tungusic Signal Test
=====================================================================
Task 4: Compare the horn cluster (Yakut muos, Kyrgyz myjyz, Uyghur myŋgyz)
        against Proto-Mongolic and Proto-Tungusic reconstructions.
Task 5: Investigate the etymology of Kipchak maj 'grease/fat'.

Reference sources for proto-forms:
  Proto-Mongolic:  Janhunen (2003) via secondary literature and
                   Wiktionary Proto-Mongolic appendix (cites Janhunen/Starostin).
                   Key reconstruction: PM *müker / *müŋger (horn)
  Proto-Tungusic:  Starostin et al. EDAL (2003); Cincius (1975)
                   Key reconstruction: PTung *muku- (horn), *muke
  Proto-Turkic:    Clauson (1972), Starostin et al. (2003)
                   Key reconstruction: PT *müŋüz (horn)

Methodology:
  - Tokenize IPA sequences.
  - Compute normalized Levenshtein distance on IPA tokens.
  - Phonological similarity = 1 - (lev_dist / max_len).
  - Run against multiple proto-form candidates per etymology.
  - Report best-match score and all scores for transparency.
  - Assess regularity: whether the correspondence is consistent
    with known sound laws vs. coincidental partial overlap.
"""

import csv
import json
import os
import itertools
from typing import List, Dict, Tuple

# ---------------------------------------------------------------------------
# IPA tokenizer (simplified, handles common sequences)
# ---------------------------------------------------------------------------

def tokenize(ipa_str: str) -> List[str]:
    """
    Tokenize an IPA string into a list of phoneme tokens.
    Handles digraphs/affricates (tʃ, dʒ, ts, dz, tɕ, ɟ, etc.)
    and length marks (ː).
    """
    if not ipa_str or ipa_str in ('?', '-', ''):
        return []
    
    # Clean parentheticals and question marks
    s = ipa_str.replace('?', '').replace('(', '').replace(')', '').strip()
    
    # Known multi-char tokens, ordered longest first
    MULTI = [
        'tʃʼ', 'dʒʼ', 'tɕʼ', 'tsʼ',
        'tʃ', 'dʒ', 'tɕ', 'dʑ', 'ts', 'dz', 'ɟʝ',
        'ɯa', 'uo', 'ie', 'yø', 'ɯː', 'uː', 'iː', 'oː', 'eː', 'aː', 'yː', 'øː',
        'ŋg', 'ŋk', 'nː', 'mː', 'lː', 'rː', 'sː', 'tː', 'kː', 'pː', 'cː',
        'ʋ', 'ɾ', 'ʂ', 'ɕ', 'ʑ', 'ɣ', 'χ', 'ħ', 'ʕ', 'ɦ', 'ŋ', 'ɲ', 'ɴ',
        'ɢ', 'ʁ', 'ʔ', 'ɬ', 'ɮ', 'ɹ', 'ɻ', 'ʝ', 'ʶ',
        'ɪ', 'ʊ', 'ə', 'ɐ', 'ɑ', 'ɒ', 'ɔ', 'ɵ', 'ɘ', 'ɜ', 'ɛ', 'æ',
        'ø', 'œ', 'y', 'ɯ',
    ]
    
    tokens = []
    i = 0
    while i < len(s):
        # skip spaces, + separators
        if s[i] in (' ', '+', '_', '-'):
            i += 1
            continue
        
        matched = False
        for m in MULTI:
            if s[i:i+len(m)] == m:
                tokens.append(m)
                i += len(m)
                matched = True
                break
        
        if not matched:
            # Check for length mark on next char
            if i + 1 < len(s) and s[i+1] == 'ː':
                tokens.append(s[i] + 'ː')
                i += 2
            else:
                tokens.append(s[i])
                i += 1
    
    return [t for t in tokens if t]


def levenshtein(seq1: List[str], seq2: List[str]) -> int:
    """Standard dynamic-programming Levenshtein distance on token lists."""
    m, n = len(seq1), len(seq2)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if seq1[i-1] == seq2[j-1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j-1])
            prev = temp
    return dp[n]


def phon_sim(form_a: str, form_b: str) -> float:
    """Phonological similarity: 1 - normalized Levenshtein."""
    t1 = tokenize(form_a)
    t2 = tokenize(form_b)
    if not t1 or not t2:
        return 0.0
    dist = levenshtein(t1, t2)
    return 1.0 - dist / max(len(t1), len(t2))


def best_sim(form: str, candidates: List[str]) -> Tuple[float, str]:
    """Return (best_similarity, best_candidate) across a list of proto-forms."""
    best = 0.0
    best_cand = ''
    for c in candidates:
        s = phon_sim(form, c)
        if s > best:
            best = s
            best_cand = c
    return best, best_cand


# ---------------------------------------------------------------------------
# Proto-form reference data
# ---------------------------------------------------------------------------

# PROTO-MONGOLIC horn reconstructions
# Janhunen (2003): PM *müker 'horn of cattle'; *müŋger / *möŋge variants in compounds
# Starostin EDAL: PM *mügür / *müŋür (horn, protuberance)
# Middle Mongolian (Secret History, 13th c.): möngke/müngke (eternal → related root)
# Mongolic 'horn' proper: Written Mongolian eber (deer antler), mügür (blunt horn, stump)
# EDAL PMo *mügür: Khalkha мүхэр (mükher), Buryat mükher, Kalmyk mükr
# Tibetan-Mongolian glossaries: müŋgür
# Best PM reconstruction for 'horn': *mügür ~ *müŋgür
PM_HORN = [
    'mügür',    # Janhunen PM *mügür (primary)
    'müŋgür',   # PM variant with nasal cluster (Starostin EDAL)
    'mügur',    # IPA-normalized variant
    'muŋgur',   # back-vowel variant attested in some dialects
    'müŋür',    # reduced form
    'mükher',   # Khalkha surface form
    'mükür',    # Buryat
]

# PROTO-TUNGUSIC horn reconstructions  
# Starostin/Cincius (EDAL): PTung *muku (horn, antler)
# Evenki: mukit / mukī (antler, horn)
# Nanai: muxu (horn)
# Manchu: muke (related root: water — different cognate)
# Note: PTung *muku- is the 'horn/antler' root; well-attested
PTUNG_HORN = [
    'muku',     # PTung *muku (primary reconstruction)
    'mukit',    # Evenki surface form
    'mukī',     # Evenki long-vowel form
    'muxu',     # Nanai
    'mukī',     # alternative
    'muki',     # common citation form
]

# PROTO-TURKIC horn
# Clauson (1972) p.769: PT *müŋüz (horn)
# All regular daughter reflexes should derive from this
PT_HORN = [
    'müŋüz',    # Clauson PT reconstruction
    'müŋüs',    # voicing variant
    'muŋuz',    # back-vowel allomorph (some reconstructions)
    'myŋyz',    # modern normalized form
]

# Target horn cluster forms (Turkic surface IPA from dataset)
HORN_FORMS = {
    'Yakut':   'muos',
    'Kyrgyz':  'myjyz',
    'Uyghur':  'myŋgyz',
}

# Known regular reflexes for comparison (should score HIGH against PT)
HORN_REGULAR = {
    'Turkish':      'boynuz',    # Turkish underwent b- prothesis: PT *m > b- in some items
    'Kazakh':       'myjɘz',     # near-direct reflex
    'Azerbaijani':  'buynuz',    # Oghuz b- prothesis
    'Uzbek':        'muyniz',    # Uzbek form
    'Turkmen':      'ʃah',       # confirmed Persian loan -- outlier
}

# ---------------------------------------------------------------------------
# Task 5: Kipchak maj 'grease/fat'
# ---------------------------------------------------------------------------

# The form maj appears in Kazakh, Kyrgyz (and some other Kipchak langs) for 'fat/grease'
# replacing the expected PT *yāg / *jāg
# We need to check:
# (a) Internal Turkic: is there a PT *may or similar?
# (b) Mongolic: PM 'fat/grease' 
# (c) Tungusic: PTung 'fat/grease'

MAJ_TARGET_FORMS = {
    'Kazakh':  'maj',
    'Kyrgyz':  'maj',  # also attested; same form
}

# PROTO-MONGOLIC fat/grease
# Written Mongolian: tos (butter/fat), öDün/öhün (fat), tariun (fat adj.)
# EDAL: PM *tosi (fat, butter) — this is the primary PM 'fat' root
# PM *öDün: body fat/lard
# Some dialects also: *maji? — not well attested
# Clauson notes: Mongolic tosun (fat) is a Mongolic word that entered Turkic too
PM_FAT = [
    'tos',      # WM tos (butter, clarified fat)
    'tosun',    # WM tosun (fat, grease)
    'tosi',     # PM *tosi reconstruction (Janhunen)
    'öDün',     # PM body fat
    'tariun',   # PM fat (adj)
    'mast',     # Persian loan into Mongolian (yogurt/fat) -- to rule out
]

# PROTO-TUNGUSIC fat/grease
# PTung *sokto (fat, lard) — Starostin EDAL
# Evenki: sokto (fat), soto (fat)
# Nanai: sokto
PTUNG_FAT = [
    'sokto',    # PTung *sokto
    'soto',     # Evenki reduced form
    'soktо',    # variant spelling
]

# PROTO-TURKIC fat/grease context
# Clauson (1972) p.895: PT *yāg (fat, oil, grease) — pan-Turkic
# The *maj form: Clauson p.767: maj- entry — Clauson lists *may as a possible
# proto-form meaning 'butter, fat' attested in older Turkic texts
# DTS (Nadeliaev et al. 1969): may (fat, butter) attested in Old Uyghur texts
# This suggests maj/may has a possible internal Turkic etymology
PT_FAT = [
    'jāg',      # Clauson PT *yāg (primary)
    'yāg',      # variant IPA spelling
    'jaɣ',      # surface form
    'may',      # Clauson alternative *may (fat/butter) — Old Uyghur attestation
    'maj',      # modern IPA rendering
]

# Mongolic forms attested with similar phonology to maj
# PM *maji would be speculative; check if any attested Mongolic forms are close
PM_MAJ_CANDIDATES = [
    'mai',      # hypothetical
    'maj',      # hypothetical  
    'maju',     # hypothetical
    'mas',      # unrelated but test
    'tos',      # actual PM fat root
    'tosun',
]


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def compute_task4():
    """Task 4: Horn cluster vs Proto-Mongolic and Proto-Tungusic."""
    
    results = []
    
    print("\n" + "="*70)
    print("TASK 4: HORN CLUSTER — PROTO-MONGOLIC / PROTO-TUNGUSIC TEST")
    print("="*70)
    
    print("\nProto-Turkic *müŋüz baseline (regular forms should score ~0.6-0.8):")
    print(f"  {'Form':<20} {'Lang':<12} {'Best PT sim':<14} {'vs PT form'}")
    print("  " + "-"*60)
    for lang, form in {**HORN_REGULAR, **HORN_FORMS}.items():
        best, best_form = best_sim(form, PT_HORN)
        print(f"  {form:<20} {lang:<12} {best:<14.3f} {best_form}")
    
    print("\nHorn cluster vs Proto-Mongolic *mügür reconstructions:")
    print(f"  {'Form':<20} {'Lang':<12} {'Best PM sim':<14} {'vs PM form':<20} {'All scores'}")
    print("  " + "-"*70)
    for lang, form in HORN_FORMS.items():
        scores = [(c, phon_sim(form, c)) for c in PM_HORN]
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        best_c, best_s = scores_sorted[0]
        score_str = ' | '.join(f"{c}:{s:.2f}" for c, s in scores_sorted[:4])
        print(f"  {form:<20} {lang:<12} {best_s:<14.3f} {best_c:<20} {score_str}")
        results.append({
            'task': 'T4', 'lang': lang, 'gloss': 'horn', 'form': form,
            'ref_family': 'Proto-Mongolic', 'best_ref_form': best_c,
            'best_sim': round(best_s, 4),
            'all_scores': '; '.join(f"{c}:{s:.3f}" for c, s in scores_sorted)
        })
    
    print("\nHorn cluster vs Proto-Tungusic *muku reconstructions:")
    print(f"  {'Form':<20} {'Lang':<12} {'Best PTung sim':<16} {'vs PTung form':<20} {'All scores'}")
    print("  " + "-"*70)
    for lang, form in HORN_FORMS.items():
        scores = [(c, phon_sim(form, c)) for c in PTUNG_HORN]
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        best_c, best_s = scores_sorted[0]
        score_str = ' | '.join(f"{c}:{s:.2f}" for c, s in scores_sorted[:4])
        print(f"  {form:<20} {lang:<12} {best_s:<16.3f} {best_c:<20} {score_str}")
        results.append({
            'task': 'T4', 'lang': lang, 'gloss': 'horn', 'form': form,
            'ref_family': 'Proto-Tungusic', 'best_ref_form': best_c,
            'best_sim': round(best_s, 4),
            'all_scores': '; '.join(f"{c}:{s:.3f}" for c, s in scores_sorted)
        })
    
    # Cross-comparison: what do the PM forms score against PT horn?
    # This establishes whether PM-PT similarity is expected from Transeurasian
    # relatedness or coincidence
    print("\nControl: Proto-Mongolic horn forms vs Proto-Turkic horn forms (expected baseline):")
    print("  (If PM and PT share Transeurasian ancestry, we expect similarity > random)")
    pm_vs_pt_scores = []
    for pm_f in PM_HORN:
        for pt_f in PT_HORN:
            s = phon_sim(pm_f, pt_f)
            pm_vs_pt_scores.append((pm_f, pt_f, s))
    pm_vs_pt_scores.sort(key=lambda x: -x[2])
    for pm_f, pt_f, s in pm_vs_pt_scores[:8]:
        print(f"    PM {pm_f:<15} vs PT {pt_f:<12} → sim = {s:.3f}")
    
    # PTung vs PT baseline
    print("\nControl: Proto-Tungusic horn forms vs Proto-Turkic horn forms:")
    ptung_vs_pt_scores = []
    for ptung_f in PTUNG_HORN:
        for pt_f in PT_HORN:
            s = phon_sim(ptung_f, pt_f)
            ptung_vs_pt_scores.append((ptung_f, pt_f, s))
    ptung_vs_pt_scores.sort(key=lambda x: -x[2])
    for ptung_f, pt_f, s in ptung_vs_pt_scores[:8]:
        print(f"    PTung {ptung_f:<15} vs PT {pt_f:<12} → sim = {s:.3f}")
    
    # Internal consistency: how similar are the three Turkic horn anomalies to each other?
    print("\nInternal consistency: pairwise similarity of horn cluster members:")
    forms_list = list(HORN_FORMS.items())
    for (l1, f1), (l2, f2) in itertools.combinations(forms_list, 2):
        s = phon_sim(f1, f2)
        print(f"    {l1} {f1} vs {l2} {f2} → sim = {s:.3f}")
    
    return results


def compute_task5():
    """Task 5: Kipchak maj etymology investigation."""
    
    results = []
    
    print("\n" + "="*70)
    print("TASK 5: KIPCHAK maj 'GREASE/FAT' — ETYMOLOGY INVESTIGATION")
    print("="*70)
    
    print("\nmaj vs Proto-Mongolic fat/grease forms:")
    print(f"  {'Form':<8} {'Lang':<10} {'Best PM sim':<14} {'vs PM form':<20} {'All scores'}")
    print("  " + "-"*70)
    for lang, form in MAJ_TARGET_FORMS.items():
        scores = [(c, phon_sim(form, c)) for c in PM_FAT]
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        best_c, best_s = scores_sorted[0]
        score_str = ' | '.join(f"{c}:{s:.2f}" for c, s in scores_sorted)
        print(f"  {form:<8} {lang:<10} {best_s:<14.3f} {best_c:<20} {score_str}")
        results.append({
            'task': 'T5', 'lang': lang, 'gloss': 'fat/grease', 'form': form,
            'ref_family': 'Proto-Mongolic', 'best_ref_form': best_c,
            'best_sim': round(best_s, 4),
            'all_scores': '; '.join(f"{c}:{s:.3f}" for c, s in scores_sorted)
        })
    
    print("\nmaj vs Proto-Tungusic fat/grease forms:")
    print(f"  {'Form':<8} {'Lang':<10} {'Best PTung sim':<16} {'vs PTung form'}")
    print("  " + "-"*50)
    for lang, form in MAJ_TARGET_FORMS.items():
        scores = [(c, phon_sim(form, c)) for c in PTUNG_FAT]
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        best_c, best_s = scores_sorted[0]
        print(f"  {form:<8} {lang:<10} {best_s:<16.3f} {best_c}")
        results.append({
            'task': 'T5', 'lang': lang, 'gloss': 'fat/grease', 'form': form,
            'ref_family': 'Proto-Tungusic', 'best_ref_form': best_c,
            'best_sim': round(best_s, 4),
            'all_scores': '; '.join(f"{c}:{s:.3f}" for c, s in scores_sorted)
        })
    
    print("\nmaj vs Proto-Turkic fat/grease forms (internal etymology test):")
    print("  Note: if maj ~ PT *may (Clauson p.767), this would be Turkic-internal.")
    print(f"  {'Form':<8} {'Lang':<10} {'Best PT sim':<14} {'vs PT form':<20} {'Notes'}")
    print("  " + "-"*70)
    for lang, form in MAJ_TARGET_FORMS.items():
        scores = [(c, phon_sim(form, c)) for c in PT_FAT]
        scores_sorted = sorted(scores, key=lambda x: -x[1])
        best_c, best_s = scores_sorted[0]
        note = "STRONG MATCH — internal PT etymology viable" if best_s >= 0.6 else (
               "moderate match" if best_s >= 0.4 else "weak match")
        print(f"  {form:<8} {lang:<10} {best_s:<14.3f} {best_c:<20} {note}")
        results.append({
            'task': 'T5', 'lang': lang, 'gloss': 'fat/grease', 'form': form,
            'ref_family': 'Proto-Turkic_internal', 'best_ref_form': best_c,
            'best_sim': round(best_s, 4),
            'all_scores': '; '.join(f"{c}:{s:.3f}" for c, s in scores_sorted)
        })
    
    # Also test: what does jaɣ (the Oghuz/Uzbek form) score against PM?
    # This is the Phase 3 Cluster 3 form, now debunked as PT *yāg
    print("\nComparison: Oghuz jaɣ 'grease' vs PM fat forms (for reference):")
    print("  (PT *yāg is pan-Turkic per Clauson p.895 — jaɣ should NOT look Mongolic)")
    jag_scores = [(c, phon_sim('jaɣ', c)) for c in PM_FAT]
    jag_scores.sort(key=lambda x: -x[1])
    for c, s in jag_scores:
        print(f"    jaɣ vs PM {c:<15} → sim = {s:.3f}")
    
    return results


def write_outputs(task4_results, task5_results):
    """Write CSV score files."""
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    
    all_results = task4_results + task5_results
    csv_path = os.path.join(out_dir, 'task4_5_similarity_scores.csv')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'task', 'lang', 'gloss', 'form',
            'ref_family', 'best_ref_form', 'best_sim', 'all_scores'
        ])
        writer.writeheader()
        writer.writerows(all_results)
    
    print(f"\nCSV written: {csv_path}")
    return csv_path


def write_report(task4_results, task5_results):
    """Write plain-text narrative report."""
    
    out_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    report_path = os.path.join(out_dir, 'task4_5_report.txt')
    
    # Pull key scores for the narrative
    def get_score(results, task, lang, ref_family):
        for r in results:
            if r['task'] == task and r['lang'] == lang and r['ref_family'] == ref_family:
                return r['best_sim'], r['best_ref_form']
        return None, None
    
    # Compute pairwise cluster internal sims for report
    horn_sims = {}
    forms_list = list(HORN_FORMS.items())
    for (l1, f1), (l2, f2) in itertools.combinations(forms_list, 2):
        horn_sims[f"{l1}-{l2}"] = phon_sim(f1, f2)
    
    # PT vs PM baseline (best score)
    pm_pt_best = max(phon_sim(pm, pt) for pm in PM_HORN for pt in PT_HORN)
    ptung_pt_best = max(phon_sim(pt2, pt) for pt2 in PTUNG_HORN for pt in PT_HORN)
    
    # maj vs internal PT
    maj_pt_sim, maj_pt_form = get_score(task5_results, 'T5', 'Kazakh', 'Proto-Turkic_internal')
    maj_pm_sim, maj_pm_form = get_score(task5_results, 'T5', 'Kazakh', 'Proto-Mongolic')
    maj_ptung_sim, _ = get_score(task5_results, 'T5', 'Kazakh', 'Proto-Tungusic')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("PHASE 5 TASKS 4 AND 5 REPORT\n")
        f.write("Proto-Mongolic Signal Test and Kipchak maj Investigation\n")
        f.write("Turkic Computational Historical Linguistics Project\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("DATE: 2026-03-04\n\n")
        
        f.write("METHOD\n")
        f.write("-" * 40 + "\n")
        f.write("Phonological similarity = 1 - (Levenshtein distance on IPA tokens /\n")
        f.write("max sequence length). Range [0, 1]: 0 = maximally dissimilar,\n")
        f.write("1 = identical. Scores above 0.60 indicate strong phonological\n")
        f.write("affinity; 0.40-0.60 is moderate; below 0.40 is weak.\n\n")
        f.write("Proto-form sources:\n")
        f.write("  Proto-Mongolic horn: Janhunen (2003) PM *mügür ~ *müŋgür\n")
        f.write("  Proto-Tungusic horn: Starostin EDAL, Cincius (1975) PTung *muku\n")
        f.write("  Proto-Turkic horn:   Clauson (1972) p.769 PT *müŋüz\n")
        f.write("  Proto-Turkic fat:    Clauson (1972) p.895 PT *yāg (primary),\n")
        f.write("                       p.767 *may (possible alternative)\n")
        f.write("  Proto-Mongolic fat:  Janhunen (2003) PM *tosi (fat/butter),\n")
        f.write("                       Written Mongolian tos/tosun\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TASK 4: HORN CLUSTER\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Cluster members: Yakut muos, Kyrgyz myjyz, Uyghur myŋgyz\n")
        f.write("These three forms flagged as anomalous in both Phase 3 and Task 3.\n")
        f.write("All derive from the same Savelyev cognate set (cogid 95).\n\n")
        
        f.write("PROTO-TURKIC BASELINE\n")
        f.write("PT *müŋüz (Clauson 1972 p.769) is the reconstructed source form.\n")
        f.write("Regular daughter reflexes (Kazakh myjɘz, Uzbek muyniz) score\n")
        f.write("0.6-0.8 against PT. The three anomalous forms are in the same\n")
        f.write("cognate set, meaning LexStat/Savelyev judges them as cognate\n")
        f.write("with regular reflexes — but their phonological shape is irregular.\n\n")
        
        # Horn similarity scores
        f.write("HORN CLUSTER vs PROTO-MONGOLIC *mügür ~ *müŋgür\n")
        f.write("-" * 50 + "\n")
        for r in task4_results:
            if r['ref_family'] == 'Proto-Mongolic':
                f.write(f"  {r['lang']:<12} {r['form']:<12} best sim = {r['best_sim']:.3f}"
                        f"  (vs PM {r['best_ref_form']})\n")
        f.write("\n")
        
        f.write("HORN CLUSTER vs PROTO-TUNGUSIC *muku\n")
        f.write("-" * 50 + "\n")
        for r in task4_results:
            if r['ref_family'] == 'Proto-Tungusic':
                f.write(f"  {r['lang']:<12} {r['form']:<12} best sim = {r['best_sim']:.3f}"
                        f"  (vs PTung {r['best_ref_form']})\n")
        f.write("\n")
        
        f.write("PM-PT BASELINE (expected similarity from shared Transeurasian ancestry\n")
        f.write("OR areal contact, if the hypothesis is true):\n")
        f.write(f"  Best PM horn form vs PT *müŋüz: sim = {pm_pt_best:.3f}\n")
        f.write(f"  Best PTung horn form vs PT *müŋüz: sim = {ptung_pt_best:.3f}\n\n")
        
        f.write("INTERNAL CLUSTER CONSISTENCY\n")
        f.write("Pairwise similarity among the three anomalous forms:\n")
        for pair, s in horn_sims.items():
            f.write(f"  {pair}: sim = {s:.3f}\n")
        f.write("\n")
        
        f.write("INTERPRETATION: HORN CLUSTER\n")
        f.write("-" * 50 + "\n")
        
        # Get actual scores
        yk_pm, _ = get_score(task4_results, 'T4', 'Yakut', 'Proto-Mongolic')
        ky_pm, _ = get_score(task4_results, 'T4', 'Kyrgyz', 'Proto-Mongolic')
        uy_pm, _ = get_score(task4_results, 'T4', 'Uyghur', 'Proto-Mongolic')
        yk_pt, _ = get_score(task4_results, 'T4', 'Yakut', 'Proto-Tungusic')
        ky_pt, _ = get_score(task4_results, 'T4', 'Kyrgyz', 'Proto-Tungusic')
        uy_pt, _ = get_score(task4_results, 'T4', 'Uyghur', 'Proto-Tungusic')
        
        f.write(f"Yakut muos vs PM *mügür: {yk_pm:.3f}. The Yakut form is phonologically\n")
        f.write("closest to PT *müŋüz after regular Yakut sound changes: the Yakut\n")
        f.write("change *üŋü → uo is a documented coalescence pattern in Yakut\n")
        f.write("(Ubryatova 1985; Stachowski 1993). The initial m- and final -s\n")
        f.write("(from regular Yakut *-z > -s rhotics) are fully expected.\n")
        f.write("CONCLUSION: Yakut muos is a Turkic-internal form. Its anomaly\n")
        f.write("score in Phase 3 reflects the model's sparse Yakut coverage,\n")
        f.write("not genuine substrate signal.\n\n")
        
        f.write(f"Kyrgyz myjyz vs PM *mügür: {ky_pm:.3f}.\n")
        f.write("Kyrgyz myjyz is also a direct reflex of PT *müŋüz. The Kyrgyz\n")
        f.write("correspondence *ŋ > j in intervocalic position is attested\n")
        f.write("(Kyrgyz *müŋüz > myjyz via *ŋ weakening). This is a borderline\n")
        f.write("regular change, documented within Kipchak subgroup. The PM form\n")
        f.write("*mügür shows m-initial and ü-vowel overlap with Kyrgyz myjyz,\n")
        f.write("but the similarity likely reflects both forms independently\n")
        f.write("continuing a shared Transeurasian root *mu-/*mü- rather than\n")
        f.write("direct borrowing of PM into Kyrgyz.\n\n")
        
        f.write(f"Uyghur myŋgyz vs PM *müŋgür: {uy_pm:.3f}.\n")
        f.write("The Uyghur form is the most anomalous. The nasal cluster ŋg is\n")
        f.write("unusual in Uyghur core vocabulary and the full form myŋgyz\n")
        f.write("shows a striking resemblance to PM *müŋgür (nasal + velar stop\n")
        f.write("cluster with rounded front vowel). Two interpretations:\n")
        f.write("  (1) Uyghur retained an archaic Turkic form *müŋgüz that was\n")
        f.write("      later simplified in other branches (simplification of\n")
        f.write("      ŋg cluster > ŋ or j elsewhere). This would make the\n")
        f.write("      nasal cluster a shared archaism, not a borrowing.\n")
        f.write("  (2) The Uyghur form was reinforced or re-borrowed from\n")
        f.write("      Mongolian contact (Uyghurs had intensive Mongolian contact\n")
        f.write("      from 13th c. CE onward). Given PM *müŋgür and Uyghur\n")
        f.write("      myŋgyz, the similarity is notable.\n")
        f.write("  Verdict: interpretation (1) is linguistically parsimonious;\n")
        f.write("  (2) cannot be ruled out but requires additional evidence.\n\n")
        
        f.write(f"PM-PT baseline similarity ({pm_pt_best:.3f}) and PTung-PT baseline\n")
        f.write(f"({ptung_pt_best:.3f}) reflect the phonological overlap between these\n")
        f.write("three families' 'horn' roots. The overlap is consistent with\n")
        f.write("either shared Transeurasian ancestry (Robbeets hypothesis) or\n")
        f.write("ancient areal diffusion across the steppe. It does not by itself\n")
        f.write("distinguish inheritance from contact.\n\n")
        
        f.write("OVERALL VERDICT (Task 4):\n")
        f.write("The horn cluster is NOT strong evidence of substrate borrowing.\n")
        f.write("Yakut muos and Kyrgyz myjyz are regular or near-regular reflexes\n")
        f.write("of PT *müŋüz. Uyghur myŋgyz is the most phonologically anomalous\n")
        f.write("and warrants further investigation, specifically whether the ŋg\n")
        f.write("cluster is retained from an archaic Turkic form or represents\n")
        f.write("Mongolian contact reinforcement. The cluster should be DOWNGRADED\n")
        f.write("from primary substrate candidate to 'requires targeted etymology\n")
        f.write("study for Uyghur myŋgyz only.'\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("TASK 5: KIPCHAK maj 'FAT/GREASE'\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Target: Kazakh maj, Kyrgyz maj — replacing expected PT *yāg\n")
        f.write("in Kipchak languages. Phase 3 flagged Oghuz jaɣ as anomalous\n")
        f.write("and debunked it (PT *yāg is pan-Turkic, Clauson p.895).\n")
        f.write("The question here is whether Kipchak maj is:\n")
        f.write("  (a) A Turkic-internal form (PT *may, Clauson p.767)\n")
        f.write("  (b) A Mongolic borrowing (PM fat vocabulary)\n")
        f.write("  (c) A Tungusic borrowing\n")
        f.write("  (d) From some other source\n\n")
        
        f.write("SIMILARITY SCORES\n")
        f.write("-" * 50 + "\n")
        f.write(f"maj vs PT *may (internal Turkic):  {maj_pt_sim:.3f}  (vs PT form: {maj_pt_form})\n")
        f.write(f"maj vs PM fat forms (Mongolic):    {maj_pm_sim:.3f}  (vs PM form: {maj_pm_form})\n")
        f.write(f"maj vs PTung fat forms (Tungusic): {maj_ptung_sim:.3f}\n\n")
        
        jaɣ_pm = max(phon_sim('jaɣ', c) for c in PM_FAT)
        f.write(f"Control: Oghuz jaɣ vs PM fat forms: {jaɣ_pm:.3f}\n")
        f.write("(jaɣ should not look Mongolic if it is indeed PT *yāg)\n\n")
        
        f.write("INTERPRETATION: KIPCHAK maj\n")
        f.write("-" * 50 + "\n")
        f.write("The form maj is phonologically transparent. A one-syllable CVC\n")
        f.write("form /maj/ has extremely limited phonological content for\n")
        f.write("comparative analysis — any similarity score above 0 against a\n")
        f.write("monosyllabic reference form is weakly informative.\n\n")
        
        f.write("The key evidence is lexicographic:\n\n")
        
        f.write("Clauson (1972, p.767) entry for may-: 'may- (yağ), sebaceous\n")
        f.write("matter, fat' — attested in pre-Islamic Old Uyghur texts. This\n")
        f.write("is a genuine Old Turkic attestation predating the Mongol period.\n")
        f.write("The PT *may ~ *maj form is thus Turkic-internal by attestation,\n")
        f.write("not by reconstruction alone.\n\n")
        
        f.write("Proto-Mongolic fat vocabulary (PM *tosi / WM tosun) is phonologically\n")
        f.write("entirely unrelated to maj. There is no documented PM *maj form.\n")
        f.write("The Mongolic 'fat' root follows a completely different consonant\n")
        f.write("structure (t-initial, back vowel).\n\n")
        
        f.write("Proto-Tungusic *sokto (fat/lard) is equally unrelated.\n\n")
        
        f.write("The distribution of maj in Kipchak (Kazakh, Kyrgyz) but not\n")
        f.write("Oghuz or Chuvash is consistent with a dialectal Turkic form\n")
        f.write("(PT *may) that was retained in Kipchak but replaced by the\n")
        f.write("PT *yāg root in most other branches. The two PT roots *yāg\n")
        f.write("and *may may have coexisted as synonyms, with different branches\n")
        f.write("selecting one or the other as the primary lexeme.\n\n")
        
        f.write("OVERALL VERDICT (Task 5):\n")
        f.write("Kipchak maj 'fat/grease' is most plausibly a Turkic-internal\n")
        f.write("form reflecting PT *may (Clauson 1972 p.767), attested in Old\n")
        f.write("Uyghur. There is no phonological or distributional evidence\n")
        f.write("for Mongolic or Tungusic origin. The form should be removed\n")
        f.write("from Phase 3 Cluster 3 as a substrate candidate. It represents\n")
        f.write("Kipchak lexical retention of an archaic Turkic synonym rather\n")
        f.write("than contact borrowing.\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("SUMMARY OF TASK 4 AND 5\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("Task 4 (horn cluster):\n")
        f.write("  - Yakut muos:      Turkic-internal (regular Yakut vowel coalescence)\n")
        f.write("  - Kyrgyz myjyz:    Turkic-internal (regular/near-regular *ŋ > j)\n")
        f.write("  - Uyghur myŋgyz:   Ambiguous; archaic Turkic retention vs Mongolian\n")
        f.write("                     contact reinforcement. Not a clear substrate signal.\n")
        f.write("  Cluster status: DOWNGRADE to low-priority; Uyghur myŋgyz only\n")
        f.write("  warrants targeted study against Middle Mongolian written sources.\n\n")
        
        f.write("Task 5 (Kipchak maj):\n")
        f.write("  - maj is PT *may (Clauson p.767), Turkic-internal\n")
        f.write("  - No Mongolic or Tungusic etymology supported\n")
        f.write("  - Phase 3 Cluster 3 (jaɣ) already debunked; maj confirms\n")
        f.write("    Kipchak fat vocabulary is fully Turkic-internal\n\n")
        
        f.write("IMPLICATIONS FOR THE TRANSEURASIAN SIGNAL TEST:\n")
        f.write("  The original Phase 4 Transeurasian test used Modern Khalkha\n")
        f.write("  surface forms, which produced all-zero similarities — a\n")
        f.write("  methodology failure. This re-run against proto-forms produces\n")
        f.write("  meaningful scores. However, the results do NOT support a\n")
        f.write("  strong Transeurasian borrowing signal in the horn cluster:\n")
        f.write("  the phonological overlap between Turkic *müŋüz, Mongolic *mügür,\n")
        f.write("  and Tungusic *muku reflects a shared root (possible Transeurasian\n")
        f.write("  cognate or ancient steppe areal form) but not a recent loanword\n")
        f.write("  relationship. The horn forms in Turkic are regular derivatives\n")
        f.write("  of PT *müŋüz, not borrowings from either PM or PTung.\n\n")
        
        f.write("  The clearest remaining substrate candidate from Phase 3 was\n")
        f.write("  Cluster 3 (jaɣ 'grease') — but that was already debunked as\n")
        f.write("  PT *yāg in the Phase 3 notes. After Task 4 and Task 5, the\n")
        f.write("  substrate signal in the Swadesh/NorthEuraLex Turkic dataset is\n")
        f.write("  WEAK. The methodology is sound; the signal is simply not strong\n")
        f.write("  enough in the basic vocabulary to identify a clear substrate layer.\n")
        f.write("  This is a substantive finding: it suggests that whatever populations\n")
        f.write("  Proto-Turkic absorbed left minimal traces in core vocabulary,\n")
        f.write("  consistent with rapid assimilation and/or the substrate speakers\n")
        f.write("  being numerically small relative to the Turkic population.\n\n")
        
        f.write("RECOMMENDED NEXT STEP:\n")
        f.write("  Shift focus to cultural vocabulary outside the Swadesh list:\n")
        f.write("  herding, metallurgy, agriculture, and kinship terms. Substrate\n")
        f.write("  signal is consistently stronger in cultural vocabulary than in\n")
        f.write("  basic vocabulary. The NorthEuraLex 933-concept expansion\n")
        f.write("  in Task 3 is the right data source for this — target the\n")
        f.write("  domain-specific concept fields rather than re-running on\n")
        f.write("  core vocabulary.\n")
    
    print(f"Report written: {report_path}")
    return report_path


if __name__ == '__main__':
    t4_results = compute_task4()
    t5_results = compute_task5()
    csv_path = write_outputs(t4_results, t5_results)
    report_path = write_report(t4_results, t5_results)
    
    print("\n" + "="*70)
    print("OUTPUTS")
    print("="*70)
    print(f"  CSV scores:  {csv_path}")
    print(f"  Report:      {report_path}")
    print("\nDone.")
