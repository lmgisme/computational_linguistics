"""
Step 1c — Task 4: Concept Overlap Assessment
=============================================
Compares EDAL meanings against the NorthEuraLex 933-concept set to determine
how much aligned data we have for the pipeline.

Also assesses proto-form availability per family for the overlapping concepts.

Output: output/concept_overlap_report.txt

Usage:
    python step1c_concept_overlap.py
"""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\lmgisme\Desktop\computational_linguistics")
OUTPUT = BASE / "output"

MASTER_CSV = OUTPUT / "edal_altaic_master.csv"
NEL_CSV = OUTPUT / "northeuralex_transeurasian.csv"

# Proto-form columns
FAMILY_PROTO_COLS = {
    "turkic": "proto_turkic",
    "mongolic": "proto_mongolic",
    "tungusic": "proto_tungusic",
    "korean": "proto_korean",
    "japanese": "proto_japanese",
}


def normalize_gloss(g):
    """Normalize a gloss for fuzzy matching."""
    g = g.lower().strip()
    # Remove articles
    g = re.sub(r'\b(a|an|the|to|of)\b', '', g)
    # Remove parenthetical content
    g = re.sub(r'\(.*?\)', '', g)
    # Remove punctuation
    g = re.sub(r'[,;:!?]', '', g)
    g = re.sub(r'\s+', ' ', g).strip()
    return g


def extract_edal_glosses(meaning):
    """
    Extract individual meaning glosses from an EDAL meaning field.
    EDAL uses comma and semicolon separators.
    e.g. "rain; air" -> ["rain", "air"]
    """
    if not meaning:
        return []
    # Split on semicolons and commas
    parts = re.split(r'[;,]', meaning)
    glosses = []
    for p in parts:
        p = p.strip()
        if p:
            glosses.append(normalize_gloss(p))
    return glosses


def main():
    print("=" * 70)
    print("Step 1c — Task 4: Concept Overlap Assessment")
    print("=" * 70)
    
    # ---- Load NorthEuraLex glosses ----
    print("\nLoading NorthEuraLex...")
    nel_glosses = set()
    nel_langs = defaultdict(set)  # lang -> set of glosses
    with open(NEL_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row["gloss"].strip()
            nel_glosses.add(gloss)
            nel_glosses.add(normalize_gloss(gloss))
            nel_langs[row["language"]].add(gloss)
    
    print(f"  NorthEuraLex concepts: {len(nel_glosses)} (normalized)")
    print(f"  Languages: {', '.join(sorted(nel_langs.keys()))}")
    for lang, glosses in sorted(nel_langs.items()):
        print(f"    {lang}: {len(glosses)} concepts")
    
    # ---- Load EDAL master ----
    print("\nLoading EDAL master...")
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        edal_rows = list(reader)
    print(f"  EDAL records: {len(edal_rows)}")
    
    # ---- Attempt matching ----
    # Strategy: For each EDAL record, check if any of its meaning glosses
    # match a NorthEuraLex concept (exact or normalized).
    
    exact_matches = []
    no_match = []
    
    for row in edal_rows:
        meaning = row.get("meaning", "")
        edal_glosses = extract_edal_glosses(meaning)
        
        matched = False
        matched_nel_gloss = None
        for eg in edal_glosses:
            if eg in nel_glosses:
                matched = True
                matched_nel_gloss = eg
                break
            # Try singular/plural variants
            if eg.endswith('s') and eg[:-1] in nel_glosses:
                matched = True
                matched_nel_gloss = eg[:-1]
                break
            if eg + 's' in nel_glosses:
                matched = True
                matched_nel_gloss = eg + 's'
                break
        
        if matched:
            # Check which families have proto-forms
            families_present = []
            for fam, col in FAMILY_PROTO_COLS.items():
                if row.get(col, "").strip():
                    families_present.append(fam)
            exact_matches.append({
                "record_id": row["record_id"],
                "edal_meaning": meaning,
                "matched_gloss": matched_nel_gloss,
                "families": families_present,
                "n_families": len(families_present),
            })
        else:
            no_match.append({
                "record_id": row["record_id"],
                "edal_meaning": meaning,
            })
    
    # ---- Analysis ----
    print(f"\n{'=' * 70}")
    print(f"OVERLAP RESULTS")
    print(f"{'=' * 70}")
    print(f"  EDAL records:            {len(edal_rows)}")
    print(f"  Matched to NorthEuraLex: {len(exact_matches)}")
    print(f"  Unmatched:               {len(no_match)}")
    print(f"  Match rate:              {len(exact_matches)/len(edal_rows)*100:.1f}%")
    
    # Family coverage within matched set
    family_counts = Counter()
    family_combo_counts = Counter()
    for m in exact_matches:
        for fam in m["families"]:
            family_counts[fam] += 1
        combo = tuple(sorted(m["families"]))
        family_combo_counts[combo] += 1
    
    print(f"\n  Proto-form availability within matched concepts:")
    for fam in ["turkic", "mongolic", "tungusic", "korean", "japanese"]:
        c = family_counts[fam]
        pct = c / len(exact_matches) * 100 if exact_matches else 0
        print(f"    {fam:12s}: {c:5d} / {len(exact_matches)} ({pct:.1f}%)")
    
    # N-family coverage
    nfam_counts = Counter(m["n_families"] for m in exact_matches)
    print(f"\n  Records by number of families with proto-forms:")
    for n in sorted(nfam_counts.keys()):
        c = nfam_counts[n]
        print(f"    {n} families: {c:5d} ({c/len(exact_matches)*100:.1f}%)")
    
    # Records with >= 3 families (most useful for cross-family comparison)
    n_useful = sum(c for n, c in nfam_counts.items() if n >= 3)
    print(f"\n  Records with >= 3 families: {n_useful} "
          f"({n_useful/len(exact_matches)*100:.1f}% of matches)")
    
    # ---- WRITE REPORT ----
    lines = []
    lines.append("=" * 78)
    lines.append("Step 1c — Concept Overlap Assessment")
    lines.append("EDAL meanings vs. NorthEuraLex 933-concept set")
    lines.append("=" * 78)
    lines.append("")
    lines.append(f"EDAL total records:        {len(edal_rows)}")
    lines.append(f"Matched to NorthEuraLex:   {len(exact_matches)}")
    lines.append(f"Unmatched:                 {len(no_match)}")
    lines.append(f"Match rate:                {len(exact_matches)/len(edal_rows)*100:.1f}%")
    lines.append("")
    
    lines.append("NorthEuraLex languages available:")
    for lang, glosses in sorted(nel_langs.items()):
        lines.append(f"  {lang}: {len(glosses)} concepts")
    lines.append("")
    
    lines.append("Proto-form availability within matched concepts:")
    for fam in ["turkic", "mongolic", "tungusic", "korean", "japanese"]:
        c = family_counts[fam]
        pct = c / len(exact_matches) * 100 if exact_matches else 0
        lines.append(f"  {fam:12s}: {c:5d} / {len(exact_matches)} ({pct:.1f}%)")
    lines.append("")
    
    lines.append("Records by number of families with proto-forms:")
    for n in sorted(nfam_counts.keys()):
        c = nfam_counts[n]
        lines.append(f"  {n} families: {c:5d} ({c/len(exact_matches)*100:.1f}%)")
    lines.append("")
    lines.append(f"Records with >= 3 families: {n_useful}")
    lines.append("")
    
    # List matched concepts
    lines.append("-" * 78)
    lines.append("MATCHED CONCEPTS (sample, first 50)")
    lines.append("-" * 78)
    for m in exact_matches[:50]:
        fams = ", ".join(m["families"]) if m["families"] else "NONE"
        lines.append(f"  [{m['record_id']:>4s}] {m['edal_meaning'][:50]:50s} -> "
                     f"{m['matched_gloss']:20s} [{fams}]")
    
    if len(exact_matches) > 50:
        lines.append(f"  ... ({len(exact_matches) - 50} more)")
    
    # Write
    report_path = OUTPUT / "concept_overlap_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to: {report_path}")
    
    # Also write matched concepts CSV for downstream use
    csv_path = OUTPUT / "edal_nel_matched_concepts.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["edal_record_id", "edal_meaning", "nel_gloss",
                         "n_families", "turkic", "mongolic", "tungusic",
                         "korean", "japanese"])
        for m in exact_matches:
            writer.writerow([
                m["record_id"],
                m["edal_meaning"],
                m["matched_gloss"],
                m["n_families"],
                "turkic" in m["families"],
                "mongolic" in m["families"],
                "tungusic" in m["families"],
                "korean" in m["families"],
                "japanese" in m["families"],
            ])
    print(f"Matched concepts CSV: {csv_path}")
    
    print("\nDone.")


if __name__ == "__main__":
    main()
