"""
Phase 5 Task 2: NorthEuraLex expansion
Expands the lexical dataset from 40-item Swadesh to NorthEuraLex concepts
for all available Turkic languages in the cached northeuralex CLDF data.

NorthEuraLex coverage for our 9 target languages:
  PRESENT:  Turkish (tur), Azerbaijani (azj), Uzbek (uzn), Kazakh (kaz),
            Yakut/Sakha (sah), Chuvash (chv)
  ABSENT:   Kyrgyz, Uyghur, Turkmen  [not in NorthEuraLex]

Strategy:
  - Extract all NorthEuraLex concepts for the 6 available languages
  - Merge with hybrid_cognates.csv (40-item base), preserving all existing rows
  - New concepts from NorthEuraLex are added as northeuralex-sourced rows
  - Existing Swadesh concepts: keep hybrid_cognates rows as ground truth,
    add NorthEuraLex forms for the 3 languages missing from hybrid (none missing,
    but NorthEuraLex may have alternate forms — flag but do not replace)
  - Output: northeuralex_merged.csv

Run from project root with venv311 active:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python phase5_substrate\\task2_northeuralex_expand.py

Output: output\\northeuralex_merged.csv
"""

import os
import csv
import json
from collections import defaultdict

BASE    = r"C:\Users\lmgisme\Desktop\computational_linguistics"
OUTPUT  = os.path.join(BASE, "output")
NEL_DIR = os.path.join(BASE, "phase1_ingestion", "output", "lexibank_cache", "northeuralex")

HYBRID_CSV  = os.path.join(OUTPUT, "hybrid_cognates.csv")
MERGED_CSV  = os.path.join(OUTPUT, "northeuralex_merged.csv")

# NorthEuraLex language IDs -> our canonical language names
NEL_LANG_MAP = {
    "tur": "Turkish",
    "azj": "Azerbaijani",
    "uzn": "Uzbek",
    "kaz": "Kazakh",
    "sah": "Yakut",
    "chv": "Chuvash",
}

# Concepts to EXCLUDE from NorthEuraLex expansion:
# month names, weekday names, numbers above 10, and abstract/cultural concepts
# that have no place in a substrate detection dataset.
# We keep: body parts, nature, animals, basic actions, basic adjectives,
# kinship, basic nouns — i.e. anything with reasonable Swadesh-adjacent value.
# Exclusion is by NorthEuraLex parameter ID prefix ranges.
EXCLUDE_CONCEPT_PREFIXES = [
    "462_", "463_", "464_", "465_", "466_", "467_", "468_", "469_",  # months Jan-Aug
    "470_", "471_", "472_", "473_",                                    # months Sep-Dec
    "474_", "475_", "476_", "477_", "478_", "479_", "480_",           # weekdays
    "663_", "664_", "665_", "666_", "667_", "668_", "669_",           # 11-69
    "670_", "671_", "672_", "673_", "674_",                            # 70-1000
]

def load_nel_forms():
    """Load NorthEuraLex forms for our 6 target languages."""
    forms_path = os.path.join(NEL_DIR, "forms.csv")
    params_path = os.path.join(NEL_DIR, "parameters.csv")

    # Load parameter ID -> concept name mapping
    param_map = {}  # param_id -> (concept_name, concepticon_gloss)
    with open(params_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            pid = row["ID"]
            # Skip excluded concepts
            skip = any(pid.startswith(pfx) for pfx in EXCLUDE_CONCEPT_PREFIXES)
            if not skip:
                param_map[pid] = row["Name"]

    # Load forms
    nel_data = []  # list of dicts
    with open(forms_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang_id = row["Language_ID"]
            if lang_id not in NEL_LANG_MAP:
                continue
            param_id = row["Parameter_ID"]
            if param_id not in param_map:
                continue
            segments = row.get("Segments", "").strip()
            form     = row.get("Form", "").strip()
            if not segments and not form:
                continue
            nel_data.append({
                "language":    NEL_LANG_MAP[lang_id],
                "nel_lang_id": lang_id,
                "gloss":       param_map[param_id],
                "nel_param_id": param_id,
                "form":        form,
                "ipa_tokens":  segments,
                "source":      "northeuralex",
            })

    print(f"Loaded {len(nel_data)} NorthEuraLex form entries for target languages.")
    return nel_data

def load_hybrid():
    """Load existing hybrid_cognates.csv."""
    rows = []
    with open(HYBRID_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Loaded {len(rows)} rows from hybrid_cognates.csv.")
    return rows

def main():
    hybrid_rows = load_hybrid()
    nel_forms   = load_nel_forms()

    # Build set of (language, gloss) already covered in hybrid
    hybrid_covered = set()
    for row in hybrid_rows:
        hybrid_covered.add((row["language"].strip().lower(), row["gloss"].strip().lower()))

    # Determine which hybrid columns exist
    hybrid_cols = list(hybrid_rows[0].keys()) if hybrid_rows else []
    print(f"Hybrid columns: {hybrid_cols}")

    # Build new rows from NorthEuraLex for concepts NOT in hybrid
    new_rows = []
    skipped_covered = 0
    lang_concept_counts = defaultdict(int)

    for entry in nel_forms:
        key = (entry["language"].lower(), entry["gloss"].lower())
        if key in hybrid_covered:
            skipped_covered += 1
            continue  # hybrid already has this language+gloss

        # Build a row compatible with hybrid_cognates schema
        new_row = {col: "" for col in hybrid_cols}
        new_row["language"]    = entry["language"]
        new_row["gloss"]       = entry["gloss"]
        new_row["form"]        = entry["form"]
        new_row["ipa_tokens"]  = entry["ipa_tokens"]
        new_row["cogid"]       = ""        # to be assigned in Task 3 by LexStat
        new_row["cogid_source"] = "northeuralex"
        new_row["loan_source"] = ""

        # carry any extra cols that exist
        if "nel_param_id" in hybrid_cols:
            new_row["nel_param_id"] = entry["nel_param_id"]

        new_rows.append(new_row)
        lang_concept_counts[entry["language"]] += 1

    print(f"\nSkipped {skipped_covered} entries already covered by hybrid (Swadesh overlap).")
    print(f"New NorthEuraLex-only rows to add: {len(new_rows)}")
    print("\nNew concept counts by language:")
    for lang, count in sorted(lang_concept_counts.items()):
        print(f"  {lang}: {count} new concepts")

    # Combine: hybrid first, then new NEL rows
    all_rows = hybrid_rows + new_rows
    total = len(all_rows)
    print(f"\nTotal rows in merged dataset: {total}")

    # Write output — use hybrid columns, drop nel_param_id if not in hybrid schema
    out_cols = hybrid_cols[:]

    with open(MERGED_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nWrote merged dataset to: {MERGED_CSV}")

    # Coverage summary
    print("\n--- Coverage summary ---")
    lang_totals = defaultdict(int)
    for row in all_rows:
        lang_totals[row["language"]] += 1
    for lang, cnt in sorted(lang_totals.items()):
        print(f"  {lang}: {cnt} total entries")

    nel_only_langs = set(NEL_LANG_MAP.values())
    absent = {"Kyrgyz", "Uyghur", "Turkmen"} 
    print(f"\nLanguages with NorthEuraLex expansion: {sorted(nel_only_langs)}")
    print(f"Languages with Swadesh-only data (no NEL coverage): {sorted(absent)}")
    print("\nTask 2 complete.")

if __name__ == "__main__":
    main()
