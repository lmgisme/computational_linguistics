"""
Transeurasian Project — Task 1: NorthEuraLex Extension
Extends the NorthEuraLex dataset to Mongolic, Tungusic, Koreanic, and Japonic
languages, using the same concept set (933 concepts) already extracted for
the Turkic substrate pipeline.

Target languages (NorthEuraLex Language_IDs):
  Mongolic:   khk (Khalkha Mongolian), bua (Buryat), xal (Kalmyk)
  Tungusic:   evn (Evenki), mnc (Manchu), gld (Nanai)
  Koreanic:   kor (Korean)
  Japonic:    jpn (Japanese)

Note on Tungusic: Even (eve) is NOT in NorthEuraLex. Manchu (mnc) is included
as a substitute. The original spec listed bxr for Buryat, but NorthEuraLex
uses bua (older ISO 639-3 code for Buriat). Same language.

Input:
  - NorthEuraLex CLDF cache: phase1_ingestion/output/lexibank_cache/northeuralex/
  - Existing concept set from: output/northeuralex_merged.csv (933 concepts)

Output:
  - output/northeuralex_transeurasian.csv

Run from project root with venv311 active:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python transeurasian\\task1_northeuralex_extend.py
"""

import os
import csv
from collections import defaultdict

BASE    = r"C:\Users\lmgisme\Desktop\computational_linguistics"
OUTPUT  = os.path.join(BASE, "output")
NEL_DIR = os.path.join(BASE, "phase1_ingestion", "output", "lexibank_cache", "northeuralex")

MERGED_CSV = os.path.join(OUTPUT, "northeuralex_merged.csv")
OUT_CSV    = os.path.join(OUTPUT, "northeuralex_transeurasian.csv")

# NorthEuraLex Language_ID -> (canonical name, family)
TARGET_LANGS = {
    "khk": ("Khalkha_Mongolian", "Mongolic"),
    "bua": ("Buryat",            "Mongolic"),
    "xal": ("Kalmyk",            "Mongolic"),
    "evn": ("Evenki",            "Tungusic"),
    "mnc": ("Manchu",            "Tungusic"),
    "gld": ("Nanai",             "Tungusic"),
    "kor": ("Korean",            "Koreanic"),
    "jpn": ("Japanese",          "Japonic"),
}

# Languages absent from NorthEuraLex (requested but not available)
ABSENT_LANGS = {
    "eve": "Even (Tungusic) — not in NorthEuraLex",
}


def load_target_concepts():
    """
    Load the 933 target concept glosses from the existing Turkic merged dataset.
    Returns a set of gloss strings (lowercased for matching).
    """
    glosses = set()
    with open(MERGED_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            glosses.add(row["gloss"].strip())
    print(f"Loaded {len(glosses)} unique target concepts from northeuralex_merged.csv.")
    return glosses


def load_nel_parameters():
    """
    Load NorthEuraLex parameters.csv.
    Returns dict: parameter Name (gloss) -> parameter ID.
    """
    params_path = os.path.join(NEL_DIR, "parameters.csv")
    name_to_id = {}
    with open(params_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name_to_id[row["Name"].strip()] = row["ID"]
    print(f"Loaded {len(name_to_id)} NorthEuraLex parameters.")
    return name_to_id


def extract_forms(target_glosses, param_name_to_id):
    """
    Extract NorthEuraLex forms for target languages and target concepts.
    Returns list of row dicts and a set of matched parameter IDs.
    """
    # Build the set of parameter IDs we want
    # Match target glosses against NorthEuraLex parameter Names
    target_param_ids = {}  # param_id -> gloss
    matched = 0
    unmatched = []
    for gloss in sorted(target_glosses):
        pid = param_name_to_id.get(gloss)
        if pid is not None:
            target_param_ids[pid] = gloss
            matched += 1
        else:
            unmatched.append(gloss)

    print(f"Concept matching: {matched} matched to NorthEuraLex parameters, "
          f"{len(unmatched)} unmatched (Swadesh-only or hybrid-only concepts).")
    if unmatched:
        print(f"  Unmatched glosses (first 20): {unmatched[:20]}")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more.")

    # Read forms
    forms_path = os.path.join(NEL_DIR, "forms.csv")
    rows = []
    with open(forms_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lang_id = row["Language_ID"]
            if lang_id not in TARGET_LANGS:
                continue
            param_id = row["Parameter_ID"]
            if param_id not in target_param_ids:
                continue

            segments = row.get("Segments", "").strip()
            form     = row.get("Form", "").strip()
            if not segments and not form:
                continue

            lang_name, family = TARGET_LANGS[lang_id]
            rows.append({
                "language":   lang_name,
                "family":     family,
                "gloss":      target_param_ids[param_id],
                "form":       form,
                "ipa_tokens": segments,
                "source":     "northeuralex",
            })

    print(f"Extracted {len(rows)} form entries for {len(TARGET_LANGS)} target languages.")
    return rows, target_param_ids


def coverage_report(rows, target_param_ids):
    """
    Print per-language coverage against the target concept set.
    Flag languages with < 200 concepts as thin-coverage.
    Flag Japanese IPA quality concern.
    """
    total_target = len(target_param_ids)

    # Count unique concepts per language
    lang_concepts = defaultdict(set)
    lang_rows     = defaultdict(int)
    for row in rows:
        lang_concepts[row["language"]].add(row["gloss"])
        lang_rows[row["language"]] += 1

    print(f"\n{'='*65}")
    print(f"COVERAGE REPORT")
    print(f"{'='*65}")
    print(f"Target concepts (NorthEuraLex-matchable): {total_target}")
    print(f"{'─'*65}")
    print(f"{'Language':<25} {'Family':<12} {'Concepts':>8} {'/ Target':>10} {'Rows':>8} {'Status'}")
    print(f"{'─'*65}")

    thin_langs = []
    for lang_id in sorted(TARGET_LANGS.keys()):
        lang_name, family = TARGET_LANGS[lang_id]
        n_concepts = len(lang_concepts.get(lang_name, set()))
        n_rows     = lang_rows.get(lang_name, 0)
        pct        = (n_concepts / total_target * 100) if total_target > 0 else 0
        flags = []
        if n_concepts < 200:
            flags.append("THIN")
            thin_langs.append(lang_name)
        if lang_id == "jpn":
            flags.append("IPA-QUALITY-FLAG")
        status = ", ".join(flags) if flags else "OK"
        print(f"  {lang_name:<23} {family:<12} {n_concepts:>8} {f'/ {total_target}':>10} {n_rows:>8}  {status}")

    print(f"{'─'*65}")

    # Absent languages
    if ABSENT_LANGS:
        print(f"\nLanguages requested but ABSENT from NorthEuraLex:")
        for code, note in ABSENT_LANGS.items():
            print(f"  {code}: {note}")

    # Thin coverage warning
    if thin_langs:
        print(f"\nWARNING: Thin coverage (< 200 concepts): {', '.join(thin_langs)}")
        print(f"  These languages may be unreliable for cross-family comparison.")

    # Japanese IPA note
    print(f"\nNOTE: Japanese IPA in NorthEuraLex is auto-generated from orthography")
    print(f"  (Dellert 2020 pipeline). Quality may be lower than hand-transcribed")
    print(f"  languages. Verify critical forms against specialist sources.")

    # Family summary
    print(f"\n{'─'*65}")
    print(f"FAMILY SUMMARY")
    print(f"{'─'*65}")
    fam_concepts = defaultdict(set)
    fam_langs    = defaultdict(int)
    for row in rows:
        fam_concepts[row["family"]].add(row["gloss"])
        fam_langs[row["family"]] = fam_langs.get(row["family"], 0)
    for lang_id in TARGET_LANGS:
        _, fam = TARGET_LANGS[lang_id]
        fam_langs[fam] = fam_langs.get(fam, 0) + 1
    # recount
    fam_langs_count = defaultdict(int)
    for lang_id in TARGET_LANGS:
        _, fam = TARGET_LANGS[lang_id]
        fam_langs_count[fam] += 1
    for fam in ["Mongolic", "Tungusic", "Koreanic", "Japonic"]:
        n_langs = fam_langs_count.get(fam, 0)
        n_concepts = len(fam_concepts.get(fam, set()))
        print(f"  {fam:<12}: {n_langs} languages, {n_concepts} unique concepts covered")

    return thin_langs


def main():
    print("Transeurasian Task 1: NorthEuraLex Extension")
    print("=" * 50)

    # Step 1: Load target concepts from existing Turkic dataset
    target_glosses = load_target_concepts()

    # Step 2: Load NorthEuraLex parameter mapping
    param_name_to_id = load_nel_parameters()

    # Step 3: Extract forms for target languages
    rows, target_param_ids = extract_forms(target_glosses, param_name_to_id)

    # Step 4: Write output
    out_cols = ["language", "family", "gloss", "form", "ipa_tokens", "source"]
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_cols)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to: {OUT_CSV}")

    # Step 5: Coverage report
    thin = coverage_report(rows, target_param_ids)

    print(f"\nTask 1 complete.")
    if thin:
        print(f"Action needed: review thin-coverage languages before proceeding to cognate detection.")


if __name__ == "__main__":
    main()
