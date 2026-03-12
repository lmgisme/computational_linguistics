"""
Step 1c — Task 1: Starling Transcription Character Inventory
=============================================================
Scans all proto-form columns in edal_altaic_master.csv and all reflex_form
columns in the 4 family reflex CSVs. Builds frequency tables of every unique
character and multi-character unit (digraphs, combining diacritics, backtick
sequences) used in Starling transcription.

Output: output/starling_char_inventory.txt  (human-readable report)
        output/starling_char_inventory.csv  (machine-readable: char, unicode_name, hex, count, source)

Usage:
    python step1c_char_inventory.py
"""

import csv
import unicodedata
import re
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(r"C:\Users\lmgisme\Desktop\computational_linguistics")
OUTPUT = BASE / "output"

MASTER_CSV = OUTPUT / "edal_altaic_master.csv"
FAMILY_CSVS = {
    "mongolic":  OUTPUT / "edal_mongolic_reflexes.csv",
    "tungusic":  OUTPUT / "edal_tungusic_reflexes.csv",
    "korean":    OUTPUT / "edal_korean_reflexes.csv",
    "japanese":  OUTPUT / "edal_japanese_reflexes.csv",
}

# Proto-form columns in master CSV
PROTO_COLS = [
    "proto_altaic", "proto_turkic", "proto_mongolic",
    "proto_tungusic", "proto_korean", "proto_japanese"
]


def extract_starling_tokens(text):
    """
    Extract transcription characters from a Starling field.
    
    Strips:
      - Leading * (proto-form marker)
      - Content in parentheses (references, alternatives)
      - Content after common bibliographic markers (L, SM, etc.)
      - Braces {} (but keeps content inside)
      - Numeric references
    
    Returns the cleaned text for character analysis.
    """
    if not text or text.strip() == "":
        return ""
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove leading asterisk (proto-form marker)
    if text.startswith("*"):
        text = text[1:]
    
    # Remove braces but keep content
    text = text.replace("{", "").replace("}", "")
    
    # Remove ~ and alternative markers (but keep the forms)
    # e.g. "*ăčaj / *ĕčej" -> keep both forms
    
    return text


def get_unicode_codepoints(text):
    """
    Decompose text into individual Unicode codepoints with their properties.
    Returns list of (char, hex_code, unicode_name, is_combining) tuples.
    """
    result = []
    for ch in text:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = f"UNKNOWN U+{ord(ch):04X}"
        is_combining = unicodedata.combining(ch) != 0
        result.append((ch, f"U+{ord(ch):04X}", name, is_combining))
    return result


def find_backtick_sequences(text):
    """Find characters followed by backtick (Starling aspiration/glottalization marker)."""
    sequences = []
    for i in range(len(text) - 1):
        if text[i+1] == '`':
            sequences.append(text[i] + '`')
    return sequences


def find_combining_sequences(text):
    """Find base characters followed by combining diacritics."""
    sequences = []
    i = 0
    while i < len(text):
        ch = text[i]
        if i + 1 < len(text) and unicodedata.combining(text[i+1]) != 0:
            # Base + combining sequence
            seq = ch
            j = i + 1
            while j < len(text) and unicodedata.combining(text[j]) != 0:
                seq += text[j]
                j += 1
            sequences.append(seq)
            i = j
        else:
            i += 1
    return sequences


def analyze_column(values, label):
    """
    Analyze a set of transcription values.
    Returns dict with character frequencies, special sequences, etc.
    """
    char_freq = Counter()
    backtick_freq = Counter()
    combining_freq = Counter()
    braces_count = 0
    tilde_count = 0
    total_forms = 0
    empty_count = 0
    
    for raw in values:
        if not raw or raw.strip() == "":
            empty_count += 1
            continue
        
        total_forms += 1
        
        # Count braces notation
        if "{" in raw or "}" in raw:
            braces_count += 1
        
        if "~" in raw:
            tilde_count += 1
        
        cleaned = extract_starling_tokens(raw)
        if not cleaned:
            continue
        
        # Individual codepoint frequencies
        for ch in cleaned:
            if ch in (' ', ',', ';', '/', '-', '(', ')', '*', '"', "'",
                       '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                       '.', ':', '?', '!', '=', '<', '>', '[', ']', '+', '#'):
                continue  # Skip punctuation and numbers
            char_freq[ch] += 1
        
        # Backtick sequences
        for seq in find_backtick_sequences(cleaned):
            backtick_freq[seq] += 1
        
        # Combining diacritic sequences
        for seq in find_combining_sequences(cleaned):
            combining_freq[seq] += 1
    
    return {
        "label": label,
        "total_forms": total_forms,
        "empty_count": empty_count,
        "char_freq": char_freq,
        "backtick_freq": backtick_freq,
        "combining_freq": combining_freq,
        "braces_count": braces_count,
        "tilde_count": tilde_count,
    }


def classify_char(ch):
    """Classify a character into transcription categories."""
    cp = ord(ch)
    name = ""
    try:
        name = unicodedata.name(ch)
    except ValueError:
        pass
    
    if unicodedata.combining(ch) != 0:
        return "combining_diacritic"
    elif ch == '`':
        return "backtick_modifier"
    elif ch in ('\u01EF', '\u01F0', '\u01E7', '\u01F5', '\u01D0', '\u01D4', '\u01D2', '\u01CE', '\u01DF'):
        # ǯ, ǰ, ǧ, ǵ, ǐ, ǔ, ǒ, ǎ, ǟ
        return "hacek_modified"
    elif ch in ('\u010D', '\u0161', '\u017E', '\u00F1', '\u014B', '\u013A', '\u0144'):
        # č, š, ž, ñ, ŋ, ĺ, ń
        return "special_consonant"
    elif ch in ('\u0263', '\u0272', '\u0268', '\u01DD', '\u0254', '\u025B', '\u0259'):
        # ɣ, ɲ, ɨ, ǝ, ɔ, ɛ, ə
        return "ipa_like"
    elif 'COMBINING' in name:
        return "combining_diacritic"
    elif ch in ('\u0101', '\u0113', '\u012B', '\u014D', '\u016B', '\u01D6'):
        # ā, ē, ī, ō, ū, ǖ
        return "long_vowel"
    elif ch in ('\u00E0', '\u00E8', '\u00EC', '\u00F2', '\u00F9',
                '\u00E1', '\u00E9', '\u00ED', '\u00F3', '\u00FA'):
        # à, è, ì, ò, ù, á, é, í, ó, ú
        return "accented_vowel"
    elif ch in ('\u0103', '\u0115', '\u012D', '\u014F', '\u016D'):
        # ă, ĕ, ĭ, ŏ, ŭ
        return "breve_vowel"
    elif ch.isascii() and ch.isalpha():
        return "basic_latin"
    else:
        return "other_special"


def main():
    print("=" * 70)
    print("Step 1c — Task 1: Starling Character Inventory")
    print("=" * 70)
    
    all_analyses = []
    all_chars_global = Counter()
    
    # ---- MASTER CSV: proto-form columns ----
    print("\nReading master CSV...")
    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    print(f"  {len(rows)} records loaded.")
    
    for col in PROTO_COLS:
        values = [r[col] for r in rows]
        analysis = analyze_column(values, f"master:{col}")
        all_analyses.append(analysis)
        all_chars_global.update(analysis["char_freq"])
        non_empty = analysis["total_forms"]
        print(f"  {col}: {non_empty} non-empty forms, "
              f"{len(analysis['char_freq'])} unique chars")
    
    # ---- FAMILY REFLEX CSVs ----
    for family, path in FAMILY_CSVS.items():
        print(f"\nReading {family} reflexes...")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fam_rows = list(reader)
        
        # Proto column
        proto_col = f"proto_{family}"
        proto_values = [r[proto_col] for r in fam_rows]
        analysis_p = analyze_column(proto_values, f"{family}:proto")
        all_analyses.append(analysis_p)
        all_chars_global.update(analysis_p["char_freq"])
        
        # Group reflexes by daughter language
        daughter_langs = set(r["daughter_language"] for r in fam_rows)
        reflex_values = [r["reflex_form"] for r in fam_rows]
        analysis_r = analyze_column(reflex_values, f"{family}:reflexes")
        all_analyses.append(analysis_r)
        all_chars_global.update(analysis_r["char_freq"])
        
        print(f"  {len(fam_rows)} rows, {len(daughter_langs)} daughter languages")
        print(f"  Proto: {analysis_p['total_forms']} forms, "
              f"{len(analysis_p['char_freq'])} unique chars")
        print(f"  Reflexes: {analysis_r['total_forms']} forms, "
              f"{len(analysis_r['char_freq'])} unique chars")
    
    # ---- BUILD REPORT ----
    print("\n" + "=" * 70)
    print("GLOBAL CHARACTER INVENTORY")
    print("=" * 70)
    
    # Classify all characters
    classified = defaultdict(list)
    for ch, count in all_chars_global.most_common():
        cat = classify_char(ch)
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = f"UNKNOWN"
        classified[cat].append((ch, count, f"U+{ord(ch):04X}", name))
    
    # Collect all backtick sequences across analyses
    all_backtick = Counter()
    all_combining = Counter()
    for a in all_analyses:
        all_backtick.update(a["backtick_freq"])
        all_combining.update(a["combining_freq"])
    
    # ---- WRITE TEXT REPORT ----
    report_lines = []
    report_lines.append("=" * 78)
    report_lines.append("Starling Transcription Character Inventory")
    report_lines.append("Step 1c — Task 1")
    report_lines.append(f"Total unique characters (excl. punctuation/digits): "
                        f"{len(all_chars_global)}")
    report_lines.append("=" * 78)
    
    # Summary per source
    report_lines.append("\nPER-SOURCE SUMMARY")
    report_lines.append("-" * 78)
    for a in all_analyses:
        bt = f", {len(a['backtick_freq'])} backtick seqs" if a['backtick_freq'] else ""
        cb = f", {len(a['combining_freq'])} combining seqs" if a['combining_freq'] else ""
        br = f", {a['braces_count']} braces" if a['braces_count'] else ""
        report_lines.append(
            f"  {a['label']:30s}  {a['total_forms']:6d} forms, "
            f"{len(a['char_freq']):4d} unique chars{bt}{cb}{br}"
        )
    
    # Characters by category
    cat_order = [
        ("basic_latin", "Basic Latin (a-z, A-Z)"),
        ("long_vowel", "Long Vowels (macron)"),
        ("accented_vowel", "Accented Vowels (grave/acute)"),
        ("breve_vowel", "Breve Vowels"),
        ("special_consonant", "Special Consonants"),
        ("hacek_modified", "Hacek/Caron Modified"),
        ("ipa_like", "IPA-like Characters"),
        ("combining_diacritic", "Combining Diacritics"),
        ("backtick_modifier", "Backtick Modifier (`)"),
        ("other_special", "Other Special Characters"),
    ]
    
    for cat_key, cat_label in cat_order:
        chars = classified.get(cat_key, [])
        if not chars:
            continue
        report_lines.append(f"\n{cat_label}")
        report_lines.append("-" * 78)
        for ch, count, hexcode, name in sorted(chars, key=lambda x: -x[1]):
            # For combining diacritics, prefix with dotted circle for display
            if unicodedata.combining(ch) != 0:
                display = f"\u25CC{ch}"
            else:
                display = ch
            report_lines.append(f"  {display:4s}  {hexcode:8s}  {count:8d}x  {name}")
    
    # Backtick sequences
    if all_backtick:
        report_lines.append(f"\nBACKTICK SEQUENCES (char + ` = glottalization/aspiration)")
        report_lines.append("-" * 78)
        for seq, count in all_backtick.most_common():
            report_lines.append(f"  {seq:6s}  {count:8d}x")
    
    # Combining sequences
    if all_combining:
        report_lines.append(f"\nCOMBINING DIACRITIC SEQUENCES (base + combining)")
        report_lines.append("-" * 78)
        for seq, count in all_combining.most_common(50):
            parts = []
            for ch in seq:
                try:
                    parts.append(f"{ch} U+{ord(ch):04X} {unicodedata.name(ch)}")
                except ValueError:
                    parts.append(f"{ch} U+{ord(ch):04X} UNKNOWN")
            report_lines.append(f"  {seq:6s}  {count:8d}x  = {' + '.join(parts)}")
    
    # CRITICAL: Characters needing IPA mapping
    report_lines.append(f"\n{'=' * 78}")
    report_lines.append("CHARACTERS REQUIRING IPA MAPPING")
    report_lines.append("(non-ASCII, non-standard, or Starling-specific)")
    report_lines.append("=" * 78)
    
    needs_mapping = []
    for ch, count in all_chars_global.most_common():
        if ch.isascii() and ch.isalpha() and ch.islower():
            continue  # Basic a-z probably map to themselves
        if ch.isascii() and ch.isalpha() and ch.isupper():
            needs_mapping.append((ch, count, "uppercase_initial"))
            continue
        if ch == '`':
            needs_mapping.append((ch, count, "backtick_modifier"))
            continue
        if unicodedata.combining(ch) != 0:
            needs_mapping.append((ch, count, "combining_diacritic"))
            continue
        if not ch.isascii():
            needs_mapping.append((ch, count, "non_ascii"))
            continue
    
    report_lines.append(f"  Total characters needing mapping: {len(needs_mapping)}")
    report_lines.append("")
    for ch, count, reason in sorted(needs_mapping, key=lambda x: -x[1]):
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNKNOWN"
        display = f"\u25CC{ch}" if unicodedata.combining(ch) != 0 else ch
        report_lines.append(f"  {display:4s}  U+{ord(ch):04X}  {count:8d}x  [{reason:20s}]  {name}")
    
    # ---- WRITE FILES ----
    report_text = "\n".join(report_lines)
    
    report_path = OUTPUT / "starling_char_inventory.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\nReport written to: {report_path}")
    
    # Machine-readable CSV
    csv_path = OUTPUT / "starling_char_inventory.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["char", "hex", "unicode_name", "category", "global_count"])
        for ch, count in all_chars_global.most_common():
            try:
                name = unicodedata.name(ch)
            except ValueError:
                name = "UNKNOWN"
            cat = classify_char(ch)
            writer.writerow([ch, f"U+{ord(ch):04X}", name, cat, count])
    print(f"CSV written to: {csv_path}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Total unique characters: {len(all_chars_global)}")
    print(f"  Characters needing IPA mapping: {len(needs_mapping)}")
    print(f"  Backtick sequences found: {len(all_backtick)}")
    print(f"  Combining sequences found: {len(all_combining)}")
    print(f"\n  Top 20 non-ASCII characters:")
    non_ascii = [(ch, c) for ch, c in all_chars_global.most_common() 
                 if not ch.isascii()]
    for ch, count in non_ascii[:20]:
        try:
            name = unicodedata.name(ch)
        except ValueError:
            name = "UNKNOWN"
        print(f"    {ch:4s}  {count:8d}x  {name}")
    
    print(f"\n  Backtick sequences:")
    for seq, count in all_backtick.most_common():
        print(f"    {seq:6s}  {count:8d}x")
    
    print("\nDone. Run the script, then we'll proceed to Task 2 (mapping table).")


if __name__ == "__main__":
    main()
