"""
Transeurasian Project — Step 1c, Task 3
Starling-to-IPA Conversion Script

Reads the mapping table (output/starling_to_ipa.json) and applies it to all
EDAL CSVs, producing IPA-converted outputs and a conversion report.

Usage:
    python transeurasian/step1c_convert_ipa.py

Outputs:
    output/edal_altaic_master_ipa.csv
    output/edal_mongolic_reflexes_ipa.csv
    output/edal_tungusic_reflexes_ipa.csv
    output/edal_korean_reflexes_ipa.csv
    output/edal_japanese_reflexes_ipa.csv
    output/ipa_conversion_report.txt
"""

import csv
import json
import os
import re
import unicodedata
from collections import Counter, defaultdict

# ============================================================================
# Configuration
# ============================================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MAPPING_FILE = os.path.join(OUTPUT_DIR, "starling_to_ipa.json")

# Files to process: (input_file, output_file, proto_columns, reflex_columns)
FILES = [
    (
        "edal_altaic_master.csv",
        "edal_altaic_master_ipa.csv",
        ["proto_altaic", "proto_turkic", "proto_mongolic", "proto_tungusic",
         "proto_korean", "proto_japanese"],
        []  # no reflex columns in master
    ),
    (
        "edal_mongolic_reflexes.csv",
        "edal_mongolic_reflexes_ipa.csv",
        ["proto_mongolic"],
        ["reflex_form"]
    ),
    (
        "edal_tungusic_reflexes.csv",
        "edal_tungusic_reflexes_ipa.csv",
        ["proto_tungusic"],
        ["reflex_form"]
    ),
    (
        "edal_korean_reflexes.csv",
        "edal_korean_reflexes_ipa.csv",
        ["proto_korean"],
        ["reflex_form"]
    ),
    (
        "edal_japanese_reflexes.csv",
        "edal_japanese_reflexes_ipa.csv",
        ["proto_japanese"],
        ["reflex_form"]
    ),
]

# ============================================================================
# Load mapping table
# ============================================================================

def load_mapping(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
# Preprocessing (Layer 0)
# ============================================================================

# Regex: parenthetical refs — e.g., (L 525), (Тод. Даг.122), (SM299), (908)
RE_PARENS = re.compile(r'\([^)]*\)')

# Regex: square bracket content — e.g., [akɨs] (phonetic transcriptions)
RE_BRACKETS = re.compile(r'\[[^\]]*\]')

# Regex: single-quoted glosses — e.g., 'to drink', 'gorgée'
RE_GLOSS = re.compile(r"'[^']*'")

# Regex: trailing sense numbers — e.g., "asī 2" → "asī", "ašǝ, asǝ 3" → "ašǝ, asǝ"
RE_SENSE_NUM = re.compile(r'\s+\d+(?:\s*,\s*\d+)*\s*$')

# Regex: Cyrillic runs (2+ Cyrillic chars, possibly with digits/punctuation/spaces)
RE_CYRILLIC_RUN = re.compile(
    r'[\u0400-\u04ff][\u0400-\u04ff0-9.,;:\s]*[\u0400-\u04ff0-9]+'
)

# Regex: single Cyrillic characters (catch strays after run removal)
RE_CYRILLIC_SINGLE = re.compile(r'[\u0400-\u04ff]')

# Regex: braces — strip braces but keep content
RE_BRACES = re.compile(r'[{}]')

# Regex: control characters
RE_CONTROL = re.compile(r'[\x00-\x1f]')

# Regex: borrowing markers — e.g., "< Turkic", "> Manchu"
RE_BORROWING = re.compile(r'[<>]\s*[A-Za-z]+\.?')

# Punctuation to strip (noise chars that are not phonological)
# Includes U+0027 APOSTROPHE — hiatus/morpheme boundary marker in Middle Korean,
# not a phonological segment; paired glosses already stripped by RE_GLOSS
STRIP_PUNCT = set("?!'\"=<>()*/~")


def _clean_common(s):
    """Common cleaning steps for both proto and reflex fields."""
    # Strip control chars
    s = RE_CONTROL.sub('', s)
    # Strip braces (keep content)
    s = RE_BRACES.sub('', s)
    # Strip borrowing markers before general punct stripping
    s = RE_BORROWING.sub('', s)
    # Strip square bracket content
    s = RE_BRACKETS.sub('', s)
    # Strip parenthetical refs
    s = RE_PARENS.sub('', s)
    # Strip single-quoted glosses
    s = RE_GLOSS.sub('', s)
    # Strip Cyrillic runs
    s = RE_CYRILLIC_RUN.sub('', s)
    # Strip single stray Cyrillic chars
    s = RE_CYRILLIC_SINGLE.sub('', s)
    # Strip remaining noise punctuation
    s = ''.join(ch for ch in s if ch not in STRIP_PUNCT)
    # Strip trailing sense numbers
    s = RE_SENSE_NUM.sub('', s)
    # Collapse whitespace
    s = ' '.join(s.split()).strip()
    # Clean orphaned commas
    s = re.sub(r',\s*,', ',', s)
    s = s.strip(',').strip()
    return s


def preprocess_proto(raw):
    """Preprocess a proto-form field."""
    if not raw or not raw.strip():
        return ""
    
    s = raw.strip()
    
    # Strip leading asterisk(s) early, before splitting
    s = s.lstrip('*')
    
    # Split on ~ (alternation) — take first variant
    if ' ~ ' in s:
        s = s.split(' ~ ')[0].strip()
    # Split on / (variant separator)
    if ' / ' in s:
        s = s.split(' / ')[0].strip()
    
    s = _clean_common(s)
    
    # Strip trailing hyphens
    s = s.strip('-').strip()
    
    return s


def preprocess_reflex(raw):
    """Preprocess a reflex_form field.
    
    Handles multiple comma-separated forms. Strips noise, preserves
    genuine alternates separated by commas.
    """
    if not raw or not raw.strip():
        return ""
    
    s = raw.strip()
    
    # Strip asterisks (shouldn't appear in reflexes, but they do)
    s = s.replace('*', '')
    
    s = _clean_common(s)
    
    return s


# ============================================================================
# IPA Conversion (Layers 1-6)
# ============================================================================

class StarlingToIPA:
    """Converts Starling transcription to IPA using the mapping table."""
    
    def __init__(self, mapping):
        self.mapping = mapping
        
        # Build multi-char lookup, sorted by length descending (longest match first)
        self.multi_char = mapping.get("multi_char", {})
        self.multi_keys_sorted = sorted(
            self.multi_char.keys(), key=len, reverse=True
        )
        
        # Combining diacritics lookup
        self.combining = {}
        for char, rule in mapping.get("combining_diacritics", {}).items():
            self.combining[char] = rule
        
        # Single-char lookup
        self.single_char = mapping.get("single_char", {})
        
        # Uppercase mapping
        self.upper_map = mapping.get("uppercase_to_lowercase", {})
        
        # Cyrillic and control char ranges for stripping
        cyr = mapping.get("cyrillic_range", [1024, 1279])
        self.cyrillic_lo, self.cyrillic_hi = cyr[0], cyr[1]
        ctrl = mapping.get("control_char_range", [0, 31])
        self.control_lo, self.control_hi = ctrl[0], ctrl[1]
        
        # Tracking
        self.unmapped = Counter()
        self.uppercase_flags = Counter()
    
    def convert_form(self, text):
        """Convert a single preprocessed form string to IPA."""
        if not text:
            return ""
        
        # Step 1: Apply multi-char sequences (longest match first)
        result = text
        for key in self.multi_keys_sorted:
            if key in result:
                result = result.replace(key, self.multi_char[key])
        
        # Step 2: Walk character-by-character, handling combining diacritics
        output = []
        i = 0
        chars = list(result)
        n = len(chars)
        
        while i < n:
            ch = chars[i]
            cp = ord(ch)
            
            # Check if this is a combining diacritic in our table
            if ch in self.combining:
                rule = self.combining[ch]
                action = rule["action"]
                
                if action == "strip":
                    i += 1
                    continue
                elif action == "keep":
                    output.append(ch)
                    i += 1
                    continue
                elif action == "replace":
                    output.append(rule["value"])
                    i += 1
                    continue
                elif action == "nonsyllabic":
                    # i + inverted breve below -> j; u -> w; else strip
                    if output and output[-1] in ('i', '\u0268'):
                        output[-1] = 'j'
                    elif output and output[-1] in ('u', 'y'):
                        output[-1] = 'w'
                    i += 1
                    continue
            
            # Cyrillic range — strip (safety net)
            if self.cyrillic_lo <= cp <= self.cyrillic_hi:
                while i < n and self.cyrillic_lo <= ord(chars[i]) <= self.cyrillic_hi:
                    i += 1
                continue
            
            # Control char range
            if self.control_lo <= cp <= self.control_hi:
                i += 1
                continue
            
            # Uppercase mapping
            if ch in self.upper_map:
                self.uppercase_flags[ch] += 1
                output.append(self.upper_map[ch])
                i += 1
                continue
            
            # Single-char mapping
            if ch in self.single_char:
                mapped = self.single_char[ch]
                if mapped:  # non-empty replacement
                    output.append(mapped)
                # else: stripped (empty string)
                i += 1
                continue
            
            # Basic Latin lowercase (a-z) and structural chars — pass through
            if ('a' <= ch <= 'z') or ch in (' ', '-', ','):
                output.append(ch)
                i += 1
                continue
            
            # IPA length marker (precomposed) — keep
            if ch == '\u02d0':  # IPA length mark
                output.append(ch)
                i += 1
                continue
            
            # Digits — strip
            if ch.isdigit():
                i += 1
                continue
            
            # Period, colon, semicolon — strip
            if ch in '.;:':
                i += 1
                continue
            
            # Anything else — track as unmapped, pass through
            self.unmapped[ch] += 1
            output.append(ch)
            i += 1
        
        return ''.join(output).strip()
    
    def convert_proto_field(self, raw):
        """Convert a proto-form field: preprocess, then convert."""
        cleaned = preprocess_proto(raw)
        if not cleaned:
            return ""
        return self.convert_form(cleaned)
    
    def convert_reflex_field(self, raw):
        """Convert a reflex_form field: preprocess, then convert each form."""
        cleaned = preprocess_reflex(raw)
        if not cleaned:
            return ""
        
        # Split on comma to handle multiple forms
        parts = [p.strip() for p in cleaned.split(',')]
        converted = []
        for part in parts:
            if part:
                c = self.convert_form(part)
                if c:
                    converted.append(c)
        
        return ', '.join(converted) if converted else ""


# ============================================================================
# File Processing
# ============================================================================

def process_file(converter, input_path, output_path, proto_cols, reflex_cols):
    """Process a single CSV file, converting specified columns to IPA."""
    stats = {
        "input_rows": 0,
        "proto_fields_total": 0,
        "proto_fields_nonempty": 0,
        "proto_fields_converted": 0,
        "reflex_fields_total": 0,
        "reflex_fields_nonempty": 0,
        "reflex_fields_converted": 0,
        "samples_before_after": [],
    }
    
    rows_out = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames)
        
        # Build output fieldnames: add _ipa columns after each target column
        out_fieldnames = []
        for col in fieldnames:
            out_fieldnames.append(col)
            if col in proto_cols or col in reflex_cols:
                out_fieldnames.append(col + "_ipa")
        out_fieldnames.append("has_uppercase")
        
        for row in reader:
            stats["input_rows"] += 1
            out_row = {}
            has_upper = False
            
            for col in fieldnames:
                out_row[col] = row[col]
                
                if col in proto_cols:
                    stats["proto_fields_total"] += 1
                    raw = row[col]
                    if raw and raw.strip():
                        stats["proto_fields_nonempty"] += 1
                        upper_before = sum(converter.uppercase_flags.values())
                        ipa = converter.convert_proto_field(raw)
                        upper_after = sum(converter.uppercase_flags.values())
                        if upper_after > upper_before:
                            has_upper = True
                        if ipa:
                            stats["proto_fields_converted"] += 1
                        out_row[col + "_ipa"] = ipa
                        
                        # Sample: collect first 20 non-trivial conversions
                        if len(stats["samples_before_after"]) < 20 and ipa and raw.strip() != ipa:
                            stats["samples_before_after"].append(
                                (col, raw.strip()[:80], ipa[:80])
                            )
                    else:
                        out_row[col + "_ipa"] = ""
                
                elif col in reflex_cols:
                    stats["reflex_fields_total"] += 1
                    raw = row[col]
                    if raw and raw.strip():
                        stats["reflex_fields_nonempty"] += 1
                        upper_before = sum(converter.uppercase_flags.values())
                        ipa = converter.convert_reflex_field(raw)
                        upper_after = sum(converter.uppercase_flags.values())
                        if upper_after > upper_before:
                            has_upper = True
                        if ipa:
                            stats["reflex_fields_converted"] += 1
                        out_row[col + "_ipa"] = ipa
                        
                        # Sample
                        if len(stats["samples_before_after"]) < 20 and ipa and raw.strip()[:40] != ipa[:40]:
                            stats["samples_before_after"].append(
                                (col, raw.strip()[:80], ipa[:80])
                            )
                    else:
                        out_row[col + "_ipa"] = ""
            
            out_row["has_uppercase"] = "1" if has_upper else "0"
            rows_out.append(out_row)
    
    # Write output
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)
    
    return stats


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(all_stats, converter, report_path):
    """Generate the conversion report."""
    lines = []
    lines.append("=" * 78)
    lines.append("Transeurasian Project - Step 1c, Task 3")
    lines.append("IPA Conversion Report")
    lines.append("=" * 78)
    lines.append("")
    
    # Per-file stats
    for filename, stats in all_stats.items():
        lines.append("-" * 78)
        lines.append(f"FILE: {filename}")
        lines.append("-" * 78)
        lines.append(f"  Input rows:             {stats['input_rows']:,}")
        
        if stats["proto_fields_total"] > 0:
            rate = (stats["proto_fields_converted"] / stats["proto_fields_nonempty"] * 100
                    if stats["proto_fields_nonempty"] > 0 else 0)
            lines.append(f"  Proto fields total:     {stats['proto_fields_total']:,}")
            lines.append(f"  Proto fields non-empty: {stats['proto_fields_nonempty']:,}")
            lines.append(f"  Proto fields converted: {stats['proto_fields_converted']:,}  ({rate:.1f}%)")
        
        if stats["reflex_fields_total"] > 0:
            rate = (stats["reflex_fields_converted"] / stats["reflex_fields_nonempty"] * 100
                    if stats["reflex_fields_nonempty"] > 0 else 0)
            lines.append(f"  Reflex fields total:    {stats['reflex_fields_total']:,}")
            lines.append(f"  Reflex fields non-empty:{stats['reflex_fields_nonempty']:,}")
            lines.append(f"  Reflex fields converted:{stats['reflex_fields_converted']:,}  ({rate:.1f}%)")
        
        lines.append("")
        lines.append("  Sample before/after pairs:")
        for col, before, after in stats["samples_before_after"][:10]:
            lines.append(f"    [{col}]")
            lines.append(f"      before: {before}")
            lines.append(f"      after:  {after}")
        lines.append("")
    
    # Unmapped characters
    lines.append("=" * 78)
    lines.append("UNMAPPED CHARACTERS (passed through as-is)")
    lines.append("=" * 78)
    if converter.unmapped:
        for ch, count in converter.unmapped.most_common():
            cp = ord(ch)
            name = unicodedata.name(ch, "UNKNOWN")
            lines.append(f"  {ch}  U+{cp:04X}  {count:>6}x  {name}")
    else:
        lines.append("  (none)")
    lines.append("")
    
    # Uppercase flags
    lines.append("=" * 78)
    lines.append("UPPERCASE CHARACTERS LOWERCASED (flagged in has_uppercase column)")
    lines.append("=" * 78)
    if converter.uppercase_flags:
        for ch, count in converter.uppercase_flags.most_common():
            lines.append(f"  {ch} -> {ch.lower()}  {count:>6}x")
    else:
        lines.append("  (none)")
    lines.append("")
    
    # Summary
    total_proto = sum(s["proto_fields_nonempty"] for s in all_stats.values())
    total_proto_conv = sum(s["proto_fields_converted"] for s in all_stats.values())
    total_reflex = sum(s["reflex_fields_nonempty"] for s in all_stats.values())
    total_reflex_conv = sum(s["reflex_fields_converted"] for s in all_stats.values())
    
    lines.append("=" * 78)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 78)
    if total_proto > 0:
        lines.append(f"  Proto fields:  {total_proto_conv:,} / {total_proto:,} converted "
                      f"({total_proto_conv/total_proto*100:.1f}%)")
    if total_reflex > 0:
        lines.append(f"  Reflex fields: {total_reflex_conv:,} / {total_reflex:,} converted "
                      f"({total_reflex_conv/total_reflex*100:.1f}%)")
    lines.append(f"  Unmapped chars: {len(converter.unmapped)} unique, "
                 f"{sum(converter.unmapped.values())} total occurrences")
    lines.append(f"  Uppercase flags: {sum(converter.uppercase_flags.values())} total")
    lines.append("")
    
    report = '\n'.join(lines)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    return report


# ============================================================================
# Main
# ============================================================================

def main():
    print("Loading mapping table...")
    mapping = load_mapping(MAPPING_FILE)
    converter = StarlingToIPA(mapping)
    
    all_stats = {}
    
    for input_name, output_name, proto_cols, reflex_cols in FILES:
        input_path = os.path.join(OUTPUT_DIR, input_name)
        output_path = os.path.join(OUTPUT_DIR, output_name)
        
        if not os.path.exists(input_path):
            print(f"  SKIP (not found): {input_name}")
            continue
        
        print(f"  Processing {input_name}...")
        stats = process_file(converter, input_path, output_path, proto_cols, reflex_cols)
        all_stats[input_name] = stats
        print(f"    -> {stats['input_rows']:,} rows, "
              f"proto={stats['proto_fields_converted']}/{stats['proto_fields_nonempty']}, "
              f"reflex={stats['reflex_fields_converted']}/{stats['reflex_fields_nonempty']}")
    
    # Generate report
    report_path = os.path.join(OUTPUT_DIR, "ipa_conversion_report.txt")
    print(f"\nGenerating report: {report_path}")
    report = generate_report(all_stats, converter, report_path)
    
    # Print summary to console
    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"Unmapped characters: {len(converter.unmapped)} unique")
    if converter.unmapped:
        for ch, count in converter.unmapped.most_common(10):
            print(f"  {ch} (U+{ord(ch):04X}): {count}x")
    print(f"\nOutputs in: {OUTPUT_DIR}")
    for _, output_name, _, _ in FILES:
        p = os.path.join(OUTPUT_DIR, output_name)
        if os.path.exists(p):
            print(f"  {output_name}")
    print(f"  ipa_conversion_report.txt")


if __name__ == "__main__":
    main()
