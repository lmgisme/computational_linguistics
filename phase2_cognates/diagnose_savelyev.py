"""
diagnose_savelyev.py
Quick diagnostic to show exactly what Form_ID parsing produces
so we can fix the regex in build_hybrid_cognates.py.
"""
import pandas as pd
import re

SV_COGNATES = "phase1_ingestion/output/lexibank_cache/savelyevturkic/cognates.csv"

SV_TO_OURS = {
    "Azeri": "Azerbaijani", "Chuvash": "Chuvash", "Kazakh": "Kazakh",
    "Kirghiz": "Kyrgyz", "Turkish": "Turkish", "Turkmen": "Turkmen",
    "Uighur": "Uyghur", "Uzbek": "Uzbek", "Yakut": "Yakut",
}

sv = pd.read_csv(SV_COGNATES)

print("=== SAMPLE Form_ID VALUES (first 20) ===")
print(sv["Form_ID"].head(20).tolist())

print("\n=== TRYING DOCULECT EXTRACTION ===")
sv["doculect"] = sv["Form_ID"].str.extract(r"^([A-Za-z]+)-\d+_")
print("Sample doculects:", sv["doculect"].head(20).tolist())

print("\n=== FILTERING TO OUR 9 LANGUAGES ===")
sv["our_lang"] = sv["doculect"].map(SV_TO_OURS)
sv_ours = sv[sv["our_lang"].notna()].copy()
print(f"Rows after filter: {len(sv_ours)}")
print("Sample Form_IDs after filter:")
print(sv_ours["Form_ID"].head(20).tolist())

print("\n=== TRYING DIFFERENT REGEX PATTERNS FOR PARAM ===")
sample = sv_ours["Form_ID"].head(30)

patterns = {
    "pattern_A": r"-(\d+_[^-]+)-\d+-\d+$",
    "pattern_B": r"^[A-Za-z]+-(\d+_[^-]+)-",
    "pattern_C": r"_(\d+_[a-z0-9]+)-",
    "pattern_D": r"-\d+_([a-z0-9]+)-",
    "pattern_E": r"[A-Za-z]+-(\d+_\w+)-\d",
}

for name, pat in patterns.items():
    results = sample.str.extract(pat)
    non_null = results[0].notna().sum()
    print(f"\n  {name}: r'{pat}'")
    print(f"  Non-null matches: {non_null}/30")
    print(f"  Sample results: {results[0].head(10).tolist()}")

print("\n=== RAW FORM_ID STRUCTURE ANALYSIS (first 5) ===")
for fid in sv_ours["Form_ID"].head(5):
    parts = fid.split("-")
    print(f"  '{fid}' -> parts={parts}")
