"""Quick verification script — spot-check transeurasian output against raw CLDF."""
import csv

BASE = r"C:\Users\lmgisme\Desktop\computational_linguistics"
NEL  = BASE + r"\phase1_ingestion\output\lexibank_cache\northeuralex"
OUT  = BASE + r"\output\northeuralex_transeurasian.csv"

# --- Check 1: Spot-check specific raw forms ---
print("=== CHECK 1: Spot-check raw forms.csv against output ===")
spot_checks = [
    ("khk", "1_eye"),      # Khalkha "eye"
    ("khk", "6_tongue"),   # Khalkha "tongue"
    ("jpn", "1_eye"),      # Japanese "eye"
    ("kor", "1_eye"),      # Korean "eye"
    ("evn", "1_eye"),      # Evenki "eye"
    ("gld", "1_eye"),      # Nanai "eye"
    ("bua", "37_water"),   # Buryat "water"
    ("xal", "37_water"),   # Kalmyk "water"
    ("mnc", "37_water"),   # Manchu "water"
]

raw_hits = {}
with open(NEL + r"\forms.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["Language_ID"], row["Parameter_ID"])
        if key in [(s[0], s[1]) for s in spot_checks]:
            raw_hits.setdefault(key, []).append({
                "Form": row["Form"],
                "Segments": row["Segments"],
            })

for lang, param in spot_checks:
    hits = raw_hits.get((lang, param), [])
    print(f"\n  RAW  {lang} / {param}:")
    for h in hits:
        print(f"    Form={h['Form']}  Segments={h['Segments']}")

# Now check same in output
print("\n--- Corresponding output rows ---")
out_checks = {
    ("Khalkha_Mongolian", "eye"), ("Khalkha_Mongolian", "tongue"),
    ("Japanese", "eye"), ("Korean", "eye"), ("Evenki", "eye"),
    ("Nanai", "eye"), ("Buryat", "water"), ("Kalmyk", "water"),
    ("Manchu", "water"),
}
with open(OUT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        key = (row["language"], row["gloss"])
        if key in out_checks:
            print(f"  OUT  {row['language']:25s} {row['gloss']:10s} form={row['form']:15s} ipa={row['ipa_tokens']}")

# --- Check 2: Verify 7 unmatched glosses ---
print("\n=== CHECK 2: Are the 7 unmatched glosses truly absent from parameters.csv? ===")
unmatched = ['all', 'flesh', 'grease', 'many', 'person', 'small', 'you_2sg']
param_names = set()
with open(NEL + r"\parameters.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        param_names.add(row["Name"].strip())

for g in unmatched:
    status = "FOUND (BUG!)" if g in param_names else "absent (correct)"
    print(f"  '{g}': {status}")

# Check near-matches
print("\n  Near-matches in parameters.csv:")
for g in unmatched:
    near = [p for p in param_names if g in p.lower() or p.lower() in g]
    if near:
        print(f"    '{g}' ~ {near}")

# --- Check 3: Row count verification ---
print("\n=== CHECK 3: Row counts per language ===")
lang_counts = {}
total = 0
with open(OUT, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        lang_counts[row["language"]] = lang_counts.get(row["language"], 0) + 1
        total += 1
for lang, cnt in sorted(lang_counts.items()):
    print(f"  {lang:25s}: {cnt}")
print(f"  {'TOTAL':25s}: {total}")
print(f"  Expected: 8796  Match: {total == 8796}")
