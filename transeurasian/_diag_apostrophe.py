"""Quick diagnostic: find apostrophe (U+0027) contexts in IPA output files."""
import csv, os, re

OUTPUT = r"C:\Users\lmgisme\Desktop\computational_linguistics\output"

files = [
    ("edal_altaic_master_ipa.csv", ["proto_altaic_ipa","proto_turkic_ipa","proto_mongolic_ipa","proto_tungusic_ipa","proto_korean_ipa","proto_japanese_ipa"]),
    ("edal_mongolic_reflexes_ipa.csv", ["proto_mongolic_ipa","reflex_form_ipa"]),
    ("edal_tungusic_reflexes_ipa.csv", ["proto_tungusic_ipa","reflex_form_ipa"]),
    ("edal_korean_reflexes_ipa.csv", ["proto_korean_ipa","reflex_form_ipa"]),
    ("edal_japanese_reflexes_ipa.csv", ["proto_japanese_ipa","reflex_form_ipa"]),
]

samples = []
for fname, cols in files:
    path = os.path.join(OUTPUT, fname)
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in cols:
                if col in row and "'" in row[col]:
                    # Get 5 chars before and after each apostrophe
                    val = row[col]
                    for m in re.finditer("'", val):
                        start = max(0, m.start()-5)
                        end = min(len(val), m.end()+5)
                        ctx = val[start:end]
                        samples.append((fname.replace("_ipa.csv",""), col, ctx, val[:60]))
                        if len(samples) >= 50:
                            break
            if len(samples) >= 50:
                break
    if len(samples) >= 50:
        break

print(f"Found {len(samples)} apostrophe contexts:\n")
for fname, col, ctx, full in samples[:50]:
    print(f"  [{fname}:{col}]  ...{ctx}...  full={full}")
