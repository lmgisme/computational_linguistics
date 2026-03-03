"""
config.py
---------
Central configuration for the Turkic substrate detection project, Phase 1.
Modify this file to add/remove target languages or adjust pipeline behavior.
"""

# ── ASJP database URL ────────────────────────────────────────────────────────
# The ASJP flat-file release (plain-text, tab-delimited) is hosted at the MPI
# linguistics server. We fetch the "listss17.txt" release which covers ~7,000
# doculects with their 40-item Swadesh lists encoded in ASJP orthography.
ASJP_URL = "https://asjp.clld.org/static/listss19.txt"
ASJP_FALLBACK_URL = "https://raw.githubusercontent.com/evolaemp/online_calculator/master/data/listss17.txt"

# ── Target Turkic doculects ───────────────────────────────────────────────────
# Keys are canonical language names used internally.
# Values are regex patterns to match the ASJP doculect identifier (case-insensitive).
# ASJP doculect names are often ISO-code-based or use the Ethnologue name, so
# we cast a broad net with alternates.
TURKIC_TARGETS = {
    "Turkish":      r"TURKISH",
    "Uzbek":        r"UZBEK",
    "Kazakh":       r"KAZAKH",
    "Kyrgyz":       r"KIRGHIZ|KYRGYZ",
    "Uyghur":       r"UIGHUR|UYGHUR",
    "Yakut":        r"YAKUT|SAKHA",
    "Chuvash":      r"CHUVASH",
    "Azerbaijani":  r"AZERBAIJANI|AZERI|AZERBAIJANI_",
    "Turkmen":      r"TURKMEN",
}

# ── ASJP 40-item Swadesh list concept labels (in ASJP list order) ─────────────
# These are the 40 core meanings in the ASJP database.
ASJP_CONCEPTS = [
    "I", "you_2sg", "we", "this", "that", "who", "what",
    "not", "all", "many", "one", "two", "big", "long", "small",
    "woman", "man", "person", "fish", "bird", "dog", "louse",
    "tree", "seed", "leaf", "root", "bark", "skin", "flesh",
    "blood", "bone", "grease", "egg", "horn", "tail", "feather",
    "hair", "head", "ear", "eye",
]

# ── IPA normalization settings ────────────────────────────────────────────────
# Whether to attempt ASJP-to-IPA symbol mapping (True) or keep raw ASJP (False).
NORMALIZE_TO_IPA = True

# Tokenization strategy: "sound_class" uses LingPy's built-in tokenizer;
# "simple" splits on spaces/hyphens after conversion.
TOKENIZER = "sound_class"

# Output paths
OUTPUT_DIR = "output"
OUTPUT_CSV  = "turkic_swadesh_phase1.csv"
REPORT_TXT  = "phase1_summary_report.txt"
