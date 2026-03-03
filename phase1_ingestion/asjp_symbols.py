"""
asjp_symbols.py
---------------
ASJP-to-IPA symbol mapping table.

ASJP uses a compact ASCII orthography designed for cross-linguistic comparison.
This module provides:
  1. A symbol-by-symbol substitution map (ASJP → IPA).
  2. A helper that applies the map and flags any residual unconverted characters.

Reference: Wichmann et al. (2022) ASJP v19 documentation,
           https://asjp.clld.org/static/ASJP_Coding_Scheme.pdf
"""

# ── ASJP → IPA character-level mapping ───────────────────────────────────────
# Ordered from longest tokens to shortest to allow greedy replacement.
# Multi-character ASJP digraphs must be handled before single chars.
ASJP_TO_IPA: dict[str, str] = {
    # ── Clicks (ASJP uses ! for dental click) ────────────────────────────────
    "!":  "ǀ",

    # ── Affricates / special consonants ──────────────────────────────────────
    "c":  "ts",    # voiceless alveolar affricate
    "C":  "tʃ",    # voiceless postalveolar affricate
    "j":  "dʒ",    # voiced postalveolar affricate
    "dz": "dz",    # voiced alveolar affricate (keep as-is in IPA)

    # ── Fricatives ────────────────────────────────────────────────────────────
    "S":  "ʃ",     # voiceless postalveolar fricative
    "Z":  "ʒ",     # voiced postalveolar fricative
    "x":  "x",     # voiceless velar fricative (already IPA)
    "G":  "ɣ",     # voiced velar fricative
    "X":  "χ",     # voiceless uvular fricative
    "q":  "q",     # voiceless uvular stop (already IPA)
    "N":  "ŋ",     # velar nasal
    "L":  "ʎ",     # palatal lateral approximant

    # ── Vowels ────────────────────────────────────────────────────────────────
    "E":  "ɛ",     # open-mid front unrounded
    "3":  "ə",     # mid central (schwa)
    "I":  "ɪ",     # near-close near-front unrounded  (ASJP I = short/lax i)
    "o":  "o",     # mid back rounded (already IPA)
    "O":  "ɔ",     # open-mid back rounded
    "u":  "u",     # close back rounded (already IPA)
    "U":  "ʊ",     # near-close near-back rounded
    "y":  "y",     # close front rounded
    "e":  "e",     # mid front unrounded (already IPA)

    # ── Prosodic / diacritic markers ──────────────────────────────────────────
    "\"": "ˈ",    # primary stress (ASJP uses " before stressed syl.)
    "'":  "ʼ",    # ejective marker / glottalization
    "~":  "̃",     # nasalization diacritic (combining)

    # ── Tone markers (ASJP uses digits for tones in some doculects) ───────────
    "7":  "ʔ",    # glottal stop (common ASJP convention)

    # ── Passthrough (already valid IPA) ──────────────────────────────────────
    # p b t d k g f v s z h m n l r w a i
    # These require no mapping; listed here for documentation only.
}

# Characters that are structural in ASJP and should be stripped or converted
# before IPA tokenization.
ASJP_STRUCTURAL_CHARS = {
    "%":  None,   # loan word marker — strip
    "$":  None,   # second element of compound — strip
    "-":  " ",    # morpheme boundary → space (treat as word-internal boundary)
    ",":  " ",    # list separator between synonyms → keep first form upstream
}

# Characters that are valid IPA passthrough (no mapping needed)
IPA_PASSTHROUGH = set("p b t d k g f v s z h m n l r w a i e o u".split())

# Known anomaly patterns — ASJP forms containing these will be flagged
ANOMALY_PATTERNS = [
    r"\d",          # residual digit after glottal-stop substitution
    r"[A-Z]{2,}",   # consecutive capitals (likely a parsing artifact)
    r"\?",          # literal question mark (missing data bled into form field)
    r"^\s*$",       # empty / whitespace-only form
    r"^0$",         # ASJP uses "0" to mark absence/missing concept
]
