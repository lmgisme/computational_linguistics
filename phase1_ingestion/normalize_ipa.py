"""
normalize_ipa.py
----------------
Module 2 of 3: Phonetic normalization.

Responsibilities:
  - Convert raw ASJP orthographic forms to IPA strings.
  - Tokenize IPA strings into discrete phone segments.
  - Flag anomalous forms for human review.

Design notes:
  - We prefer LingPy's Sound class / tokenize() for principled IPA tokenization,
    but fall back to a character-level split if LingPy is unavailable.
  - The ASJP→IPA mapping in asjp_symbols.py is applied BEFORE tokenization so
    that the tokenizer sees valid IPA input.
  - Multiple synonyms in one cell (comma-separated) were already reduced to the
    primary form in fetch_asjp.py; this module receives single forms only.
"""

import re
import logging
from typing import Optional

from asjp_symbols import ASJP_TO_IPA, ASJP_STRUCTURAL_CHARS, ANOMALY_PATTERNS

logger = logging.getLogger(__name__)

# ── Optional LingPy import ─────────────────────────────────────────────────────
try:
    from lingpy.sequence.sound_classes import token2class, ipa2tokens
    LINGPY_AVAILABLE = True
    logger.info("LingPy found — using ipa2tokens for phonetic tokenization.")
except ImportError:
    LINGPY_AVAILABLE = False
    logger.warning(
        "LingPy not found. Falling back to regex-based IPA tokenizer. "
        "Install with:  pip install lingpy"
    )


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ASJP → IPA string conversion
# ─────────────────────────────────────────────────────────────────────────────

def asjp_to_ipa(asjp_form: str) -> str:
    """
    Convert a single ASJP-orthography form string to an IPA string.

    Steps:
      1. Handle structural characters (loan markers, compound markers, etc.).
      2. Apply symbol substitutions from longest to shortest token.
      3. Return the resulting string (may still contain residual ASCII for
         passthrough IPA characters like p, b, t, d, k…).
    """
    if not asjp_form or asjp_form.strip() in ("0", "-", ""):
        return ""  # missing/absent concept

    result = asjp_form.strip()

    # Step 1: Handle structural chars
    for asjp_char, replacement in ASJP_STRUCTURAL_CHARS.items():
        if replacement is None:
            result = result.replace(asjp_char, "")
        else:
            result = result.replace(asjp_char, replacement)

    # Step 2: Apply symbol substitutions (greedy, longest first)
    # Sort by length descending to handle digraphs before single chars
    sorted_subs = sorted(ASJP_TO_IPA.items(), key=lambda x: len(x[0]), reverse=True)
    for asjp_sym, ipa_sym in sorted_subs:
        result = result.replace(asjp_sym, ipa_sym)

    # Step 3: Collapse multiple spaces introduced by structural char handling
    result = re.sub(r"\s+", " ", result).strip()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2.  IPA tokenization
# ─────────────────────────────────────────────────────────────────────────────

def tokenize_ipa(ipa_string: str) -> list[str]:
    """
    Break an IPA string into a list of discrete phone tokens.

    Uses LingPy's ipa2tokens() if available, which correctly handles:
      - Diacritics (e.g., aspirated [pʰ] as a single token)
      - Affricates (e.g., [tʃ] kept together)
      - Tone marks and length markers

    Falls back to a regex-based tokenizer that handles common IPA diacritics.
    Returns an empty list for empty/missing input.
    """
    if not ipa_string or not ipa_string.strip():
        return []

    if LINGPY_AVAILABLE:
        try:
            tokens = ipa2tokens(ipa_string, merge_vowels=False, merge_geminates=False)
            return [t for t in tokens if t.strip()]
        except Exception as exc:
            logger.debug(f"LingPy tokenization failed for '{ipa_string}': {exc}. Falling back.")

    # Regex fallback tokenizer
    # Matches: base IPA char + optional combining diacritics + optional length mark
    IPA_TOKEN_RE = re.compile(
        r"[pbtdkɡqɢʔmnŋɲɳɴlrɾɹɻjwɥʍɰfvθðszʃʒʂʐçʝxɣχʁħʕhɦ"
        r"aeiouæɛɜɞɪʊɯøœɶɑɒɔəɨʉy"
        r"tsʧdʒdzɕʑɧ]"
        r"[\u0300-\u036f\u02b0-\u02ff]*"  # combining diacritics & modifier letters
        r"ː?",                              # optional length mark
        re.UNICODE
    )
    tokens = IPA_TOKEN_RE.findall(ipa_string)
    return tokens if tokens else list(ipa_string)  # last resort: char-by-char


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────

# Compile anomaly patterns once at module load
_ANOMALY_REGEXES = [re.compile(p) for p in ANOMALY_PATTERNS]


def detect_anomalies(
    language: str,
    concept: str,
    raw_asjp: str,
    ipa_form: str,
) -> Optional[str]:
    """
    Check a (raw_asjp, ipa_form) pair for anomalies.

    Returns a human-readable anomaly description string if an issue is found,
    or None if the form looks clean.

    Checks:
      - Empty/missing form (expected vs unexpected)
      - Residual digits after glottal-stop substitution
      - Consecutive uppercase (likely a header bleed-through)
      - Known ASJP absence marker ("0")
      - Suspiciously short IPA output (< 1 phone for a non-function-word concept)
    """
    # Explicit absence marker
    if raw_asjp.strip() == "0":
        return "ASJP_ABSENT_MARKER"

    # Empty raw form
    if not raw_asjp.strip():
        return "EMPTY_RAW_FORM"

    # Pattern-based anomalies on the *raw* ASJP form
    for pattern in _ANOMALY_REGEXES:
        if pattern.search(raw_asjp):
            return f"RAW_ANOMALY:{pattern.pattern}"

    # Pattern-based anomalies on the *converted* IPA form
    if ipa_form:
        for pattern in _ANOMALY_REGEXES:
            if pattern.search(ipa_form):
                return f"IPA_ANOMALY:{pattern.pattern}"

    # Suspiciously short: a non-function concept with zero phones after conversion
    function_concepts = {"I", "you_2sg", "we", "this", "that", "who", "what", "not"}
    if concept not in function_concepts and ipa_form and len(ipa_form.replace(" ", "")) < 1:
        return "IPA_TOO_SHORT"

    return None  # no anomaly detected


# ─────────────────────────────────────────────────────────────────────────────
# 4.  High-level per-form processing entry point
# ─────────────────────────────────────────────────────────────────────────────

def process_form(
    language: str,
    concept: str,
    raw_asjp: str,
) -> dict:
    """
    Full pipeline for a single (language, concept, raw_asjp) triple.

    Returns a dict with keys:
        language, concept, raw_form, ipa_form, ipa_tokens, anomaly_flag
    """
    ipa_form = asjp_to_ipa(raw_asjp)
    tokens   = tokenize_ipa(ipa_form)
    anomaly  = detect_anomalies(language, concept, raw_asjp, ipa_form)

    return {
        "language":   language,
        "gloss":      concept,
        "form":       raw_asjp,
        "ipa_form":   ipa_form,
        "ipa_tokens": tokens,
        "anomaly":    anomaly,
    }
