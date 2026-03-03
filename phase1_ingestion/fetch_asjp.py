"""
fetch_asjp.py
-------------
Module 1 of 3: Raw data acquisition.

ASJP lists.txt format (v18+, as used in lexibank/asjp):

  Preamble block (concept index, symbol table) — skip everything before
  the first doculect header.

  Doculect header line:
    NAME{CLASSIFICATION|family@glottolog_family}
  Metadata line (immediately after header):
    <W>  <LAT>  <LON>  <POP>  <ISO>  <ISO2>
  Concept lines (one per attested meaning):
    <NUM> <LABEL>\t<form1>, <form2> //
    e.g.:  1 I\tben //
           2 you\tsen //
  Blank line terminates the doculect block.

  "XXX" means the concept is not attested.
  Synonyms are comma-separated before the " //".
  Loan markers (%) and compound markers ($) may appear inside forms.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import requests

from config import TURKIC_TARGETS, ASJP_CONCEPTS

logger = logging.getLogger(__name__)

CACHE_PATH = Path("output") / "asjp_raw_cache.txt"

GITHUB_RAW_URLS = [
    "https://raw.githubusercontent.com/lexibank/asjp/master/raw/lists.txt",
    "https://raw.githubusercontent.com/lexibank/asjp/v2.0/raw/lists.txt",
]

# Exact doculect names — no regex, no substring matching.
# These were verified against the parsed doculect list from ASJP v21.
# Order matters: first entry is preferred. Fallbacks are listed for resilience.
ASJP_DOCULECT_CANDIDATES: dict[str, list[str]] = {
    "Turkish":     ["TURKISH", "TURKISH_2", "TURKISH_3"],
    "Uzbek":       ["UZBEK", "SOUTHERN_UZBEK"],
    "Kazakh":      ["KAZAKH", "KAZAKH_2"],
    "Kyrgyz":      ["KYRGYZ", "KIRGHIZ"],
    "Uyghur":      ["UYGHUR", "UIGHUR_XINJIANG_YILI_YINING", "LOPNOR_UYGHUR"],
    "Yakut":       ["SAKHA", "SAKHA_2"],   # GILYAK/SAKHALIN excluded — unrelated languages
    "Chuvash":     ["CHUVASH", "CHUVASH_2"],
    "Azerbaijani": ["AZERBAIJANI", "AZERBAIJANI_NORTH", "AZERBAIJANI_NORTH_2"],
    "Turkmen":     ["TURKMEN", "TURKMEN_2"],
}

# ASJP numeric concept IDs → our canonical gloss labels
# Full 100-item list; we only use the 40-item Swadesh subset defined in config.
ASJP_NUM_TO_CONCEPT: dict[int, str] = {
    1:  "I",         2:  "you_2sg",  3:  "we",
    4:  "this",      5:  "that",     6:  "who",
    7:  "what",      8:  "not",      9:  "all",
    10: "many",      11: "one",      12: "two",
    13: "big",       14: "long",     15: "small",
    16: "woman",     17: "man",      18: "person",
    19: "fish",      20: "bird",     21: "dog",
    22: "louse",     23: "tree",     24: "seed",
    25: "leaf",      26: "root",     27: "bark",
    28: "skin",      29: "flesh",    30: "blood",
    31: "bone",      32: "grease",   33: "egg",
    34: "horn",      35: "tail",     36: "feather",
    37: "hair",      38: "head",     39: "ear",
    40: "eye",
    # 41-100 exist in the file but are outside our 40-item target set
}


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def fetch_turkic_data(force_download: bool = False) -> dict[str, list[dict]]:
    """
    Return dict: canonical_language → list of doculect dicts.
    Each dict: { 'doculect': str, 'forms': dict[gloss→raw_asjp_form] }
    """
    Path("output").mkdir(exist_ok=True)

    # Use cache if available
    if not force_download and CACHE_PATH.exists():
        logger.info(f"Loading from cache: {CACHE_PATH}")
        raw_text = CACHE_PATH.read_text(encoding="utf-8", errors="replace")
        return _parse_and_extract(raw_text)

    # Fetch from GitHub
    for url in GITHUB_RAW_URLS:
        raw_text = _try_get(url, timeout=60)
        if raw_text and len(raw_text) > 500_000:
            logger.info(f"Downloaded lists.txt from: {url} ({len(raw_text)/1e6:.1f} MB)")
            CACHE_PATH.write_text(raw_text, encoding="utf-8")
            return _parse_and_extract(raw_text)
        elif raw_text:
            logger.warning(f"Response from {url} looks too small ({len(raw_text)} bytes). Skipping.")

    raise RuntimeError(
        "\n\nCould not retrieve ASJP lists.txt automatically.\n\n"
        "MANUAL DOWNLOAD — do this once:\n"
        "  1. Open:  https://asjp.clld.org/download\n"
        "  2. Download the Zenodo zip linked on that page.\n"
        "  3. Inside the zip find:  raw/lists.txt\n"
        "  4. Copy it to:\n"
        f"     {CACHE_PATH.resolve()}\n"
        "  5. Re-run:  python main.py\n"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Parser for the ASJP numbered-concept flat-file format
# ─────────────────────────────────────────────────────────────────────────────

# Matches a doculect header line: WORD_CHARS optionally followed by {STUFF}
_HEADER_RE = re.compile(r"^([A-Z][A-Z0-9_\-\.]+)\{")

# Matches a concept data line: integer, optional label text, tab, forms, " //"
# e.g.  "1 I\tben //"   or   "40 eye\tgöz //"
_CONCEPT_RE = re.compile(r"^(\d+)\s+\S.*?\t(.+?)\s*//\s*$")

# Metadata line: starts with a number (W field) then lat/lon/pop/iso
_META_RE = re.compile(r"^\s*\d+\s+[-\d\.]+\s+[-\d\.]+")


def _parse_and_extract(raw_text: str) -> dict[str, list[dict]]:
    all_doculects = _parse_asjp_flatfile(raw_text)
    logger.info(f"Parsed {len(all_doculects):,} doculects from flat file.")
    return _extract_turkic(all_doculects)


def _parse_asjp_flatfile(raw_text: str) -> dict[str, dict]:
    """
    Parse the ASJP numbered-concept flat-file into a dict keyed by doculect name.
    """
    doculects: dict[str, dict] = {}
    lines = raw_text.splitlines()

    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]

        # Detect doculect header
        m = _HEADER_RE.match(line)
        if m:
            name = m.group(1).upper()

            # Next non-blank line is the metadata line
            meta_lat, meta_lon, meta_iso = None, None, ""
            j = i + 1
            while j < n and not lines[j].strip():
                j += 1
            if j < n and _META_RE.match(lines[j]):
                meta_lat, meta_lon, meta_iso = _parse_meta_line(lines[j])
                j += 1

            # Collect concept lines until blank line or next header
            forms: dict[str, str] = {}
            while j < n:
                cline = lines[j]
                # Stop at blank line between doculects
                if not cline.strip():
                    j += 1
                    # Look ahead: if next non-blank is a header, we're done
                    k = j
                    while k < n and not lines[k].strip():
                        k += 1
                    if k < n and _HEADER_RE.match(lines[k]):
                        break
                    # Otherwise blank line was internal — keep reading
                    continue

                # Stop if we hit another header
                if _HEADER_RE.match(cline):
                    break

                cm = _CONCEPT_RE.match(cline)
                if cm:
                    concept_num = int(cm.group(1))
                    raw_forms   = cm.group(2).strip()
                    gloss = ASJP_NUM_TO_CONCEPT.get(concept_num)
                    if gloss:
                        # Take first synonym, skip XXX (unattested)
                        first = raw_forms.split(",")[0].strip()
                        forms[gloss] = "" if first == "XXX" else first

                j += 1

            doculects[name] = {
                "doculect": name,
                "iso": meta_iso,
                "lat": meta_lat,
                "lon": meta_lon,
                "forms": forms,
            }
            i = j
            continue

        i += 1

    return doculects


def _parse_meta_line(line: str) -> tuple:
    """Extract lat, lon, iso from the metadata line after a doculect header."""
    # Format: <W>  <LAT>  <LON>  <POP>  <ISO>  <ISO2>
    parts = line.split()
    lat = _safe_float(parts[1]) if len(parts) > 1 else None
    lon = _safe_float(parts[2]) if len(parts) > 2 else None
    iso = parts[4].strip() if len(parts) > 4 else ""
    return lat, lon, iso


def _extract_turkic(all_doculects: dict[str, dict]) -> dict[str, list[dict]]:
    """
    Exact-name lookup using ASJP_DOCULECT_CANDIDATES.
    Tries each candidate in order; uses the first hit.
    Falls back to TURKIC_TARGETS regex only if no exact match found.
    """
    results: dict[str, list[dict]] = {lang: [] for lang in ASJP_DOCULECT_CANDIDATES}

    for lang, candidates in ASJP_DOCULECT_CANDIDATES.items():
        for name in candidates:
            if name in all_doculects:
                results[lang] = [all_doculects[name]]
                logger.info(f"  Matched {lang} → {name}")
                break
        if not results[lang]:
            logger.warning(f"  No exact match for {lang} — trying regex fallback.")
            pattern = TURKIC_TARGETS.get(lang, "")
            if pattern:
                for doculect_name, info in all_doculects.items():
                    if re.search(pattern, doculect_name, re.IGNORECASE):
                        results[lang].append(info)
            if results[lang]:
                logger.warning(f"  Regex fallback matched {lang} → {results[lang][0]['doculect']}")
            else:
                logger.warning(f"  No match at all for: {lang}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _try_get(url: str, timeout: int = 30) -> Optional[str]:
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.text
        logger.debug(f"HTTP {resp.status_code}: {url}")
        return None
    except Exception as exc:
        logger.debug(f"Request failed ({url}): {exc}")
        return None


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except (ValueError, TypeError):
        return None
