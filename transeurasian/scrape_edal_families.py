"""
Transeurasian Project — EDAL Family Sub-Database Scraper
========================================================
Scrapes daughter-language reflexes from the Starling/EDAL family-level
sub-databases (Mongolic, Tungusic, Korean, Japanese).

CIRCULARITY WARNING:
  Same as Script 1. We extract daughter-language reflexes for each proto-form,
  but we do NOT accept EDAL's cognate judgments. The cross-family groupings
  in the master table are EDAL's hypothesis; our pipeline tests these
  independently via regularity scoring.

Input:
  output\\edal_altaic_master.csv  (from scrape_edal.py)

Output:
  output\\edal_mongolic_reflexes.csv
  output\\edal_tungusic_reflexes.csv
  output\\edal_korean_reflexes.csv
  output\\edal_japanese_reflexes.csv
  output\\edal_family_scrape_errors.log

We do NOT scrape the Turkic sub-database — the Savelyev & Robbeets (2020)
dataset already provides higher-quality Turkic data for our pipeline.

Usage:
  # Explore mode — fetch one record from each family, dump HTML:
  python transeurasian\\scrape_edal_families.py --explore

  # Full scrape:
  python transeurasian\\scrape_edal_families.py

  # Limit to first N entries per family (testing):
  python transeurasian\\scrape_edal_families.py --limit 10

Run from project root with venv311 active.
"""

import argparse
import csv
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote

import requests
from bs4 import BeautifulSoup

# ── Paths ────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

MASTER_CSV = OUTPUT / "edal_altaic_master.csv"

# ── Constants ────────────────────────────────────────────────────────
RESPONSE_URL = "https://starlingdb.org/cgi-bin/response.cgi"

# Family sub-database basenames (URL-encoded in refs, decoded here)
FAMILY_CONFIG = {
    "mongolic": {
        "basename": "/data/alt/monget",
        "proto_label": "Proto-Mongolian",
    },
    "tungusic": {
        "basename": "/data/alt/tunget",
        "proto_label": "Proto-Tungus-Manchu",
    },
    "korean": {
        "basename": "/data/alt/koret",
        "proto_label": "Proto-Korean",
    },
    "japanese": {
        "basename": "/data/alt/japet",
        "proto_label": "Proto-Japanese",
    },
}

# Labels to skip when collecting daughter-language reflexes
# (these are metadata fields, not daughter languages)
SKIP_LABELS = {
    "Meaning", "Russian meaning", "Altaic etymology", "Comments",
    "References", "Nostratic",
    # Proto-form labels (handled separately)
    "Proto-Mongolian", "Proto-Tungus-Manchu", "Proto-Korean", "Proto-Japanese",
    "Proto-Turkic",
}

DELAY_SECONDS = 1.5
MAX_RETRIES   = 5
BACKOFF_BASE  = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research scraper; computational linguistics project; "
                  "contact: lukemgillespie@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

# ── Logging ──────────────────────────────────────────────────────────
error_log_path = OUTPUT / "edal_family_scrape_errors.log"
logging.basicConfig(
    filename=str(error_log_path),
    filemode="w",
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("edal_family_scraper")

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console)
logger.setLevel(logging.DEBUG)


# ── Network helpers ──────────────────────────────────────────────────
def fetch_family_entry(basename, text_number, session):
    """Fetch a single entry from a family sub-database via response.cgi."""
    params = {
        "root": "config",
        "basename": basename,
        "single": "1",
        "text_number": str(text_number),
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(RESPONSE_URL, params=params,
                               headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            elif resp.status_code in (429, 503):
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"  HTTP {resp.status_code}, backing off {wait:.0f}s")
                time.sleep(wait)
            else:
                logger.error(f"  HTTP {resp.status_code} for {basename} #{text_number}")
                return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"  Request error: {e}, backing off {wait:.0f}s")
            time.sleep(wait)
    logger.error(f"  FAILED after {MAX_RETRIES} retries: {basename} #{text_number}")
    return None


# ── HTML Parsing ─────────────────────────────────────────────────────
#
# Family sub-database entry structure (verified from edal_record1_raw.html):
#
#   <div class="results_record">
#     <div>
#       <span class="fld"><font color="green">Proto-Mongolian:</font></span>
#       <span class="unicode">*aɣu-lǯa-</span>
#     </div>
#     <div>
#       <span class="fld">Meaning:</span>
#       <span class="unicode">to meet, join</span>
#     </div>
#     <div>
#       <span class="fld">Written Mongolian:</span>
#       <span class="unicode">aɣulǯa- (L 17)</span>
#     </div>
#     <div>
#       <span class="fld">Khalkha:</span>
#       <span class="unicode">ūlʒa-</span>
#     </div>
#     ...
#     <div>
#       <span class="fld">Comments:</span>
#       <span class="unicode">...</span>
#     </div>
#   </div>
#
# NOTE: The response.cgi?single=1 for a sub-database record returns ONLY
# that sub-database's record. But etymology.cgi returns the full chain
# (all linked databases). We use response.cgi for targeted fetching.
#
# However, response.cgi?single=1 might still return multiple results_record
# blocks if the page includes linked entries. We need to find the correct
# one by matching the proto-form label for the family we're scraping.

def parse_family_entry(html, family_key):
    """
    Parse a family sub-database entry page to extract:
    - The proto-form for that family
    - The meaning
    - Each daughter-language reflex

    Returns: (proto_form, meaning, list_of_reflexes)
    where each reflex is dict: {language, form, meaning}
    """
    soup = BeautifulSoup(html, "html.parser")
    config = FAMILY_CONFIG[family_key]
    proto_label = config["proto_label"]

    # Find the correct results_record block for this family.
    # The page may contain multiple results_record blocks (e.g., linked
    # Altaic etymology, linked Turkic, etc.). We want the one that has
    # the matching proto-form label.
    record_divs = soup.find_all("div", class_="results_record")

    target_div = None
    for rd in record_divs:
        fld_spans = rd.find_all("span", class_="fld")
        for fld in fld_spans:
            label = fld.get_text(strip=True).rstrip(":").strip()
            if label == proto_label:
                target_div = rd
                break
        if target_div:
            break

    if not target_div:
        # Fallback: if only one results_record, use it
        if len(record_divs) == 1:
            target_div = record_divs[0]
        else:
            return "", "", []

    # Parse fields from the target div
    proto_form = ""
    entry_meaning = ""
    reflexes = []

    field_divs = target_div.find_all("div", recursive=False)

    for field_div in field_divs:
        if "subquery_link" in field_div.get("class", []):
            continue

        fld_span = field_div.find("span", class_="fld")
        if not fld_span:
            continue

        label = fld_span.get_text(strip=True).rstrip(":").strip()

        # Get the value from <span class="unicode">
        # (may be a direct child or inside an <a> tag)
        unicode_span = field_div.find("span", class_="unicode")
        if not unicode_span:
            continue
        value = unicode_span.get_text(strip=True)

        if label == proto_label:
            proto_form = value
        elif label == "Meaning":
            entry_meaning = value
        elif label in SKIP_LABELS:
            continue
        else:
            # This is a daughter-language reflex
            reflexes.append({
                "language": label,
                "form": value,
                "meaning": "",  # individual reflex meanings are embedded in the form text
            })

    return proto_form, entry_meaning, reflexes


# ── Load master CSV ──────────────────────────────────────────────────
def load_master_csv():
    """
    Load the master Altaic etymology CSV and extract ref info per family.
    Returns dict: family -> list of (altaic_record_id, text_number)
    """
    if not MASTER_CSV.exists():
        logger.error(f"Master CSV not found: {MASTER_CSV}")
        logger.error("Run scrape_edal.py first.")
        sys.exit(1)

    entries = {}

    with open(MASTER_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record_id = row.get("record_id", "")
            for family in FAMILY_CONFIG:
                ref_field = f"{family}_ref"
                ref_url = row.get(ref_field, "")
                if ref_url:
                    m = re.search(r"text_number=(\d+)", ref_url)
                    if m:
                        text_num = m.group(1)
                        if family not in entries:
                            entries[family] = []
                        entries[family].append((record_id, text_num))

    for fam, elist in entries.items():
        logger.info(f"  {fam}: {len(elist)} entries to scrape")

    return entries


# ── Explore mode ─────────────────────────────────────────────────────
def explore(session):
    """Fetch one sample record from each family sub-database and test parser."""
    logger.info("=== EXPLORE MODE (Family Sub-Databases) ===")

    for family, config in FAMILY_CONFIG.items():
        basename = config["basename"]
        logger.info(f"\n--- {family.upper()} ({basename}) ---")

        html = fetch_family_entry(basename, 1, session)
        if not html:
            logger.error(f"  Failed to fetch {family} record 1")
            time.sleep(DELAY_SECONDS)
            continue

        # Dump raw HTML
        dump_path = OUTPUT / f"edal_{family}_record1_raw.html"
        with open(dump_path, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"  Raw HTML dumped to: {dump_path}")

        # Structural analysis
        soup = BeautifulSoup(html, "html.parser")
        record_divs = soup.find_all("div", class_="results_record")
        logger.info(f"  results_record blocks: {len(record_divs)}")

        all_labels = []
        for rd in record_divs:
            for fld in rd.find_all("span", class_="fld"):
                all_labels.append(fld.get_text(strip=True).rstrip(":").strip())
        logger.info(f"  All <span.fld> labels: {sorted(set(all_labels))}")

        # Try parsing
        proto, meaning, reflexes = parse_family_entry(html, family)
        logger.info(f"  Proto-form: '{proto}'")
        logger.info(f"  Meaning: '{meaning}'")
        logger.info(f"  Reflexes found: {len(reflexes)}")
        for ref in reflexes:
            logger.info(f"    {ref['language']:25s}: {ref['form']}")

        time.sleep(DELAY_SECONDS)

    logger.info("\n=== EXPLORE COMPLETE ===")


# ── Full scrape ──────────────────────────────────────────────────────
def full_scrape(session, limit=None):
    """Scrape all family sub-database entries referenced in the master CSV."""
    entries = load_master_csv()

    for family, config in FAMILY_CONFIG.items():
        basename = config["basename"]
        family_entries = entries.get(family, [])
        if not family_entries:
            logger.info(f"{family}: no entries to scrape")
            continue

        if limit:
            family_entries = family_entries[:limit]

        logger.info(f"\n=== {family.upper()}: {len(family_entries)} entries ===")

        all_reflexes = []
        proto_col = f"proto_{family}"

        for idx, (record_id, text_num) in enumerate(family_entries):
            html = fetch_family_entry(basename, text_num, session)

            if html is None:
                logger.warning(f"  {family} text_number={text_num} "
                               f"(altaic #{record_id}): FAILED")
                continue

            try:
                proto, meaning, reflexes = parse_family_entry(html, family)

                for ref in reflexes:
                    all_reflexes.append({
                        "altaic_record_id": record_id,
                        proto_col: proto,
                        "daughter_language": ref["language"],
                        "reflex_form": ref["form"],
                        "meaning": meaning,
                    })

            except Exception as e:
                logger.warning(f"  Parse error for {family} text_number={text_num} "
                               f"(altaic #{record_id}): {e}")

            if (idx + 1) % 50 == 0:
                logger.info(f"  {family}: {idx+1}/{len(family_entries)} entries done, "
                             f"{len(all_reflexes)} reflexes so far")

            time.sleep(DELAY_SECONDS)

        # Write CSV
        csv_path = OUTPUT / f"edal_{family}_reflexes.csv"
        fieldnames = [
            "altaic_record_id",
            proto_col,
            "daughter_language",
            "reflex_form",
            "meaning",
        ]

        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in all_reflexes:
                writer.writerow(row)

        logger.info(f"  {family}: wrote {len(all_reflexes)} reflexes to {csv_path}")

        # Language coverage summary
        lang_counts = {}
        for r in all_reflexes:
            lang = r["daughter_language"]
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        if lang_counts:
            logger.info(f"  Daughter-language coverage:")
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                logger.info(f"    {lang:25s}: {count:4d} reflexes")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scrape EDAL family sub-databases for daughter-language reflexes"
    )
    parser.add_argument("--explore", action="store_true",
                        help="Fetch one record from each family, dump HTML")
    parser.add_argument("--limit", type=int, default=None,
                        help="Scrape only first N entries per family (testing)")
    args = parser.parse_args()

    session = requests.Session()

    if args.explore:
        explore(session)
    else:
        full_scrape(session, limit=args.limit)


if __name__ == "__main__":
    main()
