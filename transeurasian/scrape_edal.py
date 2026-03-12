"""
Transeurasian Project — EDAL Master Table Scraper
==================================================
Scrapes the Altaic etymology database from starlingdb.org (EDAL,
Starostin, Dybo & Mudrak 2003) to extract proto-forms for all five
families: Turkic, Mongolic, Tungusic, Korean, Japanese.

CIRCULARITY WARNING:
  We extract EDAL's *proto-forms* for each family, but we do NOT accept
  EDAL's cognate groupings as ground truth. EDAL groups forms across
  families under a single "Proto-Altaic" etymology — that is exactly the
  hypothesis our pipeline independently tests. We record which families
  EDAL claims each etymology spans (for reference), but regularity scoring
  is done independently by our pipeline.

Usage:
  # Explore mode — fetch page 1 only, dump raw HTML for inspection:
  python transeurasian\\scrape_edal.py --explore

  # Full scrape:
  python transeurasian\\scrape_edal.py

  # Scrape first N pages only (for testing):
  python transeurasian\\scrape_edal.py --pages 3

Output:
  output\\edal_altaic_master.csv
  output\\edal_scrape_errors.log
  (explore mode: output\\edal_page1_raw.html)

Run from project root with venv311 active:
  cd C:\\Users\\lmgisme\\Desktop\\computational_linguistics
  venv311\\Scripts\\activate
  python transeurasian\\scrape_edal.py
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

# ── Constants ────────────────────────────────────────────────────────
BASE_URL     = "https://starlingdb.org/cgi-bin/response.cgi"
BASENAME     = "/data/alt/altet"
RECORDS_PER_PAGE = 20
TOTAL_RECORDS    = 2805
TOTAL_PAGES      = (TOTAL_RECORDS + RECORDS_PER_PAGE - 1) // RECORDS_PER_PAGE  # 141

DELAY_SECONDS    = 1.5
MAX_RETRIES      = 5
BACKOFF_BASE     = 2.0

# Mapping from Starling field labels (inside <span class="fld">) to our CSV columns.
# Verified from edal_page1_raw.html — these are the exact label strings.
FAMILY_LABEL_MAP = {
    "Turkic":        ("proto_turkic",   "turkic_ref"),
    "Mongolian":     ("proto_mongolic", "mongolic_ref"),
    "Tungus-Manchu": ("proto_tungusic", "tungusic_ref"),
    "Korean":        ("proto_korean",   "korean_ref"),
    "Japanese":      ("proto_japanese",  "japanese_ref"),
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research scraper; computational linguistics project; "
                  "contact: lukemgillespie@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

# ── Logging ──────────────────────────────────────────────────────────
error_log_path = OUTPUT / "edal_scrape_errors.log"
logging.basicConfig(
    filename=str(error_log_path),
    filemode="w",
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger("edal_scraper")

console = logging.StreamHandler(sys.stdout)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console)
logger.setLevel(logging.DEBUG)


# ── Network helpers ──────────────────────────────────────────────────
def fetch_page(first_n, session):
    """Fetch a single results page with exponential backoff."""
    params = {
        "root": "config",
        "basename": BASENAME,
        "first": str(first_n),
    }
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            elif resp.status_code in (429, 503):
                wait = BACKOFF_BASE ** (attempt + 1)
                logger.warning(f"  HTTP {resp.status_code} on first={first_n}, "
                               f"backing off {wait:.0f}s (attempt {attempt+1})")
                time.sleep(wait)
            else:
                logger.error(f"  HTTP {resp.status_code} on first={first_n}")
                return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** (attempt + 1)
            logger.warning(f"  Request error on first={first_n}: {e}, "
                           f"backing off {wait:.0f}s")
            time.sleep(wait)
    logger.error(f"  FAILED after {MAX_RETRIES} retries: first={first_n}")
    return None


# ── HTML Parsing ─────────────────────────────────────────────────────
#
# Starling DOM structure (verified from edal_page1_raw.html):
#
#   <div class="results_record">
#     <div>
#       <span class="fld"><font color="green">Proto-Altaic:</font></span>
#       <span class="unicode">*ăbu</span>
#     </div>
#     <div>
#       <span class="fld">Meaning:</span>
#       <span class="unicode">interior of the mouth</span>
#     </div>
#     <div>
#       <span class="fld">Turkic:</span>
#       <a href="response.cgi?single=1&basename=...&text_number=1592&root=config">
#         <span class="unicode">*Ăburt</span>
#       </a>
#       <div class="subquery_link">...</div>
#     </div>
#     ...
#     <div>
#       <span class="fld">Comments:</span>
#       <span class="unicode">...</span>
#     </div>
#     <!-- results_record_end -->
#   </div>
#
# Key points:
#   - Records are <div class="results_record">
#   - Field labels are in <span class="fld">
#   - Values are in <span class="unicode"> (either as direct child or inside <a>)
#   - Family proto-forms that link to sub-databases are wrapped in <a href="...">
#   - Not all families are present in every record
#   - "Nostratic" field also exists but we don't need it

def parse_records_from_page(html):
    """
    Parse all etymology records from a Starling results page.
    Returns a list of dicts, one per record.
    """
    soup = BeautifulSoup(html, "html.parser")
    record_divs = soup.find_all("div", class_="results_record")
    records = []

    for rec_div in record_divs:
        try:
            record = parse_single_record(rec_div)
            if record:
                records.append(record)
        except Exception as e:
            logger.warning(f"  Error parsing record: {e}")

    return records


def parse_single_record(rec_div):
    """
    Parse one <div class="results_record"> into a dict.
    """
    record = {
        "proto_altaic": "",
        "meaning": "",
        "russian_meaning": "",
        "proto_turkic": "",   "turkic_ref": "",
        "proto_mongolic": "", "mongolic_ref": "",
        "proto_tungusic": "", "tungusic_ref": "",
        "proto_korean": "",   "korean_ref": "",
        "proto_japanese": "", "japanese_ref": "",
        "comments": "",
    }

    # Each field is in a child <div> of the record
    field_divs = rec_div.find_all("div", recursive=False)

    for field_div in field_divs:
        # Skip subquery_link divs
        if "subquery_link" in field_div.get("class", []):
            continue

        fld_span = field_div.find("span", class_="fld")
        if not fld_span:
            continue

        # Get the label text, strip colon
        label = fld_span.get_text(strip=True).rstrip(":").strip()

        if label == "Proto-Altaic":
            unicode_span = field_div.find("span", class_="unicode")
            if unicode_span:
                record["proto_altaic"] = unicode_span.get_text(strip=True)

        elif label == "Meaning":
            unicode_span = field_div.find("span", class_="unicode")
            if unicode_span:
                record["meaning"] = unicode_span.get_text(strip=True)

        elif label == "Russian meaning":
            unicode_span = field_div.find("span", class_="unicode")
            if unicode_span:
                record["russian_meaning"] = unicode_span.get_text(strip=True)

        elif label == "Comments":
            unicode_span = field_div.find("span", class_="unicode")
            if unicode_span:
                record["comments"] = unicode_span.get_text(strip=True)

        elif label in FAMILY_LABEL_MAP:
            proto_col, ref_col = FAMILY_LABEL_MAP[label]

            # Family forms are inside <a href="..."><span class="unicode">*form</span></a>
            link = field_div.find("a")
            if link:
                unicode_span = link.find("span", class_="unicode")
                if unicode_span:
                    record[proto_col] = unicode_span.get_text(strip=True)

                href = link.get("href", "")
                if href:
                    # Extract text_number from the href
                    # href looks like: response.cgi?single=1&basename=%2fdata%2falt%2fmonget&text_number=1659&root=config
                    record[ref_col] = href
            else:
                # Fallback: no link, just a unicode span (rare but possible)
                unicode_span = field_div.find("span", class_="unicode")
                if unicode_span:
                    record[proto_col] = unicode_span.get_text(strip=True)

        # We skip "Nostratic" and other labels we don't need

    # Only return if we got at least a proto-altaic form or a meaning
    if record["proto_altaic"] or record["meaning"]:
        return record
    return None


# ── Explore mode ─────────────────────────────────────────────────────
def explore(session):
    """Fetch page 1 only. Dump raw HTML and test the parser."""
    logger.info("=== EXPLORE MODE ===")
    logger.info("Fetching first page (first=1)...")

    html = fetch_page(1, session)
    if not html:
        logger.error("Failed to fetch page 1. Check network / Starling availability.")
        return

    # Dump raw HTML
    raw_path = OUTPUT / "edal_page1_raw.html"
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info(f"Raw HTML dumped to: {raw_path}")
    logger.info(f"HTML length: {len(html)} chars")

    # Parse
    records = parse_records_from_page(html)
    logger.info(f"Records parsed: {len(records)}")

    if records:
        for i, rec in enumerate(records[:3]):
            logger.info(f"\n--- Record {i+1} ---")
            for k, v in rec.items():
                if v:
                    display = v[:100] + ("..." if len(v) > 100 else "")
                    logger.info(f"  {k:20s}: {display}")
        logger.info(f"\n--- Family coverage on page 1 ---")
        for fam in ["proto_turkic", "proto_mongolic", "proto_tungusic",
                     "proto_korean", "proto_japanese"]:
            count = sum(1 for r in records if r.get(fam))
            logger.info(f"  {fam:20s}: {count}/{len(records)}")

        # Check refs
        ref_sample = []
        for r in records:
            for ref_field in ["turkic_ref", "mongolic_ref", "tungusic_ref",
                              "korean_ref", "japanese_ref"]:
                if r.get(ref_field):
                    ref_sample.append(r[ref_field])
        logger.info(f"\nSample sub-database refs ({len(ref_sample)} total on page):")
        for ref in ref_sample[:5]:
            logger.info(f"  {ref}")
    else:
        logger.error("NO RECORDS PARSED. Check edal_page1_raw.html manually.")

    logger.info("\n=== EXPLORE COMPLETE ===")
    if records:
        logger.info(f"Parser working. Run full scrape:")
        logger.info(f"  python transeurasian\\scrape_edal.py")
        logger.info(f"Or test with: python transeurasian\\scrape_edal.py --pages 3")


# ── Full scrape ──────────────────────────────────────────────────────
def full_scrape(session, max_pages=None):
    """Scrape all pages and write the master CSV."""
    pages_to_scrape = max_pages if max_pages else TOTAL_PAGES
    logger.info(f"=== FULL SCRAPE: {pages_to_scrape} pages ===")

    all_records = []
    record_counter = 0

    for page_idx in range(pages_to_scrape):
        first_n = 1 + page_idx * RECORDS_PER_PAGE
        html = fetch_page(first_n, session)

        if html is None:
            logger.error(f"Page {page_idx+1} (first={first_n}): FAILED, skipping")
            continue

        records = parse_records_from_page(html)

        for rec in records:
            record_counter += 1
            rec["record_id"] = record_counter
            all_records.append(rec)

        if (page_idx + 1) % 10 == 0 or page_idx == 0:
            logger.info(f"  Page {page_idx+1}/{pages_to_scrape} done — "
                         f"{len(records)} records on page, "
                         f"{record_counter} total so far")

        time.sleep(DELAY_SECONDS)

    logger.info(f"\nScrape complete: {record_counter} records from "
                f"{pages_to_scrape} pages")

    # Write CSV
    csv_path = OUTPUT / "edal_altaic_master.csv"
    fieldnames = [
        "record_id",
        "proto_altaic",
        "meaning",
        "russian_meaning",
        "proto_turkic",   "turkic_ref",
        "proto_mongolic",  "mongolic_ref",
        "proto_tungusic",  "tungusic_ref",
        "proto_korean",    "korean_ref",
        "proto_japanese",  "japanese_ref",
        "comments",
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in all_records:
            row = {k: rec.get(k, "") for k in fieldnames}
            writer.writerow(row)

    logger.info(f"Output written: {csv_path}")
    logger.info(f"  Total records: {record_counter}")

    # Summary stats
    filled = {fam: 0 for fam in ["turkic", "mongolic", "tungusic", "korean", "japanese"]}
    for rec in all_records:
        for fam in filled:
            if rec.get(f"proto_{fam}"):
                filled[fam] += 1
    logger.info(f"  Family coverage:")
    for fam, count in filled.items():
        logger.info(f"    {fam:12s}: {count:4d} / {record_counter} "
                     f"({100*count/max(record_counter,1):.1f}%)")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Scrape EDAL Altaic etymology database from starlingdb.org"
    )
    parser.add_argument("--explore", action="store_true",
                        help="Fetch page 1 only, dump HTML for inspection")
    parser.add_argument("--pages", type=int, default=None,
                        help="Scrape only first N pages (for testing)")
    args = parser.parse_args()

    session = requests.Session()

    if args.explore:
        explore(session)
    else:
        full_scrape(session, max_pages=args.pages)


if __name__ == "__main__":
    main()
