"""
Retry specific failed pages from the EDAL master scrape.
Hardcoded from the console output of the first run.

Usage:
  python transeurasian\\scrape_edal_retry.py
"""

import csv
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
MASTER_CSV = OUTPUT / "edal_altaic_master.csv"

# These are the first= values that 502'd during the initial scrape.
# Taken directly from the console output.
FAILED_FIRST_VALUES = [
    261, 281, 301, 321, 341,
    1361, 1381, 1401, 1421, 1441, 1461, 1521,
    1581, 1601, 1621, 1641, 1661, 1681, 1701, 1721,
    2801,
]

RECORDS_PER_PAGE = 20
TOTAL_PAGES = 141
BASE_URL = "https://starlingdb.org/cgi-bin/response.cgi"
BASENAME = "/data/alt/altet"
DELAY_SECONDS = 3.0
MAX_RETRIES = 5
BACKOFF_BASE = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research scraper; computational linguistics project; "
                  "contact: lukemgillespie@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

# ── Inline the field-label map so we don't import scrape_edal ────────
FAMILY_LABEL_MAP = {
    "Turkic":        ("proto_turkic",   "turkic_ref"),
    "Mongolian":     ("proto_mongolic", "mongolic_ref"),
    "Tungus-Manchu": ("proto_tungusic", "tungusic_ref"),
    "Korean":        ("proto_korean",   "korean_ref"),
    "Japanese":      ("proto_japanese",  "japanese_ref"),
}

def fetch_page(first_n, session):
    params = {"root": "config", "basename": BASENAME, "first": str(first_n)}
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(BASE_URL, params=params, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            elif resp.status_code in (429, 502, 503):
                wait = BACKOFF_BASE ** (attempt + 1)
                print(f"    HTTP {resp.status_code}, backing off {wait:.0f}s "
                      f"(attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
            else:
                print(f"    HTTP {resp.status_code}, giving up")
                return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** (attempt + 1)
            print(f"    Error: {e}, backing off {wait:.0f}s")
            time.sleep(wait)
    return None


def parse_records_from_page(html):
    soup = BeautifulSoup(html, "html.parser")
    record_divs = soup.find_all("div", class_="results_record")
    records = []
    for rec_div in record_divs:
        try:
            record = parse_single_record(rec_div)
            if record:
                records.append(record)
        except Exception as e:
            print(f"    Parse error: {e}")
    return records


def parse_single_record(rec_div):
    record = {
        "proto_altaic": "", "meaning": "", "russian_meaning": "",
        "proto_turkic": "", "turkic_ref": "",
        "proto_mongolic": "", "mongolic_ref": "",
        "proto_tungusic": "", "tungusic_ref": "",
        "proto_korean": "", "korean_ref": "",
        "proto_japanese": "", "japanese_ref": "",
        "comments": "",
    }
    field_divs = rec_div.find_all("div", recursive=False)
    for field_div in field_divs:
        if "subquery_link" in field_div.get("class", []):
            continue
        fld_span = field_div.find("span", class_="fld")
        if not fld_span:
            continue
        label = fld_span.get_text(strip=True).rstrip(":").strip()

        if label == "Proto-Altaic":
            us = field_div.find("span", class_="unicode")
            if us: record["proto_altaic"] = us.get_text(strip=True)
        elif label == "Meaning":
            us = field_div.find("span", class_="unicode")
            if us: record["meaning"] = us.get_text(strip=True)
        elif label == "Russian meaning":
            us = field_div.find("span", class_="unicode")
            if us: record["russian_meaning"] = us.get_text(strip=True)
        elif label == "Comments":
            us = field_div.find("span", class_="unicode")
            if us: record["comments"] = us.get_text(strip=True)
        elif label in FAMILY_LABEL_MAP:
            proto_col, ref_col = FAMILY_LABEL_MAP[label]
            link = field_div.find("a")
            if link:
                us = link.find("span", class_="unicode")
                if us: record[proto_col] = us.get_text(strip=True)
                href = link.get("href", "")
                if href: record[ref_col] = href
            else:
                us = field_div.find("span", class_="unicode")
                if us: record[proto_col] = us.get_text(strip=True)

    if record["proto_altaic"] or record["meaning"]:
        return record
    return None


def main():
    print(f"=== RETRY: {len(FAILED_FIRST_VALUES)} failed pages ===")

    # Load existing records
    existing = []
    if MASTER_CSV.exists():
        with open(MASTER_CSV, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                existing.append(dict(row))
    print(f"Existing records: {len(existing)}")

    # Build a mapping: for each successful page, what first= value did it have?
    # All pages: first=1, 21, 41, ..., 2801
    all_first = [1 + i * RECORDS_PER_PAGE for i in range(TOTAL_PAGES)]
    failed_set = set(FAILED_FIRST_VALUES)
    success_first = [f for f in all_first if f not in failed_set]

    # Tag existing records with their source page's first= value
    for idx, rec in enumerate(existing):
        page_idx = idx // RECORDS_PER_PAGE
        if page_idx < len(success_first):
            rec["_sort_key"] = success_first[page_idx]
        else:
            # Records beyond expected — put at end
            rec["_sort_key"] = success_first[-1] if success_first else 0

    # Fetch failed pages
    session = requests.Session()
    new_records = []
    still_failed = []

    for first_n in FAILED_FIRST_VALUES:
        print(f"  Fetching first={first_n}...")
        html = fetch_page(first_n, session)
        if html is None:
            print(f"    STILL FAILED")
            still_failed.append(first_n)
        else:
            recs = parse_records_from_page(html)
            print(f"    Got {len(recs)} records")
            for r in recs:
                r["_sort_key"] = first_n
            new_records.extend(recs)
        time.sleep(DELAY_SECONDS)

    print(f"\nRecovered: {len(new_records)} records")
    if still_failed:
        print(f"Still failed: {still_failed}")

    if not new_records:
        print("No new records. Done.")
        return

    # Merge and sort by page order
    all_records = existing + new_records
    all_records.sort(key=lambda r: int(r.get("_sort_key", 99999)))

    # Re-number
    for i, rec in enumerate(all_records):
        rec["record_id"] = i + 1
        rec.pop("_sort_key", None)

    # Write
    fieldnames = [
        "record_id", "proto_altaic", "meaning", "russian_meaning",
        "proto_turkic", "turkic_ref",
        "proto_mongolic", "mongolic_ref",
        "proto_tungusic", "tungusic_ref",
        "proto_korean", "korean_ref",
        "proto_japanese", "japanese_ref",
        "comments",
    ]
    with open(MASTER_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in all_records:
            row = {k: rec.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\nMerged CSV: {MASTER_CSV}")
    print(f"  Total records: {len(all_records)} (was {len(existing)})")

    filled = {}
    for fam in ["turkic", "mongolic", "tungusic", "korean", "japanese"]:
        filled[fam] = sum(1 for r in all_records if r.get(f"proto_{fam}"))
    print("  Family coverage:")
    for fam, c in filled.items():
        print(f"    {fam:12s}: {c:4d} / {len(all_records)} "
              f"({100*c/len(all_records):.1f}%)")


if __name__ == "__main__":
    main()
