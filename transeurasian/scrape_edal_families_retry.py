"""
Retry failed family sub-database entries from the EDAL family scrape.
Parses the error log for specific (family, text_number) failures,
re-fetches them, and appends to the existing family CSVs.

Usage:
  python transeurasian\\scrape_edal_families_retry.py
"""

import csv
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

ROOT   = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
ERROR_LOG = OUTPUT / "edal_family_scrape_errors.log"

RESPONSE_URL = "https://starlingdb.org/cgi-bin/response.cgi"
DELAY_SECONDS = 3.0
MAX_RETRIES   = 5
BACKOFF_BASE  = 2.0

HEADERS = {
    "User-Agent": "Mozilla/5.0 (research scraper; computational linguistics project; "
                  "contact: lukemgillespie@gmail.com)",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Encoding": "gzip, deflate",
}

FAMILY_CONFIG = {
    "mongolic": {
        "basename": "/data/alt/monget",
        "proto_label": "Proto-Mongolian",
        "csv": OUTPUT / "edal_mongolic_reflexes.csv",
        "proto_col": "proto_mongolic",
    },
    "tungusic": {
        "basename": "/data/alt/tunget",
        "proto_label": "Proto-Tungus-Manchu",
        "csv": OUTPUT / "edal_tungusic_reflexes.csv",
        "proto_col": "proto_tungusic",
    },
    "korean": {
        "basename": "/data/alt/koret",
        "proto_label": "Proto-Korean",
        "csv": OUTPUT / "edal_korean_reflexes.csv",
        "proto_col": "proto_korean",
    },
    "japanese": {
        "basename": "/data/alt/japet",
        "proto_label": "Proto-Japanese",
        "csv": OUTPUT / "edal_japanese_reflexes.csv",
        "proto_col": "proto_japanese",
    },
}

SKIP_LABELS = {
    "Meaning", "Russian meaning", "Altaic etymology", "Comments",
    "References", "Nostratic",
    "Proto-Mongolian", "Proto-Tungus-Manchu", "Proto-Korean", "Proto-Japanese",
    "Proto-Turkic",
}

# ── Network ──────────────────────────────────────────────────────────
def fetch_entry(basename, text_number, session):
    params = {"root": "config", "basename": basename,
              "single": "1", "text_number": str(text_number)}
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(RESPONSE_URL, params=params,
                               headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                resp.encoding = "utf-8"
                return resp.text
            elif resp.status_code in (429, 502, 503):
                wait = BACKOFF_BASE ** (attempt + 1)
                print(f"    HTTP {resp.status_code}, backing off {wait:.0f}s "
                      f"(attempt {attempt+1})")
                time.sleep(wait)
            else:
                return None
        except requests.RequestException as e:
            wait = BACKOFF_BASE ** (attempt + 1)
            print(f"    Error: {e}, backing off {wait:.0f}s")
            time.sleep(wait)
    return None


# ── Parser ───────────────────────────────────────────────────────────
def parse_family_entry(html, family_key):
    soup = BeautifulSoup(html, "html.parser")
    config = FAMILY_CONFIG[family_key]
    proto_label = config["proto_label"]

    record_divs = soup.find_all("div", class_="results_record")
    target_div = None
    for rd in record_divs:
        for fld in rd.find_all("span", class_="fld"):
            if fld.get_text(strip=True).rstrip(":").strip() == proto_label:
                target_div = rd
                break
        if target_div:
            break

    if not target_div:
        if len(record_divs) == 1:
            target_div = record_divs[0]
        else:
            return "", "", []

    proto_form = ""
    entry_meaning = ""
    reflexes = []

    for field_div in target_div.find_all("div", recursive=False):
        if "subquery_link" in field_div.get("class", []):
            continue
        fld_span = field_div.find("span", class_="fld")
        if not fld_span:
            continue
        label = fld_span.get_text(strip=True).rstrip(":").strip()
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
            reflexes.append({"language": label, "form": value, "meaning": ""})

    return proto_form, entry_meaning, reflexes


# ── Parse error log ─────────────────────────────────────────────────
def find_failed_entries():
    """
    Parse the error log for lines like:
      WARNING  mongolic text_number=144 (altaic #129): FAILED
    Returns dict: family -> list of (altaic_record_id, text_number)
    """
    failed = {}
    basename_to_family = {
        "/data/alt/monget": "mongolic",
        "/data/alt/tunget": "tungusic",
        "/data/alt/koret":  "korean",
        "/data/alt/japet":  "japanese",
    }

    if not ERROR_LOG.exists():
        print("Error log not found.")
        return failed

    with open(ERROR_LOG, "r", encoding="utf-8") as f:
        for line in f:
            # Match: "mongolic text_number=144 (altaic #129): FAILED"
            m = re.search(
                r"(mongolic|tungusic|korean|japanese)\s+text_number=(\d+)\s+"
                r"\(altaic\s+#(\d+)\):\s+FAILED",
                line
            )
            if m:
                family = m.group(1)
                text_num = m.group(2)
                record_id = m.group(3)
                if family not in failed:
                    failed[family] = []
                failed[family].append((record_id, text_num))

    return failed


def main():
    failed = find_failed_entries()
    if not failed:
        print("No failed entries found in error log.")
        return

    total_failed = sum(len(v) for v in failed.values())
    print(f"=== FAMILY RETRY: {total_failed} failed entries ===")
    for fam, entries in failed.items():
        print(f"  {fam}: {len(entries)} to retry")

    session = requests.Session()

    for family, entries in failed.items():
        config = FAMILY_CONFIG[family]
        basename = config["basename"]
        proto_col = config["proto_col"]
        csv_path = config["csv"]

        print(f"\n--- {family.upper()}: {len(entries)} retries ---")

        # Load existing CSV
        existing_rows = []
        existing_ids = set()
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    existing_rows.append(dict(row))
                    existing_ids.add(
                        (row.get("altaic_record_id", ""),
                         row.get("daughter_language", ""),
                         row.get("reflex_form", ""))
                    )
            print(f"  Existing rows: {len(existing_rows)}")

        new_rows = []
        still_failed = []

        for i, (record_id, text_num) in enumerate(entries):
            print(f"  [{i+1}/{len(entries)}] {family} text_number={text_num} "
                  f"(altaic #{record_id})...", end=" ")

            html = fetch_entry(basename, text_num, session)
            if html is None:
                print("STILL FAILED")
                still_failed.append((record_id, text_num))
            else:
                proto, meaning, reflexes = parse_family_entry(html, family)
                print(f"OK ({len(reflexes)} reflexes)")
                for ref in reflexes:
                    new_rows.append({
                        "altaic_record_id": record_id,
                        proto_col: proto,
                        "daughter_language": ref["language"],
                        "reflex_form": ref["form"],
                        "meaning": meaning,
                    })

            time.sleep(DELAY_SECONDS)

        if still_failed:
            print(f"  Still failed: {len(still_failed)} entries")

        if new_rows:
            # Append new rows to CSV
            all_rows = existing_rows + new_rows
            fieldnames = [
                "altaic_record_id", proto_col,
                "daughter_language", "reflex_form", "meaning",
            ]
            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames,
                                        extrasaction="ignore")
                writer.writeheader()
                for row in all_rows:
                    writer.writerow(row)
            print(f"  Wrote {len(all_rows)} total rows "
                  f"(was {len(existing_rows)}, added {len(new_rows)})")
        else:
            print(f"  No new rows recovered.")

    # Final summary
    print(f"\n=== SUMMARY ===")
    for family, config in FAMILY_CONFIG.items():
        csv_path = config["csv"]
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                count = sum(1 for _ in f) - 1  # minus header
            print(f"  {family:12s}: {count:6d} reflex rows")


if __name__ == "__main__":
    main()
