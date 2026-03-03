"""
fetch_lexibank.py
-----------------
Module: Lexibank CLDF data ingestion.

Fetches Turkic lexical data from Lexibank CLDF datasets hosted on GitHub.
This produces publication-quality data to replace or supplement the ASJP
baseline from Phase 1.

Primary source:
  lexibank/savelyevturkic  — Savelyev & Robbeets (2020), the most rigorous
  internal Turkic classification. Covers 40 languages, ~110 concepts, proper
  IPA with pre-tokenized Segments column.

Secondary source (concept gap-filling):
  lexibank/northeuralex    — NorthEuraLex (Dellert et al. 2020), broad Eurasian
  coverage, 1,016 concepts, IPA. Covers all 9 of our target languages.

CLDF format (all lexibank datasets):
  cldf/
    forms.csv       — core: Language_ID, Parameter_ID, Form, Segments, Loan
    languages.csv   — Language_ID → Name, Glottocode, ISO639P3code, Latitude, Longitude
    parameters.csv  — Parameter_ID → Name, Concepticon_ID, Concepticon_Gloss

Merge strategy:
  1. Load savelyevturkic as primary (richer IPA, cognate-annotated).
  2. Load northeuralex as secondary.
  3. For each (language, concept) pair: use savelyevturkic if available,
     fill gaps from northeuralex, flag source in a 'dataset' column.
  4. Merge with ASJP Phase 1 output for any remaining gaps.
  5. Output a unified DataFrame with source provenance per row.
"""

import io
import logging
import zipfile
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── Dataset definitions ───────────────────────────────────────────────────────

LEXIBANK_DATASETS = {
    "savelyevturkic": {
        "label":   "Savelyev & Robbeets (2020)",
        "zip_url": "https://github.com/lexibank/savelyevturkic/archive/refs/heads/master.zip",
        "cldf_subdir": "savelyevturkic-master/cldf",
        "priority": 1,
    },
    "northeuralex": {
        "label":   "NorthEuraLex (Dellert et al. 2020)",
        "zip_url": "https://github.com/lexibank/northeuralex/archive/refs/heads/master.zip",
        "cldf_subdir": "northeuralex-master/cldf",
        "priority": 2,
    },
}

# Cache directory
LEXIBANK_CACHE_DIR = Path("output") / "lexibank_cache"

# Output paths
LEXIBANK_CSV     = Path("output") / "turkic_lexibank.csv"
MERGED_CSV       = Path("output") / "turkic_merged_phase1.csv"
LEXIBANK_REPORT  = Path("output") / "lexibank_summary_report.txt"

# ── Target language mapping: canonical name → Glottocodes + ISO codes ─────────
# Glottocodes are stable identifiers across all Lexibank datasets.
# ISO codes as fallback.
TURKIC_GLOTTOCODES = {
    "Turkish":     ["nucl1301"],
    "Uzbek":       ["uzbe1247"],
    "Kazakh":      ["kaza1248"],
    "Kyrgyz":      ["kirg1245"],
    "Uyghur":      ["uigh1240"],
    "Yakut":       ["yaku1245"],
    "Chuvash":     ["chuv1255"],
    "Azerbaijani": ["nort2697", "sout2697"],  # North Azerbaijani preferred
    "Turkmen":     ["turk1304"],
}

# Swadesh-40 concept mapping via Concepticon IDs
# These are stable cross-dataset identifiers.
# Concepticon IDs verified against savelyevturkic parameters.csv (primary source).
# Where savelyevturkic uses a different CID than the canonical Concepticon value,
# we store both in SWADESH40_CONCEPTICON_ALIASES below.
SWADESH40_CONCEPTICON: dict[str, int] = {
    "I":          1209,   # confirmed
    "you_2sg":    1215,   # confirmed (THOU in savelyevturkic)
    "we":         1212,   # savelyevturkic uses 1212 (1PL pronoun); canonical=1232
    "this":       1214,   # savelyevturkic uses 1214; canonical=1421
    "that":       78,     # savelyevturkic uses 78; canonical=2166
    "who":        1235,   # savelyevturkic uses 1235; canonical=1371
    "what":       1236,   # savelyevturkic uses 1236; canonical=1372
    "not":        1240,   # savelyevturkic uses 1240; canonical=1412
    "all":        1532,   # confirmed (not present in savelyevturkic — gap)
    "many":       1198,   # savelyevturkic uses 1198; canonical=1199
    "one":        1493,   # confirmed
    "two":        1498,   # confirmed
    "big":        1202,   # savelyevturkic uses 1202; canonical=1246
    "long":       1203,   # savelyevturkic uses 1203; canonical=2122
    "small":      1246,   # savelyevturkic uses 1246; canonical=1256
    "woman":      962,    # savelyevturkic uses 962; canonical=1480
    "man":        1554,   # savelyevturkic uses 1554; canonical=1285
    "person":     683,    # savelyevturkic uses 683; canonical=1264
    "fish":       227,    # savelyevturkic uses 227; canonical=80
    "bird":       937,    # savelyevturkic uses 937; canonical=1278
    "dog":        2009,   # savelyevturkic uses 2009; canonical=1058
    "louse":      1392,   # savelyevturkic uses 1392; canonical=1282
    "tree":       906,    # savelyevturkic uses 906; canonical=1357
    "seed":       714,    # savelyevturkic uses 714; canonical=1294
    "leaf":       628,    # savelyevturkic uses 628; canonical=1284
    "root":       670,    # savelyevturkic uses 670; canonical=1306
    "bark":       1204,   # savelyevturkic uses 1204; canonical=1241
    "skin":       763,    # savelyevturkic uses 763; canonical=1314
    "flesh":      634,    # savelyevturkic=MEAT(634); no FLESH entry — closest match
    "blood":      946,    # savelyevturkic uses 946; canonical=1250
    "bone":       1394,   # savelyevturkic uses 1394; canonical=1253
    "grease":     323,    # savelyevturkic=FAT(323); no GREASE — closest match
    "egg":        744,    # savelyevturkic uses 744; canonical=1266
    "horn":       1393,   # savelyevturkic uses 1393; canonical=1279
    "tail":       1220,   # savelyevturkic uses 1220; canonical=1323
    "feather":    1201,   # savelyevturkic uses 1201; canonical=1269
    "hair":       2648,   # savelyevturkic uses 2648 (HAIR HEAD); canonical=1275
    "head":       1256,   # savelyevturkic uses 1256; canonical=1277
    "ear":        1247,   # savelyevturkic uses 1247; canonical=1265
    "eye":        1248,   # savelyevturkic uses 1248; canonical=1267
}

# Inverse: Concepticon_ID → our gloss label
CONCEPTICON_TO_GLOSS: dict[int, str] = {v: k for k, v in SWADESH40_CONCEPTICON.items()}


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Download and cache dataset zip
# ─────────────────────────────────────────────────────────────────────────────

def fetch_dataset_zip(dataset_key: str, force: bool = False) -> Optional[Path]:
    """
    Download a Lexibank dataset zip and extract it to the cache directory.
    Returns the path to the extracted cldf/ subdirectory, or None on failure.
    """
    config     = LEXIBANK_DATASETS[dataset_key]
    zip_url    = config["zip_url"]
    cldf_sub   = config["cldf_subdir"]
    cache_path = LEXIBANK_CACHE_DIR / dataset_key

    if not force and cache_path.exists() and any(cache_path.iterdir()):
        logger.info(f"  [{dataset_key}] Using cached data at {cache_path}")
        return cache_path

    logger.info(f"  [{dataset_key}] Downloading from {zip_url}...")
    try:
        resp = requests.get(zip_url, timeout=120, stream=True)
        resp.raise_for_status()
        content = b"".join(resp.iter_content(chunk_size=1_048_576))
        logger.info(f"  [{dataset_key}] Downloaded {len(content)/1e6:.1f} MB")
    except Exception as exc:
        logger.warning(f"  [{dataset_key}] Download failed: {exc}")
        return None

    cache_path.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            # Extract only the cldf/ subdirectory to save space
            cldf_members = [m for m in zf.namelist() if m.startswith(cldf_sub)]
            if not cldf_members:
                logger.warning(f"  [{dataset_key}] cldf subdir '{cldf_sub}' not found in zip.")
                logger.warning(f"  [{dataset_key}] Available paths: {zf.namelist()[:10]}")
                return None
            for member in cldf_members:
                # Strip the leading subdir prefix so files land in cache_path/
                relative = member[len(cldf_sub):].lstrip("/")
                if not relative:
                    continue
                target = cache_path / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                if not member.endswith("/"):
                    target.write_bytes(zf.read(member))
        logger.info(f"  [{dataset_key}] Extracted to {cache_path}")
        return cache_path
    except Exception as exc:
        logger.warning(f"  [{dataset_key}] Extraction failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Load and parse a CLDF dataset
# ─────────────────────────────────────────────────────────────────────────────

def load_cldf_dataset(cldf_dir: Path, dataset_key: str) -> Optional[pd.DataFrame]:
    """
    Load forms.csv, languages.csv, parameters.csv from a CLDF directory.
    Returns a DataFrame filtered to our target languages and Swadesh-40 concepts,
    with columns: language, gloss, form, ipa_tokens, loan_flag, dataset.

    Handles variation in column names across Lexibank datasets.
    """
    forms_path  = cldf_dir / "forms.csv"
    langs_path  = cldf_dir / "languages.csv"
    params_path = cldf_dir / "parameters.csv"

    # Check for alternate filenames
    for alt in ("FormTable.csv", "wordlist-forms.csv"):
        if not forms_path.exists():
            forms_path = cldf_dir / alt

    if not forms_path.exists():
        logger.warning(f"  [{dataset_key}] forms.csv not found in {cldf_dir}")
        logger.warning(f"  Available: {[f.name for f in cldf_dir.iterdir()]}")
        return None

    logger.info(f"  [{dataset_key}] Loading CLDF from {cldf_dir}")

    # Load tables
    forms  = pd.read_csv(forms_path,  encoding="utf-8", low_memory=False, dtype=str)
    langs  = pd.read_csv(langs_path,  encoding="utf-8", low_memory=False, dtype=str) if langs_path.exists() else pd.DataFrame()
    params = pd.read_csv(params_path, encoding="utf-8", low_memory=False, dtype=str) if params_path.exists() else pd.DataFrame()

    logger.info(f"  [{dataset_key}] forms={len(forms):,} rows, "
                f"languages={len(langs)}, parameters={len(params)}")

    # ── Normalize column names (datasets vary) ────────────────────────────────
    forms.columns  = [c.strip() for c in forms.columns]
    if not langs.empty:
        langs.columns  = [c.strip() for c in langs.columns]
    if not params.empty:
        params.columns = [c.strip() for c in params.columns]

    # ── Build Glottocode → canonical language name map ─────────────────────────
    glotto_to_canonical: dict[str, str] = {}
    iso_to_canonical: dict[str, str] = {}
    name_to_canonical: dict[str, str] = {}

    for canonical, glottocodes in TURKIC_GLOTTOCODES.items():
        for gc in glottocodes:
            glotto_to_canonical[gc] = canonical
        # Also map lowercase canonical name for fuzzy matching
        name_to_canonical[canonical.lower()] = canonical

    # ── Build Language_ID → canonical name map via languages.csv ──────────────
    lang_id_to_canonical: dict[str, str] = {}
    if not langs.empty:
        glotto_col = _find_col(langs, ["Glottocode", "glottocode", "GLOTTOCODE"])
        iso_col    = _find_col(langs, ["ISO639P3code", "iso639P3code", "ISO_code", "iso"])
        name_col   = _find_col(langs, ["Name", "name", "Language_name"])
        id_col     = _find_col(langs, ["ID", "id"])

        for _, row in langs.iterrows():
            lang_id = str(row.get(id_col, "")).strip() if id_col else ""
            if not lang_id:
                continue

            # Try Glottocode match
            gc = str(row.get(glotto_col, "")).strip() if glotto_col else ""
            if gc in glotto_to_canonical:
                lang_id_to_canonical[lang_id] = glotto_to_canonical[gc]
                continue

            # Try ISO match
            iso = str(row.get(iso_col, "")).strip() if iso_col else ""
            if iso in iso_to_canonical:
                lang_id_to_canonical[lang_id] = iso_to_canonical[iso]
                continue

            # Try name fuzzy match
            name = str(row.get(name_col, "")).strip().lower() if name_col else ""
            for key, canonical in name_to_canonical.items():
                if key in name or name in key:
                    lang_id_to_canonical[lang_id] = canonical
                    break

    # ── Build Parameter_ID → gloss map via Concepticon IDs ───────────────────
    param_id_to_gloss: dict[str, str] = {}
    if not params.empty:
        cid_col  = _find_col(params, ["Concepticon_ID", "concepticon_id", "CONCEPTICON_ID"])
        name_col = _find_col(params, ["Concepticon_Gloss", "Name", "name", "Gloss"])
        id_col   = _find_col(params, ["ID", "id"])

        for _, row in params.iterrows():
            pid = str(row.get(id_col, "")).strip() if id_col else ""
            if not pid:
                continue

            # Try Concepticon ID match
            if cid_col:
                cid_raw = str(row.get(cid_col, "")).strip()
                try:
                    cid = int(float(cid_raw))
                    if cid in CONCEPTICON_TO_GLOSS:
                        param_id_to_gloss[pid] = CONCEPTICON_TO_GLOSS[cid]
                        continue
                except (ValueError, TypeError):
                    pass

            # Fallback: match by concept name string
            if name_col:
                gloss_raw = str(row.get(name_col, "")).strip().upper()
                for our_gloss in SWADESH40_CONCEPTICON:
                    if our_gloss.upper() in gloss_raw or gloss_raw in our_gloss.upper():
                        param_id_to_gloss[pid] = our_gloss
                        break

    # ── Filter and build output rows ──────────────────────────────────────────
    lang_col  = _find_col(forms, ["Language_ID", "language_id", "LANGUAGE_ID"])
    param_col = _find_col(forms, ["Parameter_ID", "parameter_id", "PARAMETER_ID"])
    form_col  = _find_col(forms, ["Form", "form", "Value", "value"])
    seg_col   = _find_col(forms, ["Segments", "segments", "SEGMENTS"])
    loan_col  = _find_col(forms, ["Loan", "loan", "LOAN", "Borrowed"])

    if not all([lang_col, param_col, form_col]):
        logger.warning(f"  [{dataset_key}] Required columns missing. "
                       f"Available: {list(forms.columns)}")
        return None

    rows = []
    for _, row in forms.iterrows():
        lang_id  = str(row.get(lang_col, "")).strip()
        param_id = str(row.get(param_col, "")).strip()

        canonical = lang_id_to_canonical.get(lang_id)
        gloss     = param_id_to_gloss.get(param_id)

        if not canonical or not gloss:
            continue

        form_raw = str(row.get(form_col, "")).strip()
        if not form_raw or form_raw in ("nan", "XXX", ""):
            continue

        # IPA tokens: use Segments column if available (space-separated)
        tokens: list[str] = []
        if seg_col:
            seg_raw = str(row.get(seg_col, "")).strip()
            if seg_raw and seg_raw != "nan":
                tokens = [t for t in seg_raw.split() if t and t != "+"]

        # Loan flag
        loan = False
        if loan_col:
            loan_raw = str(row.get(loan_col, "")).strip().lower()
            loan = loan_raw in ("true", "1", "yes")

        rows.append({
            "language":   canonical,
            "gloss":      gloss,
            "form":       form_raw,
            "ipa_form":   form_raw,          # Form in Lexibank IS IPA
            "ipa_tokens": tokens,
            "loan_flag":  loan,
            "dataset":    dataset_key,
            "anomaly":    None,
        })

    if not rows:
        logger.warning(f"  [{dataset_key}] No matching rows after filtering.")
        return None

    df = pd.DataFrame(rows)
    logger.info(f"  [{dataset_key}] Extracted {len(df):,} matching rows "
                f"across {df['language'].nunique()} languages, "
                f"{df['gloss'].nunique()} glosses.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Merge datasets + ASJP baseline
# ─────────────────────────────────────────────────────────────────────────────

def build_merged_dataset(
    asjp_csv_path: Optional[Path] = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Build the merged publication-quality DataFrame:
      1. savelyevturkic (priority 1)
      2. northeuralex gap-fill (priority 2)
      3. ASJP gap-fill (priority 3, if asjp_csv_path provided)

    Returns unified DataFrame with columns:
      language, gloss, form, ipa_form, ipa_tokens, loan_flag, dataset, anomaly
    """
    LEXIBANK_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []

    # Load each Lexibank dataset in priority order
    for key in sorted(LEXIBANK_DATASETS, key=lambda k: LEXIBANK_DATASETS[k]["priority"]):
        cldf_dir = fetch_dataset_zip(key, force=force_download)
        if cldf_dir is None:
            logger.warning(f"Skipping {key} — could not retrieve data.")
            continue
        df = load_cldf_dataset(cldf_dir, key)
        if df is not None:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            "No Lexibank data could be loaded. Check network access to GitHub "
            "or manually clone the repos into output/lexibank_cache/."
        )

    # Combine and deduplicate: keep highest-priority source per (language, gloss)
    combined = pd.concat(frames, ignore_index=True)

    # Sort by dataset priority so priority-1 rows come first
    priority_map = {k: v["priority"] for k, v in LEXIBANK_DATASETS.items()}
    combined["_priority"] = combined["dataset"].map(priority_map).fillna(99)
    combined = combined.sort_values("_priority")

    # Keep first occurrence per (language, gloss) — highest priority wins
    deduped = combined.drop_duplicates(subset=["language", "gloss"], keep="first")
    deduped = deduped.drop(columns=["_priority"]).reset_index(drop=True)

    # Gap-fill from ASJP if provided
    if asjp_csv_path and Path(asjp_csv_path).exists():
        logger.info("Gap-filling from ASJP Phase 1 data...")
        asjp_df = pd.read_csv(asjp_csv_path, dtype=str)
        asjp_df["dataset"]   = "asjp"
        asjp_df["loan_flag"] = False

        # Parse ipa_tokens back to list
        if "ipa_tokens" in asjp_df.columns:
            asjp_df["ipa_tokens"] = asjp_df["ipa_tokens"].apply(
                lambda x: x.split() if isinstance(x, str) and x.strip() else []
            )

        covered = set(zip(deduped["language"], deduped["gloss"]))
        asjp_fill = asjp_df[
            asjp_df.apply(
                lambda r: (r["language"], r["gloss"]) not in covered
                and pd.notna(r.get("form"))
                and str(r.get("form", "")).strip() not in ("", "nan"),
                axis=1
            )
        ]
        if not asjp_fill.empty:
            logger.info(f"  ASJP gap-fill: {len(asjp_fill)} additional rows.")
            deduped = pd.concat([deduped, asjp_fill], ignore_index=True)

    return deduped


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Summary report
# ─────────────────────────────────────────────────────────────────────────────

def report_merged(df: pd.DataFrame) -> str:
    all_langs    = sorted(df["language"].unique())
    all_glosses  = set(SWADESH40_CONCEPTICON.keys())
    n_concepts   = len(all_glosses)

    lines = [
        "=" * 70,
        "  LEXIBANK MERGED DATASET REPORT",
        "=" * 70,
        f"  Total rows       : {len(df):,}",
        f"  Languages        : {len(all_langs)}",
        f"  Unique glosses   : {df['gloss'].nunique()}",
        f"  Loan-flagged rows: {int(df['loan_flag'].sum()):,}",
        "",
        f"  {'LANGUAGE':<16} {'PRESENT':>7}  {'COV%':>6}  {'LOANS':>6}  "
        f"{'SAVELY':>7}  {'NORTHEURA':>9}  {'ASJP':>5}",
        "-" * 70,
    ]

    for lang in all_langs:
        sub = df[df["language"] == lang]
        present = sub["gloss"].nunique()
        cov     = round(100 * present / n_concepts, 1)
        loans   = int(sub["loan_flag"].sum())
        n_sav   = int((sub["dataset"] == "savelyevturkic").sum())
        n_neu   = int((sub["dataset"] == "northeuralex").sum())
        n_asjp  = int((sub["dataset"] == "asjp").sum())
        lines.append(
            f"  {lang:<16} {present:>7}  {cov:>5.1f}%  {loans:>6}  "
            f"{n_sav:>7}  {n_neu:>9}  {n_asjp:>5}"
        )

    lines += ["", "-" * 70, "  MISSING CONCEPTS BY LANGUAGE", "-" * 70]
    for lang in all_langs:
        covered  = set(df[df["language"] == lang]["gloss"])
        missing  = sorted(all_glosses - covered)
        if missing:
            lines.append(f"  {lang}: {', '.join(missing)}")
        else:
            lines.append(f"  {lang}: (full Swadesh-40 coverage)")

    lines += ["", "=" * 70, "  END OF LEXIBANK REPORT", "=" * 70]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    """Return the first column name from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None
