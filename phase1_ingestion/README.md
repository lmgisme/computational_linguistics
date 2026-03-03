# Phase 1 — Turkic Substrate Detection: Data Ingestion & Cleaning

## Project structure

```
phase1_ingestion/
├── config.py              # All tunable parameters (URLs, target languages, paths)
├── asjp_symbols.py        # ASJP → IPA symbol mapping table
├── fetch_asjp.py          # Module 1: download and parse the ASJP flat file
├── normalize_ipa.py       # Module 2: ASJP → IPA conversion and tokenization
├── build_dataframe.py     # Module 3: DataFrame assembly and statistics
├── main.py                # Pipeline entry point
├── setup_environment.py   # Dependency checker and installer
└── output/                # Generated at runtime
    ├── asjp_raw_cache.txt          # Cached ASJP download
    ├── turkic_swadesh_phase1.csv   # Final clean DataFrame
    ├── phase1_summary_report.txt   # Coverage and anomaly report
    └── phase1_run.log              # Full run log
```

## Setup

```bash
cd phase1_ingestion
python setup_environment.py
```

This checks Python ≥ 3.9, installs `pandas`, `requests`, and `lingpy` if
missing, and runs a quick smoke test.

## Running the pipeline

```bash
python main.py
```

Optional flags:
- `--force-download` : Re-fetch ASJP even if a local cache exists
- `--no-normalize`   : Skip ASJP → IPA conversion; store raw forms

## Output DataFrame columns

| Column       | Type   | Description                                      |
|--------------|--------|--------------------------------------------------|
| `language`   | str    | Canonical language name (e.g. "Turkish")         |
| `gloss`      | str    | Swadesh concept label (e.g. "blood")             |
| `form`       | str    | Raw ASJP orthographic form                       |
| `ipa_form`   | str    | IPA string after symbol substitution             |
| `ipa_tokens` | list   | Discrete IPA phone tokens (LingPy or regex)      |
| `anomaly`    | str    | Anomaly flag (NaN if clean)                      |

## Target languages

Turkish, Uzbek, Kazakh, Kyrgyz, Uyghur, Yakut (Sakha), Chuvash, Azerbaijani, Turkmen

## Notes on ASJP data

- Source: ASJP v19 (Wichmann et al. 2022)
- 40-item Swadesh list per doculect
- ASJP orthography is a compact ASCII representation; the symbol map in
  `asjp_symbols.py` handles the most common substitutions. Some phonemic
  distinctions are collapsed in ASJP that IPA preserves — flag these for
  Phase 2 review.
- Loan-word markers (`%`) and compound markers (`$`) are stripped before IPA
  conversion; track these in Phase 2 if contact-induced borrowing is a
  substrate hypothesis variable.

## Phase 2 targets (not in scope here)

- Cognate detection via LingPy's LexStat
- Sound correspondence matrices
- Substrate candidate vocabulary isolation
