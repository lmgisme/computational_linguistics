# Turkic Substrate Detection Pipeline

A computational pipeline for detecting substrate loanwords in the Turkic language family using open-access lexical databases and tools from evolutionary biology adapted for historical linguistics.

**Associated paper:** *A Computational Pipeline for Substrate Detection in the Turkic Language Family: Methodology, Validation, and a Null Result* — Luke M. Gillespie (Independent Researcher)

---

## What This Is

This project asks whether populations absorbed during Turkic expansion left detectable lexical traces in the Turkic languages — and whether a fully automated, reproducible pipeline can find them. The pipeline builds a probabilistic sound correspondence model, identifies words that deviate from regular Turkic phonological patterns, and validates candidates through Bayesian phylogenetic inference and etymological triage.

The short answer is no confirmed substrate words were found. That null result is the main finding, and it is a substantive one: with a characterized sensitivity floor of 75% for phonologically distinct foreign material, the absence of substrate signal across both core and cultural vocabulary places a meaningful upper bound on substrate retention in the tested domains. The result is consistent with rapid linguistic assimilation during Turkic expansion.

The pipeline also makes four methodological contributions independent of the substrate question:

1. A reproducible, open-source substrate detection pipeline applicable to other language families
2. A per-branch threshold calibration method that reduces divergent-branch false positives by 40–89%
3. Quantification and correction of Persian/Arabic loan contamination effects on Turkic phylogenetic inference
4. Documented implementation solutions for non-obvious bugs in LingPy 2.6.13 and BEAST 2.7.8

---

## Languages Covered

Nine Turkic languages spanning all major branches:

| Language | Branch | Notes |
|---|---|---|
| Turkish | Oghuz | Reference language |
| Azerbaijani | Oghuz | |
| Turkmen | Oghuz | Thin NorthEuraLex coverage |
| Kazakh | Kipchak | |
| Kyrgyz | Kipchak | Thin NorthEuraLex coverage |
| Uzbek | Karluk | Heavy Persian/Arabic contact |
| Uyghur | Karluk | Thin NorthEuraLex coverage |
| Yakut/Sakha | Siberian | Most divergent Common Turkic language |
| Chuvash | Bulgar | Earliest-diverging branch; structural outgroup |

---

## Data Sources

- **savelyevturkic** (Savelyev & Robbeets 2020, via Lexibank) — primary source; publication-quality IPA, pre-tokenized, expert cognate judgments; covers the 40-item Swadesh core vocabulary
- **NorthEuraLex** (Dellert et al. 2020) — secondary source; 933 cultural vocabulary concepts (herding, metallurgy, kinship, agriculture); full coverage for 6 of 9 languages
- **ASJP v21** (Wichmann et al.) — final fallback

Sources are merged by priority (savelyevturkic > NorthEuraLex > ASJP) with provenance tracked per row. Core dataset: 353 rows, 9 languages, 40 glosses. Expanded dataset: 6,966 rows, 933 concepts.

---

## Pipeline Structure

The pipeline runs in four phases plus two infrastructure tasks. Each phase directory contains the scripts for that phase.

```
phase1_ingestion/       Data fetch, IPA normalization, merged dataframe
phase2_cognates/        Cognate detection, correspondence modeling
phase3_substrate/       Anomaly detection, regularity scoring, clustering
phase4_phylo/           BEAST XML construction, phylogenetic inference
phase5_substrate/       Proto-form comparison for surviving candidates
infra/                  Per-branch threshold calibration (6A), Uzbek loan filter (6B)
output/                 All intermediate and final outputs (tracked in repo)
```

### Phase 1 — Data Ingestion
Fetches savelyevturkic from Lexibank (CLDF zip), gap-fills with NorthEuraLex and ASJP, normalizes IPA to phoneme tokens, and builds the merged dataframe. Output: `turkic_merged_phase1.csv` (353 rows, 90–100% per-language coverage on 40 glosses).

### Phase 2 — Cognate Detection and Correspondence Modeling
Runs LexStat (LingPy 2.6.13) with Savelyev expert judgments taking priority in the hybrid cognate table (90.1% Savelyev-grounded). Extracts pairwise phoneme correspondences from SCA alignments. Output: `hybrid_cognates.csv`, `prob_model.json`, `correspondence_table.csv`.

### Phase 3 — Anomaly Detection
Computes regularity scores (mean log-probability under the correspondence model), applies statistical threshold (μ − 1.5σ), and clusters anomalous words by phonological feature vectors using Ward hierarchical clustering. Output: `regularity_scores.csv`, `anomalies.csv`, `substrate_clusters.csv`.

### Phase 4 — Bayesian Phylogenetics
Builds a BEAST 2.7.8 XML from the binary cognate matrix and runs phylogenetic inference. Lewis Mk substitution model, Gamma(4) rate heterogeneity, strict molecular clock, Yule tree prior, LogNormal root calibration (M=2100 BP, S=0.12). Output: `beast_turkic_mcc.tree`.

### Infrastructure 6A — Per-Branch Threshold Calibration
Replaces the family-wide anomaly threshold with per-branch baselines. Two-condition gate: score below branch-group mean − 1.5σ, AND below the Oghuz phoneme index threshold. Reduced Yakut candidates from 214 to 24 (89% reduction) and Chuvash from 95 to 57 (40% reduction) on the expanded dataset.

### Infrastructure 6B — Uzbek Loan Filter
Identified four Persian loanwords in the 40-item Uzbek Swadesh profile (χɔtin, daraχt, tuχum, goʃt). Removing them from the cognate matrix and rerunning BEAST corrected Uzbek's phylogenetic placement from anomalous early-diverging outgroup into the Kipchak clade at posterior 0.8673.

### Phase 5 — Proto-Form Comparison
Normalized Levenshtein distance on IPA tokens against Proto-Mongolic (Janhunen 2003) and Proto-Tungusic reconstructions for candidates surviving triage. All candidates resolved: Kazakh bɪjke = Turkic-internal derivation; Kazakh mʊdːe = Arabic loan; Yakut ɟaxtaɾ = unresolved open footnote pending Pekarskij (1959) verification.

---

## Key Results

**Phylogenetic topology recovered correctly:** Chuvash outgroup (posterior 1.0), Yakut second divergence, Kipchak clade posterior 0.9953, Azerbaijani–Turkish sisters at 0.9994. Tree height mean 2,010 BP, consistent with calibration.

**Two Bulgar sound laws independently recovered:** \*b > p devoicing and \*z > r rhotacism, the two diagnostic features of the Bulgar branch, recovered from raw data without prior specification.

**Uzbek misplacement corrected:** Four Persian loans in a 40-item profile were sufficient to move Uzbek to an incorrect phylogenetic position. Removal corrected placement at posterior 0.8673.

**Null substrate result:** No confirmed substrate words in 40 Swadesh items or 933 NorthEuraLex cultural vocabulary concepts. 340 anomaly candidates → 57 clusters → 2 shortlisted after triage → 0 confirmed after proto-form comparison.

**Pipeline sensitivity:** 75% detection rate on independently identified Persian loans (Category A, N=4). Known blind spot: phonologically nativized loans where individual phonemes are common in Turkic escape the token-level scorer.

---

## Requirements and Environment

```
Python 3.11 (required — LingPy 2.6.13 incompatible with 3.12+)
LingPy 2.6.13
BEAST 2.7.8
pandas, numpy, scipy
```

All data sources are open-access. See Data Availability below.

**Critical known bugs:**

- **LingPy `get_scorer()` crash:** Pass `runs=0` to LexStat. The permutation scorer crashes on Python 3.11 with a `TypeError` when `runs > 0`. The SCA fallback is methodologically acceptable for a shallow family like Turkic (~2,100 BP).
- **LexStat dict input:** Pass a TSV file path, not a Python dict object.
- **BEAST 2.7.8:** Use `dataType="binary"` not `"standard"`. Use strict clock — the relaxed clock (ORC) package is incompatible with v2.7.8. All `spec` attributes require fully qualified class paths. See `phase4_phylo/` for working XML.

---

## Repository Structure

```
output/
  turkic_merged_phase1.csv       Merged lexical data, 353 rows
  hybrid_cognates.csv            Savelyev-grounded cognate table
  prob_model.json                Correspondence model (1,349 distributions)
  correspondence_table.csv       Human-readable correspondence table
  regularity_scores.csv          Per-word regularity scores
  anomalies.csv                  Flagged anomalous words
  substrate_clusters.csv         Phonological clusters of anomalies
  northeuralex_merged.csv        Expanded 933-concept dataset, 6,966 rows
  anomalies_expanded.csv         Anomalies on expanded dataset
  beast_turkic_final.xml         BEAST input (loan-corrected)
  beast_turkic_mcc.tree          Maximum clade credibility tree
  task4_5_similarity_scores.csv  Proto-form comparison results
  project_overview_v2_2.docx     Running project documentation
```

---

## Limitations

- **Token-level scoring** assesses phoneme regularity without alignment context. Phonologically nativized loans escape detection because individual phonemes are common in Turkic even when the combination is etymologically foreign.
- **Thin-coverage languages:** Kyrgyz, Turkmen, and Uyghur are absent from NorthEuraLex. Cultural vocabulary analysis covers 6 of 9 target languages.
- **Category A sensitivity estimate** is based on 4 items from one language and one source language. Sensitivity for substrate items from other sources may differ.
- **40-item Swadesh list** is the binding constraint for the core vocabulary analysis. Insufficient for a statistically powered Transeurasian signal test.

---

## Intended Next Application

The pipeline architecture is designed for transfer. The intended next target is Indo-Iranian substrate detection: the BMAC (Bactria-Margiana Archaeological Complex) problem. The BMAC application has a methodological advantage this project lacks — well-defined ground truth via Lubotsky's 55 candidate substrate words and Kuiper's 383 non-Indo-European Rigvedic words — making validation more tractable. The data situation is harder (dead languages, deeper time depth ~4,000 BP, sandhi resolution required), but the pipeline structure transfers directly.

---

## References

- Clauson, G. (1972). *An Etymological Dictionary of Pre-Thirteenth-Century Turkish.* Oxford University Press.
- Dellert, J. et al. (2020). NorthEuraLex: A wide-coverage lexical database of Northern Eurasia. *Language Resources and Evaluation*, 54, 273–301.
- Janhunen, J. (2003). *The Mongolic Languages.* Routledge.
- Kuiper, F.B.J. (1991). *Aryans in the Rigveda.* Rodopi.
- Lubotsky, A. (2001). The Indo-Iranian substratum. In *Early Contacts between Uralic and Indo-European.* Mémoires de la Société Finno-Ougrienne, 261–277.
- Robbeets, M. et al. (2021). Triangulation supports agricultural spread of the Transeurasian languages. *Nature*, 599, 616–621.
- Savelyev, A. & Robbeets, M. (2020). Bayesian phylolinguistics infers the internal structure and time-depth of the Turkic language family. *Journal of Language Evolution*, 5(1), 39–53.
- Tian, Z. et al. (2022). Triangulation fails when neither linguistic, genetic, nor archaeological data support the Transeurasian narrative. *bioRxiv.* https://doi.org/10.1101/2022.06.09.495471
- Wichmann, S., Holman, E.W., & Brown, C. (2025). ASJP Database, v21. https://asjp.clld.org

---

## Contact

Luke M. Gillespie — lukemgillespie@gmail.com

Feedback from specialists in Turkic historical linguistics and computational phylogenetics is welcome.
