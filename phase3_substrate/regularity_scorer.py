"""
Phase 3 - Step 1: Regularity Scorer
=====================================
Computes a per-word regularity score: how well each word's phoneme tokens
fit the expected Turkic correspondence distribution from Phase 2.

Score logic
-----------
For each word W in language L with cogid C:
  1. Find all other members of cogid C (the cognate group).
  2. For each pair (L, L') and each token t in W, look up P(t | L->L', context).
     We use the pairwise prob_model: key = "LangA|LangB|phoneme_a" -> {phoneme_b: prob}.
     Since we don't have aligned positions here (alignment is in alignments.html,
     not easily parsed), we use a token-frequency approach:
       For each token t in W, find all model entries "L|L'|t" across all partner
       languages L', and compute mean log P of t appearing as a source phoneme
       in any context it was observed in.
  3. regularity_score = mean log P across all tokens (floor = LOG_PROB_FLOOR).
     Higher (closer to 0) = more regular. Lower (more negative) = more anomalous.

Adjustments
-----------
- cogid_source == 'lexstat': subtract LEXSTAT_PENALTY from raw score
- Yakut s->e artifact: Yakut words where s appears and corresponds to e in other
  languages get flagged separately; their Yakut-specific anomaly is excluded from
  threshold decisions.
- Oghuz k->q/ɢ shift: Turkish/Azerbaijani/Turkmen k/q/ɢ alternations are treated
  as regular (high-probability correspondences); no penalty applied.

Output
------
output/regularity_scores.csv with columns:
  language, gloss, form, ipa_tokens, cogid, cogid_source, raw_score,
  adjusted_score, lexstat_penalized, yakut_artifact_flag, n_tokens, n_lookups
"""

import json
import math
import re
import pandas as pd
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "output"
HYBRID_CSV    = BASE / "hybrid_cognates.csv"
PROB_MODEL    = BASE / "prob_model.json"
OUTPUT_CSV    = BASE / "regularity_scores.csv"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LEXSTAT_PENALTY  = 0.30   # subtract from adjusted_score when source is lexstat
LOG_PROB_FLOOR   = -6.0   # floor for phonemes not seen in any correspondence
OGHUZ_LANGS      = {"Turkish", "Azerbaijani", "Turkmen"}
# Known Yakut s->e artifact: Yakut 'e' reflex of *s is an alignment artifact
# (Phase 2 confirmed). We flag Yakut words where e-tokens are dominant but do
# not apply extra penalty for them in the Yakut-specific score.
YAKUT_ARTIFACT_TOKENS = {"e"}  # Yakut tokens implicated in s>e artifact

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    df = pd.read_csv(HYBRID_CSV, dtype=str).fillna("")
    with open(PROB_MODEL, "r", encoding="utf-8") as f:
        prob_model = json.load(f)
    return df, prob_model

# ---------------------------------------------------------------------------
# Tokenize IPA (reuse Phase 1/2 convention: space-separated in ipa_tokens col)
# ---------------------------------------------------------------------------
def get_tokens(token_str: str) -> list:
    """ipa_tokens column is space-separated.
    Strips leading '?' uncertainty markers and parenthetical morphology
    before tokenizing, so they don't pollute phoneme lookup.
    """
    t = token_str.strip()
    if not t:
        return []
    # Remove leading '?' uncertainty flag (e.g. '? m y ŋ g y z' -> 'm y ŋ g y z')
    t = re.sub(r'^\?\s*', '', t)
    # Remove parenthetical suffixes carried over from source data
    # e.g. 'k i ʨ i ( ʥ i k )' -> 'k i ʨ i'
    t = re.sub(r'\(.*?\)', '', t)
    # Collapse any resulting extra whitespace
    t = ' '.join(t.split())
    return t.split() if t else []

# ---------------------------------------------------------------------------
# Build a token-level score index from prob_model
# For each (lang, token) pair, collect all observed log-probs as a source phoneme.
# We want: given that token t appears in language L, how regular is it?
# We compute mean log P across all partner-language contexts where t was observed.
# ---------------------------------------------------------------------------
def build_token_score_index(prob_model: dict) -> dict:
    """
    Returns: {(lang, token): mean_log_prob}
    mean_log_prob = average of log(P) for each entry "lang|lang'|token" summed
    over all partner languages, then averaged.
    This gives a baseline regularity for each (language, phoneme) pair.
    """
    accum = defaultdict(list)
    for key, reflex_dist in prob_model.items():
        parts = key.split("|")
        if len(parts) != 3:
            continue
        lang_a, lang_b, phoneme_a = parts
        # Sum probability mass for this phoneme as source
        total_p = sum(reflex_dist.values())
        # We use the probability that this phoneme has ANY regular reflex,
        # i.e., total_p (should be ~1.0 for well-attested pairs)
        # More useful: the entropy-weighted regularity = max P of top reflex.
        top_p = max(reflex_dist.values())
        accum[(lang_a, phoneme_a)].append(math.log(max(top_p, 1e-6)))
    # Average across all language-pair contexts
    index = {}
    for (lang, token), log_probs in accum.items():
        index[(lang, token)] = sum(log_probs) / len(log_probs)
    return index

# ---------------------------------------------------------------------------
# Oghuz uvular filter: k/q/ɢ alternations among Oghuz languages are regular
# ---------------------------------------------------------------------------
OGHUZ_UVULAR_TOKENS = {"k", "q", "ɢ", "g"}

def is_oghuz_uvular_context(lang: str, token: str) -> bool:
    """Returns True if this token is the regular Oghuz k->q/ɢ shift context."""
    return lang in OGHUZ_LANGS and token in OGHUZ_UVULAR_TOKENS

# ---------------------------------------------------------------------------
# Yakut artifact flag
# ---------------------------------------------------------------------------
def is_yakut_artifact(lang: str, tokens: list) -> bool:
    """
    Flag Yakut words where 'e' dominates the token set.
    The s->e Yakut pattern is an alignment artifact from Phase 2.
    """
    if lang != "Yakut":
        return False
    if not tokens:
        return False
    e_count = tokens.count("e")
    return e_count / len(tokens) >= 0.4  # 40%+ 'e' tokens in Yakut = artifact flag

# ---------------------------------------------------------------------------
# Compute regularity score for a single word
# ---------------------------------------------------------------------------
def score_word(lang: str, tokens: list, token_index: dict) -> tuple:
    """
    Returns (raw_score, n_tokens, n_lookups).
    raw_score = mean log P across tokens (LOG_PROB_FLOOR for unknown pairs).
    """
    if not tokens:
        return (LOG_PROB_FLOOR, 0, 0)

    log_probs = []
    n_lookups = 0

    for token in tokens:
        # Skip gap tokens
        if token in {"-", "+", "?", ""}:
            continue

        # Oghuz uvular: these are regular, assign max score
        if is_oghuz_uvular_context(lang, token):
            log_probs.append(0.0)  # log(1.0) = 0
            n_lookups += 1
            continue

        key = (lang, token)
        if key in token_index:
            log_probs.append(token_index[key])
            n_lookups += 1
        else:
            # Unseen (lang, token) pair -> floor
            log_probs.append(LOG_PROB_FLOOR)
            n_lookups += 1

    if not log_probs:
        return (LOG_PROB_FLOOR, len(tokens), 0)

    raw_score = sum(log_probs) / len(log_probs)
    return (raw_score, len(tokens), n_lookups)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("Loading data...")
    df, prob_model = load_data()

    print("Building token score index from correspondence model...")
    token_index = build_token_score_index(prob_model)
    print(f"  {len(token_index)} (language, phoneme) pairs indexed.")

    results = []
    for _, row in df.iterrows():
        lang         = row["language"]
        gloss        = row["gloss"]
        form         = row["form"]
        token_str    = row["ipa_tokens"]
        cogid        = row["cogid"]
        cogid_source = row["cogid_source"]

        tokens = get_tokens(token_str)

        # Yakut artifact flag (does NOT remove record, just marks it)
        yakut_flag = is_yakut_artifact(lang, tokens)

        raw_score, n_tokens, n_lookups = score_word(lang, tokens, token_index)

        # LexStat penalty
        lexstat_penalized = (cogid_source == "lexstat")
        adjusted_score = raw_score - (LEXSTAT_PENALTY if lexstat_penalized else 0.0)

        results.append({
            "language":           lang,
            "gloss":              gloss,
            "form":               form,
            "ipa_tokens":         token_str,
            "cogid":              cogid,
            "cogid_source":       cogid_source,
            "raw_score":          round(raw_score, 4),
            "adjusted_score":     round(adjusted_score, 4),
            "lexstat_penalized":  lexstat_penalized,
            "yakut_artifact_flag": yakut_flag,
            "n_tokens":           n_tokens,
            "n_lookups":          n_lookups,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nRegularity scores written to: {OUTPUT_CSV}")

    # Quick summary
    print(f"\nScore distribution:")
    print(f"  Mean adjusted score:   {out_df['adjusted_score'].mean():.3f}")
    print(f"  Median adjusted score: {out_df['adjusted_score'].median():.3f}")
    print(f"  Std dev:               {out_df['adjusted_score'].std():.3f}")
    print(f"  Min:                   {out_df['adjusted_score'].min():.3f}")
    print(f"  Max:                   {out_df['adjusted_score'].max():.3f}")
    print(f"\n  Yakut artifact flags:  {out_df['yakut_artifact_flag'].sum()}")
    print(f"  LexStat penalized:     {out_df['lexstat_penalized'].sum()}")

    return out_df


if __name__ == "__main__":
    main()
