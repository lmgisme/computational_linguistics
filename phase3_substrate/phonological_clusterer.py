"""
Phase 3 - Step 3: Phonological Feature Profiler and Hierarchical Clustering
=============================================================================
Takes unknown_anomalies.csv and clusters the anomalous words by phonological
feature profile to identify candidate substrate groups.

Feature encoding
----------------
Each word is encoded as a feature vector over phonological properties:
  - Presence/absence of phoneme classes (nasals, stops, fricatives, etc.)
  - Syllable structure features (CV ratio, consonant cluster presence)
  - Vowel features (front/back, round, low, height)
  - Initial consonant place/manner
  - Word-final features

Clustering
----------
Hierarchical agglomerative clustering (Ward linkage) on cosine distance.
Number of clusters determined by dendrogram cut or user-specified k.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent.parent / "output"
UNKNOWN_CSV   = BASE / "unknown_anomalies.csv"
CLUSTERS_CSV  = BASE / "substrate_clusters.csv"
PROFILES_CSV  = BASE / "cluster_profiles.csv"

# ---------------------------------------------------------------------------
# Phonological feature definitions
# ---------------------------------------------------------------------------

# Phoneme class membership - covers IPA segments likely in the data
NASALS      = {"m", "n", "ŋ", "ɲ", "ɱ"}
STOPS       = {"p", "b", "t", "d", "k", "g", "q", "ɢ", "ʔ", "kː", "tː"}
AFFRICATES  = {"tʃ", "dʒ", "ts", "dz", "tɕ", "dʑ", "tɕ", "ʨ", "ʥ"}
FRICATIVES  = {"f", "v", "s", "z", "ʃ", "ʒ", "x", "ɣ", "χ", "ʁ", "h", "ħ", "ʕ", "ɕ", "ʑ", "ɸ", "β"}
LIQUIDS     = {"l", "r", "ɫ", "ɾ", "ʎ", "ɭ"}
GLIDES      = {"j", "w", "ɥ"}
FRONT_V     = {"i", "e", "æ", "y", "ø", "ĭ", "iː", "eː", "yː", "øː", "ĕ"}
BACK_V      = {"u", "o", "ɯ", "ɑ", "ɔ", "ɵ", "uː", "oː", "ɯː", "aː"}
MID_V       = {"ə", "ɘ", "ɤ", "ɐ", "a", "ŏ"}
ROUND_V     = {"u", "o", "y", "ø", "ɔ", "œ", "ɵ", "uː", "oː", "yː", "øː", "ŏ"}
# Phonemes rare/absent in proto-Turkic core (flag as non-Turkic if present)
NON_TURKIC_MARKERS = {"f", "v", "ħ", "ʕ", "ʔ", "ts", "dz", "ɸ", "β", "ʡ"}
# Uvular/pharyngeal inventory (Mongolic substrate signal)
UVULARS     = {"q", "ɢ", "χ", "ʁ", "ɣ"}
# Chuvash-type phones (Bulgar branch markers)
BULGAR_PHONES = {"ɕ", "ʑ", "ĭ", "ĕ", "ŏ", "ə", "ʑ"}

def featurize_word(token_str: str, form: str) -> dict:
    """
    Returns a feature dictionary for a word given its IPA token string.
    """
    tokens = token_str.strip().split() if token_str.strip() else []
    if not tokens:
        return {}

    token_set = set(tokens)
    n = len(tokens)

    # Vowels and consonants
    vowels = [t for t in tokens if any(t in s for s in [FRONT_V, BACK_V, MID_V])]
    consonants = [t for t in tokens if t not in FRONT_V | BACK_V | MID_V and t not in {"-", "+"}]

    feats = {
        # Phoneme class presence
        "has_nasal":         int(bool(token_set & NASALS)),
        "has_stop":          int(bool(token_set & STOPS)),
        "has_affricate":     int(bool(token_set & AFFRICATES)),
        "has_fricative":     int(bool(token_set & FRICATIVES)),
        "has_liquid":        int(bool(token_set & LIQUIDS)),
        "has_glide":         int(bool(token_set & GLIDES)),
        "has_uvular":        int(bool(token_set & UVULARS)),
        "has_non_turkic":    int(bool(token_set & NON_TURKIC_MARKERS)),
        "has_bulgar_phone":  int(bool(token_set & BULGAR_PHONES)),

        # Vowel features
        "has_front_vowel":   int(bool(token_set & FRONT_V)),
        "has_back_vowel":    int(bool(token_set & BACK_V)),
        "has_round_vowel":   int(bool(token_set & ROUND_V)),
        "vowel_count":       len(vowels),
        "consonant_count":   len(consonants),

        # Structural
        "word_length":       n,
        "cv_ratio":          round(len(vowels) / max(n, 1), 2),

        # Initial consonant features
        "initial_stop":      int(tokens[0] in STOPS) if tokens else 0,
        "initial_nasal":     int(tokens[0] in NASALS) if tokens else 0,
        "initial_liquid":    int(tokens[0] in LIQUIDS) if tokens else 0,
        "initial_glide":     int(tokens[0] in GLIDES) if tokens else 0,
        "initial_fricative": int(tokens[0] in FRICATIVES) if tokens else 0,
        "initial_affricate": int(tokens[0] in AFFRICATES) if tokens else 0,

        # Final consonant features
        "final_nasal":       int(tokens[-1] in NASALS) if tokens else 0,
        "final_stop":        int(tokens[-1] in STOPS) if tokens else 0,
        "final_liquid":      int(tokens[-1] in LIQUIDS) if tokens else 0,
    }
    return feats

# ---------------------------------------------------------------------------
# Build feature matrix
# ---------------------------------------------------------------------------
def build_feature_matrix(df: pd.DataFrame):
    """
    Returns (feature_matrix, feature_names, valid_indices).
    Rows where featurization fails are dropped.
    """
    records = []
    valid_idx = []

    for i, row in df.iterrows():
        feats = featurize_word(str(row.get("ipa_tokens", "")), str(row.get("form", "")))
        if feats:
            records.append(feats)
            valid_idx.append(i)

    if not records:
        return None, [], []

    feat_df = pd.DataFrame(records).fillna(0)
    feature_names = list(feat_df.columns)
    matrix = feat_df.values.astype(float)

    return matrix, feature_names, valid_idx

# ---------------------------------------------------------------------------
# Hierarchical clustering
# ---------------------------------------------------------------------------
def cluster_words(matrix: np.ndarray, n_clusters: int = 5):
    """
    Hierarchical agglomerative clustering (Ward linkage).
    Returns cluster label array.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist

    if len(matrix) < n_clusters:
        return np.zeros(len(matrix), dtype=int)

    # Normalize features to unit scale
    std = matrix.std(axis=0)
    std[std == 0] = 1
    matrix_norm = (matrix - matrix.mean(axis=0)) / std

    dist = pdist(matrix_norm, metric="euclidean")
    Z = linkage(dist, method="ward")
    labels = fcluster(Z, t=n_clusters, criterion="maxclust")
    return labels

# ---------------------------------------------------------------------------
# Profile each cluster
# ---------------------------------------------------------------------------
def profile_clusters(df_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster, summarize:
    - size, languages, glosses, mean adjusted_score
    - top phonological features
    - most common forms
    """
    profiles = []
    for cluster_id in sorted(df_clustered["cluster"].unique()):
        cdf = df_clustered[df_clustered["cluster"] == cluster_id]

        lang_counts  = cdf["language"].value_counts().to_dict()
        gloss_counts = cdf["gloss"].value_counts().head(5).to_dict()
        mean_score   = cdf["adjusted_score"].mean()
        top_forms    = cdf["form"].value_counts().head(5).index.tolist()

        profiles.append({
            "cluster":        cluster_id,
            "size":           len(cdf),
            "mean_score":     round(mean_score, 4),
            "languages":      str(lang_counts),
            "top_glosses":    str(gloss_counts),
            "sample_forms":   ", ".join(top_forms),
            "loan_sources":   str(cdf["loan_source"].value_counts().to_dict()),
        })

    return pd.DataFrame(profiles)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(unknown_df: pd.DataFrame = None, n_clusters: int = 5):
    if unknown_df is None:
        print("Loading unknown anomalies...")
        unknown_df = pd.read_csv(UNKNOWN_CSV, dtype=str)
        unknown_df["adjusted_score"] = unknown_df["adjusted_score"].astype(float)

    if len(unknown_df) < 3:
        print("Not enough unknown anomalies to cluster.")
        return unknown_df, pd.DataFrame()

    print(f"\nFeaturizing {len(unknown_df)} words...")
    matrix, feature_names, valid_idx = build_feature_matrix(unknown_df)

    if matrix is None or len(matrix) < 3:
        print("Featurization produced insufficient data.")
        return unknown_df, pd.DataFrame()

    print(f"Feature matrix: {matrix.shape[0]} words x {matrix.shape[1]} features")

    print(f"Clustering into {n_clusters} groups (Ward linkage)...")
    try:
        labels = cluster_words(matrix, n_clusters=n_clusters)
    except Exception as e:
        print(f"Clustering failed: {e}")
        print("Assigning all words to cluster 1.")
        labels = np.ones(len(matrix), dtype=int)

    # Map labels back to df
    clustered_subset = unknown_df.loc[valid_idx].copy()
    clustered_subset["cluster"] = labels

    # Merge back (words that failed featurization get cluster=-1)
    unknown_df["cluster"] = -1
    unknown_df.loc[valid_idx, "cluster"] = labels

    # Write clustered words
    unknown_df.to_csv(CLUSTERS_CSV, index=False)
    print(f"Clustered words written to: {CLUSTERS_CSV}")

    # Profile clusters
    profiles_df = profile_clusters(clustered_subset)
    profiles_df.to_csv(PROFILES_CSV, index=False)
    print(f"Cluster profiles written to: {PROFILES_CSV}")

    print("\nCluster profiles:")
    print(profiles_df.to_string(index=False))

    return unknown_df, profiles_df


if __name__ == "__main__":
    main()
