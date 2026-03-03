"""
Phase 3 - Main Entry Point
===========================
Runs the full Phase 3 pipeline in sequence:
  1. regularity_scorer.py    -> output/regularity_scores.csv
  2. anomaly_detector.py     -> output/anomalies.csv, unknown_anomalies.csv
  3. phonological_clusterer  -> output/substrate_clusters.csv, cluster_profiles.csv
  4. substrate_report.py     -> output/phase3_report.txt

Usage (from project root, with venv311 active):
  cd phase3_substrate
  python main.py

Or specify number of clusters:
  python main.py --clusters 6
"""

import argparse
import sys
from pathlib import Path

# Make sure sibling modules are importable regardless of cwd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import regularity_scorer
import anomaly_detector
import phonological_clusterer
import substrate_report


def main():
    parser = argparse.ArgumentParser(description="Phase 3: Substrate Anomaly Detection")
    parser.add_argument(
        "--clusters", type=int, default=5,
        help="Number of phonological clusters (default: 5)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("PHASE 3 PIPELINE START")
    print("=" * 60)

    # Step 1: Score all words
    print("\n--- Step 1: Regularity Scoring ---")
    scores_df = regularity_scorer.main()

    # Step 2: Anomaly detection and loan separation
    print("\n--- Step 2: Anomaly Detection ---")
    scores_df, anomalies_df, unknown_df, threshold = anomaly_detector.main(scores_df)

    # Step 3: Cluster unknown anomalies
    print(f"\n--- Step 3: Phonological Clustering (k={args.clusters}) ---")
    unknown_df, profiles_df = phonological_clusterer.main(
        unknown_df, n_clusters=args.clusters
    )

    # Step 4: Generate report
    print("\n--- Step 4: Substrate Report ---")
    substrate_report.main()

    print("\n" + "=" * 60)
    print("PHASE 3 COMPLETE")
    print("Outputs written to output/ directory:")
    print("  regularity_scores.csv   - all words with regularity scores")
    print("  anomalies.csv           - all anomalous words with loan classification")
    print("  unknown_anomalies.csv   - unknown-source anomalies only")
    print("  substrate_clusters.csv  - anomalies with cluster assignments")
    print("  cluster_profiles.csv    - per-cluster summary statistics")
    print("  phase3_report.txt       - full narrative report")
    print("=" * 60)


if __name__ == "__main__":
    main()
