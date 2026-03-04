"""
Phase 4 - Main Entry Point
===========================
Runs the full Phase 4 pipeline in sequence:

  Step 1: build_nexus.py        -> output/turkic_cognates.nex
                                   output/nexus_summary.txt
  Step 2: build_beast_xml.py    -> output/beast_turkic.xml
  Step 3+4: transeurasian_test.py -> output/similarity_scores.csv
                                     output/ncd_distances.csv
                                     output/origin_probabilities.csv
                                     output/transeurasian_test.txt

Usage (from project root, with venv311 active):
  cd phase4_phylo
  python main.py

BEAST itself is not run from this script — it requires a separate
installation and GUI/CLI invocation. See output/beast_turkic.xml for
setup instructions.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import build_nexus
import build_beast_xml
import transeurasian_test


def main():
    print("\n" + "=" * 62)
    print("PHASE 4 PIPELINE START")
    print("=" * 62)

    print("\n--- Step 1: Build NEXUS Matrix ---")
    taxa, char_ids, matrix, glosses_per_cog = build_nexus.main()

    print("\n--- Step 2: Generate BEAST XML ---")
    build_beast_xml.main()

    print("\n--- Steps 3 & 4: Transeurasian Signal Test ---")
    sim_rows, ncd_rows, origin_results = transeurasian_test.main()

    print("\n" + "=" * 62)
    print("PHASE 4 COMPLETE")
    print("=" * 62)
    print("Outputs in output/:")
    print("  turkic_cognates.nex       NEXUS matrix (BEAST/MrBayes input)")
    print("  nexus_summary.txt         Matrix dimensions and coverage")
    print("  beast_turkic.xml          BEAST 2 run configuration")
    print("  similarity_scores.csv     Per-word phonological similarity")
    print("  ncd_distances.csv         Cluster-level NCD distances")
    print("  origin_probabilities.csv  Transeurasian origin probability table")
    print("  transeurasian_test.txt    Full narrative report")
    print()
    print("BEAST NEXT STEPS:")
    print("  Install BEAST 2 (https://www.beast2.org/)")
    print("  Install BEAST_CLASSIC package via BEAUti Package Manager")
    print("  Load turkic_cognates.nex in BEAUti, configure, and run XML")
    print("  Diagnose convergence in Tracer; target ESS > 200")
    print("=" * 62)


if __name__ == "__main__":
    main()
