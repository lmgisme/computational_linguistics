"""
Generate Figure 2: Regularity score distribution histogram.
Shows the distribution of regularity scores across the expanded dataset,
with per-branch thresholds marked and the missed loan (xotin) annotated.

Run from project root:
  venv311\Scripts\python.exe phase7_sensitivity\make_figures.py

Output: output/fig2_regularity_distribution.png
        output/fig2_regularity_distribution.pdf
"""

import csv
import os

# Check for matplotlib
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
except ImportError:
    print("ERROR: matplotlib not installed.")
    print("  pip install matplotlib")
    raise SystemExit(1)

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT = os.path.join(BASE, "output")

REG_FILE = os.path.join(OUTPUT, "regularity_expanded.csv")
THRESH_FILE = os.path.join(OUTPUT, "infra6a_thresholds.csv")


def load_data():
    scores = []
    with open(REG_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scores.append({
                "language": row["language"],
                "gloss": row["gloss"],
                "form": row["form"],
                "score": float(row["adjusted_score"]),
            })
    thresholds = {}
    with open(THRESH_FILE, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            thresholds[row["group"]] = {
                "threshold": float(row["threshold"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
            }
    return scores, thresholds


def make_histogram():
    scores, thresholds = load_data()
    all_scores = [s["score"] for s in scores]

    fig, ax = plt.subplots(figsize=(10, 5.5))

    # Main histogram
    n, bins, patches = ax.hist(
        all_scores, bins=80, color="#4878A8", alpha=0.85,
        edgecolor="white", linewidth=0.3,
    )

    # Per-branch threshold lines
    colors = {
        "A_oghuz": ("#D4443B", "Oghuz threshold"),
        "A_kipchak": ("#E88D2A", "Kipchak threshold"),
        "B_yakut": ("#6B3FA0", "Yakut threshold"),
        "C_chuvash": ("#2A9D5C", "Chuvash threshold"),
    }
    for group, info in thresholds.items():
        color, label = colors.get(group, ("gray", group))
        ax.axvline(
            info["threshold"], color=color, linestyle="--",
            linewidth=1.5, alpha=0.9, label=f"{label} ({info['threshold']:.3f})",
        )

    # Family-wide threshold
    ax.axvline(
        -1.1851, color="black", linestyle=":",
        linewidth=1.2, alpha=0.7, label="Family-wide threshold (−1.185)",
    )

    # Annotate the missed loan (xotin)
    xotin_score = -0.7618
    ax.annotate(
        "Uzbek χɔtin\n(missed loan)",
        xy=(xotin_score, 0), xytext=(xotin_score + 0.15, max(n) * 0.65),
        fontsize=8.5, ha="left",
        arrowprops=dict(arrowstyle="->", color="#D4443B", lw=1.2),
        color="#D4443B", fontweight="bold",
    )

    # Annotate a caught loan
    gosht_score = -1.2322
    ax.annotate(
        "Uzbek goʃt\n(caught loan)",
        xy=(gosht_score, 0), xytext=(gosht_score - 0.05, max(n) * 0.50),
        fontsize=8.5, ha="right",
        arrowprops=dict(arrowstyle="->", color="#2A9D5C", lw=1.2),
        color="#2A9D5C", fontweight="bold",
    )

    ax.set_xlabel("Regularity Score (mean log-probability)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("Distribution of Regularity Scores (N = 6,966)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left", framealpha=0.9)
    ax.set_xlim(-2.5, 0.05)
    ax.tick_params(labelsize=9)

    plt.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(OUTPUT, f"fig2_regularity_distribution.{ext}")
        fig.savefig(path, dpi=300)
        print(f"Saved: {path}")
    plt.close()


if __name__ == "__main__":
    make_histogram()
    print("Done.")
