"""E2b figure: R@20 against C_KL@20 for the ML-1M seed-1 grid, labelled (kappa, gamma).

Reads out/e2b_frontier.csv only. Needs matplotlib:
    python analysis/eval_package/e2b_plot.py
Output: analysis/eval_package/out/e2b_frontier.pdf, .png
"""
import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
rows = [r for r in csv.DictReader(open(os.path.join(OUT, "e2b_frontier.csv"), encoding="utf-8"))
        if r["rescored"] == "True"]
col = {"inf": "#1f5fa8", "10": "#c97b12", "0": "#b8322a"}
name = {"inf": "\u03ba = \u221e (population prior)", "10": "\u03ba = 10", "0": "\u03ba = 0 (personal)"}
fig, ax = plt.subplots(figsize=(5.2, 3.8))
for k in ("inf", "10", "0"):
    pts = [r for r in rows if r["kappa"] == k]
    if not pts:
        continue
    ax.scatter([float(r["C_KL@20"]) for r in pts], [float(r["R@20"]) for r in pts], s=42, color=col[k],
               marker="o", label=name[k], zorder=3)
    w = [r for r in pts if r["prior"] == "windowed"]
    ax.scatter([float(r["C_KL@20"]) for r in w], [float(r["R@20"]) for r in w], s=42, facecolors="none",
               edgecolors="black", zorder=4)
    for r in pts:
        lab = "(%s, %s)%s" % ("\u221e" if k == "inf" else k, r["gamma"], " w" if r["prior"] == "windowed" else "")
        lab = "(\u221e, \u2013) DecRS" if r["gamma"] == "-" else lab
        ax.annotate(lab, (float(r["C_KL@20"]), float(r["R@20"])), textcoords="offset points", xytext=(5, 3),
                    fontsize=7.5, color="#333333")
ax.set_xlabel("C$_{KL}$@20 against the training genre mix (lower = better calibrated)")
ax.set_ylabel("Recall@20 (test)")
ax.set_title("ML-1M seed 1: accuracy vs calibration, (\u03ba, \u03b3)", fontsize=10)
ax.grid(alpha=0.3)
ax.legend(fontsize=7.5, frameon=False, loc="lower right")
fig.text(0.01, 0.01, "Outlined: windowed population prior. Cells without a surviving checkpoint are omitted.",
         fontsize=6.5, color="#555555")
fig.tight_layout(rect=(0, 0.03, 1, 1))
for ext in ("pdf", "png"):
    fig.savefig(os.path.join(OUT, "e2b_frontier." + ext), dpi=200)
print("wrote", os.path.join(OUT, "e2b_frontier.pdf"))
