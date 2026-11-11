"""Heatmaps of the D1b (gamma, kappa) surface, one PDF per dataset. Needs matplotlib.

    python analysis/eval_package/diag/d1b_plot.py
Output: analysis/eval_package/diag/out/d1b_surface_<ds>.pdf (+ .png)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
PANELS = (("full_valid_w", "full test, w from validation"), ("full_test_w", "full test, w from test (upper bound)"),
          ("halfB_valid_w", "half-B, w from validation"), ("halfB_B_w", "half-B, w from B (upper bound)"))


def main():
    for ds in ("ml_1m", "ml_10m", "goodreads"):
        f = os.path.join(OUT, "d1b_%s.json" % ds)
        if not os.path.exists(f):
            continue
        J = json.load(open(f))
        G, K = J["gammas"], J["kappa_labels"]
        cell = {(c["gamma"], c["kappa_label"]): c for c in J["cells"]}
        Z = {p: np.array([[cell[(g, k)][p]["rel_pct"] for k in K] for g in G]) for p, _ in PANELS}
        lo = {p: np.array([[cell[(g, k)][p]["ci95"][0] for k in K] for g in G]) for p, _ in PANELS}
        lim = max(1.0, max(np.abs(z).max() for z in Z.values()))
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), sharey=True)
        for ax, (p, title) in zip(axes, PANELS):
            z = Z[p]
            im = ax.imshow(z, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
            for i in range(len(G)):
                for j in range(len(K)):
                    sig = lo[p][i, j] > 0
                    ax.text(j, i, "%+.1f%s" % (z[i, j], "*" if sig else ""), ha="center", va="center", fontsize=6.5,
                            color="white" if abs(z[i, j]) > 0.6 * lim else "black")
            i, j = np.unravel_index(np.argmax(z), z.shape)
            ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False, ec="black", lw=2))
            gi, kj = G.index(J["gamma_star"]), K.index([k for k in K if k.startswith("a0=")][0])
            ax.add_patch(plt.Rectangle((kj - .5, gi - .5), 1, 1, fill=False, ec="#d4a017", lw=2, ls="--"))
            ax.set_xticks(range(len(K)))
            ax.set_xticklabels(K, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(len(G)))
            ax.set_yticklabels([str(g) for g in G], fontsize=7)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("kappa")
        axes[0].set_ylabel("gamma")
        fig.colorbar(im, ax=axes, shrink=0.8, label="R@20 vs FM (%)")
        fig.suptitle("%s: genre-mix re-ranker on seed-1 FM, m_u(gamma, kappa).  black = argmax, "
                     "gold dashed = theory (gamma*, kappa = alpha0), * = 95%% CI > 0" % J["dataset"], fontsize=10)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(OUT, "d1b_surface_%s.%s" % (ds, ext)), bbox_inches="tight", dpi=130)
        plt.close(fig)
        print("wrote", ds)


if __name__ == "__main__":
    main()
