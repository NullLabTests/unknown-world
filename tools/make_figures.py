"""Generate README figures from the Experiment 007 record (byte-verified).

Data are transcribed from the report block in README.md; every number here
matches the byte-verbatim record printed by ``python3 run.py`` (seed base 7,
n_seeds=16). Run:  python3 tools/make_figures.py
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

OUT = "charts"

# --------------------------------------------------------------------------
# per-arm vs null: (block, arm, n, mean d, lo, hi, p, dz, seeds_helping)
# --------------------------------------------------------------------------
ARMS = [
    ("A-learn", "grid", 80, -0.50, -0.65, -0.35, 0.0003, -0.69, "16/16"),
    ("A-learn", "exact", 80, -0.50, -0.65, -0.35, 0.0003, -0.69, "16/16"),
    ("A-learn", "anticipate", 80, -0.33, -0.44, -0.23, 0.0003, -0.69, "16/16"),
    ("A-transfer", "grid", 80, -0.34, -0.51, -0.16, 0.0003, -0.42, "13/16"),
    ("A-transfer", "exact", 80, -0.34, -0.51, -0.16, 0.0003, -0.42, "13/16"),
    ("A-transfer", "anticipate", 80, -0.30, -0.45, -0.15, 0.0003, -0.47, "14/16"),
    ("B-home", "grid", 320, -0.44, -0.53, -0.36, 0.0003, -0.53, "16/16"),
    ("B-home", "exact", 320, -0.44, -0.53, -0.36, 0.0003, -0.53, "16/16"),
    ("B-home", "anticipate", 320, -0.41, -0.49, -0.33, 0.0003, -0.50, "16/16"),
    ("B-slow", "grid", 512, -0.51, -0.59, -0.44, 0.0003, -0.61, "16/16"),
    ("B-slow", "exact", 512, -0.51, -0.59, -0.44, 0.0003, -0.61, "16/16"),
    ("B-slow", "anticipate", 512, -0.51, -0.59, -0.44, 0.0003, -0.61, "16/16"),
    ("B-fast", "grid", 256, -0.07, -0.17, 0.04, 0.1278, -0.08, "12/16"),
    ("B-fast", "exact", 256, -0.07, -0.17, 0.04, 0.1278, -0.08, "12/16"),
    ("B-fast", "anticipate", 256, -0.09, -0.18, -0.01, 0.0215, -0.13, "12/16"),
    ("B-switchrate", "grid", 1280, -0.38, -0.43, -0.33, 0.0003, -0.43, "16/16"),
    ("B-switchrate", "exact", 1280, -0.38, -0.43, -0.33, 0.0003, -0.43, "16/16"),
    ("B-switchrate", "anticipate", 1280, -0.31, -0.35, -0.27, 0.0003, -0.45, "16/16"),
    ("C-mixed", "grid", 160, 0.05, -0.07, 0.17, 0.8127, 0.06, "7/16"),
    ("C-mixed", "exact", 160, 0.05, -0.07, 0.17, 0.8127, 0.06, "7/16"),
    ("C-mixed", "anticipate", 160, -0.05, -0.12, 0.03, 0.1260, -0.10, "13/16"),
]

# --------------------------------------------------------------------------
# effective hazard E[h] posterior mean over seeds by capture point
# --------------------------------------------------------------------------
EH_CAPTURE = ["post-A", "post-home", "post-slow", "post-fast", "post-sr1", "post-srfast", "post-srmid"]
EH = {
    "grid": [0.1257, 0.2163, 0.1559, 0.2970, 0.2523, 0.3231, 0.3155],
    "exact": [0.3198, 0.3489, 0.2523, 0.6190, 0.2691, 0.6251, 0.4521],
    "anticipate": [0.3195, 0.3487, 0.2526, 0.6196, 0.2675, 0.6263, 0.4523],
}
EH_SR2 = {"grid": [0.3056], "exact": [0.2782], "anticipate": [0.2775]}

# --------------------------------------------------------------------------
# gate CIs: (name, mean d, lo, hi, boundary kind, bound)
# --------------------------------------------------------------------------
GATES = [
    ("A_transfer exact-null", -0.34, -0.50, -0.15, "upper<0", 0.0),
    ("A_transfer anticipate-null", -0.30, -0.44, -0.16, "upper<0", 0.0),
    ("H_order (fast − slow)", 0.367, 0.365, 0.368, "lower>0.04", 0.04),
    ("H_fall (second calm)", 0.137, 0.133, 0.140, "upper<0", 0.0),
    ("A_dividend (fast worlds)", -0.03, -0.08, 0.02, "upper<0", 0.0),
    ("C_mixed exact-null", 0.05, -0.07, 0.17, "upper<0.25", 0.25),
    ("C_mixed anticipate-null", -0.05, -0.13, 0.03, "upper<0.25", 0.25),
]
# --------------------------------------------------------------------------
# calibration readout (reported, not gated)
# --------------------------------------------------------------------------
CAL = [
    ("grid", "home", 0.173, 0.158, 0.1433),
    ("grid", "slow", 0.179, 0.032, 0.0541),
    ("grid", "fast", 0.228, 0.467, 0.3143),
    ("grid", "switchrate", 0.295, 0.139, 0.1455),
    ("exact", "home", 0.335, 0.158, 0.1752),
    ("exact", "slow", 0.290, 0.032, 0.0999),
    ("exact", "fast", 0.441, 0.467, 0.2889),
    ("exact", "switchrate", 0.425, 0.139, 0.2087),
    ("anticipate", "home", 0.334, 0.158, 0.1751),
    ("anticipate", "slow", 0.290, 0.032, 0.1000),
    ("anticipate", "fast", 0.441, 0.467, 0.2890),
    ("anticipate", "switchrate", 0.425, 0.139, 0.2087),
]

ARM_COLOR = {"grid": "#b07aa1", "exact": "#1f6f8b", "anticipate": "#e07b39", "null": "#8b8b8b"}
ARM_LABEL = {"grid": "grid", "exact": "exact", "anticipate": "anticipate"}


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    return ax


def hero():
    """Per-block mean d (n_exp arm − n_exp null), CI bars, 18 careful bars."""
    blocks = ["A-learn", "A-transfer", "B-home", "B-slow", "B-fast", "B-switchrate", "C-mixed"]
    height = 0.26
    fig, ax = plt.subplots(figsize=(9.2, 6.4), dpi=200)
    y = 0
    for b in blocks:
        row = [r for r in ARMS if r[0] == b]
        for r in row:
            _, arm, _, m, lo, hi, *_ = r
            ax.barh(y, hi - lo, left=lo, height=height, color=ARM_COLOR[arm], alpha=0.32, zorder=2)
            ax.barh(y, 0.02, left=m - 0.01, height=height, color=ARM_COLOR[arm], zorder=3)
            ax.plot(m, y, "o", ms=5, color=ARM_COLOR[arm], zorder=4)
            y += 1
        # group separator after each complete block (skip before last)
        if b != "C-mixed":
            ax.axhline(y - 0.5, 0, 1, color="#dddddd", lw=0.8, zorder=1)
    ax.set_yticks(list(range(len(ARMS))))
    ax.set_yticklabels([f"{r[0]}  ·  {r[1]}" for r in ARMS], fontsize=9)
    ax.axvline(0.0, color="#555555", lw=1.1, zorder=1)
    ax.axvline(0.25, color="#eeeeee", lw=1.0, ls=":", zorder=1)
    ax.set_xlim(-1.0, 0.6)
    ax.set_ylim(-1.0, len(ARMS))
    ax.set_xlabel("n_exp(arm) − n_exp(null), 95% CI   (negative = fewer experiments than null)")
    ax.set_title("Experiment 007 — every arm beats null on nearly every block", loc="left", fontsize=13, weight="bold", pad=14)
    legend = [Line2D([0], [0], marker="o", color="w", mfc=ARM_COLOR[a], ms=7, ls="", label=f"{a}") for a in ("grid", "exact", "anticipate")]
    legend.append(Line2D([0], [0], color="#666666", lw=4, alpha=0.3, label="95% CI"))
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=10, ncol=4)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hero_advantage.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/hero_advantage.png")


def trajectory():
    """E[h] across capture points: the hierarchy's signature is the fast spike."""
    fig, ax = plt.subplots(figsize=(9.2, 5.4), dpi=200)
    x = list(range(len(EH_CAPTURE)))
    for arm, ys in EH.items():
        ax.plot(x, ys, "-o", ms=5, color=ARM_COLOR[arm], lw=2, label=arm, zorder=3)
    x2 = len(x)
    for arm, ys in EH_SR2.items():
        ax.plot([x2], ys, "o", ms=7, color=ARM_COLOR[arm], zorder=3)
        ax.plot([x2 - 1, x2], [EH[arm][-1], ys[0]], "--", color=ARM_COLOR[arm], lw=1.2, zorder=2)
    xt = EH_CAPTURE + ["post-sr2"]
    ax.set_xticks(range(x2 + 1))
    ax.set_xticklabels(xt, rotation=20, ha="right", fontsize=9)
    ax.axvspan(EH_CAPTURE.index("post-sr1") - 0.4, x2 + 0.4, color="#f6f0e8", zorder=0)
    ax.annotate("switchrate block: rate changes 1/16 → 1/2 → 1/16", xy=(6.3, 0.72), fontsize=9.5, color="#7a5c3a", ha="right")
    ax.axhline(1 / 16, color="#555555", lw=1.0, ls=":")
    ax.annotate("true calm rate = 1/16", xy=(0.1, 1 / 16 + 0.012), fontsize=8.5, color="#555555")
    ax.set_ylabel("effective hazard E[h] (posterior mean over seeds)")
    ax.set_ylim(0.0, 0.78)
    ax.set_title("The hierarchy spikes on fast, and falls back when calm returns", loc="left", fontsize=13, weight="bold", pad=14)
    ax.legend(loc="upper left", frameon=False, fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/hazard_trajectory.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/hazard_trajectory.png")


def gates():
    """Forest plot of the pre-committed gate CIs; pass/fail coloring by boundary."""
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=200)
    names = [g[0] for g in GATES]
    rows = list(range(len(GATES)))[::-1]
    for row, (name, m, lo, hi, kind, bound) in zip(rows, GATES):
        if "upper" in kind:
            ok = hi < bound
        else:
            ok = lo > bound
        color = "#2a7a3e" if ok else "#b02a2a"
        ax.plot([lo, hi], [row, row], color=color, lw=3.5, zorder=2)
        ax.plot(m, row, "o", ms=6, color=color, zorder=3)
        ax.axvline(bound, color="#999999", lw=1.0, ls=":", zorder=1)
        side = 1 if kind.startswith("upper") else -1
        ax.annotate("PASS" if ok else "FAIL", xy=(m + side * 0.02, row + 0.18), fontsize=8, ha="left", color=color, weight="bold")
        ax.annotate(f"bound: {bound:g}", xy=(bound, row + 0.15), fontsize=8, color="#888888", ha=("left" if side == 1 else "right"))
    ax.set_yticks(rows)
    ax.set_yticklabels(names, fontsize=9.5)
    ax.set_xlabel("mean d (or E[h] difference), 95% CI")
    ax.set_title("Pre-committed gates — the verdict is drawn on these boundaries", loc="left", fontsize=13, weight="bold", pad=14)
    legend = [
        Line2D([0], [0], color="#2a7a3e", lw=3.5, label="gate passed"),
        Line2D([0], [0], color="#b02a2a", lw=3.5, label="gate failed"),
        Line2D([0], [0], color="#999999", lw=1.0, ls=":", label="boundary"),
    ]
    ax.legend(handles=legend, loc="lower left", frameon=False, fontsize=9)
    _style(ax)
    fig.tight_layout()
    fig.savefig(f"{OUT}/gates.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/gates.png")


def calibration():
    """Predicted vs realized switches, per arm and pace (reported, not gated)."""
    paces = ["home", "slow", "fast", "switchrate"]
    fig, axes = plt.subplots(1, 4, figsize=(11.6, 4.0), dpi=200, sharey=True)
    for ax, pace in zip(axes, paces):
        for arm in ("grid", "exact", "anticipate"):
            row = [c for c in CAL if c[0] == arm and c[1] == pace][0]
            ax.plot(row[2], row[3], "o", ms=8, color=ARM_COLOR[arm], zorder=3)
        ax.plot([0, 0.5], [0, 0.5], color="#cccccc", ls="--", zorder=1)
        ax.set_title(pace, fontsize=11)
        ax.set_xlim(0.1, 0.5)
        ax.set_ylim(0.0, 0.5)
        ax.set_aspect("equal", adjustable="box")
        _style(ax)
    axes[0].set_ylabel("realized switch rate")
    fig.text(0.06, 0.02, "predicted switch rate", ha="center", fontsize=10)
    legend = [Line2D([0], [0], marker="o", color="w", mfc=ARM_COLOR[a], ms=8, ls="", label=f"{a}") for a in ("grid", "exact", "anticipate")]
    axes[0].legend(handles=legend, loc="upper left", frameon=False, fontsize=9)
    fig.suptitle("Calibration (reported, not gated): boundary belief vs realized switches", fontsize=13, weight="bold", x=0.54)
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.95])
    fig.savefig(f"{OUT}/calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/calibration.png")


def timeline():
    """001..007 outcomes as a single strip: kept (green) vs inconclusive/not-kept (amber)."""
    exps = [
        ("001", "ACT narrows the version space", "kept"),
        ("002", "salience prior", "inconclusive"),
        ("003", "paired rototest harness", "kept"),
        ("004", "change-aware prior learns", "kept"),
        ("005", "hazard from winner seq", "inconclusive"),
        ("006", "hazard per observation", "kept"),
        ("007", "closed loop: hazard steers ACT", "inconclusive"),
    ]
    keep_color = "#2a7a3e"
    keep_label = "kept"
    amber = "#d29a37"
    fig, ax = plt.subplots(figsize=(10.6, 3.4), dpi=200)
    y = 0.0
    for x, (num, claim, outcome) in enumerate(exps, start=1):
        color = keep_color if outcome == "kept" else amber
        ax.plot(x, 0, "o", ms=26, color=color, zorder=2)
        ax.text(x, 0, num, ha="center", va="center", fontsize=11, color="white", weight="bold", zorder=3)
        ax.annotate(num, xy=(x, 0.34), ha="center", fontsize=10, weight="bold", color="#222222")
        ax.annotate(claim, xy=(x, 0.56), ha="center", fontsize=7.6, color="#555555")
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-0.45, 1.15)
    ax.axis("off")
    legend = [
        Line2D([0], [0], marker="o", color="w", mfc=keep_color, ms=11, ls="", label="kept"),
        Line2D([0], [0], marker="o", color="w", mfc=amber, ms=11, ls="", label="inconclusive — not kept"),
    ]
    ax.legend(handles=legend, loc="lower right", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(1.0, -0.1))
    ax.set_title("Seven experiments, one measurement discipline — kept only where the effect is real", loc="left", fontsize=13, weight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(f"{OUT}/timeline.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote charts/timeline.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    hero()
    trajectory()
    gates()
    calibration()
    timeline()