"""Architecture figure for the paper (paper/figures/fig0_architecture.png)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.path import Path
import matplotlib.patches as mpatches

BLUE, ORANGE, TEAL, AMBER, PINK, GREY = ("#4361ee", "#e4572e", "#2a9d8f",
                                         "#f3a712", "#d81159", "#6b7280")
plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, ax = plt.subplots(figsize=(13.6, 7.4))
ax.set_xlim(0, 13.6); ax.set_ylim(0, 7.4); ax.axis("off")


def arrow(pts, col, lw=1.7, ls="-", z=4):
    """Elbow arrow through a list of points; head on the last segment."""
    for k in range(len(pts) - 2):
        ax.plot([pts[k][0], pts[k + 1][0]], [pts[k][1], pts[k + 1][1]],
                color=col, lw=lw, ls=ls, solid_capstyle="round", zorder=z)
    ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle="-|>",
                 mutation_scale=17, color=col, lw=lw, linestyle=ls,
                 shrinkA=0, shrinkB=0, zorder=z))


def box(x, y, w, h, title, formula, fc, ec, fs=11.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.12",
                 fc=fc, ec=ec, lw=2.0, zorder=3))
    ax.text(x + w / 2, y + h * 0.72, title, ha="center", va="center",
            fontsize=10.4, zorder=5, color=ec, fontweight="bold")
    ax.text(x + w / 2, y + h * 0.34, formula, ha="center", va="center",
            fontsize=fs, zorder=5)


# ----------------------------------------------------------------- input
ax.add_patch(Circle((0.78, 4.15), 0.44, fc=BLUE, ec="#1b2a49", lw=2, zorder=5))
ax.text(0.78, 4.15, r"$t$", ha="center", va="center", fontsize=17,
        color="white", zorder=6)
ax.text(0.78, 3.38, "input\n" + r"$t\in\mathbb{T}$", ha="center", va="top",
        fontsize=10)

# ----------------------------------------------------------- hidden layer
HX, HW, HH = 2.30, 2.75, 0.62
ys = [6.35, 5.45, 4.55, 2.95]
labs = [r"$\phi_1=e_{\ominus p_1}(t,c_1)$", r"$\phi_2=e_{\ominus p_2}(t,c_2)$",
        r"$\phi_3=e_{\ominus p_3}(t,c_3)$", r"$\phi_m=e_{\ominus p_m}(t,c_m)$"]
for y, l in zip(ys, labs):
    ax.add_patch(FancyBboxPatch((HX, y - HH / 2), HW, HH,
                 boxstyle="round,pad=0.07", fc="#fdece6", ec=ORANGE,
                 lw=1.8, zorder=3))
    ax.text(HX + HW / 2, y, l, ha="center", va="center", fontsize=11, zorder=5)
    arrow([(1.24, 4.15), (HX - 0.06, y)], GREY, lw=1.1, z=2)
ax.text(HX + HW / 2, 3.72, r"$\vdots$", ha="center", fontsize=16)
ax.text(HX + HW / 2, 7.05, "hidden layer", ha="center", fontsize=11.5,
        color=ORANGE, fontweight="bold")
ax.text(HX + HW / 2, 6.80, r"$p_j(\tau)=\alpha_j h_1(\tau,c_j)=\alpha_j(\tau-c_j)$",
        ha="center", fontsize=10.3)
# placed under the input node so it cannot collide with the feedback line
ax.text(0.78, 2.55,
        r"centres $c_j\in\mathbb{T}$" "\n" r"are fixed;" "\n"
        r"widths $\alpha_j$ trained," "\n"
        r"admissible" "\n" r"$0<\alpha_j<1/W(c_j)$",
        ha="center", va="top", fontsize=9.4, linespacing=1.45)

# --------------------------------------------------------- linear output
SX, SY = 6.15, 4.15
ax.add_patch(Circle((SX, SY), 0.50, fc=TEAL, ec="#1b2a49", lw=2, zorder=5))
ax.text(SX, SY, r"$\Sigma$", ha="center", va="center", fontsize=19,
        color="white", zorder=6)
for y, l in zip(ys, [r"$v_1$", r"$v_2$", r"$v_3$", r"$v_m$"]):
    arrow([(HX + HW + 0.06, y), (SX - 0.52, SY)], TEAL, lw=1.2, z=2)
    ax.text(HX + HW + 0.40, y + (0.20 if y > SY else -0.20), l, fontsize=10,
            color="#1d7a6e", zorder=6)
ax.text(SX, 3.42, "linear output", ha="center", va="top", fontsize=10.2,
        color="#1d7a6e", fontweight="bold")
ax.text(SX, 3.14, r"$N(t,p)=\sum_j v_j\phi_j(t)$", ha="center", va="top",
        fontsize=10.5)

# --------------------------------------------------- right-hand column
BX, BW, BH = 7.85, 5.35, 1.20
box(BX, 5.55, BW, BH, "trial solution — satisfies the ICs identically",
    r"$y_a(t,p)=\sum_{k=0}^{n-1}h_k(t,t_0)\,y_k+h_n(t,t_0)\,N(t,p)$",
    "#eef3fd", BLUE)
box(BX, 3.75, BW, BH, "residual",
    r"$R(t,p)=F\left(t,\,y_a,\,y_a^{\Delta},\,\ldots,\,y_a^{\Delta^n}\right)$",
    "#fff8e8", AMBER)
box(BX, 1.95, BW, BH, "error functional",
    r"$E(p)=\frac{1}{2}\sum_{i=1}^{M}R^{2}(t_i,p)$", "#fdeaf1", PINK)

#   Sigma  ->  trial solution   (right, then up, then into the left edge)
arrow([(SX + 0.52, SY), (7.30, SY), (7.30, 6.15), (BX - 0.10, 6.15)],
      "#41485a", lw=1.9)
#   trial solution -> residual        (straight down)
arrow([(BX + BW / 2, 5.50), (BX + BW / 2, 5.02)], BLUE, lw=1.9)
#   residual -> error functional      (straight down)
arrow([(BX + BW / 2, 3.70), (BX + BW / 2, 3.22)], AMBER, lw=1.9)

# ------------------------------------------------------------- feedback
#   error functional -> down -> left along the bottom -> up into the hidden layer
FB_Y = 0.85
arrow([(BX - 0.10, 2.55), (7.05, 2.55), (7.05, FB_Y),
       (HX + HW / 2, FB_Y), (HX + HW / 2, 2.55)], PINK, lw=2.0, ls="--")
ax.text((7.05 + HX + HW / 2) / 2, FB_Y - 0.20,
        r"update $(v,\log\alpha)$ — trust-region reflective",
        ha="center", va="top", fontsize=10.4, color=PINK, fontweight="bold")

fig.tight_layout()
for p in [os.path.join(os.path.dirname(__file__), "..", "results", "figures",
                       "fig0_architecture.png")]:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    fig.savefig(p, dpi=240, bbox_inches="tight", facecolor="white")
print("architecture figure written")
