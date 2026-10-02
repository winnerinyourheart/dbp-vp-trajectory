# -*- coding: utf-8 -*-
"""Figure 4（方案 A）：eICU-CRD 独立复制。"""
import os, sys, warnings
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- path & credential configuration (added for public release) ---
DATA_DIR = os.environ.get("DBP_DATA_DIR", os.path.dirname(os.path.abspath(__file__)))
DB_PASSWORD = os.environ.get("DBP_DB_PASSWORD", "postgres")


BASE = DATA_DIR
OUT = os.path.join(BASE, "submission_clep", "Figures_rev")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "savefig.bbox": "tight"})

# MIMIC-IV（未变）
mm = np.array([2.1, 7.2, 14.7, 23.8, 24.4, 27.7])
ms = np.array([5.7, 6.9, 7.5, 7.0, 8.0, 9.7])
# eICU-CRD（方案 A + 升压药单位修正：3,624 + 源表 DBP + LOCF）
em = np.array([6.4, 14.8, 17.3, 18.5, 20.1, 23.1])
es = np.array([4.9, 5.5, 5.8, 6.2, 6.2, 6.5])
x = np.arange(6)
BLUE, RED = "#1f4e9c", "#b02418"

fig, ax = plt.subplots(2, 2, figsize=(7.4, 5.6))

a = ax[0, 0]
a.plot(x, mm, "o-", color=BLUE, ms=4.5, lw=1.4, label="MIMIC-IV")
a.set_xticks(x); a.set_xticklabels([f"C{i}" for i in range(6)])
a.set_ylabel("28-day mortality (%)"); a.set_ylim(0, 32)
a.set_title("A", fontsize=9, loc="left")
a.grid(axis="y", alpha=.25, lw=.6)

b = ax[0, 1]
b.plot(x, em, "s-", color=RED, ms=4.5, lw=1.4)
b.set_xticks(x); b.set_xticklabels([f"C{i}" for i in range(6)])
b.set_ylabel("In-ICU mortality (%)"); b.set_ylim(0, 32)
b.set_title("B", fontsize=9, loc="left")
b.grid(axis="y", alpha=.25, lw=.6)

c = ax[1, 0]
w = 0.38
c.bar(x - w / 2, ms, w, color=BLUE, label="MIMIC-IV")
c.bar(x + w / 2, es, w, color=RED, label="eICU-CRD")
c.set_xticks(x); c.set_xticklabels([f"C{i}" for i in range(6)])
c.set_ylabel("Mean baseline SOFA"); c.set_ylim(0, 12)
c.set_title("C", fontsize=9, loc="left")
c.legend(frameon=False, fontsize=7.5)
c.grid(axis="y", alpha=.25, lw=.6)

d = ax[1, 1]
lbl = ["Class only", "Class + SOFA"]
mv = [0.693, 0.779]; ev = [0.597, 0.694]
xx = np.arange(2)
d.bar(xx - w / 2, mv, w, color=BLUE, label="MIMIC-IV")
d.bar(xx + w / 2, ev, w, color=RED, label="eICU-CRD")
for i, (p, q) in enumerate(zip(mv, ev)):
    d.annotate(f"{p:.3f}", (i - w / 2, p), xytext=(0, 2), textcoords="offset points",
               ha="center", fontsize=7)
    d.annotate(f"{q:.3f}", (i + w / 2, q), xytext=(0, 2), textcoords="offset points",
               ha="center", fontsize=7)
d.set_xticks(xx); d.set_xticklabels(lbl)
d.set_ylabel("Harrell's C-statistic"); d.set_ylim(0.5, 0.9)
d.set_title("D", fontsize=9, loc="left")
d.legend(frameon=False, fontsize=7.5)
d.grid(axis="y", alpha=.25, lw=.6)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "Figure_4.png"), dpi=300)
fig.savefig(os.path.join(OUT, "Figure_4.tif"), dpi=600, pil_kwargs={"compression": "tiff_lzw"})
print("[OK] Figure_4 已重制（方案 A）")
