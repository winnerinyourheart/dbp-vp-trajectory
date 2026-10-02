# -*- coding: utf-8 -*-
"""Figure S2 (revision) - cohort flow diagram.

B7-compliant: no in-artwork title and no explanatory notes (they are in the figure legend).
Layout: box centres are computed from the height of the neighbouring boxes, so that no two
boxes overlap and every arrow runs between box edges rather than through text.
Numbers: MIMIC-IV 94,458 -> ... -> 4,883 -> 4,869 (verified against mimiciv_icu.icustays and
public.ds_mimic_shock); eICU-CRD 5,728 -> 5,132 -> 3,794 -> 3,624 (reproduced by re-running the
original eICU pipeline).
"""
import os, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'submission_clep', 'Figures_rev')
os.makedirs(OUT, exist_ok=True)

DARK, BLUE, GREEN, GRAY = '#2c3e50', '#2980b9', '#27ae60', '#95a5a6'
LBLUE, LGREEN = '#eaf2f8', '#e8f8f5'

BOXW = 3.75
GAP = 0.30          # vertical gap between box edges
LINE = 0.185        # height contributed by each line of text
PADH = 0.30         # vertical padding inside a box


def h_of(lines):
    return max(2, lines) * LINE + PADH


def layout(items, top):
    """items: list of (text, lines) -> list of (text, centre_y, height), top to bottom."""
    out = []
    for i, (txt, ln) in enumerate(items):
        h = h_of(ln)
        if i == 0:
            y = top - h / 2
        else:
            prev_h = out[-1][2]
            y = out[-1][1] - prev_h / 2 - GAP - h / 2
        out.append((txt, y, h))
    return out


MIMIC = [('ICU admissions\n(n = 94,458)', 2),
         ('Meeting Sepsis-3 criteria\n(n = 41,295)', 2),
         ('Vasopressor requirement\nwithin 48 h\n(n = 17,543)', 3),
         ('Lactate >2 mmol/L\nwithin 24 h\n(n = 9,934)', 3),
         ('Excluded (n = 3,134):\ncardiac arrest, cirrhosis,\nacute myocardial infarction, ECMO', 3),
         ('First ICU stay, age \u226518 y,\nICU LOS \u226524 h\n(n = 4,883)', 3),
         ('Excluded (n = 14):\nfewer than four evaluable\nDBP windows', 3)]
EICU = [('Norepinephrine infusion with a sepsis\nor septic shock diagnosis\n(n = 5,728)', 3),
        ('Excluded (n = 596):\ncardiac arrest, cirrhosis,\nacute myocardial infarction, ECMO', 3),
        ('Age \u226518 y, ICU LOS \u226524 h,\nfirst ICU stay\n(n = 3,794)', 3),
        ('Excluded (n = 170):\nfewer than four evaluable\nDBP windows', 3)]

TOP = 10.0
mim = layout(MIMIC, TOP)
eic = layout(EICU, TOP)
bottom = min(mim[-1][1] - mim[-1][2] / 2, eic[-1][1] - eic[-1][2] / 2)
fin_h = h_of(2) + 0.10
fin_y = bottom - GAP - fin_h / 2
BOT = fin_y - fin_h / 2 - 0.25
H = TOP + 0.75 - BOT

fig, ax = plt.subplots(1, 1, figsize=(8.6, 8.6 * H / 10.8))
ax.set_xlim(0, 10)
ax.set_ylim(BOT, TOP + 0.75)
ax.axis('off')
cx, cx2 = 2.62, 7.38


def box(x, y, w, h, text, fc='white', ec=DARK, lw=1.1, fs=8, fw='normal', tc=DARK):
    ax.add_patch(mpatches.FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                 boxstyle="round,pad=0.045", facecolor=fc, edgecolor=ec, linewidth=lw))
    ax.text(x, y, text, ha='center', va='center', fontsize=fs, fontweight=fw,
            color=tc, linespacing=1.35)


def arrow(x, y0, y1):
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.0, shrinkA=0, shrinkB=0))


ax.text(cx, TOP + 0.42, 'MIMIC-IV v3.0', ha='center', fontsize=10.5, fontweight='bold', color=BLUE)
ax.text(cx2, TOP + 0.42, 'eICU-CRD', ha='center', fontsize=10.5, fontweight='bold', color=GREEN)

for items, x, fc, ec, tc, label in [
        (mim, cx, LBLUE, BLUE, BLUE, 'MIMIC-IV analysis cohort\nN = 4,869'),
        (eic, cx2, LGREEN, GREEN, GREEN, 'eICU-CRD analysis cohort\nN = 3,624')]:
    for txt, y, h in items:
        box(x, y, BOXW, h, txt, fs=8)
    for i in range(len(items) - 1):
        arrow(x, items[i][1] - items[i][2] / 2, items[i + 1][1] + items[i + 1][2] / 2)
    arrow(x, items[-1][1] - items[-1][2] / 2, fin_y + fin_h / 2)
    box(x, fin_y, BOXW + 0.42, fin_h, label, fc=fc, ec=ec, lw=2, fs=10, fw='bold', tc=tc)

fig.savefig(os.path.join(OUT, 'Figure_S2.tif'), dpi=600, format='tiff',
            bbox_inches='tight', pad_inches=0.25, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(os.path.join(OUT, 'Figure_S2.png'), dpi=300, format='png',
            bbox_inches='tight', pad_inches=0.25)
plt.close()
print('Saved Figure_S2.tif/.png ->', OUT)
