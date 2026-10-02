# -*- coding: utf-8 -*-
"""Figure S5 (revision) - missing-data sensitivity analysis, MIMIC-IV.
Panel A: missing DBP proportion per window, before vs after LOCF.
Panel B: incremental C-statistic of class over SOFA under four strategies.
Data from rev_results_v1.json.
"""
import os, json, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, 'submission_clep', 'Figures_rev')
R = json.load(open(os.path.join(BASE, 'rev_results_v1.json'), encoding='utf-8'))

mr = R['missing_rates']['mimic']
before = np.array([mr[f'dbp_w{i}'] for i in range(8)]) * 100.0
after = np.zeros(8)                      # forward+backward LOCF -> complete
w = np.arange(8)
lbl = [f'W{i}' for i in range(8)]

ms = R['missing_sensitivity']; mi = R['mi']
locf = ms['LOCF (\u524d\u5411+\u540e\u5411)']; cc = ms['\u5b8c\u6574\u75c5\u4f8b (n=3416)']
zf = ms['\u96f6\u503c\u586b\u8865 (\u539f\u6295\u7a3f\u65b9\u6848, \u5bf9\u7167)']
strat = [('LOCF\n(primary)', locf['dC'], 0.0289, 0.048, True),
         ('Complete\ncase', cc['dC'], None, None, False),
         ('Multiple\nimputation', mi['dC'][0], mi['dC'][0] - 1.96 * mi['dC'][1],
          mi['dC'][0] + 1.96 * mi['dC'][1], False),
         ('Zero-fill\n(original)', zf['dC'], None, None, False)]

fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.4))

# ---- Panel A ----
BW = 0.38
ax[0].bar(w - BW / 2, before, BW, color='#c0392b', label='Before LOCF')
ax[0].bar(w + BW / 2, after, BW, color='#2980b9', label='After LOCF')
ax[0].set_title('A', loc='left', fontweight='bold', fontsize=10)
ax[0].set_xticks(w); ax[0].set_xticklabels(lbl)
ax[0].set_ylabel('Missing DBP (%)'); ax[0].set_xlabel('6-hour window')
ax[0].legend(frameon=False, fontsize=7.5); ax[0].grid(axis='y', ls=':', alpha=.5)

# ---- Panel B ----
names = [s[0] for s in strat]; vals = [s[1] for s in strat]
cols = ['#2980b9' if s[4] else '#95a5a6' for s in strat]
xb = np.arange(4)
ax[1].bar(xb, vals, 0.6, color=cols)
for i, s in enumerate(strat):
    ax[1].text(i, s[1] + 0.0015, f'{s[1]:.3f}', ha='center', fontsize=7.5,
               fontweight='bold' if s[4] else 'normal')
    if s[2] is not None and s[3] is not None:
        ax[1].errorbar(i, s[1], yerr=[[s[1] - s[2]], [s[3] - s[1]]], fmt='none',
                       ecolor='black', capsize=3, lw=1)
ax[1].set_title('B', loc='left', fontweight='bold', fontsize=10)
ax[1].set_xticks(xb); ax[1].set_xticklabels(names, fontsize=7.5)
ax[1].set_ylabel('\u0394C-statistic (class over SOFA)')
ax[1].set_ylim(0, max(vals) * 1.25); ax[1].grid(axis='y', ls=':', alpha=.5)

plt.tight_layout()
fig.savefig(os.path.join(OUT, 'Figure_S5.tif'), dpi=600, format='tiff',
            bbox_inches='tight', pad_inches=0.2, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(os.path.join(OUT, 'Figure_S5.png'), dpi=300, format='png',
            bbox_inches='tight', pad_inches=0.2)
plt.close()
print('Saved Figure_S5.tif/.png ->', OUT)
