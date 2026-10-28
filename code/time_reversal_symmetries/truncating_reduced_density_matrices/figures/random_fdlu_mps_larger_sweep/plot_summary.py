# Run from truncating_reduced_density_matrices:
# ../../num_spt_venv_p314/bin/python figures/random_fdlu_mps_larger_sweep/plot_summary.py
"""Summarize the saved results used by the larger-sweep analysis notebook.

Run with the repository's num_spt_venv_p314 Python (numpy, matplotlib, quimb).
No optimizations are rerun. Pickles are trusted local simulation outputs.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/mpl-summary')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/numba-summary')
import csv
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
SOURCE = OUT.parents[2] / 'truncating_reduced_density_matrices' / 'results'
COLORS = ['#2574ab', '#d46b32']
LABELS = ['Product', 'Cluster']
data = []
for state in ['product', 'cluster']:
    with (SOURCE / f'z_2_t_{state}_results.pkl').open('rb') as f:
        results = pickle.load(f)
    # Same tuple fields as the analysis notebook: final loss, L/R invariant, overlap.
    values = np.array([[r[0], r[5], r[6], r[8]] for r in results])
    assert values.shape == (100, 4) and np.isfinite(values).all()
    assert (values[:, 0].real > 0).all()
    data.append(values)

with (OUT / 'summary_data.csv').open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['state', 'run', 'final_loss', 'left_invariant_real',
                     'left_invariant_imag', 'right_invariant_real',
                     'right_invariant_imag', 'defect_overlap'])
    for label, values in zip(LABELS, data):
        for run, (loss, left, right, overlap) in enumerate(values, 1):
            writer.writerow([label, run, loss.real, left.real, left.imag,
                             right.real, right.imag, overlap.real])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.titleweight': 'normal', 'savefig.facecolor': 'white'})
fig, axes = plt.subplots(1, 3, figsize=(14, 5.4))
fig.subplots_adjust(left=.065, right=.98, bottom=.12, top=.92, wspace=.35)
rng = np.random.default_rng(42)

ax = axes[1]
for i, (values, color) in enumerate(zip(data, COLORS)):
    y = values[:, 0].real
    ax.scatter(i+rng.uniform(-.19, .19, len(y)), y, s=19, alpha=.55,
               color=color, linewidths=0, zorder=3)
    ax.boxplot(y, positions=[i], widths=.5, showfliers=False,
               medianprops={'color': '#202630', 'linewidth': 2},
               boxprops={'color': color}, whiskerprops={'visible': False},
               showcaps=False)
ax.set(yscale='log', ylim=(3e-7, 3e-4), xticks=[0, 1], xticklabels=LABELS,
       ylabel=r'Final loss $\|\rho-\rho_A\|_F^2$', title='B) Reconstruction loss')
ax.grid(axis='y', alpha=.18)

ax = axes[0]
for i, (values, color) in enumerate(zip(data, COLORS)):
    for j, end in enumerate(['L', 'R']):
        x = i + (-.16 if j == 0 else .16)
        y = values[:, j+1].real
        ax.scatter(x+rng.uniform(-.045, .045, len(y)), y,
                   s=22, color=color, alpha=.5, linewidths=0, zorder=3)
        ax.annotate(end, (x, np.median(y)), xytext=(0, 10),
                    textcoords='offset points', color=color, ha='center',
                    fontsize=11, weight='bold')
for y in [-1, 1]:
    ax.axhline(y, color='#8c939b', ls='--', lw=1, zorder=0)
ax.set(ylim=(-1.3, 1.4), xlim=(-.5, 1.5), xticks=[0, 1], xticklabels=LABELS,
       yticks=[-1, 0, 1], ylabel='Real part of $\omega(t,t)$', title='A) $\omega(t,t)$ via defect operators')

# Full complex deviation from the expected invariant, as in the Z2T figures.
inset = ax.inset_axes([.23, .32, .72, .38])
for i, (values, color, target) in enumerate(zip(data, COLORS, [1, -1])):
    for j in range(2):
        error = np.abs(values[:, j+1] - target)
        x = 2*i+j
        q = np.quantile(np.maximum(error, 1e-16), [0, .5, 1])
        inset.plot([x, x], q[[0, 2]], color=color, lw=1.5)
        inset.plot(x, q[1], marker='_', color=color, ms=8, mew=2)
inset.set(yscale='log', xlim=(-.5, 3.5), xticks=range(4), xticklabels=['P:L', 'P:R', 'C:L', 'C:R'],
          ylabel=r'$|\omega(t,t)-\omega(t,t)_{\mathrm{target}}|$')
inset.set_title('Invariant error: median + range', fontsize=9, pad=4)
inset.tick_params(labelsize=8)
inset.yaxis.label.set_size(9)
inset.grid(axis='y', alpha=.18)

ax = axes[2]
for i, (values, color) in enumerate(zip(data, COLORS)):
    overlap_error = 1 - values[:, 3].real
    assert (overlap_error > 0).all(), 'Log axis requires positive overlap errors'
    ax.scatter(i+rng.uniform(-.19, .19, len(values)), overlap_error,
               s=19, color=color, alpha=.55, linewidths=0, zorder=3)
    ax.boxplot(overlap_error, positions=[i], widths=.5, showfliers=False,
               medianprops={'color': '#202630', 'linewidth': 2},
               boxprops={'color': color}, whiskerprops={'visible': False},
               showcaps=False)
ax.set(yscale='log', ylim=(5e-4, .6), ylabel='1 − c',
       xticks=[0, 1], xticklabels=LABELS, title='C)  Defect overlap error')
ax.grid(axis='y', alpha=.18)

for ext in ['png', 'pdf', 'svg']:
    fig.savefig(OUT / f'mps_factorization_bosonic_z2_t_summary.{ext}', dpi=200)
print('Output:', OUT)
for label, values in zip(LABELS, data):
    print(label, 'median loss:', np.median(values[:, 0].real),
          'median overlap:', np.median(values[:, 3].real),
          'max imaginary invariant:', np.max(np.abs(values[:, 1:3].imag)))
