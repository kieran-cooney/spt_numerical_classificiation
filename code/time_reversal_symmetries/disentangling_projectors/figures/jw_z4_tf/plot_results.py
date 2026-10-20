"""Reproduce figures from the two saved result pickles; no simulations are rerun.

Run from the disentangling_projectors directory:
    ../../num_spt_venv_p314/bin/python figures/jw_z4_tf/plot_results.py

Outputs are written beside this script. Matplotlib and Numba caches default
to /tmp via the environment settings below.
"""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/jw-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/jw-numba')
import pickle
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
COLORS = {'cluster': '#2864a0', 'product': '#d87822'}
FLOOR = 1e-16
plt.rcParams.update({'font.size': 12, 'axes.spines.top': False,
                     'axes.spines.right': False, 'savefig.dpi': 190})

def floor(x):
    return np.maximum(x, FLOOR)

def save(fig, name):
    for ext in ('png', 'pdf'):
        fig.savefig(OUT / f'{name}.{ext}', bbox_inches='tight')

data = {}
summary = {}
for state in COLORS:
    with open(ROOT / 'results' / f'z_4_tf_{state}_results.pkl', 'rb') as f:
        rows = pickle.load(f)
    a = lambda key: np.asarray([r[key] for r in rows])
    inv = np.stack([a('left_defect_op_invariant'), a('right_defect_op_invariant')], axis=1)
    decoration = np.asarray([r['cocycles'][:, 0, 1] for r in rows])
    assert decoration.shape == inv.shape, 'Decoration columns must be left, right'
    full = a('cocycle_equation_output')
    bos = a('bosonic_cocycle_equation_output')
    target = np.ones(bos.shape[1:])
    # Targets used in the original generation notebook, separately for each state.
    if state == 'cluster':
        target[..., -1, -1, -1] = -1
    scores = np.asarray([[r['defect_ops_scores'][j][-1] for j in range(3)] for r in rows])
    d = dict(decoration=decoration,
             decoration_err=np.abs(decoration - (-1 if state == 'cluster' else 1)),
             inv=inv, inv_err=np.abs(inv - (-1 if state == 'cluster' else 1)),
             full=np.max(np.abs(full-1).reshape(len(rows), -1), axis=1),
             bos=np.max(np.abs(bos-target).reshape(len(rows), -1), axis=1),
             purity=np.abs(1-a('sub_cut_rho_purity')), defect=np.abs(1-scores),
             cocycles=a('cocycles')[:, 0:2, 0, 0])
    assert all(np.all(np.isfinite(v)) for v in d.values())
    data[state] = d
    summary[state] = {'n': len(rows)}
    for key in ('full', 'bos', 'purity', 'inv_err', 'decoration_err', 'defect'):
        summary[state][key] = {'median': float(np.median(d[key])),
                               'max': float(np.max(d[key])),
                               'at_or_below_plot_floor': int(np.sum(d[key] <= FLOOR))}
    del rows
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2))

fig, axs = plt.subplots(2, 2, figsize=(14, 10), layout='constrained')
ax = axs[0, 0]
rng = np.random.default_rng(20261014)
centers = [0, 1, 2.6, 3.6]
labels = ['Defect operator\nFP','Cocycle\ninvariant',
          'Defect operator\nFP','Cocycle\ninvariant',]
ins = ax.inset_axes([.10,.34,.83,.32])
for si, (state, d) in enumerate(data.items()):
    for ki, key in enumerate(('inv', 'decoration')):
        center = centers[2*si+ki]
        for side in range(2):
            x = center + (side-.5)*.27
            values = d[key][:,side]
            ax.scatter(x+rng.uniform(-.035,.035,len(values)), values.real,
                       s=10, alpha=.16, color=COLORS[state], rasterized=True)
            median = np.median(values.real)
            #ax.plot(x, median, marker='_', color='black', ms=13, mew=2)
            ax.annotate(('L','R')[side], (x, median), xytext=(0,13 if si == 0 else -22),
                        textcoords='offset points', ha='center', fontsize=12)
            q=np.quantile(floor(d[key+'_err'][:,side]),[0,.5,1])
            ins.plot([x,x],q[[0,2]],color=COLORS[state],lw=1.5)
            ins.plot(x,q[1],marker='_',color=COLORS[state],ms=7,mew=2)
            ins.annotate(('L','R')[side], (x,q[1]), xytext=(5,0),
                         textcoords='offset points', va='center', fontsize=10)
for y in (-1,1): ax.axhline(y, ls=':', color='.6', zorder=0)
ax.set(xticks=centers, xticklabels=labels, ylim=(-1.2,1.2),
       ylabel='Real part of $(-1)^n$', title='A) Extraction of $(-1)^n$ from defect operators')
ax.tick_params(axis='x',labelsize=11)
ax.legend(handles=[Line2D([],[],color=c,lw=5,label=s.title()) for s,c in COLORS.items()],
          loc='upper left')
ins.set(yscale='log',xticks=centers,
        xticklabels=['Defect operator\nFP','Cocycle\ninvariant',
                     'Defect operator\nFP','Cocycle\ninvariant',],
        xlim=(-.35,4.05),title='Absolute error from ±1: median + range')
ins.tick_params(labelsize=10); ins.title.set_fontsize(11)

ax=axs[0,1]
for state,d in data.items():
    for key,ls,label in [('full','-','Fermionic'),('bos','--','Bosonic')]:
        x=np.sort(floor(d[key])); y=np.arange(1,len(x)+1)/len(x)
        ax.step(x,y,where='post',color=COLORS[state],ls=ls,lw=2,label=f'{state.title()} · {label}')
ax.set(xscale='log',ylim=(0,1.02),xlabel='Maximum absolute cocycle equation deviation',
       ylabel='Fraction of runs with deviation ≤ x',title='B) Cocycle equation deviation')
ax.legend(loc='upper center',fontsize=11); ax.grid(alpha=.15)

ax=axs[1,0]
for si,(state,d) in enumerate(data.items()):
    vals=[d['defect'][:,j] for j in range(3)]
    positions=np.arange(3)+(si-.5)*.28
    ax.boxplot([floor(v) for v in vals],positions=positions,orientation='horizontal',widths=.22,
               patch_artist=True,manage_ticks=False,whis=(5,95),
               flierprops={'marker':'.','markersize':2,'alpha':.35,'markeredgecolor':COLORS[state]},
               medianprops={'color':'black'},boxprops={'facecolor':COLORS[state],'alpha':.65})
ax.set(xscale='log',yticks=range(3),yticklabels=['$t$','$t^2=p$','$t^3=tp$'],
       xlabel='1 - $c$',title='C) Defect operator overlap $c$ boxplots')
ax.invert_yaxis(); ax.grid(axis='x',alpha=.15)
ax.legend(handles=[Line2D([],[],color=c,lw=5,label=s.title()) for s,c in COLORS.items()],loc='center right')

ax=axs[1,1]
d=data['cluster']
ax.scatter(floor(d['purity']),floor(d['full']),color=COLORS['cluster'],
           s=10,alpha=.2,edgecolors='none',label='Cluster',rasterized=True)
p=data['product']
ax.scatter(floor(p['purity']),floor(p['full']),color=COLORS['product'],marker='x',
           s=10,alpha=.2,linewidths=.6,label='Product',rasterized=True)
ax.set(xscale='log',yscale='log',xlabel=r'$\left|1-\mathrm{Tr}\left(\rho^2\right)\right|$',
       ylabel='Maximum fermionic cocycle deviation',title='D) Purity deviation versus cocycle residual')
legend = ax.legend(loc='upper left')
for handle in legend.legend_handles:
    handle.set_alpha(1)
ax.grid(alpha=.15)
fig.suptitle(r'JW $\mathbb{Z}_4^{TF}$ defect operators via disentangling projectors',fontsize=20)
#fig.supxlabel(r'Log plots display values ≤ $10^{-16}$ at $10^{-16}$; absolute deviations retain complex roundoff.',fontsize=12)
save(fig,'disentangling_projectors_z4_tf_summary')

fig2, axes=plt.subplots(1,2,figsize=(10,5),layout='constrained')
t=np.linspace(0,2*np.pi,400)
for ax,(state,d) in zip(axes,data.items()):
    ax.plot(np.cos(t),np.sin(t),color='.8',ls='--',lw=1)
    for side,marker in enumerate(('o','^')):
        z=d['cocycles'][:,side]
        ax.scatter(z.real,z.imag,s=22,alpha=.2,marker=marker,
                   color=('#2864a0','#b94e67')[side],label=('Left','Right')[side],rasterized=True)
    ax.axhline(0,color='.9',lw=.8); ax.axvline(0,color='.9',lw=.8)
    ax.set(aspect='equal',xlim=(-1.15,1.15),ylim=(-1.15,1.15),
           xlabel='Real part',ylabel='Imaginary part',title=state.title())
    ax.legend()
fig2.suptitle('Selected cocycle entry: cocycles[:, 0, 0]',fontsize=18)
fig2.supxlabel('All 1,000 runs shown per state; coincident points overlap. Dashed circle: unit modulus.',fontsize=12)
save(fig2,'cocycle_complex_plane')
print(json.dumps(summary,indent=2))
