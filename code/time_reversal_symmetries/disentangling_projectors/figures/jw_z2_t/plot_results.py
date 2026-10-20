"""Plot the four saved Z_2^T datasets without rerunning simulations.

Run from disentangling_projectors:
    ../../num_spt_venv_p314/bin/python figures/jw_z2_t/plot_results.py

PNG/PDF figures and JSON summaries are written beside this script, with
_filtered and _unfiltered suffixes. The older files without these suffixes
are not updated. Each saved figure's full path is printed to the terminal.
The filtered version discards samples with
min_cocycle_abs < 0.975 from every panel. Errors use complex absolute differences from the signed targets
below, not distance to whichever of +/-1 is nearest.
"""
from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/jw-mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/jw-numba")

import json
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
COLORS = dict(product="#d87822", xzx="#2864a0", yzy="#a64d79", xzzzx="#32936f")
LABELS = dict(product="Z", xzx="XZX", yzy="YZY", xzzzx="XZZZX")
# (fermion decoration, Kramers sign), in the generation notebook's convention.
TARGETS = dict(product=(1, 1), xzx=(-1, 1), yzy=(-1, -1), xzzzx=(1, -1))
FLOOR = 1e-16
MIN_COCYCLE_ABS = 0.975
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "savefig.dpi": 190,
})


def floor(values):
    return np.maximum(values, FLOOR)


def save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}", flush=True)
    plt.close(fig)


def state_handles():
    return [Line2D([], [], color=color, lw=4, label=LABELS[state])
            for state, color in COLORS.items()]


def load_data(filtered=True):
    data, summary = {}, {}
    for state in COLORS:
        with (ROOT / "results" / f"z_2_t_{state}_results.pkl").open("rb") as f:
            rows = pickle.load(f)
        n = len(rows)

        def array(key):
            return np.asarray([row[key] for row in rows])

        cocycles = array("cocycles")
        assert cocycles.shape == (n, 2, 2, 2, 2, 2)
        min_cocycle_abs = np.abs(cocycles).reshape(n, -1).min(axis=1)
        assert np.all(np.isfinite(min_cocycle_abs)), state
        below_cutoff = min_cocycle_abs < MIN_COCYCLE_ABS
        keep = ~below_cutoff if filtered else np.ones(n, dtype=bool)
        original_n = n
        rows = [row for row, retain in zip(rows, keep) if retain]
        cocycles = cocycles[keep]
        n = len(rows)
        if not n:
            raise ValueError(f"{state}: no samples pass min_cocycle_abs >= {MIN_COCYCLE_ABS}")
        inv = np.stack([array("left_defect_op_invariant"),
                        array("right_defect_op_invariant")], axis=1)
        # Use the full ratio defined by the generation notebook, retaining
        # deviations of its denominator from one.
        decoration = array("fermion_decoration_from_cocycle")
        np.testing.assert_allclose(
            decoration, cocycles[:, :, 0, 1, 1, 0] / cocycles[:, :, 1, 0, 0, 1])
        kramers = cocycles[:, :, 0, 1, 0, 1]
        fermion_target, kramers_target = TARGETS[state]
        scores = np.asarray([[score[-1] for score in row["defect_ops_scores"]]
                             for row in rows])
        assert scores.shape == (n, 3)  # t, p, pt (not t, t^2, t^3).
        d = dict(
            inv=inv, inv_err=np.abs(inv - fermion_target),
            decoration=decoration, decoration_err=np.abs(decoration - fermion_target),
            kramers=kramers, kramers_err=np.abs(kramers - kramers_target),
            full=np.max(np.abs(array("cocycle_equation_output") - 1).reshape(n, -1), axis=1),
            # For Z_2^T the bosonic twisted cocycle equation has target +1
            # for every state, including those with Kramers sign -1.
            bos=np.max(np.abs(array("bosonic_cocycle_equation_output") - 1).reshape(n, -1), axis=1),
            purity=np.abs(1 - array("sub_cut_rho_purity")),
            defect=np.abs(1 - scores),
        )
        assert all(np.all(np.isfinite(v)) for v in d.values()), state
        data[state] = d
        summary[state] = dict(n=n, original_n=original_n,
                              discarded_n=original_n - n,
                              below_cutoff_n=int(np.sum(below_cutoff)),
                              filter_applied=filtered,
                              min_cocycle_abs_threshold=MIN_COCYCLE_ABS,
                              fermion_target=fermion_target,
                              kramers_target=kramers_target)
        for key in ("full", "bos", "purity", "inv_err", "decoration_err",
                    "kramers_err", "defect"):
            summary[state][key] = dict(
                median=float(np.median(d[key])), max=float(np.max(d[key])),
                at_or_below_plot_floor=int(np.sum(d[key] <= FLOOR)))
        del rows
    return data, summary


def invariant_panel(fig, slot, data, keys, title, ylabel):
    """Raw L/R samples above a separate aligned error strip, keeping outliers visible."""
    grid = slot.subgridspec(2, 1, height_ratios=(2.4, 1), hspace=.05)
    ax = fig.add_subplot(grid[0])
    err_ax = fig.add_subplot(grid[1], sharex=ax)
    rng = np.random.default_rng(20261020)
    ticks, labels = [], []
    key_labels = dict(inv="Defect FP", decoration="Cocycle",
                      kramers="Cocycle invariant")
    for si, (state, d) in enumerate(data.items()):
        for ki, key in enumerate(keys):
            center = si * (len(keys) + .7) + ki
            ticks.append(center)
            labels.append(f"{LABELS[state]}\n{key_labels[key]}" if len(keys) > 1
                          else LABELS[state])
            for side, marker in enumerate(("o", "^")):
                x = center + (side - .5) * .28
                values = d[key][:, side]
                ax.scatter(x + rng.uniform(-.035, .035, len(values)), values.real,
                           s=10, alpha=.22, color=COLORS[state], marker=marker,
                           rasterized=True)
                ax.annotate(("L", "R")[side], (x, np.median(values.real)),
                            xytext=(0, 10 if np.median(values.real) > 0 else -18),
                            textcoords="offset points", ha="center", fontsize=10)
                q = np.quantile(floor(d[key + "_err"][:, side]), [0, .5, 1])
                err_ax.plot([x, x], q[[0, 2]], color=COLORS[state], lw=1.5)
                err_ax.plot(x, q[1], marker="_", color=COLORS[state], ms=8, mew=2)
    for y in (-1, 1):
        ax.axhline(y, ls=":", color=".6", zorder=0)
    all_values = np.concatenate([d[k].real.ravel() for d in data.values() for k in keys])
    ax.set(ylim=(min(-1.25, all_values.min() - .15),
                 max(1.25, all_values.max() + .15)),
           ylabel=ylabel, title=title)
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    err_ax.set(yscale="log", xticks=ticks, xticklabels=labels,
               ylabel="Absolute error\nmedian + range")
    err_ax.tick_params(axis="x", labelsize=9)
    err_ax.grid(axis="y", alpha=.15)
    return ax


def plot_results(data, summary, filtered, purity_bins):
    suffix = "filtered" if filtered else "unfiltered"
    (OUT / f"summary_{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig = plt.figure(figsize=(16, 11), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.15, 1))
    invariant_panel(
        fig, grid[0, 0], data, ("inv", "decoration"),
        r"A) Extraction of $(-1)^n$ from defect operators",
        r"Real part of $(-1)^n$")
    invariant_panel(
        fig, grid[0, 1], data, ("kramers",),
        r"B) Kramers invariant $\omega_f(t,t)$",
        r"Real part of $\omega_f(t,t)$")

    ax = fig.add_subplot(grid[1, 0])
    for si, (state, d) in enumerate(data.items()):
        ax.boxplot([floor(d["defect"][:, j]) for j in range(3)],
                   positions=np.arange(3) + (si - 1.5) * .18,
                   orientation="horizontal", widths=.14,
                   patch_artist=True, manage_ticks=False, whis=(5, 95),
                   flierprops=dict(marker=".", markersize=2, alpha=.35,
                                   markeredgecolor=COLORS[state]),
                   medianprops=dict(color="black"),
                   boxprops=dict(facecolor=COLORS[state], alpha=.65))
    ax.set(xscale="log", yticks=range(3), yticklabels=["$t$", "$p$", "$pt$"],
           xlabel=r"$|1-c|$", title="C) Defect operator overlap errors")
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=.15)
    ax.legend(handles=state_handles(), fontsize=9, loc="center")

    ax = fig.add_subplot(grid[1, 1])
    for state, d in data.items():
        ax.hist(floor(d["purity"]), bins=purity_bins, histtype="step",
                color=COLORS[state], linewidth=1.8, label=LABELS[state])
    ax.set(xscale="log", xlabel=r"$|1-\mathrm{Tr}(\rho^2)|$",
           ylabel="Number of runs", title="D) Purity histogram")
    ax.legend(fontsize=9)
    ax.grid(alpha=.15)
    if filtered:
        fig.suptitle(
            r"JW $\mathbb{Z}_2^F \times \mathbb{Z}_2^T$ defect operators via disentangling projectors",
            fontsize=20
        )
    else:
        fig.suptitle(
            r"JW $\mathbb{Z}_2^F \times \mathbb{Z}_2^T$ defect operators via disentangling projectors - unfiltered",
            fontsize=20
        )
    counts = "; ".join(
        f"{LABELS[s]}: {summary[s]['discarded_n']:,}"
        for s in data)
    if filtered:
        filter_caption = (
            f"Discarded: {counts}.")
    else:
        counts = "; ".join(
            f"{LABELS[s]}: {summary[s]['below_cutoff_n']:,}"
            for s in data)
        filter_caption = (
            f"Unfiltered: all runs retained (0 discarded). "
            f"Below {MIN_COCYCLE_ABS:g} / total: {counts}.")
    maxima = []
    for key, label in (("full", "Fermionic"), ("bos", "Bosonic")):
        values = "; ".join(f"{LABELS[s]}: {summary[s][key]['max']:.3g}" for s in data)
        maxima.append(f"{label} max cocycle deviation: {values}.")
    if filtered:
        fig.supxlabel(
            filter_caption + "\n" + "\n".join(maxima),
            fontsize=10
        )
    else:
        fig.supxlabel(
            "\n".join(maxima),
            fontsize=10
        )
    save(fig, f"disentangling_projectors_z2_t_summary_{suffix}")

    print(json.dumps(summary, indent=2))


def main():
    # Apply the identical selection to every diagnostic in each version.
    data, summary = load_data(filtered=False)
    # Common logarithmic bins allow direct comparison of the two histograms.
    purity_values = np.concatenate([floor(d["purity"]) for d in data.values()])
    purity_bins = np.logspace(np.floor(np.log10(purity_values.min())),
                              np.ceil(np.log10(purity_values.max())), 36)
    plot_results(data, summary, filtered=False, purity_bins=purity_bins)
    data, summary = load_data(filtered=True)
    plot_results(data, summary, filtered=True, purity_bins=purity_bins)


if __name__ == "__main__":
    main()
