#!/usr/bin/env python3
"""
replot_two_losses.py

Reload saved results and regenerate the log-log scaling plots
without retraining.

Usage:
    python replot_two_losses.py <path/to/results_models_data_scaling_two_losses.json>

Output plots are saved next to the JSON file under plots/.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -------------------------
# Plotting config
# -------------------------
LINESTYLE  = {"joint": "-", "prior": "--"}
YLABEL = r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(\cdot)\right)\right]$"


def _add_power_reference(Ks, ys, exponent):
    Ks = np.asarray(Ks, dtype=float)
    ys = np.asarray(ys, dtype=float)
    mask = np.isfinite(ys) & (ys > 0) & np.isfinite(Ks) & (Ks > 0)
    if not np.any(mask):
        return None, None
    i0 = np.where(mask)[0][0]
    c = ys[i0] * Ks[i0] ** exponent
    ref = c / Ks ** exponent
    return c, ref


def plot_models_data_scaling_two_losses(all_results_by_loss, plot_dir,
                                        split="test", logx=True, logy=True):
    assert split in ("val", "test")
    key_mean = f"avgED_{split}_mean"
    key_std  = f"avgED_{split}_std"

    fig = plt.figure()

    unique_configs = set()
    for results in all_results_by_loss.values():
        for r in results:
            unique_configs.add((int(r["width"]), int(r["depth"])))
    unique_configs = sorted(unique_configs, key=lambda x: (x[1], x[0]))

    cmap = cm.get_cmap("tab10", len(unique_configs))
    color_map = {cfg: cmap(i) for i, cfg in enumerate(unique_configs)}

    ref_Ks, ref_ys = None, None

    for loss_mode in sorted(all_results_by_loss.keys()):
        results = all_results_by_loss[loss_mode]
        linestyle = LINESTYLE.get(loss_mode, "-")

        groups = {}
        for r in results:
            wd = (int(r["width"]), int(r["depth"]))
            groups.setdefault(wd, []).append(r)
        for wd in groups:
            groups[wd] = sorted(groups[wd], key=lambda rr: rr["K_train"])

        for (w, d), rows in sorted(groups.items(), key=lambda t: (t[0][1], t[0][0])):
            Ks = [rr["K_train"] for rr in rows]
            ys = [rr[key_mean]  for rr in rows]
            es = [rr[key_std]   for rr in rows]

            if ref_Ks is None:
                ref_Ks, ref_ys = Ks, ys

            color = color_map[(w, d)]
            label = f"{loss_mode}: layers={d} (width={w})"

            plt.errorbar(
                Ks, ys,
                yerr=es,
                marker="o",
                capsize=3,
                linestyle=linestyle,
                color=color,
                label=label,
            )

    if ref_Ks is not None:
        _, ref_half = _add_power_reference(ref_Ks, ref_ys, 0.5)
        if ref_half is not None:
            plt.plot(
                ref_Ks, ref_half,
                linestyle=":",
                color="black",
                linewidth=2,
                label=r"$\propto 1/\sqrt{N}$",
            )
        _, ref_one = _add_power_reference(ref_Ks, ref_ys, 1.0)
        if ref_one is not None:
            plt.plot(
                ref_Ks, ref_one,
                linestyle="-.",
                color="black",
                linewidth=2,
                label=r"$\propto 1/N$",
            )

    plt.xlabel("Training set size $N$", fontsize=16)
    plt.ylabel(YLABEL, fontsize=16)
    plt.tick_params(axis="both", labelsize=12)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.0)

    plt.legend(frameon=False, fontsize=12, loc="lower left", ncol=1)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    fname = f"scaling_models_avgED_vs_dataK_{split.upper()}_TWO_LOSSES.png"
    outpath = os.path.join(plot_dir, fname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved {split.upper()} plot to: {os.path.abspath(outpath)}")


def main():
    if len(sys.argv) < 2:
        # default: look for the JSON in the same directory as this script
        default = os.path.join(
            os.path.dirname(__file__),
            "data_scaling_saved_20260224_164620",
            "results_models_data_scaling_two_losses.json",
        )
        json_path = default
        print(f"No path given. Using default:\n  {json_path}")
    else:
        json_path = sys.argv[1]

    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"Results file not found: {json_path}")

    with open(json_path) as f:
        all_results_by_loss = json.load(f)

    plot_dir = os.path.join(os.path.dirname(json_path), "plots")

    plot_models_data_scaling_two_losses(all_results_by_loss, plot_dir, split="val",  logx=True, logy=True)
    plot_models_data_scaling_two_losses(all_results_by_loss, plot_dir, split="test", logx=True, logy=True)


if __name__ == "__main__":
    main()
