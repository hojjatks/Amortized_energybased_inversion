#!/usr/bin/env python3
"""
replot_param_scaling.py

Reload saved results and regenerate the parameter-scaling log-log plots
without retraining.

Usage:
    python replot_param_scaling.py <path/to/results_models_param_scaling_two_losses.json>

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
LINESTYLE = {"joint": "-", "prior": "--"}
YLABEL = r"$\mathbb{E}^{y^\dagger\sim\kappa}\!\left[D_E^2\!\left(T(\cdot,\cdot; y^\dagger)_\#\gamma,\;\pi^{y^\dagger}(\cdot)\right)\right]$"


def plot_models_param_scaling_two_losses(all_results_by_loss, plot_dir,
                                         split="test", logx=True, logy=True):
    assert split in ("val", "test")
    key_mean = f"avgED_{split}_mean"
    key_std  = f"avgED_{split}_std"

    fig = plt.figure()

    unique_Ks = set()
    for results in all_results_by_loss.values():
        for r in results:
            unique_Ks.add(int(r["K_train"]))
    unique_Ks = sorted(unique_Ks)

    cmap = cm.get_cmap("tab10", len(unique_Ks))
    color_map = {K: cmap(i) for i, K in enumerate(unique_Ks)}

    for loss_mode in sorted(all_results_by_loss.keys()):
        results = all_results_by_loss[loss_mode]
        linestyle = LINESTYLE.get(loss_mode, "-")

        groups = {}
        for r in results:
            K = int(r["K_train"])
            groups.setdefault(K, []).append(r)
        for K in groups:
            groups[K] = sorted(groups[K], key=lambda rr: rr["n_params"])

        for K, rows in sorted(groups.items()):
            xs = [rr["n_params"] for rr in rows]
            ys = [rr[key_mean]   for rr in rows]
            es = [rr[key_std]    for rr in rows]

            plt.errorbar(
                xs, ys,
                yerr=es,
                marker="o",
                capsize=3,
                linestyle=linestyle,
                color=color_map[K],
                label=f"{loss_mode}: K={K}",
            )

    plt.xlabel("Number of parameters", fontsize=14)
    plt.ylabel(YLABEL, fontsize=14)
    plt.tick_params(axis="both", labelsize=12)
    if logx:
        plt.xscale("log")
    if logy:
        plt.yscale("log")

    ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax * 3.0)

    plt.legend(frameon=False, fontsize=12, loc="upper right", ncol=2)
    plt.tight_layout()

    os.makedirs(plot_dir, exist_ok=True)
    fname = f"scaling_models_avgED_vs_nparams_{split.upper()}_TWO_LOSSES.png"
    outpath = os.path.join(plot_dir, fname)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)
    print(f"Saved {split.upper()} plot to: {os.path.abspath(outpath)}")


def main():
    if len(sys.argv) < 2:
        default = os.path.join(
            os.path.dirname(__file__),
            "param_scaling_saved_20260331_210837",
            "results_models_param_scaling_two_losses.json",
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

    plot_models_param_scaling_two_losses(all_results_by_loss, plot_dir, split="val",  logx=True, logy=True)
    plot_models_param_scaling_two_losses(all_results_by_loss, plot_dir, split="test", logx=True, logy=True)


if __name__ == "__main__":
    main()
