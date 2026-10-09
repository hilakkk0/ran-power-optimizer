"""Matplotlib figures. Every function returns the figure so apps can reuse it."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from .scenario import Scenario
from .model import NetworkState, KPIs

C_BASE, C_OPT, C_SLEEP = "#7f8c9b", "#1f9d6b", "#d64545"


def _overlay(ax, scn: Scenario, state: NetworkState) -> None:
    """Sites as triangles; sector markers show boresight, red x = sleeping cell."""
    ax.scatter(scn.site_xy[:, 0], scn.site_xy[:, 1], marker="^", s=45, c="white",
               edgecolors="black", linewidths=0.8, zorder=3)
    cxy = scn.cell_xy()
    az = np.radians(scn.cell_az)
    mx = cxy[:, 0] + 110.0 * np.cos(az)
    my = cxy[:, 1] + 110.0 * np.sin(az)
    on = state.active
    ax.scatter(mx[on], my[on], marker="o", s=14, c="black", zorder=3)
    ax.scatter(mx[~on], my[~on], marker="x", s=36, c=C_SLEEP, linewidths=1.8, zorder=4)


def _map(fig, ax, scn, data, title, cmap, vmin, vmax, label, state):
    im = ax.imshow(data, origin="lower", extent=scn.extent, cmap=cmap, vmin=vmin,
                   vmax=vmax, interpolation="bilinear")
    _overlay(ax, scn, state)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7)


def plot_comparison(scn, base_state, base_maps, opt_state, opt_maps):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    specs = [("rsrp", "RSRP", "viridis", -120, -60, "dBm"),
             ("sinr", "SINR", "plasma", -5, 25, "dB"),
             ("tput", "User throughput", "magma", 0, 8, "Mbit/s")]
    for row, (name, state, maps) in enumerate(
            [("Baseline", base_state, base_maps), ("Optimised", opt_state, opt_maps)]):
        for col, (key, title, cmap, lo, hi, unit) in enumerate(specs):
            _map(fig, axes[row, col], scn, maps[key], f"{name}: {title}", cmap,
                 lo, hi, unit, state)
    fig.suptitle("Coverage, quality and capacity: baseline vs. AI-optimised "
                 "(red x = cell in sleep mode)", fontsize=13)
    return fig


def plot_summary(scn, history, base_k: KPIs, opt_k: KPIs,
                 base_state: NetworkState, opt_state: NetworkState):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    ax = axes[0]
    gens = np.arange(1, len(history["best_fitness"]) + 1)
    ax.plot(gens, np.array(history["best_power_ratio"]) * 100, color=C_OPT, lw=2)
    ax.axhline(100, color=C_BASE, ls="--", lw=1)
    ax.set_xlabel("Generation"); ax.set_ylabel("Network power (% of baseline)")
    ax.set_title("Genetic algorithm convergence", fontsize=10)
    ax2 = ax.twinx()
    ax2.plot(gens, np.array(history["best_coverage"]) * 100, color="#3b6fd6", lw=1.5)
    ax2.set_ylabel("Coverage (%)", color="#3b6fd6")
    ax.grid(alpha=0.25)

    ax = axes[1]
    idx = np.arange(scn.n_cells)
    ax.axhline(base_state.power_dbm[0], color=C_BASE, ls="--", lw=1.2, label="Baseline")
    colors = [C_OPT if a else C_SLEEP for a in opt_state.active]
    heights = np.where(opt_state.active, opt_state.power_dbm, 0.0)
    ax.bar(idx, heights, color=colors)
    for i in idx[~opt_state.active]:
        ax.text(i, 1, "sleep", rotation=90, ha="center", va="bottom", fontsize=7,
                color=C_SLEEP)
    ax.set_xlabel("Cell"); ax.set_ylabel("TX power (dBm)")
    ax.set_title("Optimised power per cell (idle headroom = baseline minus bar)",
                 fontsize=10)
    ax.set_ylim(0, 54); ax.legend(fontsize=8, loc="upper right")

    ax = axes[2]
    labels = ["Power", "Coverage", "Cell-edge\nthroughput", "Mean\nthroughput"]
    vals = [opt_k.power_w / base_k.power_w, opt_k.coverage / base_k.coverage,
            opt_k.edge_tput_mbps / base_k.edge_tput_mbps,
            opt_k.mean_tput_mbps / base_k.mean_tput_mbps]
    cols = [C_OPT] + ["#3b6fd6"] * 3
    bars = ax.bar(labels, np.array(vals) * 100, color=cols)
    ax.axhline(100, color=C_BASE, ls="--", lw=1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 1.5, f"{v * 100:.0f}%",
                ha="center", fontsize=9)
    ax.set_ylim(0, max(130, max(vals) * 100 + 12))
    ax.set_ylabel("Optimised as % of baseline")
    ax.set_title("KPI trade-off", fontsize=10)
    return fig
