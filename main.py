"""Command line entry point: build a scenario, optimise it, save figures and reports.

    python main.py                       # defaults (7 sites, 21 cells)
    python main.py --rings 2 --gens 150  # 19 sites, 57 cells
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ranopt import GAConfig, ModelParams, ScenarioConfig, build_scenario, optimize
from ranopt.viz import plot_comparison, plot_summary


def cell_report(scn, res) -> pd.DataFrame:
    b, o = res.baseline, res.best
    status = np.where(~o.active, "SLEEP",
                      np.where(o.power_dbm < b.power_dbm - 0.5, "REDUCED",
                               np.where(o.power_dbm > b.power_dbm + 0.5, "INCREASED", "UNCHANGED")))
    return pd.DataFrame({
        "cell": np.arange(scn.n_cells),
        "site": scn.cell_site,
        "azimuth_deg": scn.cell_az,
        "baseline_tx_dbm": b.power_dbm,
        "optimised_tx_dbm": np.where(o.active, o.power_dbm.round(1), np.nan),
        "baseline_tilt_deg": b.tilt_deg,
        "optimised_tilt_deg": o.tilt_deg.round(1),
        "status": status,
        "baseline_users": res.baseline_eval.cell_users.round(1),
        "optimised_users": res.best_eval.cell_users.round(1),
        "baseline_power_w": res.baseline_eval.cell_power_w.round(0),
        "optimised_power_w": res.best_eval.cell_power_w.round(0),
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=7, help="scenario seed")
    ap.add_argument("--rings", type=int, default=1, help="hex rings: 1 = 7 sites, 2 = 19")
    ap.add_argument("--hotspots", type=int, default=4)
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gens", type=int, default=120)
    ap.add_argument("--max-cov-drop", type=float, default=1.0,
                    help="allowed coverage drop in percentage points")
    ap.add_argument("--max-edge-drop", type=float, default=15.0,
                    help="allowed cell-edge throughput drop in percent")
    ap.add_argument("--max-mean-drop", type=float, default=20.0,
                    help="allowed mean throughput drop in percent")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scn = build_scenario(ScenarioConfig(seed=args.seed, n_rings=args.rings,
                                        n_hotspots=args.hotspots))
    ga = GAConfig(pop_size=args.pop, generations=args.gens,
                  max_coverage_drop=args.max_cov_drop / 100.0,
                  max_edge_drop=args.max_edge_drop / 100.0,
                  max_mean_drop=args.max_mean_drop / 100.0, seed=args.seed)
    print(f"Scenario: {len(scn.site_xy)} sites, {scn.n_cells} cells, "
          f"{scn.n_pixels} grid points, {scn.cfg.n_users} active users")
    t0 = time.time()
    res = optimize(scn, ModelParams(), ga,
                   progress=lambda g, n: print(f"\rOptimising: generation {g}/{n}",
                                               end="", flush=True))
    print(f"\nDone in {time.time() - t0:.1f} s")

    b, o = res.baseline_eval.kpis, res.best_eval.kpis
    saving = 100.0 * (1.0 - o.power_w / b.power_w)
    print(f"\n{'KPI':<28}{'Baseline':>12}{'Optimised':>12}")
    for name, bv, ov, fmt in [
        ("Network power (kW)", b.power_w / 1e3, o.power_w / 1e3, "{:.2f}"),
        ("Coverage (%)", b.coverage * 100, o.coverage * 100, "{:.1f}"),
        ("Cell-edge throughput (Mbit/s)", b.edge_tput_mbps, o.edge_tput_mbps, "{:.2f}"),
        ("Mean throughput (Mbit/s)", b.mean_tput_mbps, o.mean_tput_mbps, "{:.2f}"),
        ("Mean SINR (dB)", b.mean_sinr_db, o.mean_sinr_db, "{:.1f}")]:
        print(f"{name:<28}{fmt.format(bv):>12}{fmt.format(ov):>12}")
    n_sleep = int((~res.best.active).sum())
    print(f"\nEnergy saving: {saving:.1f} %  ({n_sleep} of {scn.n_cells} cells in sleep mode)")

    fig = plot_comparison(scn, res.baseline, res.baseline_eval.maps,
                          res.best, res.best_eval.maps)
    fig.savefig(out / "coverage_comparison.png", dpi=140)
    fig = plot_summary(scn, res.history, b, o, res.baseline, res.best)
    fig.savefig(out / "optimization_summary.png", dpi=140)
    plt.close("all")

    report = cell_report(scn, res)
    report.to_csv(out / "cell_report.csv", index=False)
    with open(out / "summary.json", "w") as f:
        json.dump({"energy_saving_percent": round(saving, 2),
                   "cells_in_sleep_mode": n_sleep,
                   "baseline": b.as_dict(), "optimised": o.as_dict()}, f, indent=2)
    print(f"Saved figures, cell_report.csv and summary.json to '{out}/'")


if __name__ == "__main__":
    main()
