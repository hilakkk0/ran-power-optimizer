"""Radio, capacity and energy model: evaluates a network configuration."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .scenario import Scenario


@dataclass
class ModelParams:
    gmax_dbi: float = 15.0
    vertical_hpbw_deg: float = 10.0
    sidelobe_att_db: float = 30.0
    rsrp_min_dbm: float = -110.0
    sinr_min_db: float = -3.0
    interference_load: float = 0.6
    se_alpha: float = 0.75
    se_max_bps_hz: float = 7.4
    n_trx: int = 2
    p0_w: float = 130.0
    delta_p: float = 4.7
    p_sleep_w: float = 75.0
    base_power_dbm: float = 43.0
    base_tilt_deg: float = 6.0


@dataclass
class NetworkState:
    """The decision variables: one entry per cell."""
    power_dbm: np.ndarray
    tilt_deg: np.ndarray
    active: np.ndarray

    def copy(self) -> "NetworkState":
        return NetworkState(self.power_dbm.copy(), self.tilt_deg.copy(), self.active.copy())


@dataclass
class KPIs:
    coverage: float
    mean_tput_mbps: float
    edge_tput_mbps: float
    power_w: float
    mean_sinr_db: float
    mean_rsrp_dbm: float

    def as_dict(self) -> dict:
        return {k: float(v) for k, v in self.__dict__.items()}


@dataclass
class Evaluation:
    kpis: KPIs
    cell_users: np.ndarray
    cell_power_w: np.ndarray
    maps: dict = field(default_factory=dict)


def baseline_state(scn: Scenario, params: ModelParams) -> NetworkState:
    c = scn.n_cells
    return NetworkState(
        power_dbm=np.full(c, params.base_power_dbm),
        tilt_deg=np.full(c, params.base_tilt_deg),
        active=np.ones(c, dtype=bool),
    )


def weighted_percentile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    if values.size == 0:
        return 0.0
    order = np.argsort(values)
    v, w = values[order], weights[order]
    cum = np.cumsum(w) / w.sum()
    idx = min(int(np.searchsorted(cum, q / 100.0)), len(v) - 1)
    return float(v[idx])


def cell_input_power_w(state: NetworkState, params: ModelParams) -> np.ndarray:
    p_out_w = 10.0 ** (state.power_dbm / 10.0) / 1000.0
    active_w = params.n_trx * params.p0_w + params.delta_p * p_out_w
    sleep_w = params.n_trx * params.p_sleep_w
    return np.where(state.active, active_w, sleep_w)


def evaluate(scn: Scenario, params: ModelParams, state: NetworkState,
             maps: bool = False) -> Evaluation:
    p = scn.n_pixels
    sla = params.sidelobe_att_db

    vatt = np.minimum(
        12.0 * ((scn.dep_deg - state.tilt_deg[:, None]) / params.vertical_hpbw_deg) ** 2, sla)
    att = np.minimum(scn.hatt + vatt, sla)

    rsrp = (state.power_dbm[:, None] - scn.re_db + params.gmax_dbi - att - scn.pl_db)
    rsrp = np.where(state.active[:, None], rsrp, -300.0)

    serving = rsrp.argmax(axis=0)
    rsrp_s = rsrp[serving, np.arange(p)]
    mw = 10.0 ** (rsrp / 10.0)
    s_mw = mw[serving, np.arange(p)]
    interf = params.interference_load * (mw.sum(axis=0) - s_mw)
    noise_mw = 10.0 ** (scn.noise_dbm_re / 10.0)
    sinr_lin = s_mw / (interf + noise_mw)
    sinr_db = 10.0 * np.log10(sinr_lin)

    covered = (rsrp_s >= params.rsrp_min_dbm) & (sinr_db >= params.sinr_min_db)

    w = scn.traffic
    cell_users = np.bincount(serving[covered], weights=w[covered], minlength=scn.n_cells)
    eff = np.minimum(params.se_alpha * np.log2(1.0 + sinr_lin), params.se_max_bps_hz)
    tput = scn.cfg.bandwidth_hz * eff / 1e6 / np.maximum(cell_users[serving], 1.0)
    tput = np.where(covered, tput, 0.0)

    wc = w[covered]
    coverage = float(wc.sum() / w.sum())
    if wc.size:
        mean_t = float((tput[covered] * wc).sum() / wc.sum())
        edge_t = weighted_percentile(tput[covered], wc, 5.0)
    else:
        mean_t = edge_t = 0.0

    cell_p = cell_input_power_w(state, params)
    kpis = KPIs(
        coverage=coverage, mean_tput_mbps=mean_t, edge_tput_mbps=edge_t,
        power_w=float(cell_p.sum()),
        mean_sinr_db=float((sinr_db * w).sum() / w.sum()),
        mean_rsrp_dbm=float((rsrp_s * w).sum() / w.sum()),
    )

    out = Evaluation(kpis=kpis, cell_users=cell_users, cell_power_w=cell_p)
    if maps:
        shape = scn.shape
        out.maps = {
            "rsrp": rsrp_s.reshape(shape),
            "sinr": sinr_db.reshape(shape),
            "tput": tput.reshape(shape),
            "serving": serving.reshape(shape),
            "covered": covered.reshape(shape),
        }
    return out
