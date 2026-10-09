"""Synthetic 3-sector macro network scenario.

Everything here is generated from a random seed. No operator data is used.

Propagation follows a simplified 3GPP TR 38.901 Urban Macro (UMa) model with a
LOS-probability-weighted mean path loss, spatially correlated shadowing, and the
3GPP sector antenna pattern (horizontal 65 deg HPBW, vertical electrical tilt).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SECTOR_AZIMUTHS_DEG = (30.0, 150.0, 270.0)


@dataclass
class ScenarioConfig:
    isd_m: float = 500.0          # inter-site distance
    n_rings: int = 1              # 1 -> 7 sites (21 cells), 2 -> 19 sites (57 cells)
    area_m: float = 2000.0        # square evaluation area, centred on the origin
    step_m: float = 40.0          # grid resolution
    fc_ghz: float = 2.1
    bs_height_m: float = 25.0
    ue_height_m: float = 1.5
    bandwidth_hz: float = 20e6
    n_subcarriers: int = 1200     # 100 PRB x 12 subcarriers (20 MHz)
    shadow_sigma_db: float = 5.0
    shadow_corr_m: float = 80.0
    n_users: int = 300            # simultaneously active users in the whole area
    n_hotspots: int = 4
    seed: int = 7


@dataclass
class Scenario:
    cfg: ScenarioConfig
    shape: tuple                  # (ny, nx)
    extent: tuple                 # (xmin, xmax, ymin, ymax) for plotting
    site_xy: np.ndarray           # (S, 2)
    cell_site: np.ndarray         # (C,) site index of each cell
    cell_az: np.ndarray           # (C,) boresight azimuth in degrees
    traffic: np.ndarray           # (P,) active users per pixel, sums to n_users
    pl_db: np.ndarray             # (C, P) path loss incl. shadowing
    hatt: np.ndarray              # (C, P) horizontal antenna attenuation, dB
    dep_deg: np.ndarray           # (C, P) depression angle from antenna to pixel
    noise_dbm_re: float           # thermal noise per resource element
    re_db: float                  # 10*log10(subcarriers): per-RE power split

    @property
    def n_cells(self) -> int:
        return len(self.cell_site)

    @property
    def n_pixels(self) -> int:
        return self.pl_db.shape[1]

    def cell_xy(self) -> np.ndarray:
        return self.site_xy[self.cell_site]

    def traffic_map(self) -> np.ndarray:
        return self.traffic.reshape(self.shape)


def _hex_sites(n_rings: int, isd: float) -> np.ndarray:
    pts = []
    for q in range(-n_rings, n_rings + 1):
        for r in range(max(-n_rings, -q - n_rings), min(n_rings, -q + n_rings) + 1):
            pts.append((isd * (q + r / 2.0), isd * (r * np.sqrt(3) / 2.0)))
    return np.array(pts)


def _correlated_field(rng, shape, sigma_db: float, corr_px: float) -> np.ndarray:
    """Zero-mean Gaussian random field with Gaussian spatial correlation."""
    noise = rng.standard_normal(shape)
    ky = np.fft.fftfreq(shape[0])
    kx = np.fft.fftfreq(shape[1])
    KX, KY = np.meshgrid(kx, ky)
    kernel = np.exp(-2.0 * (np.pi * corr_px) ** 2 * (KX ** 2 + KY ** 2))
    field = np.fft.ifft2(np.fft.fft2(noise) * kernel).real
    return field * (sigma_db / field.std())


def _uma_path_loss_db(d2d, cfg: ScenarioConfig) -> np.ndarray:
    """Mean UMa path loss: LOS/NLOS blended with the 3GPP LOS probability."""
    h_bs, h_ut, fc = cfg.bs_height_m, cfg.ue_height_m, cfg.fc_ghz
    dh = h_bs - h_ut
    d3d = np.sqrt(d2d ** 2 + dh ** 2)
    d_bp = 4.0 * (h_bs - 1.0) * (h_ut - 1.0 if h_ut > 1.0 else 0.5) * fc * 1e9 / 3e8
    pl1 = 28.0 + 22.0 * np.log10(d3d) + 20.0 * np.log10(fc)
    pl2 = (28.0 + 40.0 * np.log10(d3d) + 20.0 * np.log10(fc)
           - 9.0 * np.log10(d_bp ** 2 + dh ** 2))
    pl_los = np.where(d2d <= d_bp, pl1, pl2)
    pl_nlos = np.maximum(
        pl_los,
        13.54 + 39.08 * np.log10(d3d) + 20.0 * np.log10(fc) - 0.6 * (h_ut - 1.5),
    )
    d = np.maximum(d2d, 1.0)
    p_los = np.where(d <= 18.0, 1.0, 18.0 / d + np.exp(-d / 63.0) * (1.0 - 18.0 / d))
    return p_los * pl_los + (1.0 - p_los) * pl_nlos


def build_scenario(cfg: ScenarioConfig | None = None) -> Scenario:
    cfg = cfg or ScenarioConfig()
    rng = np.random.default_rng(cfg.seed)

    half = cfg.area_m / 2.0
    xs = np.arange(-half, half + 1e-6, cfg.step_m)
    X, Y = np.meshgrid(xs, xs)
    shape = X.shape
    px, py = X.ravel(), Y.ravel()

    site_xy = _hex_sites(cfg.n_rings, cfg.isd_m)
    n_sites = len(site_xy)
    cell_site = np.repeat(np.arange(n_sites), 3)
    cell_az = np.tile(np.array(SECTOR_AZIMUTHS_DEG), n_sites)
    cxy = site_xy[cell_site]

    dx = px[None, :] - cxy[:, 0:1]
    dy = py[None, :] - cxy[:, 1:2]
    d2d = np.hypot(dx, dy)
    phi = np.degrees(np.arctan2(dy, dx)) - cell_az[:, None]
    phi = (phi + 180.0) % 360.0 - 180.0
    hatt = np.minimum(12.0 * (phi / 65.0) ** 2, 30.0)
    dep = np.degrees(np.arctan2(cfg.bs_height_m - cfg.ue_height_m, d2d))

    corr_px = cfg.shadow_corr_m / cfg.step_m
    shadow = np.stack([
        _correlated_field(rng, shape, cfg.shadow_sigma_db, corr_px).ravel()
        for _ in range(n_sites)
    ])
    pl = _uma_path_loss_db(d2d, cfg) + shadow[cell_site]

    density = np.ones(shape)
    for _ in range(cfg.n_hotspots):
        hx, hy = rng.uniform(-0.35, 0.35, 2) * cfg.area_m
        sig = rng.uniform(100.0, 220.0)
        amp = rng.uniform(5.0, 10.0)
        density += amp * np.exp(-((X - hx) ** 2 + (Y - hy) ** 2) / (2 * sig ** 2))
    traffic = density.ravel()
    traffic = traffic / traffic.sum() * cfg.n_users

    noise_dbm_re = -174.0 + 10.0 * np.log10(15e3) + 7.0   # 15 kHz RE, NF = 7 dB

    return Scenario(
        cfg=cfg, shape=shape, extent=(-half, half, -half, half),
        site_xy=site_xy, cell_site=cell_site, cell_az=cell_az, traffic=traffic,
        pl_db=pl, hatt=hatt, dep_deg=dep, noise_dbm_re=noise_dbm_re,
        re_db=10.0 * np.log10(cfg.n_subcarriers),
    )
