# Bare SHG fiber brightness model

import numpy as np

OPACITY_DEFAULTS = {
    # Global brightness anchor + per-fiber jitter around it.
    "intensity_range": (0.35, 0.40),
    "fiber_log_std": 0.05,   # was 0.22 -- turned down by default

    # Along-fiber SHG phase interference -- the main modulation effect,
    # left at its original strength.
    "shg_floor": 0.30,
    "shg_phase_cycles": (20.5, 60.0),

    # Connectivity gap modulation (uses aux_L_conn per fiber).
    "gap_depth": 0.08,   # was 0.35 -- turned down by default
    "gap_min": 0.85,     # was 0.06 -- turned down by default
    "gap_cycles_base": 20.0,
    "gap_cycles_scale": 4.5,

    # Many oversampled points overlap along a single fiber's own length (radius
    # stamps stacking as you walk along the spline), so this just keeps one
    # fiber from self-saturating regardless of oversample rate.
    "overlap_damp": 8.0,
}


class FiberOpacityTable:
    # Per-fiber brightness parameters for rasterization.
    __slots__ = ("fiber_base", "phase_offset", "n_phase")

    def __init__(self, fiber_base, phase_offset, n_phase):
        self.fiber_base = np.asarray(fiber_base, dtype=np.float64)
        self.phase_offset = np.asarray(phase_offset, dtype=np.float64)
        self.n_phase = np.asarray(n_phase, dtype=np.float64)


def build_fiber_opacity_table(n_fibers, rng, cfg=None):
    #Build per-fiber brightness baseline + SHG phase parameters.

    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    n = int(n_fibers)
    if n == 0:
        return FiberOpacityTable(np.zeros(0), np.zeros(0), np.zeros(0))

    lo, hi = cfg["intensity_range"]
    global_anchor = 0.5 * (lo + hi)
    fiber_base = global_anchor * np.exp(cfg["fiber_log_std"] * rng.normal(size=n))
    fiber_base = np.clip(fiber_base, lo * 0.5, hi * 1.4)

    phase_lo, phase_hi = cfg["shg_phase_cycles"]
    phase_offset = rng.uniform(0.0, 2.0 * np.pi, size=n)
    n_phase = rng.uniform(phase_lo, phase_hi, size=n)

    return FiberOpacityTable(fiber_base, phase_offset, n_phase)


def shg_along_fiber(table, fiber_i, s_frac, cfg=None):
    # Along-fiber SHG coherent-interference factor in [shg_floor, 1].
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    shg_floor = float(np.clip(cfg["shg_floor"], 0.0, 0.9))
    phase_offset = table.phase_offset[fiber_i]
    n_phase = table.n_phase[fiber_i]
    wave = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_phase * s_frac + phase_offset)
    return shg_floor + (1.0 - shg_floor) * (wave ** 2)


def connectivity_gap(s_frac, aux_L_conn, cfg=None):
    # Connectivity gap opacity in [gap_min, 1]; separate from SHG phase.
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    cn = float(np.clip(aux_L_conn, 0.0, 1.0))
    n_slow = cfg["gap_cycles_base"] + cfg["gap_cycles_scale"] * cn
    conn_phase = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_slow * s_frac)
    gap = 1.0 - cn * cfg["gap_depth"] * (1.0 - conn_phase ** 2.2)
    return float(np.clip(gap, cfg["gap_min"], 1.0))


# Added modulate to take care of rasterization issue
# modulate=True makes the bumpy looking fibers
def stamp_opacity(table, fiber_i, s_frac, aux_L_conn, cfg=None, modulate=False):
    # Combined brightness multiplier at one stamp position:
    #   fiber_base[i] -- per-fiber average brightness
    #   shg           -- along-fiber phase pulsation (primary effect)
    #   gap           -- connectivity breaks
    if not modulate:
        return table.fiber_base[fiber_i]
    shg = shg_along_fiber(table, fiber_i, s_frac, cfg)
    gap = connectivity_gap(s_frac, aux_L_conn, cfg)
    return table.fiber_base[fiber_i] * shg * gap
