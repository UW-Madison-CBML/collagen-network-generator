import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import Opacity as opacity


def _prep_aux(values, n_sp, default):
    """Coerce a per-spline array (or None) to length n_sp, padding with default."""
    if values is None:
        return np.full(n_sp, float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < n_sp:
        arr = np.pad(arr, (0, n_sp - arr.size), constant_values=default)
    return arr[:n_sp]


def rasterize_splines(
    H,
    W,
    splines,
    thickness=3.0,
    oversample=4.0,
    out_H=None,
    out_W=None,
    intensity_seed=0,
    # spline waviness -- ground-truth sinusoidal wobble.
    # Amplitude and wavelength are independent knobs in pixel-space, each
    # optionally scaled per-spline by a [0,1] field.
    wave_amplitude_px=2.8,      # max wobble amplitude, in output pixels
    wave_wavelength_px=None,    # wobble wavelength, in output pixels
                                # (defaults to max(4 * thickness, 6.0))
    aux_wave_amp=None,         # per-spline [0,1]; scales amplitude (e.g. an aux curve field)
    aux_wave_freq=None,        # per-spline [0,1]; scales wavelength (e.g. an aux freq field)
    # connectivity gaps (fiber breaks), independent of the wave above
    L_conn=0.3,
    aux_L_conn=None,
    # brightness model: per-fiber jitter + SHG phase wave + gap dimming
    opacity_table=None,
    opacity_cfg=None,
):
    # Rasterize a list of polyline splines into a grayscale ground-truth image.

    # Barebones by design: splines -> pixels. Two effects are deliberately
    # kept because real SHG collagen images show them too:

    #   1. Sinusoidal wobble along each spline (wavy-fiber geometry).
    #      Amplitude (`wave_amplitude_px` / `aux_wave_amp`) and wavelength
    #      (`wave_wavelength_px` / `aux_wave_freq`) are fully decoupled --
    #      change one without disturbing the other.
    #   2. A brightness model (see Opacity.py): per-fiber jitter, an
    #      along-fiber SHG phase wave, and connectivity-gap dimming.

    # No tone-mapping of any kind -- the raster accumulator is just clipped
    # to [0, 1] at the end.
    
    if out_H is None:
        out_H = H
    if out_W is None:
        out_W = W

    cfg = {**opacity.OPACITY_DEFAULTS, **(opacity_cfg or {})}

    scale_y = out_H / max(H, 1)
    scale_x = out_W / max(W, 1)
    img = np.zeros((out_H, out_W), dtype=np.float32)

    rng = np.random.default_rng(intensity_seed)
    n_sp = len(splines)

    # aux_wave_amp defaults to "full amplitude" (1.0), aux_wave_freq defaults
    # to "base wavelength" (0.5, the midpoint of the 0.5x-1.5x range below).
    aux_wave_amp = _prep_aux(aux_wave_amp, n_sp, 1.0)
    aux_wave_freq = _prep_aux(aux_wave_freq, n_sp, 0.5)
    aux_L_conn = _prep_aux(aux_L_conn, n_sp, L_conn)

    if opacity_table is None:
        opacity_table = opacity.build_fiber_opacity_table(n_sp, rng, cfg)

    stamp_damp = 1.0 / max(cfg["overlap_damp"], 1.0)
    base_wavelength = (
        wave_wavelength_px if wave_wavelength_px is not None
        else max(4.0 * thickness, 6.0)
    )

    for i, spline in enumerate(splines):
        pts = np.asarray(spline)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue

        amp_i = float(np.clip(aux_wave_amp[i], 0.0, 1.0))
        freq_i = float(np.clip(aux_wave_freq[i], 0.0, 1.0))
        cn = float(aux_L_conn[i])

        pts_out = np.empty_like(pts, dtype=np.float64)
        pts_out[:, 0] = pts[:, 0] * scale_y
        pts_out[:, 1] = pts[:, 1] * scale_x

        seg = np.diff(pts_out, axis=0)
        seg_len = np.sqrt((seg**2).sum(axis=1))
        total_len = float(seg_len.sum())
        if total_len < 1e-6:
            continue
        s_vert = np.concatenate([[0.0], np.cumsum(seg_len)])

        # wavelength ranges 0.5x-1.5x the base as freq_i goes 0 -> 1.
        # n_cycles falls out naturally from total spline length / wavelength
        # -- no more arbitrary clipping of the cycle count.
        wavelength = base_wavelength * (0.5 + freq_i)
        n_cycles = total_len / wavelength
        wave_amp = wave_amplitude_px * amp_i

        for seg_idx, ((y0, x0), (y1, x1)) in enumerate(zip(pts_out[:-1], pts_out[1:])):
            dy = y1 - y0
            dx = x1 - x0
            sl = float(np.hypot(dy, dx))
            if sl < 1e-9:
                continue
            ty, tx = dy / sl, dx / sl
            ny_n, nx_n = -tx, ty

            s0 = s_vert[seg_idx]
            num = max(2, int(sl * oversample))
            for t in np.linspace(0.0, 1.0, num=num):
                s = s0 + t * sl
                s_frac = s / total_len
                y = y0 + t * dy
                x = x0 + t * dx
                if wave_amp > 0.0:
                    wobble = wave_amp * np.sin(2.0 * np.pi * n_cycles * s_frac)
                    y += wobble * ny_n
                    x += wobble * nx_n

                op = opacity.stamp_opacity(opacity_table, i, s_frac, cn, cfg) #modulate=False is default
                stamp = op * stamp_damp

                iy, ix = int(round(y)), int(round(x))
                if 0 <= iy < out_H and 0 <= ix < out_W:
                    r = int(np.ceil(thickness))
                    for ddy in range(-r, r + 1):
                        for ddx in range(-r, r + 1):
                            jy, jx = iy + ddy, ix + ddx
                            if 0 <= jy < out_H and 0 <= jx < out_W:
                                d2 = ddy * ddy + ddx * ddx
                                if d2 <= thickness * thickness:
                                    tt = d2 / (thickness * thickness + 1e-8)
                                    wgt = np.exp(-6.0 * tt * tt)
                                    img[jy, jx] += stamp * wgt

    return np.clip(img, 0.0, 1.0).astype(np.float32)

# OPACITY_DEFAULTS = {
#     "intensity_range": (0.35, 0.40),
#     "fiber_log_std": 0.05,
#     "shg_floor": 0.30,
#     "shg_phase_cycles": (20.5, 60.0),
#     "gap_depth": 0.08,
#     "gap_min": 0.85,
#     "gap_cycles_base": 20.0,
#     "gap_cycles_scale": 4.5,
#     "overlap_damp": 8.0,
# }

# def rasterize_splines(
#     H: int, W: int, splines: List[np.ndarray], thickness: float = 3.0, oversample: float = 4.0,
#     out_H: Optional[int] = None, out_W: Optional[int] = None, intensity_seed: int = 0,

#     # wave_amplitude_px: float = 2.8, wave_wavelength_px: Optional[float] = None,
#     # aux_wave_amp: Optional[np.ndarray] = None, aux_wave_freq: Optional[np.ndarray] = None,


#     L_conn: float = 0.3, aux_L_conn: Optional[np.ndarray] = None,
#     # opacity_table: Optional[FiberOpacityTable] = None,
#     opacity_cfg: Optional[Dict[str, Any]] = None
# ) -> np.ndarray:
#     """Rasterize splines into a 2D float32 image with brightness and wobble models."""
#     out_H = out_H or H
#     out_W = out_W or W
#     cfg = {**OPACITY_DEFAULTS, **(opacity_cfg or {})}

#     scale_y, scale_x = out_H / max(H, 1), out_W / max(W, 1)
#     img = np.zeros((out_H, out_W), dtype=np.float32)

#     rng = np.random.default_rng(intensity_seed)
#     n_sp = len(splines)

#     # aux_wave_amp = _prep_aux(aux_wave_amp, n_sp, 1.0)
#     # aux_wave_freq = _prep_aux(aux_wave_freq, n_sp, 0.5)
#     aux_L_conn = _prep_aux(aux_L_conn, n_sp, L_conn)

#     # if opacity_table is None:
#     #     opacity_table = build_fiber_opacity_table(n_sp, rng, cfg)

#     stamp_damp = 1.0 / max(cfg["overlap_damp"], 1.0)
#     # base_wavelength = wave_wavelength_px if wave_wavelength_px is not None else max(4.0 * thickness, 6.0)

#     for i, spline in enumerate(splines):
#         pts = np.asarray(spline)
#         if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
#             continue

#         # amp_i = float(np.clip(aux_wave_amp[i], 0.0, 1.0))
#         # freq_i = float(np.clip(aux_wave_freq[i], 0.0, 1.0))
#         cn = float(aux_L_conn[i])

#         pts_out = np.empty_like(pts, dtype=np.float64)
#         pts_out[:, 0] = pts[:, 0] * scale_y
#         pts_out[:, 1] = pts[:, 1] * scale_x

#         seg = np.diff(pts_out, axis=0)
#         seg_len = np.sqrt((seg**2).sum(axis=1))
#         total_len = float(seg_len.sum())
#         if total_len < 1e-6:
#             continue
#         s_vert = np.concatenate([[0.0], np.cumsum(seg_len)])

#         # wavelength = base_wavelength * (0.5 + freq_i)
#         # n_cycles = total_len / wavelength
#         # wave_amp = wave_amplitude_px * amp_i

#         for seg_idx, ((y0, x0), (y1, x1)) in enumerate(zip(pts_out[:-1], pts_out[1:])):
#             dy, dx = y1 - y0, x1 - x0
#             sl = float(np.hypot(dy, dx))
#             if sl < 1e-9:
#                 continue
#             ty, tx = dy / sl, dx / sl
#             ny_n, nx_n = -tx, ty

#             s0 = s_vert[seg_idx]
#             num = max(2, int(sl * oversample))
#             for t in np.linspace(0.0, 1.0, num=num):
#                 s = s0 + t * sl
#                 s_frac = s / total_len
#                 y, x = y0 + t * dy, x0 + t * dx

#                 # if wave_amp > 0.0:
#                 #     wobble = wave_amp * np.sin(2.0 * np.pi * n_cycles * s_frac)
#                 #     y += wobble * ny_n
#                 #     x += wobble * nx_n

#                 op = stamp_opacity(opacity_table, i, s_frac, cn, cfg)
#                 stamp = op * stamp_damp

#                 iy, ix = int(round(y)), int(round(x))
#                 if 0 <= iy < out_H and 0 <= ix < out_W:
#                     r = int(np.ceil(thickness))
#                     for ddy in range(-r, r + 1):
#                         for ddx in range(-r, r + 1):
#                             jy, jx = iy + ddy, ix + ddx
#                             if 0 <= jy < out_H and 0 <= jx < out_W:
#                                 d2 = ddy * ddy + ddx * ddx
#                                 if d2 <= thickness * thickness:
#                                     tt = d2 / (thickness * thickness + 1e-8)
#                                     wgt = np.exp(-6.0 * tt * tt)
#                                     img[jy, jx] += stamp * wgt

#     return np.clip(img, 0.0, 1.0).astype(np.float32)

