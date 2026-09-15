import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_splprep
from typing import List, Tuple, Optional, Dict, Any # added for sinusoidal func

def sample_vector(vx, vy, pos, ref_dir=None):
    # Sample the axial/nematic orientation field (vx=Qx=cos(2*theta),
    # vy=Qy=sin(2*theta)) at pos and decode it back to a real direction unit
    # vector (dy, dx).

    # Bilinearly interpolating Qx/Qy directly (as done below) is correct for
    # an axial field -- it's *why* the double-angle encoding exists, so
    # orientations 180 degrees apart average out instead of cancelling. But
    # the interpolated result must be decoded (theta = 0.5*atan2(Qy,Qx))
    # before it's used as a direction to walk along, or you get a field that
    # effectively rotates at 2x the true rate -- harmless where theta is
    # slowly varying, but produces spurious spirals/inward-pointing paths
    # wherever theta turns sharply (e.g. wrapping around a well).

    # Since axial data is headless (theta and theta+pi are the same
    # orientation), decoding gives two opposite candidate directions. When
    # ref_dir is provided (the previous step's heading), whichever candidate
    # is more aligned with it is returned, so the streamline doesn't
    # randomly flip 180 degrees step to step.
    h, w = vx.shape
    y, x = pos
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None

    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0

    def interp(arr):
        return (
            (1 - dx) * (1 - dy) * arr[y0, x0]
            + dx * (1 - dy) * arr[y0, x0 + 1]
            + (1 - dx) * dy * arr[y0 + 1, x0]
            + dx * dy * arr[y0 + 1, x0 + 1]
        )

    Qx = interp(vx)
    Qy = interp(vy)
    if Qx == 0.0 and Qy == 0.0:
        return None

    theta = 0.5 * np.arctan2(Qy, Qx)
    d = np.array([np.sin(theta), np.cos(theta)])  # (dy, dx), matches pos=(y, x)

    if ref_dir is not None and (d[0] * ref_dir[0] + d[1] * ref_dir[1]) < 0.0:
        d = -d

    return d

# Added to move sinusoidal offset to the fiber points
def sinusoidal_fiber_offset(pts: np.ndarray, rng: np.random.Generator,
                            wave_amplitude_px: float = 2.8, wave_wavelength_px: Optional[float] = None,
                            wave_amp: Optional[np.ndarray] = None, wave_freq: Optional[np.ndarray] = None
                            ) -> np.ndarray:
    if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
        return pts

    base_wavelength = wave_wavelength_px if wave_wavelength_px is not None else 12.0#max(4.0 * thickness, 6.0)
    wavelength = base_wavelength * (0.5 + wave_freq)

    wave_amp = wave_amplitude_px * wave_amp

    seg = np.diff(pts, axis=0)
    seg_len = np.sqrt((seg ** 2).sum(axis=1))
    cum_len = np.concatenate([[0.0], np.cumsum(seg_len)])
    total_len = cum_len[-1]

    if total_len < 1e-6:
        return pts

    n_cycles = total_len / wavelength

    phase_offset = rng.uniform(0, 2 * np.pi)
    s_frac = cum_len / total_len
    wave = wave_amp * np.sin(2.0 * np.pi * n_cycles * s_frac + phase_offset)

    tangent = seg / (seg_len[:, None] + 1e-8)
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal = np.vstack([normal[0], normal])

    offset_pts = pts + normal * wave[:, None]
    return offset_pts


def resolve_streamline_steps(spline_length=None, max_steps=None, default=300):
    """Max integration steps per direction from spline_length or max_steps."""
    if spline_length is not None:
        return max(1, int(spline_length))
    if max_steps is not None:
        return max(1, int(max_steps))
    return max(1, int(default))


def integrate_streamline(
    vx, vy, seed, step_size=1.0, max_steps=None, spline_length=None,
    direction=1, L_curve=0.5, susceptibility=1.0, rng=None,
):
    # susceptibility in [0,1]: how strongly this fiber follows the local
    # vector field at each step.
    #   1.0 -> follows the field faithfully (original behavior).
    #   0.0 -> ignores the field entirely and walks in a straight line along
    #          whatever direction it first sampled at the seed -- this is
    #          what lets a fiber stay straight and cross paths with curvier
    #          neighbors, regardless of how the field bends further along.
    # Values in between blend the two, so a fiber can be "mostly straight
    # but nudged by the field" rather than an all-or-nothing switch.
    pts = []
    pos = np.array(seed, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    max_angle = 0.35 * L_curve
    n_steps = resolve_streamline_steps(spline_length=spline_length, max_steps=max_steps)
    susceptibility = float(np.clip(susceptibility, 0.0, 1.0))

    ref_dir = None       # previous step's heading, for headless (axial) continuity
    preferred_dir = None  # this fiber's own straight-line heading (set from its first field sample)

    for _ in range(n_steps):
        v1 = sample_vector(vx, vy, pos, ref_dir=ref_dir)
        if v1 is None:
            break
        if preferred_dir is None:
            preferred_dir = v1.copy()

        mid = pos + 0.5 * step_size * direction * v1
        v2 = sample_vector(vx, vy, mid, ref_dir=ref_dir if ref_dir is not None else v1)
        if v2 is None:
            break

        if susceptibility < 1.0:
            # preferred_dir is itself headless -- flip it to match v2's
            # current sign before blending so they don't cancel out.
            pref = preferred_dir if (preferred_dir[0] * v2[0] + preferred_dir[1] * v2[1]) >= 0.0 else -preferred_dir
            blended = susceptibility * v2 + (1.0 - susceptibility) * pref
            n = np.linalg.norm(blended)
            v2 = blended / n if n > 1e-8 else pref

        if max_angle > 0:
            a = rng.normal(0.0, max_angle)
            ca, sa = np.cos(a), np.sin(a)
            vy2, vx2 = v2[0], v2[1]
            v2 = np.array([ca * vy2 - sa * vx2, sa * vy2 + ca * vx2])
            v2 = v2 / (np.linalg.norm(v2) + 1e-8)

        pos = pos + step_size * direction * v2
        pts.append(pos.copy())
        ref_dir = v2

    return np.array(pts)


def generate_fiber(
    vx, vy, seed, step_size=1.0, max_steps=None, spline_length=None,
    L_curve=0.5, susceptibility=1.0, rng=None,
):
    forward = integrate_streamline(
        vx, vy, seed, step_size, max_steps=max_steps, spline_length=spline_length,
        direction=1, L_curve=L_curve, susceptibility=susceptibility, rng=rng,
    )
    backward = integrate_streamline(
        vx, vy, seed, step_size, max_steps=max_steps, spline_length=spline_length,
        direction=-1, L_curve=L_curve, susceptibility=susceptibility, rng=rng,
    )

    pts = []
    if backward is not None and len(backward) > 0:
        pts.append(backward[::-1])
    pts.append(np.asarray(seed, dtype=float)[None, :])
    if forward is not None and len(forward) > 0:
        pts.append(forward)
    return np.vstack(pts)


def fit_spline(points, smoothing=2.0, num_samples=400, k=3):
    if len(points) < k + 2:
        return points
    x = points[:, 1]
    y = points[:, 0]
    try:
        spl, _ = make_splprep([x, y], s=smoothing, k=k)
        u_new = np.linspace(0, 1, num_samples)
        x_s, y_s = spl(u_new)
        return np.vstack([y_s, x_s]).T
    except Exception:
        return points

def per_spline_auxiliary_values(global_p, n, rng):
    p = float(np.clip(global_p, 0.0, 1.0))
    n = int(n)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    if p <= 0.0:
        return np.zeros(n, dtype=np.float64)
    if p >= 1.0:
        return np.ones(n, dtype=np.float64)
    eps = 1e-9
    edge = 4.0 * p * (1.0 - p) + eps
    concentration = 0.35 / edge
    a = max(p * concentration, eps)
    b = max((1.0 - p) * concentration, eps)
    x = rng.beta(a, b, size=n)
    return np.clip(x.astype(np.float64), 0.0, 1.0)


def sample_seeds_from_density(D, spline_num, L_density, rng):
    D = np.asarray(D, dtype=np.float64)
    H, W = D.shape
    n = int(max(0, spline_num))
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)

    D_clipped = np.clip(D, 0.0, None)
    if np.all(D_clipped == 0):
        probs = np.full(D_clipped.size, 1.0 / D_clipped.size, dtype=np.float64)
    else:
        dens_power = 0.8 + 2.2 * float(L_density)
        probs = np.power(D_clipped.ravel(), dens_power)
        probs = probs / (probs.sum() + 1e-12)

    expected = n * probs
    counts = np.floor(expected).astype(np.int64)
    missing = int(n - counts.sum())

    if missing > 0:
        frac = expected - counts
        frac_sum = frac.sum()
        if frac_sum <= 0:
            pick = rng.choice(counts.size, size=missing, replace=False)
        else:
            pick_probs = frac / frac_sum
            pick = rng.choice(counts.size, size=missing, replace=False, p=pick_probs)
        counts[pick] += 1

    idx = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rng.shuffle(idx)
    seeds = np.column_stack([idx // W, idx % W]).astype(np.int64)
    return seeds

