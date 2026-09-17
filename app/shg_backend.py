import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_splprep
import plotly.graph_objects as go


OPACITY_DEFAULTS = {
    "intensity_range": (0.35, 0.40),
    "fiber_log_std": 0.05,
    "shg_floor": 0.30,
    "shg_phase_cycles": (20.5, 60.0),
    "gap_depth": 0.08,
    "gap_min": 0.85,
    "gap_cycles_base": 20.0,
    "gap_cycles_scale": 4.5,
    "overlap_damp": 8.0,
}

class FiberOpacityTable:
    """Per-fiber brightness parameters for rasterization."""
    __slots__ = ("fiber_base", "phase_offset", "n_phase")

    def __init__(self, fiber_base: np.ndarray, phase_offset: np.ndarray, n_phase: np.ndarray):
        self.fiber_base = np.asarray(fiber_base, dtype=np.float64)
        self.phase_offset = np.asarray(phase_offset, dtype=np.float64)
        self.n_phase = np.asarray(n_phase, dtype=np.float64)


def build_fiber_opacity_table(n_fibers: int, rng: np.random.Generator, cfg: Optional[Dict[str, Any]] = None) -> FiberOpacityTable:
    """Build per-fiber brightness baseline + SHG phase parameters."""
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


def shg_along_fiber(table: FiberOpacityTable, fiber_i: int, s_frac: float, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Along-fiber SHG coherent-interference factor in [shg_floor, 1]."""
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    shg_floor = float(np.clip(cfg["shg_floor"], 0.0, 0.9))
    phase_offset = table.phase_offset[fiber_i]
    n_phase = table.n_phase[fiber_i]
    wave = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_phase * s_frac + phase_offset)
    return float(shg_floor + (1.0 - shg_floor) * (wave ** 2))


def connectivity_gap(s_frac: float, aux_L_conn: float, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Connectivity gap opacity in [gap_min, 1]."""
    cfg = {**OPACITY_DEFAULTS, **(cfg or {})}
    cn = float(np.clip(aux_L_conn, 0.0, 1.0))
    n_slow = cfg["gap_cycles_base"] + cfg["gap_cycles_scale"] * cn
    conn_phase = 0.5 + 0.5 * np.cos(2.0 * np.pi * n_slow * s_frac)
    gap = 1.0 - cn * cfg["gap_depth"] * (1.0 - conn_phase ** 2.2)
    return float(np.clip(gap, cfg["gap_min"], 1.0))


def stamp_opacity(table: FiberOpacityTable, fiber_i: int, s_frac: float, aux_L_conn: float, cfg: Optional[Dict[str, Any]] = None) -> float:
    """Combined brightness multiplier at one stamp position."""
    shg = shg_along_fiber(table, fiber_i, s_frac, cfg)
    gap = connectivity_gap(s_frac, aux_L_conn, cfg)
    return float(table.fiber_base[fiber_i] * shg * gap)


# =====================================================================
# 2. VECTOR FIELD & SYNTHETIC MAP GENERATION
# =====================================================================

WELL_HARD_SCALE = 1
WELL_OUTER_DECAY = 0.75
WELL_INFLUENCE_TAU = 0.15


def normalize(Qx: np.ndarray, Qy: np.ndarray, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
    mag = np.sqrt(Qx * Qx + Qy * Qy) + eps
    return Qx / mag, Qy / mag


def create_grid(image_size: int, resolution_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    M = int(image_size * resolution_factor)
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)
    return np.meshgrid(x, y)


def _tangent_q_from_gradient(gx: np.ndarray, gy: np.ndarray, Qx_g: np.ndarray, Qy_g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    grad_len = np.sqrt(gx * gx + gy * gy) + 1e-8
    rx, ry = gx / grad_len, gy / grad_len
    tx1, ty1 = -ry, rx
    tx2, ty2 = ry, -rx
    inward_x, inward_y = -rx, -ry

    def inward_score(tx, ty):
        phi = np.arctan2(ty, tx)
        return np.cos(phi) * inward_x + np.sin(phi) * inward_y

    score1, score2 = inward_score(tx1, ty1), inward_score(tx2, ty2)
    pick1 = (score1 < score2) | ((np.abs(score1 - score2) < 1e-6) & (Qx_g * tx1 + Qy_g * ty1 >= Qx_g * tx2 + Qy_g * ty2))
    tx = np.where(pick1, tx1, tx2)
    ty = np.where(pick1, ty1, ty2)

    theta = np.arctan2(ty, tx)
    return np.cos(2.0 * theta), np.sin(2.0 * theta)


def make_wells(X: np.ndarray, Y: np.ndarray, G_conn: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    max_wells = 20
    K = int(max_wells * (1 - G_conn))
    base_sigma = 0.08

    if K > 0:
        sx = base_sigma * rng.uniform(0.5, 1.6, size=K)
        sy = base_sigma * rng.uniform(0.5, 1.6, size=K)
        phi = rng.uniform(0, 2 * np.pi, size=K)
        max_hard_axis = WELL_HARD_SCALE * np.max(np.maximum(sx, sy))
        min_dist = 2.0 * max_hard_axis
        centers = []
        target, attempts, max_attempts = K, 0, 8000
        while len(centers) < target and attempts < max_attempts:
            cand = rng.random(2)
            if not centers or np.all(np.sum((np.asarray(centers) - cand) ** 2, axis=1) >= min_dist * min_dist):
                centers.append(cand)
            attempts += 1
            if attempts % 1200 == 0 and len(centers) < target:
                min_dist *= 0.92
        while len(centers) < target:
            centers.append(rng.random(2))
        centers = np.asarray(centers, dtype=float)
        wells = np.column_stack([centers, sx, sy, phi])
    else:
        wells = np.empty((0, 5))

    W = np.zeros_like(X)
    hard_zero_mask = np.zeros_like(X, dtype=bool)

    for wx, wy, ax, ay, ang in wells:
        dx, dy = X - wx, Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr, yr = ca * dx + sa * dy, -sa * dx + ca * dy
        r2_hard = (xr / (WELL_HARD_SCALE * ax + 1e-8)) ** 2 + (yr / (WELL_HARD_SCALE * ay + 1e-8)) ** 2
        hard_zero_mask |= (r2_hard <= 1.0)
        r = np.sqrt(np.maximum(r2_hard, 1e-12))
        outside = np.maximum(r - 1.0, 0.0)
        W += np.exp(-outside / WELL_OUTER_DECAY)

    return wells, W, hard_zero_mask

# ORiginal density function that initiates density as 0
# def make_density(W: np.ndarray, hard_zero_mask: np.ndarray, G_density: float, L_density: float, rng: np.random.Generator) -> np.ndarray:
#     G_density, L_density = float(np.clip(G_density, 0.0, 1.0)), float(np.clip(L_density, 0.0, 1.0))
#     mean_D = np.exp(-4.0 * W)

#     if L_density > 0.0:
#         z = gaussian_filter(rng.normal(size=W.shape), 3.0 + 18.0 * L_density)
#         z = (z - z.mean()) / (z.std() + 1e-8)
#         multiplier = np.exp(L_density * (0.25 + 0.55 * L_density) * z)
#     else:
#         multiplier = np.ones_like(W)

#     D = np.where(hard_zero_mask, 0.0, G_density * mean_D * multiplier)
#     valid = ~hard_zero_mask
#     if np.any(valid) and D[valid].mean() > 1e-8:
#         D = np.where(valid, D / D[valid].mean() * G_density, 0.0)
#     return np.clip(D, 0.0, None)

def make_density(W: np.ndarray, hard_zero_mask: np.ndarray, G_density: float, L_density: float, rng: np.random.Generator) -> np.ndarray:
    G_density, L_density = float(np.clip(G_density, 0.0, 1.0)), float(np.clip(L_density, 0.0, 1.0))
    
    # Base density set directly to a very low initial floor (e.g., 0.01 * G_density)
    baseline_density = 0.01 * G_density
    mean_D = np.full_like(W, baseline_density)

    if L_density > 0.0:
        z = gaussian_filter(rng.normal(size=W.shape), 3.0 + 18.0 * L_density)
        z = (z - z.mean()) / (z.std() + 1e-8)
        multiplier = np.exp(L_density * 0.15 * z)  # Scaled down multiplier swing
    else:
        multiplier = np.ones_like(W)

    D = np.where(hard_zero_mask, 0.0, mean_D * multiplier)
    
    # Ensure background stays very low (capped at near zero default)
    return np.clip(D, 0.0, None)


# def make_global_orientation(shape: Tuple[int, int], G_align: float, G_curve: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
#     theta0 = rng.uniform(0, np.pi)
#     ny, nx = shape
#     X, Y = np.meshgrid(np.linspace(0, 1, nx), np.linspace(0, 1, ny))
#     g = float(np.clip(G_curve, 0.0, 1.0))

#     psi = rng.uniform(0, 2 * np.pi)
#     u = np.array([np.cos(psi), np.sin(psi)])
#     r_center = g * 1.28 + (1.0 - g) * 34.0
#     cx, cy = 0.5 + r_center * u[0], 0.5 + r_center * u[1]

#     theta_circle = np.arctan2(Y - cy, X - cx) + (np.pi / 2)
#     theta_base = (1.0 - g) * theta0 + g * theta_circle

#     phi = gaussian_filter(rng.normal(size=shape), sigma=40 * G_align + 5)
#     phi = (phi - phi.mean()) / (phi.std() + 1e-8)
#     theta = theta_base + 0.6 * (1 - G_align) * phi

#     return np.cos(2 * theta), np.sin(2 * theta)


def _well_weight_and_tangent(X: np.ndarray, Y: np.ndarray, wells: np.ndarray, Qx_ref: np.ndarray, Qy_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    influence = np.zeros_like(X)
    Qx, Qy = np.zeros_like(X), np.zeros_like(Y)
    if len(wells) == 0:
        return Qx, Qy, influence

    best_weight = np.full(X.shape, -1.0, dtype=np.float64)
    for wx, wy, ax, ay, ang in wells:
        dx, dy = X - wx, Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr, yr = ca * dx + sa * dy, -sa * dx + ca * dy

        r2_hard = (xr / (WELL_HARD_SCALE * ax + 1e-8)) ** 2 + (yr / (WELL_HARD_SCALE * ay + 1e-8)) ** 2
        r = np.sqrt(np.maximum(r2_hard, 1e-12))
        weight = np.exp(-np.maximum(r - 1.0, 0.0) / WELL_OUTER_DECAY)

        gx = ca * (2.0 * xr / (ax ** 2 + 1e-8)) - sa * (2.0 * yr / (ay ** 2 + 1e-8))
        gy = sa * (2.0 * xr / (ax ** 2 + 1e-8)) + ca * (2.0 * yr / (ay ** 2 + 1e-8))
        qx_t, qy_t = _tangent_q_from_gradient(gx, gy, Qx_ref, Qy_ref)

        replace = weight > best_weight
        best_weight = np.where(replace, weight, best_weight)
        Qx = np.where(replace, qx_t, Qx)
        Qy = np.where(replace, qy_t, Qy)
        influence = np.where(replace, weight, influence)

    influence /= influence.max() + 1e-8
    return Qx, Qy, influence

# Add this near the Vector Field section of shg_backend.py

def wells_from_canvas_objects(objects: list, canvas_width: int, canvas_height: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parses Fabric.js shapes (circles/rectangles) from streamlit-drawable-canvas 
    into normalized well tuples [wx, wy, ax, ay, phi] and a hard zero mask.
    """
    wells = []
    hard_zero_mask = np.zeros((canvas_height, canvas_width), dtype=bool)
    
    Y_grid, X_grid = np.ogrid[:canvas_height, :canvas_width]
    
    for obj in objects:
        shape_type = obj.get("type")
        left = obj.get("left", 0)
        top = obj.get("top", 0)
        width = obj.get("width", 1) * obj.get("scaleX", 1)
        height = obj.get("height", 1) * obj.get("scaleY", 1)
        angle_deg = obj.get("angle", 0)
        phi = np.radians(angle_deg)
        
        # Center coordinates normalized to [0, 1]
        cx = (left + width / 2.0) / canvas_width
        cy = (top + height / 2.0) / canvas_height
        
        # Radii normalized to grid scale
        ax = (width / 2.0) / canvas_width
        ay = (height / 2.0) / canvas_height
        
        wells.append([cx, cy, ax, ay, phi])
        
        # Calculate hard mask in pixel space
        dx = (X_grid - (left + width / 2.0))
        dy = (Y_grid - (top + height / 2.0))
        ca, sa = np.cos(-phi), np.sin(-phi)
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        
        r2 = (xr / (width / 2.0 + 1e-8))**2 + (yr / (height / 2.0 + 1e-8))**2
        hard_zero_mask |= (r2 <= 1.0)
        
    return np.array(wells) if wells else np.empty((0, 5)), hard_zero_mask

def compute_drawn_well_potential(X: np.ndarray, Y: np.ndarray, wells: np.ndarray) -> np.ndarray:
    """Computes well potential field (W) for user-drawn wells."""
    W = np.zeros_like(X)
    if len(wells) == 0:
        return W
        
    for wx, wy, ax, ay, ang in wells:
        dx, dy = X - wx, Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr, yr = ca * dx + sa * dy, -sa * dx + ca * dy
        r2_hard = (xr / (ax + 1e-8)) ** 2 + (yr / (ay + 1e-8)) ** 2
        r = np.sqrt(np.maximum(r2_hard, 1e-12))
        outside = np.maximum(r - 1.0, 0.0)
        W += np.exp(-outside / WELL_OUTER_DECAY)

    return W

# Add or update in shg_backend.py

# Add/update in shg_backend.py

def generate_per_point_gaussian_density(
    point_configs: list, 
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a density field map and orientation wells where each point 
    has its own custom width, height, position, and intensity parameters.
    
    Parameters:
        point_configs: List of dicts, e.g., [{"x": 100, "y": 200, "w": 30, "h": 50, "intensity": 1.2}, ...]
        shape: Tuple (H, W) canvas size.
        
    Returns:
        D: Combined Continuous Density Map.
        wells: Formatted well array for vector alignment [cx, cy, ax, ay, phi].
        hard_zero_mask: Mask of well centers where density is zeroed out.
    """
    H, W = shape
    D = np.zeros((H, W), dtype=np.float64)
    hard_zero_mask = np.zeros((H, W), dtype=bool)
    
    if not point_configs:
        return D, np.empty((0, 5)), hard_zero_mask

    Y_grid, X_grid = np.ogrid[:H, :W]
    wells_list = []
    
    for cfg in point_configs:
        pt_x = cfg["x"]
        pt_y = cfg["y"]
        sigma_x = max(1.0, float(cfg["w"]))
        sigma_y = max(1.0, float(cfg["h"]))
        intensity = float(cfg["intensity"])
        
        # 1. Individual 2D Gaussian density distribution
        gauss = intensity * np.exp(
            -(((X_grid - pt_x) ** 2) / (2.0 * sigma_x ** 2) + ((Y_grid - pt_y) ** 2) / (2.0 * sigma_y ** 2))
        )
        D += gauss
        
        # 2. Derive well properties normalized to [0, 1]
        cx, cy = pt_x / W, pt_y / H
        ax, ay = sigma_x / W, sigma_y / H
        wells_list.append([cx, cy, ax, ay, 0.0])
        
        # Hard zero mask inside the center core
        r2_core = ((X_grid - pt_x) / (0.5 * sigma_x + 1e-8))**2 + ((Y_grid - pt_y) / (0.5 * sigma_y + 1e-8))**2
        hard_zero_mask |= (r2_core <= 1.0)
        
    wells = np.array(wells_list, dtype=np.float64)
    return D, wells, hard_zero_mask

def relax(Qx: np.ndarray, Qy: np.ndarray, Qx_global: np.ndarray, Qy_global: np.ndarray,
          Qx_well: np.ndarray, Qy_well: np.ndarray, well_influence: np.ndarray, D: np.ndarray,
          G_align: float, L_align: float, wells: Optional[np.ndarray] = None,
          X: Optional[np.ndarray] = None, Y: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    alpha0, local_sigma, tau = 0.2 + 0.8 * L_align, 1.0 + 4.0 * L_align, WELL_INFLUENCE_TAU
    near_well = well_influence > tau

    for _ in range(25):
        alpha = np.where(near_well, alpha0 * D * 0.25, alpha0 * D)
        beta = np.where(near_well, 0.0, G_align * (1.0 - well_influence))
        eta = np.where(near_well, 1.0, well_influence.copy())

        total = alpha + beta + eta
        scale = np.where(total > 1.0, 1.0 / (total + 1e-8), 1.0)
        alpha, beta, eta = alpha * scale, beta * scale, eta * scale

        Qx_loc, Qy_loc = gaussian_filter(Qx, local_sigma), gaussian_filter(Qy, local_sigma)
        Qx = (1.0 - alpha - beta - eta) * Qx + alpha * Qx_loc + beta * Qx_global + eta * Qx_well
        Qy = (1.0 - alpha - beta - eta) * Qy + alpha * Qy_loc + beta * Qy_global + eta * Qy_well
        Qx, Qy = normalize(Qx, Qy)

    if wells is not None and X is not None and Y is not None and len(wells) > 0:
        mask = well_influence > tau
        if np.any(mask):
            qx_t, qy_t, _ = _well_weight_and_tangent(X, Y, wells, Qx, Qy)
            blend = np.where(mask, np.clip(well_influence, 0.0, 1.0), 0.0)
            Qx, Qy = normalize((1.0 - blend) * Qx + blend * qx_t, (1.0 - blend) * Qy + blend * qy_t)

    return Qx, Qy


def _normalize_field01(field: np.ndarray) -> np.ndarray:
    lo, hi = field.min(), field.max()
    return np.full_like(field, 0.5) if hi - lo < 1e-8 else (field - lo) / (hi - lo)


def make_fiber_aux_fields(shape: Tuple[int, int], L_curve: float, L_conn: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    L_curve, L_conn = float(np.clip(L_curve, 0.0, 1.0)), float(np.clip(L_conn, 0.0, 1.0))
    raw_curve = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), 2.0 + 16.0 * L_curve))
    raw_conn = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), 2.0 + 16.0 * L_conn))

    curve_field = L_curve + (0.15 + 0.35 * L_curve) * (raw_curve - 0.5) * 2.0
    conn_field = L_conn + (0.15 + 0.35 * L_conn) * (raw_conn - 0.5) * 2.0
    return np.clip(curve_field, 0.0, 1.0), np.clip(conn_field, 0.0, 1.0)


def make_wave_freq_field(shape: Tuple[int, int], L_wave_freq: float, rng: np.random.Generator) -> np.ndarray:
    L_wave_freq = float(np.clip(L_wave_freq, 0.0, 1.0))
    raw = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), 2.0 + 16.0 * L_wave_freq))
    field = L_wave_freq + (0.15 + 0.35 * L_wave_freq) * (raw - 0.5) * 2.0
    return np.clip(field, 0.0, 1.0)


def sample_field_at_seeds(seeds: np.ndarray, field: np.ndarray) -> np.ndarray:
    seeds = np.asarray(seeds, dtype=np.float64)
    h, w = field.shape
    out = np.zeros(seeds.shape[0], dtype=np.float64)

    for i, (row, col) in enumerate(seeds):
        if row < 0 or col < 0 or row >= h - 1 or col >= w - 1:
            out[i] = float(field[int(np.clip(round(row), 0, h - 1)), int(np.clip(round(col), 0, w - 1))])
        else:
            r0, c0 = int(row), int(col)
            dr, dc = row - r0, col - c0
            out[i] = float(
                (1.0 - dr) * (1.0 - dc) * field[r0, c0]
                + (1.0 - dr) * dc * field[r0, c0 + 1]
                + dr * (1.0 - dc) * field[r0 + 1, c0]
                + dr * dc * field[r0 + 1, c0 + 1]
            )
    return out


# =====================================================================
# 3. STREAMLINE TRACING & RASTERIZATION PIPELINE
# =====================================================================

def sample_vector(vx: np.ndarray, vy: np.ndarray, pos: np.ndarray, ref_dir: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    h, w = vx.shape
    y, x = pos
    if x < 0 or x >= w - 1 or y < 0 or y >= h - 1:
        return None

    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0

    Qx = (1 - dx) * (1 - dy) * vx[y0, x0] + dx * (1 - dy) * vx[y0, x0 + 1] + (1 - dx) * dy * vx[y0 + 1, x0] + dx * dy * vx[y0 + 1, x0 + 1]
    Qy = (1 - dx) * (1 - dy) * vy[y0, x0] + dx * (1 - dy) * vy[y0, x0 + 1] + (1 - dx) * dy * vy[y0 + 1, x0] + dx * dy * vy[y0 + 1, x0 + 1]

    if Qx == 0.0 and Qy == 0.0:
        return None

    theta = 0.5 * np.arctan2(Qy, Qx)
    d = np.array([np.sin(theta), np.cos(theta)])

    if ref_dir is not None and (d[0] * ref_dir[0] + d[1] * ref_dir[1]) < 0.0:
        d = -d
    return d


def integrate_streamline(vx: np.ndarray, vy: np.ndarray, seed: np.ndarray, step_size: float = 1.0,
                         max_steps: Optional[int] = None, spline_length: Optional[int] = None,
                         direction: int = 1, L_curve: float = 0.5, susceptibility: float = 1.0,
                         rng: Optional[np.random.Generator] = None) -> np.ndarray:
    pts = []
    pos = np.array(seed, dtype=float)
    rng = np.random.default_rng() if rng is None else rng
    max_angle = 0.35 * L_curve
    n_steps = max(1, int(spline_length if spline_length is not None else (max_steps if max_steps is not None else 300)))
    susceptibility = float(np.clip(susceptibility, 0.0, 1.0))

    ref_dir, preferred_dir = None, None

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

        pos += step_size * direction * v2
        pts.append(pos.copy())
        ref_dir = v2

    return np.array(pts)


def generate_fiber(vx: np.ndarray, vy: np.ndarray, seed: np.ndarray, step_size: float = 1.0,
                   max_steps: Optional[int] = None, spline_length: Optional[int] = None,
                   L_curve: float = 0.5, susceptibility: float = 1.0, rng: Optional[np.random.Generator] = None) -> np.ndarray:
    forward = integrate_streamline(vx, vy, seed, step_size, max_steps, spline_length, 1, L_curve, susceptibility, rng)
    backward = integrate_streamline(vx, vy, seed, step_size, max_steps, spline_length, -1, L_curve, susceptibility, rng)

    pts = []
    if backward is not None and len(backward) > 0:
        pts.append(backward[::-1])
    pts.append(np.asarray(seed, dtype=float)[None, :])
    if forward is not None and len(forward) > 0:
        pts.append(forward)
    return np.vstack(pts)

# adds sinusoidal offset to the fiber points
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

def fit_spline(points: np.ndarray, smoothing: float = 2.0, num_samples: int = 400, k: int = 3) -> np.ndarray:
    if len(points) < k + 2:
        return points
    try:
        spl, _ = make_splprep([points[:, 1], points[:, 0]], s=smoothing, k=k)
        x_s, y_s = spl(np.linspace(0, 1, num_samples))
        return np.vstack([y_s, x_s]).T
    except Exception:
        return points


def sample_seeds_from_density(D: np.ndarray, spline_num: int, L_density: float, rng: np.random.Generator) -> np.ndarray:
    D = np.asarray(D, dtype=np.float64)
    H, W = D.shape
    n = int(max(0, spline_num))
    if n == 0:
        return np.zeros((0, 2), dtype=np.int64)

    D_clipped = np.clip(D, 0.0, None)
    if np.all(D_clipped == 0):
        probs = np.full(D_clipped.size, 1.0 / D_clipped.size, dtype=np.float64)
    else:
        probs = np.power(D_clipped.ravel(), 0.8 + 2.2 * float(L_density))
        probs /= (probs.sum() + 1e-12)

    expected = n * probs
    counts = np.floor(expected).astype(np.int64)
    missing = int(n - counts.sum())

    if missing > 0:
        frac = expected - counts
        frac_sum = frac.sum()
        pick_probs = frac / frac_sum if frac_sum > 0 else None
        pick = rng.choice(counts.size, size=missing, replace=False, p=pick_probs)
        counts[pick] += 1

    idx = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    if idx.size == 0:
        return np.zeros((0, 2), dtype=np.int64)

    rng.shuffle(idx)
    return np.column_stack([idx // W, idx % W]).astype(np.int64)


def _prep_aux(values: Optional[np.ndarray], n_sp: int, default: float) -> np.ndarray:
    if values is None:
        return np.full(n_sp, float(default), dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).ravel()
    if arr.size < n_sp:
        arr = np.pad(arr, (0, n_sp - arr.size), constant_values=default)
    return arr[:n_sp]


def rasterize_splines(
    H: int, W: int, splines: List[np.ndarray], thickness: float = 3.0, oversample: float = 4.0,
    out_H: Optional[int] = None, out_W: Optional[int] = None, intensity_seed: int = 0,

    # wave_amplitude_px: float = 2.8, wave_wavelength_px: Optional[float] = None,
    # aux_wave_amp: Optional[np.ndarray] = None, aux_wave_freq: Optional[np.ndarray] = None,


    L_conn: float = 0.3, aux_L_conn: Optional[np.ndarray] = None,
    opacity_table: Optional[FiberOpacityTable] = None, opacity_cfg: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """Rasterize splines into a 2D float32 image with brightness and wobble models."""
    out_H = out_H or H
    out_W = out_W or W
    cfg = {**OPACITY_DEFAULTS, **(opacity_cfg or {})}

    scale_y, scale_x = out_H / max(H, 1), out_W / max(W, 1)
    img = np.zeros((out_H, out_W), dtype=np.float32)

    rng = np.random.default_rng(intensity_seed)
    n_sp = len(splines)

    # aux_wave_amp = _prep_aux(aux_wave_amp, n_sp, 1.0)
    # aux_wave_freq = _prep_aux(aux_wave_freq, n_sp, 0.5)
    aux_L_conn = _prep_aux(aux_L_conn, n_sp, L_conn)

    if opacity_table is None:
        opacity_table = build_fiber_opacity_table(n_sp, rng, cfg)

    stamp_damp = 1.0 / max(cfg["overlap_damp"], 1.0)
    # base_wavelength = wave_wavelength_px if wave_wavelength_px is not None else max(4.0 * thickness, 6.0)

    for i, spline in enumerate(splines):
        pts = np.asarray(spline)
        if pts.ndim != 2 or pts.shape[1] != 2 or pts.size == 0:
            continue

        # amp_i = float(np.clip(aux_wave_amp[i], 0.0, 1.0))
        # freq_i = float(np.clip(aux_wave_freq[i], 0.0, 1.0))
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

        # wavelength = base_wavelength * (0.5 + freq_i)
        # n_cycles = total_len / wavelength
        # wave_amp = wave_amplitude_px * amp_i

        for seg_idx, ((y0, x0), (y1, x1)) in enumerate(zip(pts_out[:-1], pts_out[1:])):
            dy, dx = y1 - y0, x1 - x0
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
                y, x = y0 + t * dy, x0 + t * dx

                # if wave_amp > 0.0:
                #     wobble = wave_amp * np.sin(2.0 * np.pi * n_cycles * s_frac)
                #     y += wobble * ny_n
                #     x += wobble * nx_n

                op = stamp_opacity(opacity_table, i, s_frac, cn, cfg)
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


# =====================================================================
# 4. SERVICE PIPELINE ENTRY POINT
# =====================================================================

def generate_shg_image(
    image_size: int = 512,
    num_fibers: int = 300,
    spline_length: int = 150,
    G_align: float = 0.5,
    G_curve: float = 0.5,
    G_density: float = 0.5,
    G_conn: float = 0.5,
    L_align: float = 0.5,
    L_curve: float = 0.5,
    L_density: float = 0.5,
    L_conn: float = 0.5,
    L_wave_freq: float = 0.5,
    seed: int = 42
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Execution wrapper to run the full synthesis and rasterization process."""
    rng = np.random.default_rng(seed)
    
    # 1. Coordinate maps and vector field initialization
    X, Y = create_grid(image_size, resolution_factor=1.0)
    shape = X.shape

    wells, W, hard_zero_mask = make_wells(X, Y, G_conn, rng)
    D = make_density(W, hard_zero_mask, G_density, L_density, rng)

    Qx_g, Qy_g = make_global_orientation(shape, G_align, G_curve, rng)
    Qx_w, Qy_w, influence = _well_weight_and_tangent(X, Y, wells, Qx_g, Qy_g)

    # 2. Relax local vector fields
    Qx, Qy = relax(Qx_g, Qy_g, Qx_g, Qy_g, Qx_w, Qy_w, influence, D, G_align, L_align, wells, X, Y)

    # 3. Auxiliary spatial fields
    curve_field, conn_field = make_fiber_aux_fields(shape, L_curve, L_conn, rng)
    wave_freq_field = make_wave_freq_field(shape, L_wave_freq, rng)

    # 4. Seed generation and trace integration
    seeds = sample_seeds_from_density(D, num_fibers, L_density, rng)
    aux_curve = sample_field_at_seeds(seeds, curve_field)
    aux_conn = sample_field_at_seeds(seeds, conn_field)
    aux_wave_freq = sample_field_at_seeds(seeds, wave_freq_field)

    splines = []
    for i, seed_pt in enumerate(seeds):
        raw_fiber = generate_fiber(
            Qx, Qy, seed_pt,
            step_size=1.0,
            spline_length=spline_length,
            L_curve=L_curve,
            susceptibility=1.0 - aux_curve[i],
            rng=rng
        )
        fitted = fit_spline(raw_fiber, num_samples=max(50, spline_length * 2))
        splines.append(fitted)

    # 5. Rasterize image
    raster = rasterize_splines(
        H=shape[0], W=shape[1],
        splines=splines,
        thickness=2.5,
        # aux_wave_amp=aux_curve,
        # aux_wave_freq=aux_wave_freq,
        aux_L_conn=aux_conn,
        intensity_seed=seed
    )

    return raster, splines

def generate_custom_fields_from_canvas(
    density_configs: list, 
    well_configs: list,
    shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates continuous density maps and well array from interactive GUI inputs.
    
    Parameters:
        density_configs: List of dicts for density points [{"x", "y", "w", "h", "intensity"}, ...]
        well_configs: List of dicts for wells [{"x", "y", "w", "h", "angle"}, ...]
        shape: Tuple (H, W) canvas dimensions.
        
    Returns:
        D: Density Map
        wells: Formatted well array for alignment [cx, cy, ax, ay, phi]
        hard_zero_mask: Mask of well interiors
    """
    H, W = shape
    D = np.zeros((H, W), dtype=np.float64)
    hard_zero_mask = np.zeros((H, W), dtype=bool)
    Y_grid, X_grid = np.ogrid[:H, :W]

    # 1. Process Density Points
    for cfg in density_configs:
        pt_x, pt_y = cfg["x"], cfg["y"]
        sigma_x = max(1.0, float(cfg["w"]))
        sigma_y = max(1.0, float(cfg["h"]))
        intensity = float(cfg["intensity"])

        gauss = intensity * np.exp(
            -(((X_grid - pt_x) ** 2) / (2.0 * sigma_x ** 2) + ((Y_grid - pt_y) ** 2) / (2.0 * sigma_y ** 2))
        )
        D += gauss

    # 2. Process Wells
    wells_list = []
    for cfg in well_configs:
        wx, wy = cfg["x"], cfg["y"]
        ax_px = max(1.0, float(cfg["w"]))
        ay_px = max(1.0, float(cfg["h"]))
        phi = np.radians(float(cfg.get("angle", 0.0)))

        # Center and radii normalized to [0, 1]
        cx, cy = wx / W, wy / H
        ax, ay = ax_px / W, ay_px / H
        wells_list.append([cx, cy, ax, ay, phi])

        # Compute hard interior zero-mask for fiber exclusion
        dx = X_grid - wx
        dy = Y_grid - wy
        ca, sa = np.cos(-phi), np.sin(-phi)
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        
        r2 = (xr / (ax_px + 1e-8))**2 + (yr / (ay_px + 1e-8))**2
        hard_zero_mask |= (r2 <= 1.0)

    # Zero out density inside structural wells
    D[hard_zero_mask] = 0.0
    wells = np.array(wells_list, dtype=np.float64) if wells_list else np.empty((0, 5))

    return D, wells, hard_zero_mask

def plot_density_vector_and_splines(
    ax, 
    D: np.ndarray, 
    Qx: np.ndarray, 
    Qy: np.ndarray, 
    splines: list, 
    image_size: int,
    subsample_ratio: int = 15
):
    """
    Plots the continuous density field, vector quiver arrows, and traced fiber splines.
    """
    shape = D.shape
    
    # 1. Density Map
    im = ax.imshow(D, cmap="viridis", origin="upper", extent=[0, image_size, image_size, 0])
    
    # 2. Quiver Plot for Vector Field
    step = max(1, image_size // 24)
    Y_sub, X_sub = np.mgrid[0:shape[0]:step, 0:shape[1]:step]
    angles = 0.5 * np.arctan2(Qy[::step, ::step], Qx[::step, ::step])
    u, v = np.cos(angles), np.sin(angles)
    
    ax.quiver(
        X_sub, Y_sub, u, v, 
        color="white", headlength=0, headaxislength=0, 
        pivot="middle", scale=25, alpha=0.6
    )

    # 3. Overlaid Fiber Splines (Subsampled for clarity)
    # sub_splines = splines[::max(1, len(splines) // subsample_ratio)]
    sub_splines = splines
    for sp in sub_splines:
        if len(sp) > 0:
            # Spline coordinates are (y, x) -> plot as (x, y)
            ax.plot(sp[:, 1], sp[:, 0], color="white", linewidth=1.2, alpha=0.8)

    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.axis("off")
    return im


# def plot_splines_3d(splines, image_size, depth_scale=0.15):
#     """
#     Renders 2D splines as 3D paths by assigning a pseudo Z-depth curve,
#     allowing interactive 3D rotation in Streamlit.
#     """
#     fig = go.Figure()

#     for idx, sp in enumerate(splines):
#         if len(sp) == 0:
#             continue
        
#         y_coords = sp[:, 0]
#         x_coords = sp[:, 1]
        
#         # Synthesize Z-axis depth using path progression
#         s_frac = np.linspace(0, 1, len(sp))
#         z_coords = np.sin(np.pi * s_frac) * (image_size * depth_scale) + (idx % 10)

#         fig.add_trace(go.Scatter3d(
#             x=x_coords,
#             y=y_coords,
#             z=z_coords,
#             mode='lines',
#             line=dict(width=3, color='cyan'),
#             hoverinfo='none',
#             showlegend=False
#         ))

#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(range=[0, image_size], title="X (px)"),
#             yaxis=dict(range=[image_size, 0], title="Y (px)"),  # Inverted for image orientation
#             zaxis=dict(title="Z Depth"),
#             aspectmode='cube',
#             bgcolor='black'
#         ),
#         margin=dict(l=0, r=0, b=0, t=0),
#         paper_bgcolor='black'
#     )
#     return fig