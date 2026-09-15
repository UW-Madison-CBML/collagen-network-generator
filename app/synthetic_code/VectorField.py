import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import make_splprep

WELL_HARD_SCALE = 1
WELL_OUTER_DECAY = 0.75
WELL_INFLUENCE_TAU = 0.15

def normalize(Qx, Qy, eps=1e-8):
    mag = np.sqrt(Qx * Qx + Qy * Qy) + eps
    return Qx / mag, Qy / mag


def axial_to_theta(Qx, Qy):
    return 0.5 * np.arctan2(Qy, Qx)


def create_grid(image_size, resolution_factor):
    M = int(image_size * resolution_factor)
    x = np.linspace(0, 1, M)
    y = np.linspace(0, 1, M)
    return np.meshgrid(x, y)

# Can remove this function
def _tangent_q_from_gradient(gx, gy, Qx_g, Qy_g):
    # Return axial Q for fiber tangent perpendicular to the well potential gradient.
    grad_len = np.sqrt(gx * gx + gy * gy) + 1e-8
    rx = gx / grad_len
    ry = gy / grad_len

    tx1, ty1 = -ry, rx
    tx2, ty2 = ry, -rx

    inward_x, inward_y = -rx, -ry

    def inward_score(tx, ty):
        phi = np.arctan2(ty, tx)
        return np.cos(phi) * inward_x + np.sin(phi) * inward_y

    score1 = inward_score(tx1, ty1)
    score2 = inward_score(tx2, ty2)
    pick1 = (score1 < score2) | ((np.abs(score1 - score2) < 1e-6) & (Qx_g * tx1 + Qy_g * ty1 >= Qx_g * tx2 + Qy_g * ty2))
    tx = np.where(pick1, tx1, tx2)
    ty = np.where(pick1, ty1, ty2)

    theta = np.arctan2(ty, tx)
    return np.cos(2.0 * theta), np.sin(2.0 * theta)


def make_wells(X, Y, G_conn, rng):
    max_wells = 20
    K = int(max_wells * (1 - G_conn))

    base_sigma = 0.08
    if K > 0:
        sx = base_sigma * rng.uniform(0.5, 1.6, size=K)
        sy = base_sigma * rng.uniform(0.5, 1.6, size=K)
        phi = rng.uniform(0, 2 * np.pi, size=K)
    else:
        sx = np.empty((0,))
        sy = np.empty((0,))
        phi = np.empty((0,))

    if K > 0:
        max_hard_axis = WELL_HARD_SCALE * np.max(np.maximum(sx, sy))
        min_dist = 2.0 * max_hard_axis
        centers = []
        target = K
        attempts = 0
        max_attempts = 8000
        while len(centers) < target and attempts < max_attempts:
            cand = rng.random(2)
            if not centers:
                centers.append(cand)
            else:
                dist2 = np.sum((np.asarray(centers) - cand) ** 2, axis=1)
                if np.all(dist2 >= min_dist * min_dist):
                    centers.append(cand)
            attempts += 1
            if attempts % 1200 == 0 and len(centers) < target:
                min_dist *= 0.92
        if len(centers) < target:
            while len(centers) < target:
                centers.append(rng.random(2))
        centers = np.asarray(centers, dtype=float)
    else:
        centers = np.empty((0, 2))

    wells = np.column_stack([centers, sx, sy, phi]) if K > 0 else np.empty((0, 5))
    W = np.zeros_like(X)
    hard_zero_mask = np.zeros_like(X, dtype=bool)

    for wx, wy, ax, ay, ang in wells:
        dx = X - wx
        dy = Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        r2_hard = (xr / (WELL_HARD_SCALE * ax + 1e-8)) ** 2 + (yr / (WELL_HARD_SCALE * ay + 1e-8)) ** 2
        inside_hard = r2_hard <= 1.0
        hard_zero_mask |= inside_hard

        r = np.sqrt(np.maximum(r2_hard, 1e-12))
        outside = np.maximum(r - 1.0, 0.0)
        W += np.exp(-outside / WELL_OUTER_DECAY)

    return wells, W, hard_zero_mask


def make_density(W, hard_zero_mask, G_density, L_density, rng):
    G_density = float(np.clip(G_density, 0.0, 1.0))
    L_density = float(np.clip(L_density, 0.0, 1.0))

    beta_wells = 4.0
    mean_D = np.exp(-beta_wells * W)

    if L_density > 0.0:
        noise_sigma = 3.0 + 18.0 * L_density
        z = gaussian_filter(rng.normal(size=W.shape), noise_sigma)
        z = (z - z.mean()) / (z.std() + 1e-8)
        z_std = 0.25 + 0.55 * L_density
        multiplier = np.exp(L_density * z_std * z)
    else:
        multiplier = np.ones_like(W)

    D = G_density * mean_D * multiplier
    D = np.where(hard_zero_mask, 0.0, D)

    valid = ~hard_zero_mask
    # Mean normalize (redundant now)
    if np.any(valid):
        mean_valid = D[valid].mean()
        if mean_valid > 1e-8:
            D = np.where(valid, D / mean_valid * G_density, 0.0)

    D = np.clip(D, 0.0, None) # clipping
    # Mean normalize (redundant now)
    if np.any(valid):
        mean_valid = D[valid].mean()
        if mean_valid > 1e-8:
            D = np.where(valid, D / mean_valid * G_density, 0.0)
    return np.clip(D, 0.0, None)


def make_global_orientation(shape, G_align, G_curve, rng):
    theta0 = rng.uniform(0, np.pi)
    ny, nx = shape
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    g = float(np.clip(G_curve, 0.0, 1.0))

    psi = rng.uniform(0, 2 * np.pi)
    u = np.array([np.cos(psi), np.sin(psi)])
    d_close = 1.28
    d_far = 34.0
    r_center = g * d_close + (1.0 - g) * d_far
    cx = 0.5 + r_center * u[0]
    cy = 0.5 + r_center * u[1]

    rx = X - cx
    ry = Y - cy
    theta_circle = np.arctan2(ry, rx) + (np.pi / 2)

    theta_base = (1.0 - g) * theta0 + g * theta_circle

    phi = gaussian_filter(rng.normal(size=shape), sigma=40 * G_align + 5)
    phi = (phi - phi.mean()) / (phi.std() + 1e-8)
    theta = theta_base + 0.6 * (1 - G_align) * phi

    Qx = np.cos(2 * theta)
    Qy = np.sin(2 * theta)
    return Qx, Qy


def _well_weight_and_tangent(X, Y, wells, Qx_ref, Qy_ref):
    # Per-pixel nearest-well influence and tangent axial Q.
    influence = np.zeros_like(X)
    Qx = np.zeros_like(X)
    Qy = np.zeros_like(Y)

    if len(wells) == 0:
        return Qx, Qy, influence

    best_weight = np.full(X.shape, -1.0, dtype=np.float64)

    for wx, wy, ax, ay, ang in wells:
        dx = X - wx
        dy = Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr = ca * dx + sa * dy
        yr = -sa * dx + ca * dy

        r2_hard = (xr / (WELL_HARD_SCALE * ax + 1e-8)) ** 2 + (yr / (WELL_HARD_SCALE * ay + 1e-8)) ** 2
        r = np.sqrt(np.maximum(r2_hard, 1e-12))
        outside = np.maximum(r - 1.0, 0.0)
        weight = np.exp(-outside / WELL_OUTER_DECAY)

        gx_l = 2.0 * xr / (ax ** 2 + 1e-8)
        gy_l = 2.0 * yr / (ay ** 2 + 1e-8)
        gx = ca * gx_l - sa * gy_l
        gy = sa * gx_l + ca * gy_l
        qx_t, qy_t = _tangent_q_from_gradient(gx, gy, Qx_ref, Qy_ref)
        replace = weight > best_weight
        best_weight = np.where(replace, weight, best_weight)
        Qx = np.where(replace, qx_t, Qx)
        Qy = np.where(replace, qy_t, Qy)
        influence = np.where(replace, weight, influence)

    influence /= influence.max() + 1e-8
    return Qx, Qy, influence


def make_well_orientation(X, Y, Qx_g, Qy_g, wells):
    if len(wells) == 0:
        return Qx_g.copy(), Qy_g.copy(), np.zeros_like(X)

    Qx, Qy, influence = _well_weight_and_tangent(X, Y, wells, Qx_g, Qy_g)
    Qx, Qy = normalize(Qx, Qy)
    return Qx, Qy, influence


def enforce_well_perpendicular(Qx, Qy, X, Y, wells, well_influence, tau=WELL_INFLUENCE_TAU):
    if len(wells) == 0:
        return Qx, Qy

    mask = well_influence > tau
    if not np.any(mask):
        return Qx, Qy

    qx_t, qy_t, _ = _well_weight_and_tangent(X, Y, wells, Qx, Qy)
    blend = np.where(mask, np.clip(well_influence, 0.0, 1.0), 0.0)
    Qx_out = (1.0 - blend) * Qx + blend * qx_t
    Qy_out = (1.0 - blend) * Qy + blend * qy_t
    return normalize(Qx_out, Qy_out)


def relax(Qx, Qy, Qx_global, Qy_global, Qx_well, Qy_well, well_influence, D, G_align, L_align,
          wells=None, X=None, Y=None):
    alpha0 = 0.2 + 0.8 * L_align
    local_sigma = 1.0 + 4.0 * L_align
    tau = WELL_INFLUENCE_TAU

    near_well = well_influence > tau

    for _ in range(25):
        alpha = alpha0 * D
        beta = G_align * (1.0 - well_influence)
        eta = well_influence.copy()

        beta = np.where(near_well, 0.0, beta)
        eta = np.where(near_well, 1.0, eta)
        alpha = np.where(near_well, alpha * 0.25, alpha)

        total = alpha + beta + eta
        overflow = total > 1.0
        scale = np.where(overflow, 1.0 / (total + 1e-8), 1.0)
        alpha = alpha * scale
        beta = beta * scale
        eta = eta * scale

        Qx_loc = gaussian_filter(Qx, local_sigma)
        Qy_loc = gaussian_filter(Qy, local_sigma)

        Qx = (1.0 - alpha - beta - eta) * Qx + alpha * Qx_loc + beta * Qx_global + eta * Qx_well
        Qy = (1.0 - alpha - beta - eta) * Qy + alpha * Qy_loc + beta * Qy_global + eta * Qy_well
        Qx, Qy = normalize(Qx, Qy)

    if wells is not None and X is not None and Y is not None and len(wells) > 0:
        Qx, Qy = enforce_well_perpendicular(Qx, Qy, X, Y, wells, well_influence, tau=tau)

    return Qx, Qy


def _normalize_field01(field):
    lo, hi = field.min(), field.max()
    if hi - lo < 1e-8:
        return np.full_like(field, 0.5)
    return (field - lo) / (hi - lo)


def make_fiber_aux_fields(shape, L_curve, L_conn, rng):
    L_curve = float(np.clip(L_curve, 0.0, 1.0))
    L_conn = float(np.clip(L_conn, 0.0, 1.0))

    sigma_curve = 2.0 + 16.0 * L_curve
    sigma_conn = 2.0 + 16.0 * L_conn

    raw_curve = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), sigma_curve))
    raw_conn = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), sigma_conn))

    spread_curve = 0.15 + 0.35 * L_curve
    spread_conn = 0.15 + 0.35 * L_conn

    curve_field = L_curve + spread_curve * (raw_curve - 0.5) * 2.0
    conn_field = L_conn + spread_conn * (raw_conn - 0.5) * 2.0

    return np.clip(curve_field, 0.0, 1.0), np.clip(conn_field, 0.0, 1.0)


def make_wave_freq_field(shape, L_wave_freq, rng):
    # Spatial field controlling per-fiber wobble wavelength.

    # Mirrors make_fiber_aux_fields: smooth, clustered noise centered on
    # L_wave_freq with spread scaling with L_wave_freq. Sample this at seed
    # locations and pass the result to rasterize_splines(aux_wave_freq=...).

    L_wave_freq = float(np.clip(L_wave_freq, 0.0, 1.0))
    sigma = 2.0 + 16.0 * L_wave_freq
    raw = _normalize_field01(gaussian_filter(rng.uniform(0.0, 1.0, shape), sigma))
    spread = 0.15 + 0.35 * L_wave_freq
    field = L_wave_freq + spread * (raw - 0.5) * 2.0
    return np.clip(field, 0.0, 1.0)


def _bilinear_sample(field, row, col):
    h, w = field.shape
    if row < 0 or col < 0 or row >= h - 1 or col >= w - 1:
        r = int(np.clip(round(row), 0, h - 1))
        c = int(np.clip(round(col), 0, w - 1))
        return float(field[r, c])

    r0, c0 = int(row), int(col)
    dr, dc = row - r0, col - c0
    return float(
        (1.0 - dr) * (1.0 - dc) * field[r0, c0]
        + (1.0 - dr) * dc * field[r0, c0 + 1]
        + dr * (1.0 - dc) * field[r0 + 1, c0]
        + dr * dc * field[r0 + 1, c0 + 1]
    )


def sample_field_at_seeds(seeds, field):
    # Generic bilinear sampling of any scalar field at seed (row, col) locations.
    seeds = np.asarray(seeds, dtype=np.float64)
    n = seeds.shape[0]
    out = np.zeros(n, dtype=np.float64)
    for i, (row, col) in enumerate(seeds):
        out[i] = _bilinear_sample(field, float(row), float(col))
    return out


def sample_aux_at_seeds(seeds, curve_field, conn_field):
    aux_curve = sample_field_at_seeds(seeds, curve_field)
    aux_conn = sample_field_at_seeds(seeds, conn_field)
    return aux_curve, aux_conn
