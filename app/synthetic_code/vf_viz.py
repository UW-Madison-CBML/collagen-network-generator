import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from . import VectorField as vecfield

# Pulled from VectorField.py directly rather than hardcoded, so these plots
# never go stale relative to whatever you've tuned WELL_OUTER_DECAY /
# WELL_INFLUENCE_TAU to (both used to be hardcoded copies here -- e.g. the
# 0.75 decay length and 0.15 tau threshold -- which silently drifted out of
# sync once those constants got tuned in VectorField.py).
WELL_HARD_SCALE = vecfield.WELL_HARD_SCALE
WELL_OUTER_DECAY = vecfield.WELL_OUTER_DECAY
WELL_INFLUENCE_TAU = vecfield.WELL_INFLUENCE_TAU


def _quiver_stride(shape, n_arrows=28):
    # Return a stride so that roughly n_arrows x n_arrows arrows are drawn.
    return max(1, shape[0] // n_arrows)


def _axial_to_headless_uv(Qx, Qy, stride):
    # Subsample an axial (director) field and convert to +-(u,v) pairs so that
    # quiver draws headless line segments (both directions shown).
    # Returns xs, ys, us, vs all 1-D.

    sl = (slice(None, None, stride), slice(None, None, stride))
    qx = Qx[sl]
    qy = Qy[sl]
    theta = 0.5 * np.arctan2(qy, qx)
    u = np.cos(theta).ravel()
    v = np.sin(theta).ravel()
    ny, nx = Qx[sl].shape
    xs = np.linspace(0, 1, nx)
    ys = np.linspace(0, 1, ny)
    gx, gy = np.meshgrid(xs, ys)
    return gx.ravel(), gy.ravel(), u, v


def _well_ellipse_patch(wx, wy, ax, ay, ang, scale=WELL_HARD_SCALE, **kwargs):
    # Return a matplotlib Ellipse patch for one well's hard boundary
    width  = 2 * scale * ax
    height = 2 * scale * ay
    angle_deg = np.degrees(ang)
    return mpatches.Ellipse(
        (wx, wy), width, height,
        angle=angle_deg,
        **kwargs
    )


def _save_or_show(fig, save_path):
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# Well potential field

def plot_well_potential(W, X, Y, wells, hard_mask,
                        show_boundaries=True, save_path=None):
    # Heatmap of the aggregate well potential W, with optional hard-boundary
    # ellipses and well-centre markers overlaid.

    # Parameters
    # ----------
    # W            : (M, M) array  — output of make_wells
    # X, Y         : (M, M) arrays — coordinate grids
    # wells        : (K, 5) array  — [cx, cy, sx, sy, phi]
    # hard_mask    : (M, M) bool   — True inside any hard ellipse
    # show_boundaries : overlay hard ellipse outlines

    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    im = ax.imshow(W, origin="lower", extent=extent,
                   cmap="magma", interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Well potential W")

    if show_boundaries and len(wells) > 0:
        for wx, wy, ax_, ay, ang in wells:
            patch = _well_ellipse_patch(
                wx, wy, ax_, ay, ang,
                linewidth=1.2, edgecolor="cyan", facecolor="none"
            )
            ax.add_patch(patch)
        ax.scatter(wells[:, 0], wells[:, 1],
                   s=18, c="cyan", zorder=5, label="Well centres")
        ax.legend(fontsize=8, loc="upper right")

    ax.set_title("Well potential field W")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Single well

def plot_single_well(wells, index, X, Y, save_path=None):

    # Isolate and visualise one well: its own potential slice, hard boundary,
    # and the exponential decay profile along the major axis.

    # Parameters
    # ----------
    # wells : (K, 5) array
    # index : int — which well to highlight (0-based)
    # X, Y  : coordinate grids

    if len(wells) == 0:
        raise ValueError("No wells to plot.")
    if index >= len(wells):
        raise IndexError(f"index {index} out of range for {len(wells)} wells.")

    wx, wy, ax_, ay, ang = wells[index]

    # recompute this well's potential in isolation
    dx = X - wx
    dy = Y - wy
    ca, sa = np.cos(ang), np.sin(ang)
    xr =  ca * dx + sa * dy
    yr = -sa * dx + ca * dy
    r2 = (xr / (WELL_HARD_SCALE * ax_ + 1e-8))**2 + \
         (yr / (WELL_HARD_SCALE * ay  + 1e-8))**2
    r  = np.sqrt(np.maximum(r2, 1e-12))
    outside = np.maximum(r - 1.0, 0.0)
    W_single = np.exp(-outside / WELL_OUTER_DECAY)
    hard_single = r2 <= 1.0

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # left: 2-D heatmap
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    im = axes[0].imshow(W_single, origin="lower", extent=extent,
                        cmap="inferno", interpolation="bilinear", vmin=0, vmax=1)
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04, label="Potential")
    patch = _well_ellipse_patch(wx, wy, ax_, ay, ang,
                                linewidth=1.5, edgecolor="cyan", facecolor="none")
    axes[0].add_patch(patch)
    axes[0].scatter([wx], [wy], s=30, c="cyan", zorder=5)
    axes[0].set_title(f"Well {index}  (cx={wx:.3f}, cy={wy:.3f})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    # right: 1-D radial decay profile along the major semi-axis
    # sample along the rotated major axis direction
    t = np.linspace(-3.5 * max(ax_, ay), 3.5 * max(ax_, ay), 400)
    # major axis direction in world coords
    major_dir = np.array([ca, sa])  # ang rotates x-axis → major
    sample_x = wx + t * major_dir[0]
    sample_y = wy + t * major_dir[1]
    # clamp to [0,1]
    in_bounds = (sample_x >= 0) & (sample_x <= 1) & \
                (sample_y >= 0) & (sample_y <= 1)
    xr_line =  ca * t + 0.0   # wy term cancels in relative coords
    r_line  = np.abs(xr_line) / (WELL_HARD_SCALE * ax_ + 1e-8)
    out_line = np.maximum(r_line - 1.0, 0.0)
    w_line  = np.exp(-out_line / WELL_OUTER_DECAY)

    r_signed = t / (WELL_HARD_SCALE * ax_ + 1e-8)
    axes[1].plot(r_signed[in_bounds], w_line[in_bounds],
                 color="tab:orange", lw=2, label="Potential W")
    axes[1].axvspan(-1, 1, alpha=0.15, color="cyan", label="Hard interior")
    axes[1].axhline(np.exp(-1), color="gray", ls="--", lw=1,
                    label=f"e⁻¹ ≈ {np.exp(-1):.2f}")
    axes[1].set_xlabel("Normalised distance from centre (major axis)")
    axes[1].set_ylabel("W")
    axes[1].set_title("Radial decay profile")
    axes[1].legend(fontsize=8)
    axes[1].set_ylim(-0.05, 1.05)

    fig.suptitle(f"Single-well inspection — well {index}", fontsize=12)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# All wells

def plot_all_wells(wells, X, Y, hard_mask, save_path=None):
    # One panel per well, each showing its isolated potential + hard boundary.
    # Arranges into a grid of subplots automatically.

    # Parameters
    # ----------
    # wells     : (K, 5) array
    # X, Y      : coordinate grids
    # hard_mask : aggregate hard mask (shown as light overlay in background)

    K = len(wells)
    if K == 0:
        print("No wells to plot.")
        return None

    ncols = min(K, 4)
    nrows = int(np.ceil(K / ncols))
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(3.5 * ncols, 3.2 * nrows),
                             squeeze=False)
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    for idx in range(K):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        wx, wy, ax_, ay, ang = wells[idx]

        dx = X - wx; dy = Y - wy
        ca, sa = np.cos(ang), np.sin(ang)
        xr =  ca * dx + sa * dy
        yr = -sa * dx + ca * dy
        r2 = (xr / (WELL_HARD_SCALE * ax_ + 1e-8))**2 + \
             (yr / (WELL_HARD_SCALE * ay  + 1e-8))**2
        outside = np.maximum(np.sqrt(np.maximum(r2, 1e-12)) - 1.0, 0.0)
        W_i = np.exp(-outside / WELL_OUTER_DECAY)

        ax.imshow(W_i, origin="lower", extent=extent,
                  cmap="inferno", interpolation="bilinear", vmin=0, vmax=1)
        patch = _well_ellipse_patch(wx, wy, ax_, ay, ang,
                                    linewidth=1.2, edgecolor="cyan",
                                    facecolor="none")
        ax.add_patch(patch)
        ax.scatter([wx], [wy], s=15, c="cyan", zorder=5)
        ax.set_title(f"Well {idx}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # hide unused axes
    for idx in range(K, nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].set_visible(False)

    fig.suptitle(f"All {K} wells (isolated potentials)", fontsize=12)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Hard boundaries
def plot_hard_boundaries(wells, X, Y, hard_mask, save_path=None):
    # Two-panel plot: (left) the boolean hard_mask as an image,
    # (right) each ellipse outline drawn analytically on a clean axes.

    # Parameters
    # ----------
    # wells     : (K, 5) array
    # X, Y      : coordinate grids
    # hard_mask : (M, M) bool array

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    # left: rasterised mask
    axes[0].imshow(hard_mask.astype(float), origin="lower", extent=extent,
                   cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
    axes[0].set_title("Hard-zero mask (rasterised)")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")

    # right: analytic ellipse outlines, coloured by well index
    axes[1].set_xlim(0, 1); axes[1].set_ylim(0, 1)
    axes[1].set_aspect("equal")
    axes[1].set_facecolor("#1a1a2e")
    cmap = plt.get_cmap("tab20")
    for idx, (wx, wy, ax_, ay, ang) in enumerate(wells):
        color = cmap(idx % 20)
        # hard boundary
        patch_hard = _well_ellipse_patch(
            wx, wy, ax_, ay, ang,
            linewidth=1.8, edgecolor=color, facecolor=color, alpha=0.35,
            label=f"Well {idx}"
        )
        # soft-influence boundary: WELL_INFLUENCE_TAU corresponds to
        # e^(-outside/WELL_OUTER_DECAY) = WELL_INFLUENCE_TAU
        # -> outside = -WELL_OUTER_DECAY * ln(WELL_INFLUENCE_TAU)
        tau_r = 1.0 + (-WELL_OUTER_DECAY * np.log(WELL_INFLUENCE_TAU))
        patch_soft = _well_ellipse_patch(
            wx, wy, ax_ * tau_r, ay * tau_r, ang,
            linewidth=1.0, edgecolor=color, facecolor="none",
            linestyle="--", alpha=0.6
        )
        axes[1].add_patch(patch_hard)
        axes[1].add_patch(patch_soft)
        axes[1].scatter([wx], [wy], s=20, color=color, zorder=5)
        axes[1].text(wx, wy, f" {idx}", fontsize=7,
                     color=color, va="center", zorder=6)

    axes[1].set_title("Analytic hard (solid) & influence (dashed) boundaries")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")

    # legend only if ≤ 12 wells (avoid clutter)
    if len(wells) <= 12:
        handles = [mpatches.Patch(color=cmap(i % 20), label=f"Well {i}")
                   for i in range(len(wells))]
        axes[1].legend(handles=handles, fontsize=7,
                       loc="upper right", ncol=2)

    fig.suptitle("Well hard boundaries", fontsize=12)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Density field

def plot_density(D, hard_mask=None, wells=None, save_path=None):
    # Heatmap of the fiber density field D.

    # Parameters
    # ----------
    # D         : (M, M) float array — output of make_density
    # hard_mask : optional (M, M) bool — overlays hard-zero regions
    # wells     : optional (K, 5) array — overlays well centres

    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(D, origin="lower", cmap="viridis",
                   interpolation="bilinear")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density D")

    if hard_mask is not None:
        overlay = np.zeros((*D.shape, 4))
        overlay[hard_mask] = [1, 0.2, 0.2, 0.5]   # red tint for hard zeros
        ax.imshow(overlay, origin="lower", interpolation="nearest")

    if wells is not None and len(wells) > 0:
        ax.scatter(wells[:, 0], wells[:, 1],
                   s=18, c="white", edgecolors="black",
                   linewidths=0.6, zorder=5, label="Well centres")
        ax.legend(fontsize=8)

    ax.set_title("Fiber density field D")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Orientation field

def plot_orientation_field(Qx, Qy, density_mask=None, wells=None,
                           n_arrows=28, title="Orientation field",
                           save_path=None):
    # HSV colormap where hue encodes fiber angle θ = ½ arctan2(Qy, Qx),
    # overlaid with a headless quiver plot.

    # Parameters
    # ----------
    # Qx, Qy       : (M, M) axial Q-tensor components
    # density_mask : optional (M, M) float — modulates alpha of quiver arrows
    # wells        : optional (K, 5) — overlay hard boundaries
    # n_arrows     : approximate number of arrows per axis

    theta = 0.5 * np.arctan2(Qy, Qx)          # in [-π/2, π/2]
    hue   = (theta / np.pi + 0.5) % 1.0        # map to [0, 1]

    # HSV image: hue = angle, sat = 1, val = 1
    hsv = np.ones((*theta.shape, 3))
    hsv[..., 0] = hue
    rgb = plt.matplotlib.colors.hsv_to_rgb(hsv)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb, origin="lower", interpolation="bilinear")

    # quiver (headless: draw +dir and -dir)
    M = Qx.shape[0]
    stride = _quiver_stride(Qx.shape, n_arrows)
    sl = (slice(None, None, stride), slice(None, None, stride))
    qx_s = Qx[sl]; qy_s = Qy[sl]
    theta_s = 0.5 * np.arctan2(qy_s, qx_s)
    u = np.cos(theta_s); v = np.sin(theta_s)
    ny, nx = u.shape
    xs = np.linspace(0, M - 1, nx)
    ys = np.linspace(0, M - 1, ny)
    gx, gy = np.meshgrid(xs, ys)

    qkw = dict(angles="xy", scale_units="xy", scale=0.045 * M / n_arrows,
               width=0.003, headwidth=0, headlength=0, headaxislength=0,
               color="white", alpha=0.65)
    ax.quiver(gx, gy,  u,  v, **qkw)
    ax.quiver(gx, gy, -u, -v, **qkw)

    if wells is not None and len(wells) > 0:
        for wx, wy, ax_, ay, ang in wells:
            # convert [0,1] coords to pixel coords
            px = wx * (M - 1); py = wy * (M - 1)
            ax_px = ax_ * (M - 1); ay_px = ay * (M - 1)
            patch = mpatches.Ellipse(
                (px, py), 2 * ax_px, 2 * ay_px,
                angle=np.degrees(ang),
                linewidth=1.2, edgecolor="black",
                facecolor="none", linestyle="--"
            )
            ax.add_patch(patch)

    # colorbar for angle
    sm = ScalarMappable(cmap="hsv", norm=Normalize(-90, 90))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Fiber angle θ (°)")
    cbar.set_ticks([-90, -45, 0, 45, 90])

    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Well influence field

def plot_well_influence(influence, wells, X, Y, tau=None, save_path=None):
    # Heatmap of the per-pixel well influence weight, with the tau threshold
    # contour and well boundaries overlaid.

    # Parameters
    # ----------
    # influence : (M, M) float — output of make_well_orientation (3rd return)
    # wells     : (K, 5) array
    # X, Y      : coordinate grids
    # tau       : influence threshold. Defaults to VectorField.WELL_INFLUENCE_TAU
    #             (the actual threshold used in relax()'s hard-snap cutoff).

    if tau is None:
        tau = WELL_INFLUENCE_TAU
    fig, ax = plt.subplots(figsize=(6, 6))
    extent = [X.min(), X.max(), Y.min(), Y.max()]

    im = ax.imshow(influence, origin="lower", extent=extent,
                   cmap="plasma", interpolation="bilinear", vmin=0, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Well influence")

    # tau contour
    ax.contour(X, Y, influence, levels=[tau],
               colors=["lime"], linewidths=1.2, linestyles="--")

    if len(wells) > 0:
        for wx, wy, ax_, ay, ang in wells:
            patch = _well_ellipse_patch(
                wx, wy, ax_, ay, ang,
                linewidth=1.2, edgecolor="white", facecolor="none"
            )
            ax.add_patch(patch)
        ax.scatter(wells[:, 0], wells[:, 1],
                   s=18, c="white", zorder=5, label="Well centres")

    # legend entries
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="lime", lw=1.2, ls="--",
               label=f"τ = {tau} contour"),
        Line2D([0], [0], color="white", lw=1.2,
               label="Hard boundary"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper right")
    ax.set_title("Well influence field")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Aux fields (curve / conn)

def plot_aux_fields(curve_field, conn_field, wave_freq_field=None, seeds=None, save_path=None):
    # Side-by-side heatmaps of the curve, conn, and (optionally) wave-frequency
    # auxiliary fields.

    # Parameters
    # ----------
    # curve_field     : (M, M) float — output of make_fiber_aux_fields
    # conn_field      : (M, M) float — output of make_fiber_aux_fields
    # wave_freq_field : optional (M, M) float — output of make_wave_freq_field.
    #                   This controls each fiber's sinusoidal wobble
    #                   wavelength in rasterize_splines (aux_wave_freq).
    # seeds           : optional (N, 2) array of (row, col) seed positions to overlay

    fields = [curve_field, conn_field]
    titles = ["Curve field (local curviness)", "Conn field (local connectivity)"]
    cmaps = ["YlOrRd", "YlGnBu"]
    if wave_freq_field is not None:
        fields.append(wave_freq_field)
        titles.append("Wave-freq field (wobble wavelength)")
        cmaps.append("PuBuGn")

    fig, axes = plt.subplots(1, len(fields), figsize=(5.5 * len(fields), 5), squeeze=False)
    axes = axes[0]

    for ax, field, title, cmap in zip(axes, fields, titles, cmaps):
        im = ax.imshow(field, origin="lower", cmap=cmap,
                       interpolation="bilinear", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if seeds is not None:
            # seeds are (row, col) — imshow origin="lower" maps row→y, col→x
            ax.scatter(seeds[:, 1], seeds[:, 0],
                       s=8, c="white", alpha=0.7, zorder=5)
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Fiber auxiliary fields", fontsize=12)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Per-fiber susceptibility (field-following vs. straight/rogue crossover fibers)

def plot_susceptibility(seeds, aux_susceptibility, D=None, save_path=None):
    # Visualize per-fiber susceptibility values: seed locations colored by how
    # strongly each fiber follows the local vector field (1 = follows it
    # faithfully, 0 = walks straight/independent of it, driving crossover).
    # Low L_align pushes this toward a bimodal split -- see the histogram
    # panel for whether that's actually showing up in a given run.

    # Parameters
    # ----------
    # seeds               : (N, 2) array of (row, col) seed positions
    # aux_susceptibility  : (N,) float in [0,1] — per-fiber susceptibility,
    #                       e.g. SyntheticGen's `res["aux_susceptibility"]`
    # D                   : optional (M, M) density field, shown as a faint backdrop

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    ax = axes[0]
    if D is not None:
        ax.imshow(D, cmap="magma", alpha=0.35, origin="lower")
    sc = ax.scatter(seeds[:, 1], seeds[:, 0], c=aux_susceptibility,
                    cmap="coolwarm_r", vmin=0, vmax=1, s=14, edgecolors="none")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="Susceptibility")
    ax.set_title("Per-fiber susceptibility\n(blue=follows field, red=straight/rogue)")
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.hist(aux_susceptibility, bins=40, range=(0, 1),
            color="steelblue", edgecolor="none")
    ax.set_xlabel("Susceptibility")
    ax.set_ylabel("Fiber count")
    frac_rogue = float(np.mean(aux_susceptibility < 0.2))
    frac_faithful = float(np.mean(aux_susceptibility > 0.8))
    ax.set_title(f"Distribution  (rogue <0.2: {frac_rogue:.0%}, faithful >0.8: {frac_faithful:.0%})")

    fig.suptitle("Fiber susceptibility / crossover diagnostics", fontsize=12)
    fig.tight_layout()
    return _save_or_show(fig, save_path)


# Full overview

def plot_overview(W, hard_mask, D, Qx, Qy, influence, wells, X, Y,
                  save_path=None):
    # Six-panel summary: potential, hard mask, density, orientation,
    # well influence, and per-angle histogram.

    # Parameters
    # ----------
    # W, hard_mask, D, Qx, Qy, influence : standard VectorField.py outputs
    # wells, X, Y                         : as usual
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    extent = [X.min(), X.max(), Y.min(), Y.max()]
    M = Qx.shape[0]

    # 1. Well potential
    ax = axes[0][0]
    ax.imshow(W, origin="lower", extent=extent, cmap="magma",
              interpolation="bilinear")
    for wx, wy, ax_, ay, ang in wells:
        ax.add_patch(_well_ellipse_patch(wx, wy, ax_, ay, ang,
                     linewidth=1, edgecolor="cyan", facecolor="none"))
    ax.set_title("Well potential W"); ax.set_xticks([]); ax.set_yticks([])

    # 2. Hard mask
    ax = axes[0][1]
    ax.imshow(hard_mask.astype(float), origin="lower", extent=extent,
              cmap="Greys", interpolation="nearest")
    for wx, wy, ax_, ay, ang in wells:
        ax.add_patch(_well_ellipse_patch(wx, wy, ax_, ay, ang,
                     linewidth=1.2, edgecolor="red", facecolor="none"))
    ax.set_title("Hard-zero mask"); ax.set_xticks([]); ax.set_yticks([])

    # 3. Density
    ax = axes[0][2]
    ax.imshow(D, origin="lower", extent=extent, cmap="viridis",
              interpolation="bilinear")
    ax.set_title("Fiber density D"); ax.set_xticks([]); ax.set_yticks([])

    # 4. Orientation field (HSV)
    ax = axes[1][0]
    theta = 0.5 * np.arctan2(Qy, Qx)
    hue   = (theta / np.pi + 0.5) % 1.0
    hsv   = np.ones((*theta.shape, 3))
    hsv[..., 0] = hue
    rgb   = plt.matplotlib.colors.hsv_to_rgb(hsv)
    ax.imshow(rgb, origin="lower", interpolation="bilinear")
    stride = _quiver_stride(Qx.shape, 20)
    sl = (slice(None, None, stride), slice(None, None, stride))
    qx_s = Qx[sl]; qy_s = Qy[sl]
    theta_s = 0.5 * np.arctan2(qy_s, qx_s)
    u = np.cos(theta_s); v = np.sin(theta_s)
    ny2, nx2 = u.shape
    xs = np.linspace(0, M - 1, nx2); ys = np.linspace(0, M - 1, ny2)
    gx2, gy2 = np.meshgrid(xs, ys)
    qkw = dict(angles="xy", scale_units="xy", scale=0.055 * M / 20,
               width=0.003, headwidth=0, headlength=0, headaxislength=0,
               color="white", alpha=0.5)
    ax.quiver(gx2, gy2,  u,  v, **qkw)
    ax.quiver(gx2, gy2, -u, -v, **qkw)
    ax.set_title("Orientation (HSV + directors)")
    ax.set_xticks([]); ax.set_yticks([])

    # 5. Well influence
    ax = axes[1][1]
    ax.imshow(influence, origin="lower", extent=extent, cmap="plasma",
              interpolation="bilinear", vmin=0, vmax=1)
    ax.contour(X, Y, influence, levels=[WELL_INFLUENCE_TAU],
               colors=["lime"], linewidths=1.0, linestyles="--")
    ax.set_title("Well influence"); ax.set_xticks([]); ax.set_yticks([])

    # 6. Angle histogram
    ax = axes[1][2]
    theta_deg = np.degrees(theta).ravel()
    ax.hist(theta_deg, bins=90, range=(-90, 90),
            color="steelblue", edgecolor="none", density=True)
    ax.set_xlabel("Fiber angle θ (°)")
    ax.set_ylabel("Density")
    ax.set_title("Angle distribution")
    ax.set_xlim(-90, 90)

    fig.suptitle("VectorField overview", fontsize=14, y=1.01)
    fig.tight_layout()
    return _save_or_show(fig, save_path)
