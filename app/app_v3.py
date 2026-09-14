"""
SHG Fiber Simulator — app.py
Custom canvas component with density painting, vector field painting, and well placement.
"""
import base64, io, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from typing import Optional

# from shg_backend import (
#     _well_weight_and_tangent, create_grid, fit_spline,
#     generate_custom_fields_from_canvas, generate_fiber,
#     make_fiber_aux_fields, make_global_orientation, make_wave_freq_field,
#     rasterize_splines, relax, sample_field_at_seeds,
#     sample_seeds_from_density, sinusoidal_fiber_offset,
# )

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 4. Now use an absolute import (remove the dots "..")
from VectorField import _well_weight_and_tangent, create_grid, make_fiber_aux_fields, make_global_orientation, make_wave_freq_field, relax, sample_field_at_seeds
from SplineSample import fit_spline, sample_seeds_from_density, generate_fiber, sinusoidal_fiber_offset
from shg_backend import ( generate_custom_fields_from_canvas)
from Rasterize import rasterize_splines

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHG Simulator", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""<style>
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:#0d1117!important;color:#e2e8f0!important}
[data-testid="stAppViewBlockContainer"]{padding-top:0.8rem!important}
#MainMenu,footer,header,[data-testid="stDeployButton"],[data-testid="stToolbar"],[data-testid="collapsedControl"]{display:none!important}
*{font-family:Inter,system-ui,sans-serif!important}
[data-testid="stSlider"] label{color:#94a3b8!important;font-size:12px!important}
[data-testid="stButton"]>button{background:#1e2535!important;border:1.5px solid #2d3650!important;color:#94a3b8!important;border-radius:7px!important;transition:all .15s!important}
[data-testid="stButton"]>button:hover{border-color:#4ade8066!important;color:#4ade80!important}
button[kind="primary"]{background:#4ade8022!important;border-color:#4ade80!important;color:#4ade80!important;font-weight:700!important}
button[kind="primary"]:hover{background:#4ade8033!important}
[data-testid="stExpander"]{background:#161b27!important;border:1px solid #2d3650!important;border-radius:8px!important}
[data-testid="stExpander"] summary{color:#94a3b8!important}
hr{border-color:#2d3650!important;margin:0.6rem 0!important}
[data-testid="stDownloadButton"]>button{background:#60a5fa18!important;border:1.5px solid #60a5fa!important;color:#60a5fa!important;border-radius:7px!important;width:100%!important}
[data-testid="stNumberInput"] input{background:#1e2535!important;border:1px solid #2d3650!important;color:#e2e8f0!important;border-radius:6px!important}
[data-testid="stCheckbox"] label{color:#94a3b8!important;font-size:12px!important}
iframe[title="shg_canvas"]{min-height:660px!important}
</style>""", unsafe_allow_html=True)

# ── Component ──────────────────────────────────────────────────────────────────
_COMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_component")
_canvas_fn = components.declare_component("shg_canvas", path=_COMP_DIR)

CANVAS_SIZE = 512

def shg_canvas(mode, canvas_size, wells, density_b64="",
               vec_qx_b64="", vec_qy_b64="", vec_mag_b64="", key=None):
    return _canvas_fn(
        mode=mode, canvas_size=canvas_size,
        wells=json.dumps(wells),
        density_b64=density_b64,
        vec_qx_b64=vec_qx_b64, vec_qy_b64=vec_qy_b64, vec_mag_b64=vec_mag_b64,
        key=key, default=None,
    )

# ── Session state ──────────────────────────────────────────────────────────────
def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

_z = lambda: np.zeros(CANVAS_SIZE * CANVAS_SIZE, dtype=np.float32)
_ss("density_arr",  _z())
_ss("vec_qx_arr",   _z())
_ss("vec_qy_arr",   _z())
_ss("vec_mag_arr",  _z())
_ss("wells",        [])
_ss("active_tab",   "density")
_ss("shg_image",    None)
_ss("shg_splines",  None)
_ss("shg_D",        None)
_ss("shg_Qx",       None)
_ss("shg_Qy",       None)
_ss("active_params",{})
_ss("num_fibers",   300)
_ss("spline_length",150)
_ss("thickness",    2.5)
_ss("G_align",      0.50)
_ss("L_align",      0.50)
_ss("G_curve",      0.30)
_ss("L_curve",      0.50)
_ss("L_conn",       0.50)
_ss("L_wave_freq",  0.25)
_ss("seed",         42)
_ss("show_density", True)
_ss("show_vectors", True)
_ss("show_splines", True)
_ss("show_quiver",  True)

# ── Encode / decode helpers ────────────────────────────────────────────────────
def to_b64(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

def from_b64(b64, size):
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        if arr.size == size * size: return arr
    except Exception: pass
    return np.zeros(size * size, dtype=np.float32)

def arr_to_png_bytes(arr_2d, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(5,5), facecolor="#0d1117")
    ax.imshow(arr_2d, cmap=cmap, origin="upper")
    ax.axis("off"); fig.tight_layout(pad=0)
    buf = io.BytesIO(); fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor="#0d1117")
    plt.close(fig); buf.seek(0); return buf.getvalue()

# ── Orientation field blending ─────────────────────────────────────────────────
def blend_orientation(Qx_g, Qy_g, vec_qx, vec_qy, vec_mag):
    """
    Blend user-painted Q-field (double-angle cos2θ, sin2θ) onto global Q-field.
    vec_mag=0 → pure global, vec_mag=1 → pure painted.
    Both fields are in double-angle space; result is renormalized to unit vectors.
    """
    mag2d = vec_mag.reshape(Qx_g.shape).astype(np.float64)
    qx_p  = vec_qx.reshape(Qx_g.shape).astype(np.float64)
    qy_p  = vec_qy.reshape(Qy_g.shape).astype(np.float64)

    # Only blend where the user has actually painted
    has_paint = mag2d > 0.01
    Qx = np.where(has_paint, (1.0 - mag2d) * Qx_g + mag2d * qx_p, Qx_g)
    Qy = np.where(has_paint, (1.0 - mag2d) * Qy_g + mag2d * qy_p, Qy_g)

    # Renormalize — a linear blend of unit vectors is not itself a unit vector
    norm = np.sqrt(Qx**2 + Qy**2)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return Qx / norm, Qy / norm

# ── Custom orientation field plot ─────────────────────────────────────────────
def plot_fields(ax, D, Qx, Qy, splines, image_size,
                show_density=True, show_quiver=True, show_splines=True):
    ax.set_xlim(0, image_size); ax.set_ylim(image_size, 0); ax.axis("off")

    if show_density:
        ax.imshow(D, cmap="viridis", origin="upper",
                  extent=[0, image_size, image_size, 0], alpha=1.0)

    if show_quiver and Qx is not None:
        shape = Qx.shape
        step = max(1, image_size // 24)
        rows = np.arange(step//2, shape[0], step)
        cols = np.arange(step//2, shape[1], step)
        R, C = np.meshgrid(rows, cols, indexing="ij")
        # Recover fiber angle from double-angle representation
        # Q = (cos2θ, sin2θ)  →  θ = 0.5 * arctan2(Qy, Qx)
        theta = 0.5 * np.arctan2(Qy[R, C], Qx[R, C])
        # In image coords: x=col (rightward), y=row (downward)
        # Fiber tangent: (cos θ, sin θ) in standard math coords
        # In image display with ylim inverted: u=cosθ (right), v=sinθ (down matches image y-down)
        u =  np.cos(theta)
        v =  np.sin(theta)
        ax.quiver(C, R, u, v,
                  color="white", headlength=0, headaxislength=0,
                  pivot="middle", scale=26, alpha=0.55, width=0.003)

    if show_splines and splines:
        for sp in splines:
            if len(sp) > 0:
                ax.plot(sp[:, 1], sp[:, 0], color="white", linewidth=1.0, alpha=0.75)

    return ax

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px">
  <div style="width:9px;height:9px;border-radius:50%;background:#4ade80;box-shadow:0 0 12px #4ade80;flex-shrink:0"></div>
  <span style="font-size:15px;font-weight:700;letter-spacing:-0.02em">SHG Fiber Simulator</span>
  <span style="color:#475569;font-size:12px">Interactive Field Editor</span>
</div><hr>""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([2.1, 2.1, 1.1], gap="medium")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — Canvas
# ══════════════════════════════════════════════════════════════════════════════
with col_left:
    tc1, tc2 = st.columns(2, gap="small")
    with tc1:
        if st.button("🟢 Density / Vectors", use_container_width=True,
                     type="primary" if st.session_state.active_tab=="density" else "secondary"):
            if st.session_state.active_tab != "density":
                st.session_state.active_tab = "density"; st.rerun()
    with tc2:
        if st.button("🔵 Orientation Wells", use_container_width=True,
                     type="primary" if st.session_state.active_tab=="wells" else "secondary"):
            if st.session_state.active_tab != "wells":
                st.session_state.active_tab = "wells"; st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    result = shg_canvas(
        mode        = st.session_state.active_tab,
        canvas_size = CANVAS_SIZE,
        wells       = st.session_state.wells,
        density_b64 = to_b64(st.session_state.density_arr),
        vec_qx_b64  = to_b64(st.session_state.vec_qx_arr),
        vec_qy_b64  = to_b64(st.session_state.vec_qy_arr),
        vec_mag_b64 = to_b64(st.session_state.vec_mag_arr),
        key         = "shg_canvas",
    )

    if result is not None:
        rt = result.get("type", "")
        sz = result.get("size", CANVAS_SIZE)
        if rt == "density":
            b64 = result.get("density_b64", "")
            if b64: st.session_state.density_arr = from_b64(b64, sz)
        elif rt == "vec":
            if result.get("vec_qx_b64"):  st.session_state.vec_qx_arr  = from_b64(result["vec_qx_b64"],  sz)
            if result.get("vec_qy_b64"):  st.session_state.vec_qy_arr  = from_b64(result["vec_qy_b64"],  sz)
            if result.get("vec_mag_b64"): st.session_state.vec_mag_arr = from_b64(result["vec_mag_b64"], sz)
        elif rt == "wells":
            st.session_state.wells = result.get("wells", [])
        elif rt == "sync":
            # Sent after loading a file in the JS — sync all state to Python
            if result.get("density_b64"):  st.session_state.density_arr = from_b64(result["density_b64"], sz)
            if result.get("vec_qx_b64"):   st.session_state.vec_qx_arr  = from_b64(result["vec_qx_b64"],  sz)
            if result.get("vec_qy_b64"):   st.session_state.vec_qy_arr  = from_b64(result["vec_qy_b64"],  sz)
            if result.get("vec_mag_b64"):  st.session_state.vec_mag_arr = from_b64(result["vec_mag_b64"], sz)
            if result.get("wells") is not None: st.session_state.wells = result["wells"]

    # Legend
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;margin-top:6px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:72px;height:6px;border-radius:3px;background:linear-gradient(to right,#440154,#3b528b,#21908d,#5dc963,#fde725)"></div>
        <span style="color:#475569;font-size:10px">density</span>
      </div>
      <div style="display:flex;align-items:center;gap:6px">
        <div style="width:18px;height:6px;border-radius:3px;background:#a78bfa"></div>
        <span style="color:#475569;font-size:10px">painted orientation</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Status
    has_d   = float(st.session_state.density_arr.max()) > 1e-3
    has_v   = float(st.session_state.vec_mag_arr.max()) > 1e-3
    n_wells = len(st.session_state.wells)
    st.markdown(f"""
    <div style="background:#1e2535;border:1px solid #2d3650;border-radius:8px;padding:9px 12px;font-size:11px;margin-top:8px">
      <div style="display:flex;gap:16px;flex-wrap:wrap">
        <span style="color:#64748b">Density: <span style="color:{'#4ade80' if has_d else '#f87171'}">{'painted' if has_d else 'empty'}</span></span>
        <span style="color:#64748b">Vectors: <span style="color:{'#a78bfa' if has_v else '#475569'}">{'painted' if has_v else 'empty'}</span></span>
        <span style="color:#64748b">Wells: <span style="color:#60a5fa">{n_wells}</span></span>
      </div>
    </div>""", unsafe_allow_html=True)

# ── Presets ─────────────────────────────────────────────────────────────────────
# Mapped from legacy synthetic_class_params. Fields not present in current UI
# (G_density, L_density, G_conn) are stored but not used by sliders.
# spline_num  → num_fibers
# fiber_width_px → thickness (default 2.5 kept where not specified)
PRESETS = {
    "RED": {
        "label": "RED",
        "color": "#f87171",
        "desc":  "High density, high alignment, short fibers",
        "params": {
            "G_align":       0.7,
            "L_align":       0.6,
            "G_curve":       0.5,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
    "YELLOW_BAD": {
        "label": "YELLOW BAD",
        "color": "#fbbf24",
        "desc":  "Low alignment, moderate density, short fibers",
        "params": {
            "G_align":       0.4,
            "L_align":       0.0,
            "G_curve":       0.5,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
    "YELLOW_GOOD": {
        "label": "YELLOW GOOD",
        "color": "#fde68a",
        "desc":  "Very high density, moderate curve, many short fibers",
        "params": {
            "G_align":       0.2,
            "L_align":       0.6,
            "G_curve":       0.7,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    800,   # capped at slider max 800 (orig 3000)
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
    "INTERESTING": {
        "label": "INTERESTING",
        "color": "#a78bfa",
        "desc":  "Balanced alignment and density",
        "params": {
            "G_align":       0.5,
            "L_align":       0.5,
            "G_curve":       0.5,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
    "CYAN": {
        "label": "CYAN",
        "color": "#22d3ee",
        "desc":  "Balanced — same base as INTERESTING",
        "params": {
            "G_align":       0.5,
            "L_align":       0.5,
            "G_curve":       0.5,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
    "ORANGE": {
        "label": "ORANGE",
        "color": "#fb923c",
        "desc":  "Balanced — test configuration",
        "params": {
            "G_align":       0.5,
            "L_align":       0.5,
            "G_curve":       0.5,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 50,
            "thickness":     2.5,
        },
    },
}

def apply_preset(key):
    p = PRESETS[key]["params"]
    for k, v in p.items():
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Parameters
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    def _sec(label):
        st.markdown(f"<div style='color:#475569;font-size:10px;font-weight:700;letter-spacing:.07em;"
                    f"text-transform:uppercase;margin:10px 0 6px'>{label}</div>", unsafe_allow_html=True)

    # ── Preset selector ────────────────────────────────────────────────────────
    _sec("Presets")

    # Build colored preset buttons via HTML — one per row for legibility
    for preset_key, preset in PRESETS.items():
        col_dot, col_btn = st.columns([0.08, 0.92], gap="small")
        with col_dot:
            st.markdown(
                f"<div style='width:10px;height:10px;border-radius:50%;"
                f"background:{preset['color']};margin-top:10px'></div>",
                unsafe_allow_html=True,
            )
        with col_btn:
            if st.button(preset["label"], key=f"preset_{preset_key}", use_container_width=True):
                apply_preset(preset_key)
                st.rerun()

    # Show description of active preset (whichever matches current params)
    active_preset = None
    for pk, pv in PRESETS.items():
        if all(abs(st.session_state.get(k, 0) - v) < 0.01
               for k, v in pv["params"].items()):
            active_preset = pv
            break
    if active_preset:
        st.markdown(
            f"<div style='background:#1e2535;border:1px solid {active_preset['color']}44;"
            f"border-radius:6px;padding:6px 9px;font-size:10px;color:#94a3b8;margin-top:2px'>"
            f"<span style='color:{active_preset['color']};font-weight:700'>{active_preset['label']}</span>"
            f" — {active_preset['desc']}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    _sec("Fiber")
    st.session_state.num_fibers    = st.slider("Count",     20,  800, st.session_state.num_fibers,    step=10)
    st.session_state.spline_length = st.slider("Length",    20,  400, st.session_state.spline_length, step=5)
    st.session_state.thickness     = st.slider("Thickness", 0.5, 10.0,st.session_state.thickness,    step=0.5)
    st.session_state.wave_amplitude_px     = st.slider("Wave amplitude", 0.5, 5.0,st.session_state.wave_amplitude_px,    step=0.5)
    st.session_state.wave_wavelength_px     = st.slider("Wave wavelength", 0.5, .0,st.session_state.wave_wavelength_px,    step=0.5)

    st.markdown("---")
    _sec("Alignment")
    st.session_state.G_align = st.slider("Global align", 0.0, 1.0, st.session_state.G_align, step=0.01)
    st.session_state.L_align = st.slider("Local relax",  0.0, 1.0, st.session_state.L_align, step=0.01)
    st.session_state.G_curve = st.slider("Global curve", 0.0, 1.0, st.session_state.G_curve, step=0.01)
    st.session_state.L_curve = st.slider("Local curve",  0.0, 1.0, st.session_state.L_curve, step=0.01)
    st.markdown("---")
    _sec("Texture")
    st.session_state.L_conn      = st.slider("Connectivity", 0.0, 1.0, st.session_state.L_conn,      step=0.01)
    st.session_state.L_wave_freq = st.slider("Wobble freq",  0.0, 1.0, st.session_state.L_wave_freq, step=0.01)
    st.markdown("---")
    _sec("Seed")
    st.session_state.seed = st.number_input("Random seed", value=int(st.session_state.seed), step=1)

# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE — Render & Output
# ══════════════════════════════════════════════════════════════════════════════
with col_mid:
    st.markdown("<div style='color:#475569;font-size:10px;font-weight:700;letter-spacing:.07em;"
                "text-transform:uppercase;margin-bottom:8px'>Synthetic SHG Output</div>",
                unsafe_allow_html=True)

    render_btn = st.button("⟳  Render SHG Image", type="primary", use_container_width=True)

    if render_btn:
        with st.spinner("Generating SHG image…"):
            rng   = np.random.default_rng(int(st.session_state.seed))
            X, Y  = create_grid(CANVAS_SIZE, resolution_factor=1.0)
            shape = X.shape

            well_configs = [{"x":w["x"],"y":w["y"],"w":w.get("rx",45),"h":w.get("ry",28),"angle":w.get("angle",0)}
                            for w in st.session_state.wells]
            _, wells_arr, hard_mask = generate_custom_fields_from_canvas(
                density_configs=[], well_configs=well_configs, shape=shape)

            D_raw = st.session_state.density_arr.reshape(CANVAS_SIZE, CANVAS_SIZE).astype(np.float64)
            if D_raw.max() < 1e-4: D_raw = np.full((CANVAS_SIZE, CANVAS_SIZE), 0.3)
            D = np.where(hard_mask, 0.0, D_raw)

            # Global orientation field
            Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng)

            # Blend with user-painted vector field
            Qx_g, Qy_g = blend_orientation(
                Qx_g, Qy_g,
                st.session_state.vec_qx_arr,
                st.session_state.vec_qy_arr,
                st.session_state.vec_mag_arr,
            )

            # Well influence
            if len(wells_arr) > 0:
                Qx_w, Qy_w, influence = _well_weight_and_tangent(X, Y, wells_arr, Qx_g, Qy_g)
            else:
                Qx_w = Qy_w = np.zeros(shape); influence = np.zeros(shape)

            Qx, Qy = relax(Qx_g, Qy_g, Qx_g, Qy_g, Qx_w, Qy_w, influence,
                           D, st.session_state.G_align, st.session_state.L_align, wells_arr, X, Y)

            curve_field, conn_field = make_fiber_aux_fields(shape, st.session_state.L_curve, st.session_state.L_conn, rng)
            wave_freq_field = make_wave_freq_field(shape, st.session_state.L_wave_freq, rng)

            wave_amplitude_px = int(st.sessions_state.wave_amplitude_px)
            wave_wavelength_px = int(st.sessions_state.wave_wavelength_px)

            num_fibers    = int(st.session_state.num_fibers)
            spline_length = int(st.session_state.spline_length)
            seeds         = sample_seeds_from_density(D, num_fibers, L_density=0.5, rng=rng)
            aux_curve     = sample_field_at_seeds(seeds, curve_field)
            aux_conn      = sample_field_at_seeds(seeds, conn_field)
            aux_wave_freq = sample_field_at_seeds(seeds, wave_freq_field)

            splines = []
            for i, seed_pt in enumerate(seeds):
                raw = generate_fiber(Qx, Qy, seed_pt, step_size=1.0, spline_length=spline_length,
                                     L_curve=st.session_state.L_curve, susceptibility=1.0-aux_curve[i], rng=rng)
                off = sinusoidal_fiber_offset(
                    raw, wave_amp=aux_curve[i], wave_freq=aux_wave_freq[i], 
                    wave_amplitude_px=wave_amplitude_px,
                    wave_wavelength_px=wave_wavelength_px,
                    rng=rng
                )
                splines.append(fit_spline(off, num_samples=max(50, spline_length*2)))

            raster = rasterize_splines(H=shape[0], W=shape[1], splines=splines,
                                       thickness=float(st.session_state.thickness),
                                       aux_L_conn=aux_conn, intensity_seed=int(st.session_state.seed))

            st.session_state.shg_image   = raster
            st.session_state.shg_splines = splines
            st.session_state.shg_D       = D
            st.session_state.shg_Qx      = Qx
            st.session_state.shg_Qy      = Qy
            st.session_state.active_params = dict(
                seed=int(st.session_state.seed), num_fibers=num_fibers,
                spline_length=spline_length, thickness=float(st.session_state.thickness),
                G_align=float(st.session_state.G_align), L_align=float(st.session_state.L_align),
                G_curve=float(st.session_state.G_curve), L_curve=float(st.session_state.L_curve),
                L_conn=float(st.session_state.L_conn), L_wave_freq=float(st.session_state.L_wave_freq),
                n_wells=len(st.session_state.wells), vec_painted=has_v,
            )

    # ── Display ────────────────────────────────────────────────────────────────
    if st.session_state.shg_image is not None:
        img_arr = st.session_state.shg_image
        st.image(img_arr, clamp=True, use_container_width=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            norm = img_arr if img_arr.dtype==np.uint8 else (
                (img_arr-img_arr.min())/(img_arr.max()-img_arr.min()+1e-8)*255).astype(np.uint8)
            buf = io.BytesIO(); Image.fromarray(norm).save(buf,format="PNG")
            st.download_button("↓ SHG PNG", data=buf.getvalue(),
                file_name=f"shg_{st.session_state.active_params.get('seed',0)}.png",
                mime="image/png", use_container_width=True)
        with c2:
            st.download_button("↓ Splines JSON",
                data=json.dumps({"parameters":st.session_state.active_params,
                                 "splines":[s.tolist() for s in (st.session_state.shg_splines or [])]},indent=2),
                file_name=f"shg_splines_{st.session_state.active_params.get('seed',0)}.json",
                mime="application/json", use_container_width=True)

        with st.expander("Parameters"):
            st.json(st.session_state.active_params)

        # ── Orientation field preview ─────────────────────────────────────────
        st.markdown("---")
        st.markdown("<div style='color:#475569;font-size:10px;font-weight:700;letter-spacing:.07em;"
                    "text-transform:uppercase;margin-bottom:6px'>Orientation Field</div>",
                    unsafe_allow_html=True)

        # Visibility checkboxes
        cb1, cb2, cb3 = st.columns(3)
        with cb1: st.session_state.show_density = st.checkbox("Density",  value=st.session_state.show_density)
        with cb2: st.session_state.show_quiver  = st.checkbox("Vectors",  value=st.session_state.show_quiver)
        with cb3: st.session_state.show_splines = st.checkbox("Splines",  value=st.session_state.show_splines)

        fig, ax = plt.subplots(figsize=(5,5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        plot_fields(
            ax=ax, D=st.session_state.shg_D,
            Qx=st.session_state.shg_Qx, Qy=st.session_state.shg_Qy,
            splines=st.session_state.shg_splines,
            image_size=CANVAS_SIZE,
            show_density=st.session_state.show_density,
            show_quiver=st.session_state.show_quiver,
            show_splines=st.session_state.show_splines,
        )
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Download orientation field PNG
        c3, c4 = st.columns(2, gap="small")
        with c3:
            fig2, ax2 = plt.subplots(figsize=(5,5), facecolor="#0d1117")
            ax2.set_facecolor("#0d1117")
            plot_fields(ax=ax2, D=st.session_state.shg_D,
                        Qx=st.session_state.shg_Qx, Qy=st.session_state.shg_Qy,
                        splines=st.session_state.shg_splines, image_size=CANVAS_SIZE,
                        show_density=True, show_quiver=True, show_splines=True)
            fig2.tight_layout(pad=0)
            buf2 = io.BytesIO(); fig2.savefig(buf2,format="png",bbox_inches="tight",pad_inches=0,facecolor="#0d1117")
            plt.close(fig2); buf2.seek(0)
            st.download_button("↓ Field PNG", data=buf2.getvalue(),
                file_name="orientation_field.png", mime="image/png", use_container_width=True)
        with c4:
            # Download vector field as numpy
            vf_export = {
                "Qx": st.session_state.shg_Qx.tolist(),
                "Qy": st.session_state.shg_Qy.tolist(),
                "shape": list(st.session_state.shg_Qx.shape),
            }
            st.download_button("↓ Vector JSON", data=json.dumps(vf_export),
                file_name="vector_field.json", mime="application/json", use_container_width=True)

    else:
        st.markdown("""
        <div style="background:#161b27;border:1px solid #2d3650;border-radius:8px;
             padding:40px 20px;text-align:center;margin-top:8px">
          <div style="font-size:32px;margin-bottom:10px;opacity:0.4">⟳</div>
          <div style="color:#64748b;font-size:12px;line-height:1.6">
            Paint density, orientation vectors,<br>optionally place wells,<br>
            then click <b style="color:#4ade80">Render</b>.
          </div>
        </div>""", unsafe_allow_html=True)