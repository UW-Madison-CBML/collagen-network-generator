"""
SHG Fiber Simulator : app.py
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
from scipy.ndimage import zoom

from typing import List, Tuple, Optional, Dict, Any

from typing import Optional

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from synthetic_code.VectorField import *
from synthetic_code.VectorField import _well_weight_and_tangent
from synthetic_code.SplineSample import *
from synthetic_code.Rasterize import *

def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_B64 = img_to_base64(os.path.join(current_dir, "assets", "uw-logo-vertical-color-web-digital.png"))
LOGO_B64_horizontal = img_to_base64(os.path.join(current_dir, "assets", "uw-logo-horizontal-color-web-digital.png"))

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

    # Process Density Points
    for cfg in density_configs:
        pt_x, pt_y = cfg["x"], cfg["y"]
        sigma_x = max(1.0, float(cfg["w"]))
        sigma_y = max(1.0, float(cfg["h"]))
        intensity = float(cfg["intensity"])

        gauss = intensity * np.exp(
            -(((X_grid - pt_x) ** 2) / (2.0 * sigma_x ** 2) + ((Y_grid - pt_y) ** 2) / (2.0 * sigma_y ** 2))
        )
        D += gauss

    # Process Wells
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

RESOLUTION_FACTOR = 2.0

def upsample_canvas(arr_flat, canvas_size, target_shape):
    """Upsample a flat canvas array to match the simulation grid shape."""
    arr_2d = arr_flat.reshape(canvas_size, canvas_size).astype(np.float64)
    zoom_y = target_shape[0] / canvas_size
    zoom_x = target_shape[1] / canvas_size
    return zoom(arr_2d, (zoom_y, zoom_x), order=1)  # bilinear

# ── Session state ──────────────────────────────────────────────────────────────
def _ss(k, v):
    if k not in st.session_state: st.session_state[k] = v

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
_ss("show_density", False)
_ss("show_vectors", True)
_ss("show_splines", True)
_ss("show_quiver",  False)
_ss("wave_amplitude_px", 2.8)
_ss("wave_wavelength_px", None)

_ss("n_layers",  1)
_ss("depth", 1.0)
_ss("focal_plane", 0.0)

ACCENT = "#cc1543"


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHG Simulator", layout="wide", page_icon="assets/uw-logo-vertical-color-web-digital.png")
st.markdown(f"""
<style>
:root {{
  --shg-surface: var(--secondary-background-color, #f7f7f7);
  --shg-text: var(--text-color, #222222);
  --shg-muted: color-mix(in srgb, var(--text-color, #222222) 80%, var(--background-color, #ffffff));
  --shg-border: color-mix(in srgb, var(--text-color, #222222) 24%, var(--background-color, #ffffff));
}}
[data-testid="stMainBlockContainer"] {{ padding: 4.5rem 2.5rem 2rem; max-width: 1900px; }}
.research-header {{ display:flex; align-items:center; justify-content:space-between;
  gap:24px; flex-wrap:wrap; border-bottom:3px solid #c5050c; padding-bottom:20px; margin-bottom:20px; }}
.research-header h1 {{ font-size:36px; font-weight:600; padding:0; margin:0; line-height:1.25; }}
.research-header p {{ margin:6px 0 0; font-size:20px; color:var(--shg-muted); }}
.research-header img {{ width:190px; height:auto; }}
[data-testid="stHeadingWithActionElements"] h3 {{ font-size:28px; font-weight:650; padding-top:0; }}
[data-testid="stExpander"] summary p {{ font-size:22px!important; font-weight:600; }}
[data-testid="stCaptionContainer"] p {{ font-size:20px; line-height:1.5; color:var(--shg-muted); }}
[data-testid="stWidgetLabel"] p, [data-testid="stButton"] button p,
[data-testid="stDownloadButton"] button p, [data-testid="stNumberInput"] input,
[data-testid="stExpander"] summary p, [data-testid="stTabs"] button p {{ font-size:21px; line-height:1.45; }}
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {{
  min-height:44px; padding:8px 14px; border-radius:4px;
}}
button[kind="primary"] {{ background:#c5050c!important; border-color:#c5050c!important; color:#fff!important; }}
button[kind="primary"]:hover {{ background:#9b0000!important; color:#fff!important; }}
button:focus-visible, input:focus-visible, summary:focus-visible {{ outline:3px solid #c5050c; outline-offset:3px; }}
button[aria-label^="Help for"] {{ color:#c5050c; min-width:28px; min-height:28px; }}
button[aria-label^="Help for"] svg {{ width:22px; height:22px; }}
[data-baseweb="select"] *, [role="listbox"] *, [role="option"] {{ font-size:22px!important; line-height:1.4!important; }}
[data-testid="stTabs"] [role="tablist"] {{ order:2; }}
[data-testid="stTabs"] [role="tabpanel"] {{ order:1; padding-top:0; }}
[role="tooltip"] {{ font-size:20px; line-height:1.5; }}
[data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"],
[data-testid="stSliderThumbValue"] {{ font-size:20px; }}
[data-testid="stExpander"] {{ border-radius:4px; }}
[data-testid="stExpander"] summary {{ min-height:48px; }}
.output-empty {{ aspect-ratio:1; display:flex; align-items:center; justify-content:center;
  border:1px solid var(--shg-border); background:var(--shg-surface); padding:32px;
  font-size:21px; line-height:1.6; text-align:center; color:var(--shg-muted); }}
@media (max-width:900px) {{
  [data-testid="stMainBlockContainer"] {{ padding:4.5rem 1rem 2rem; }}
  .research-header h1 {{ font-size:26px; }}
  .research-header img {{ width:160px; }}
}}
</style>
<header class="research-header">
  <div><h1>SHG Fiber Simulator</h1><p>Bhaskar Lab · Computational Biology &amp; Machine Learning</p></div>
  <img src="data:image/png;base64,{LOGO_B64_horizontal}" alt="University of Wisconsin–Madison">
</header>
""", unsafe_allow_html=True)

# Readable native inputs and contextual help.
st.markdown("""
<style>
[data-testid="stSelectbox"] input, [data-testid="stSelectbox"] div,
[data-testid="stSelectbox"] span, [role="combobox"], [role="option"] * {
  font-size:22px!important; line-height:1.4!important;
}
[data-testid="stTooltipContent"] *, [data-baseweb="tooltip"] *, [role="tooltip"] * {
  font-size:22px!important; line-height:1.5!important;
}
.preset-description { font-size:21px; line-height:1.5; color:var(--shg-text); margin:0 0 12px; }
</style>
""", unsafe_allow_html=True)

from contextlib import contextmanager
from html import escape

@contextmanager
def parameter_group(label, description):
    with st.expander(label, expanded=False):
        st.caption(description)
        yield

def help_icon(label, description):
    return f'''<span class="parameter-help">
      <span class="help-circle" tabindex="0" aria-label="Help for {escape(label)}">?</span>
      <span class="parameter-help-card" role="tooltip">{escape(description)}</span>
    </span>'''

def preset_description(label, description):
    st.markdown(f'<div class="preset-help-row">{help_icon(label, description)}</div>', unsafe_allow_html=True)

def parameter_label(label, description):
    st.markdown(f'<div class="parameter-label"><span>{escape(label)}</span>{help_icon(label, description)}</div>', unsafe_allow_html=True)

def parameter_slider(label, *args, help, **kwargs):
    parameter_label(label, help)
    return st.slider(label, *args, label_visibility="collapsed", **kwargs)

def parameter_number(label, *args, help, **kwargs):
    parameter_label(label, help)
    return st.number_input(label, *args, label_visibility="collapsed", **kwargs)

st.markdown("""
<style>
.parameter-label { display:flex; align-items:center; justify-content:space-between;
  gap:16px; font-size:21px; font-weight:500; line-height:1.45; margin-top:4px; }
.parameter-help { display:inline-flex; position:relative; flex:0 0 22px; }
.help-circle { box-sizing:border-box; display:inline-flex; align-items:center; justify-content:center;
  width:22px; height:22px; border:2px solid #737780; border-radius:50%;
  font:600 15px/1 sans-serif; color:#737780; cursor:help; }
.preset-help-row { display:flex; justify-content:center; padding-top:9px; }
.parameter-help-card { display:none; position:absolute; right:0; top:100%; z-index:1000;
  width:min(360px,75vw); padding:16px; border:1px solid var(--shg-border);
  border-radius:6px; background:var(--background-color,white); color:var(--shg-text);
  font-size:21px; font-weight:400; line-height:1.5; box-shadow:0 3px 10px #0002; }
.parameter-help:hover .parameter-help-card,
.help-circle:focus-visible + .parameter-help-card { display:block; }
.help-circle:focus-visible { outline:3px solid #c5050c; outline-offset:3px; }
[data-testid="stSlider"] { padding-top:14px; padding-bottom:4px; }
[data-testid="stSliderThumbValue"] { font-size:20px!important; line-height:1.3; }
[data-testid="stNumberInput"] { margin-top:4px; margin-bottom:6px; }
[data-testid="stMarkdownContainer"] .preset-status { height:48px; min-height:48px; min-width:0; box-sizing:border-box;
  padding:8px 14px; border-left:4px solid #c5050c; display:flex; flex-wrap:nowrap;
  align-items:center; gap:10px; overflow:hidden; background:var(--shg-surface); cursor:default; }
.preset-status .status-label { flex:0 0 auto; white-space:nowrap; font-size:20px!important; font-weight:500; color:var(--shg-muted); }
.preset-status .status-value { min-width:0; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-size:24px!important; font-weight:600; line-height:1.4; color:var(--shg-text); }
</style>
""", unsafe_allow_html=True)

# ── Component ──────────────────────────────────────────────────────────────────
_COMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_component")
_canvas_fn = components.declare_component("shg_canvas", path=_COMP_DIR)

CANVAS_SIZE = 512

_z = lambda: np.zeros(CANVAS_SIZE * CANVAS_SIZE, dtype=np.float32)
_ss("density_arr",  _z())
_ss("vec_qx_arr",   _z())
_ss("vec_qy_arr",   _z())
_ss("vec_mag_arr",  _z())

def shg_canvas(mode, canvas_size, wells, density_b64="",
               vec_qx_b64="", vec_qy_b64="", vec_mag_b64="", key=None):
    return _canvas_fn(
        mode=mode, canvas_size=canvas_size,
        wells=json.dumps(wells),
        density_b64=density_b64,
        vec_qx_b64=vec_qx_b64, vec_qy_b64=vec_qy_b64, vec_mag_b64=vec_mag_b64,
        key=key, default=None,
    )


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

    # Renormalize : a linear blend of unit vectors is not itself a unit vector
    norm = np.sqrt(Qx**2 + Qy**2)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return Qx / norm, Qy / norm

# ── Custom orientation field plot ─────────────────────────────────────────────
def plot_fields(ax, D, Qx, Qy, splines, image_size, grid_size=None,
                show_density=True, show_quiver=True, show_splines=True):
    
    # grid_size is the actual simulation resolution (e.g. 2048).
    # image_size is the display size we want to plot at (e.g. 512).
    # If they differ, spline coords need scaling down for display.
    if grid_size is None:
        grid_size = image_size
    display_scale = image_size / grid_size  # e.g. 512/2048 = 0.25

    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)
    ax.axis("off")

    if show_density:
        ax.imshow(D, cmap="magma", origin="upper",
                  extent=[0, image_size, image_size, 0], alpha=1.0)

    if show_quiver and Qx is not None:
        shape = Qx.shape
        step = max(1, grid_size // 24)
        rows = np.arange(step//2, shape[0], step)
        cols = np.arange(step//2, shape[1], step)
        R, C = np.meshgrid(rows, cols, indexing="ij")
        theta = 0.5 * np.arctan2(Qy[R, C], Qx[R, C])
        u =  np.cos(theta)
        v = -np.sin(theta)
        # Scale quiver positions to display space
        ax.quiver(C * display_scale, R * display_scale, u, v,
                  color="#f59e0b",
                  headlength=4,
                  headaxislength=3.5,
                  headwidth=3,
                  pivot="middle",
                  scale=28,
                  alpha=0.75,
                  width=0.004)

    if show_splines and splines:
        for i, sp in enumerate(splines):
            if len(sp) > 0:
                # sp[:,0] = row (y), sp[:,1] = col (x) in grid coords
                ax.plot(sp[:, 1] * display_scale,
                        sp[:, 0] * display_scale,
                        color=plt.get_cmap("tab20")(i % 20), linewidth=1.1, alpha=0.9)

    return ax

# ── Header ─────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px">
#   <div style="width:9px;height:9px;border-radius:50%;background:#cc1543;box-shadow:0 0 12px #cc154388;flex-shrink:0"></div>
#   <span style="font-size:20px;font-weight:700;letter-spacing:-0.02em">SHG Fiber Simulator</span>
#   <span style="color:#475569;font-size:12px">Interactive Field Editor</span>
# </div><hr>""", unsafe_allow_html=True)

# st.markdown(f"""
# <div class="brand-bar">
#   <div class="brand-left">
#     <img src="data:image/png;base64,{LOGO_B64}" style="height:44px;opacity:0.95" alt="UW–Madison">
#     <div class="brand-divider"></div>
#     <div class="brand-lab">
#       <div class="brand-lab-name">CBML</div>
#       <div class="brand-lab-sub">Computational Biology &amp; Machine Learning</div>
#       <div class="brand-lab-sub" style="color:#475569">Bhaskar Lab · UW–Madison</div>
#     </div>
#     <div class="brand-divider"></div>
#     <div class="brand-app">
#       <div class="brand-app-title">SHG Fiber Simulator</div>
#       <div class="brand-app-sub">Interactive Field Editor</div>
#     </div>
#   </div>
#   <div class="brand-right" style="display:flex;align-items:center;gap:10px">
#     <span style="font-size:11px;color:{MUTED}">{'🌙 Dark' if st.session_state.dark_mode else '☀️ Light'}</span>
#   </div>
# </div>
# """, unsafe_allow_html=True)

# Toggle button : sits in top-right via Streamlit columns trick
# _, toggle_col = st.columns([0.85, 0.15])
# with toggle_col:
#     label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
#     if st.button(label, key="theme_toggle", use_container_width=True):
#         st.session_state.dark_mode = not st.session_state.dark_mode
#         st.rerun()

# ── Layout ─────────────────────────────────────────────────────────────────────
header_columns = st.columns([1.3, 1.3, 1], gap="medium")
for column, title, description in zip(header_columns,
        ("Field editor", "Output", "Parameters"),
        ("Edit density and fiber direction.", "View the image and fiber paths.",
         "Adjust settings, then select Render image.")):
    with column:
        st.subheader(title)
        st.caption(description)
editor_actions, output_actions, parameter_actions = st.columns([1.3, 1.3, 1], gap="medium")
col_left, col_mid, settings_area = st.columns([1.3, 1.3, 1], gap="medium")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT : Canvas
# ══════════════════════════════════════════════════════════════════════════════
with col_left:
    with editor_actions:
        tc1, tc2 = st.columns(2, gap="small")
        with tc1:
            if st.button("Paint fields", use_container_width=True,
                         type="primary" if st.session_state.active_tab=="density" else "secondary"):
                if st.session_state.active_tab != "density":
                    st.session_state.active_tab = "density"; st.rerun()
        with tc2:
            if st.button("Fiber guides", use_container_width=True,
                         type="primary" if st.session_state.active_tab=="wells" else "secondary"):
                if st.session_state.active_tab != "wells":
                    st.session_state.active_tab = "wells"; st.rerun()


    result = shg_canvas(
        mode        = st.session_state.active_tab,
        canvas_size = CANVAS_SIZE,
        wells       = st.session_state.wells,
        density_b64 = to_b64(st.session_state.density_arr) if np.any(st.session_state.density_arr) else "",
        vec_qx_b64  = to_b64(st.session_state.vec_qx_arr) if np.any(st.session_state.vec_mag_arr) else "",
        vec_qy_b64  = to_b64(st.session_state.vec_qy_arr) if np.any(st.session_state.vec_mag_arr) else "",
        vec_mag_b64 = to_b64(st.session_state.vec_mag_arr) if np.any(st.session_state.vec_mag_arr) else "",
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
            # Sent after loading a file in the JS : sync all state to Python
            if result.get("density_b64"):  st.session_state.density_arr = from_b64(result["density_b64"], sz)
            if result.get("vec_qx_b64"):   st.session_state.vec_qx_arr  = from_b64(result["vec_qx_b64"],  sz)
            if result.get("vec_qy_b64"):   st.session_state.vec_qy_arr  = from_b64(result["vec_qy_b64"],  sz)
            if result.get("vec_mag_b64"):  st.session_state.vec_mag_arr = from_b64(result["vec_mag_b64"], sz)
            if result.get("wells") is not None: st.session_state.wells = result["wells"]

    has_d = float(st.session_state.density_arr.max()) > 1e-3
    has_v = float(st.session_state.vec_mag_arr.max()) > 1e-3
    n_wells = len(st.session_state.wells)

# ── Presets ─────────────────────────────────────────────────────────────────────
# Mapped from legacy synthetic_class_params. Fields not present in current UI
# (G_density, L_density, G_conn) are stored but not used by sliders.
# spline_num  → num_fibers
# fiber_width_px → thickness (default 2.5 kept where not specified)

PRESETS = {

    #syn.generate_synthetic_shg(seed=11, G_align=0.7, G_density=0.7, G_curve=0.5, G_conn=1, L_align=0.6, L_density=0, L_conn=0.3, L_curve=0, L_intensity=0.5, spline_length=50, spline_num=1200)
    #
    "RED": {
        "label": "Aligned short fibers",
        "color": "#f87171",
        "desc":  "1,200 short fibers with strong overall alignment and moderate broad curvature.",
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
    "YELLOW_BAD": { # syn.generate_synthetic_shg(seed=10, G_align=0.4, G_density=0.7, G_curve=0, G_conn=1, L_align=0.3, L_density=0.4, L_conn=1, L_curve=0, spline_length=60, spline_num=1200)

        "label": "Loosely aligned fibers",
        "color": "#fbbf24",
        "desc":  "1,200 short fibers with weaker alignment and little added waviness.",
        "params": {
            "G_align":       0.4,
            "L_align":       0.3,
            "G_curve":       0.0,
            "L_curve":       0.0,
            "L_conn":        1.0,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 60,
            "thickness":     2.5,
        },
    },
    "YELLOW_GOOD": {# syn.generate_synthetic_shg(seed=10, G_align=0.2, G_density=1, G_curve=0.7, G_conn=1, L_align=1, L_density=0.6, L_conn=0.3, L_curve=0, spline_length=70, spline_num=1200)
        "label": "Curved, field-following fibers",
        "color": "#fde68a",
        "desc":  "800 fibers with broad curvature and strong following of the underlying direction field.",
        "params": {
            "G_align":       0.2,
            "L_align":       1.0,
            "G_curve":       0.7,
            "L_curve":       0.0,
            "L_conn":        0.3,
            "L_wave_freq":   0.25,
            "num_fibers":    800,   # capped at slider max 800 (orig 3000)
            "spline_length": 70,
            "thickness":     2.5,
        },
    },
    "INTERESTING": { # syn.generate_synthetic_shg(seed=10, G_align=0.2, G_density=1, G_curve=1, G_conn=1, L_align=0.6, L_density=0.55, L_conn=1, L_curve=0, spline_length=40, spline_num=3000)
        "label": "Dense curved network",
        "color": "#a78bfa",
        "desc":  "3,000 short fibers with strong broad curvature and weak overall alignment.",
        "params": {
            "G_align":       0.2,
            "L_align":       0.6,
            "G_curve":       1.0,
            "L_curve":       0.0,
            "L_conn":        1.0,
            "L_wave_freq":   0.25,
            "num_fibers":    3000,
            "spline_length": 40,
            "thickness":     2.5,
        },
    },
    "CYAN": { #syn.generate_synthetic_shg(seed=10, G_align=0.95, G_density=0.7, G_curve=0, G_conn=1, L_align=0.6, L_density=0.4, L_conn=1, L_curve=0.6, spline_length=60, spline_num=1200, wave_amplitude_px=6, wave_wavelength_px=30, L_wave_freq=0.5)

        "label": "Aligned wavy fibers",
        "color": "#22d3ee",
        "desc":  "1,200 short fibers with strong overall alignment and more local waviness.",
        "params": {
            "G_align":       0.95,
            "L_align":       0.6,
            "G_curve":       0.0,
            "L_curve":       0.6,
            "L_conn":        1.0,
            "L_wave_freq":   0.25,
            "num_fibers":    1200,
            "spline_length": 60,
            "thickness":     2.5,
        },
    },
    # "ORANGE": {
    #     "label": "ORANGE",
    #     "color": "#fb923c",
    #     "desc":  "Balanced : test configuration",
    #     "params": {
    #         "G_align":       0.5,
    #         "L_align":       0.5,
    #         "G_curve":       0.5,
    #         "L_curve":       0.0,
    #         "L_conn":        0.3,
    #         "L_wave_freq":   0.25,
    #         "num_fibers":    1200,
    #         "spline_length": 50,
    #         "thickness":     2.5,
    #     },
    # },
}

PARAMETER_KEYS = ("num_fibers", "spline_length", "thickness", "G_align", "L_align",
                  "G_curve", "L_curve", "L_wave_freq", "n_layers", "depth", "focal_plane", "seed")

def apply_preset(key):
    p = PRESETS[key]["params"]
    for k, v in p.items():
        st.session_state[k] = v
    st.session_state.applied_preset = key
    st.session_state.preset_snapshot = {k: st.session_state[k] for k in PARAMETER_KEYS}

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT : Parameters
# ══════════════════════════════════════════════════════════════════════════════
with settings_area:
    selected_preset = st.session_state.get("applied_preset")
    snapshot = st.session_state.get("preset_snapshot", {})
    preset_modified = bool(selected_preset) and any(st.session_state[k] != snapshot.get(k) for k in PARAMETER_KEYS)
    status = (PRESETS[selected_preset]["label"] + (" (modified)" if preset_modified else "")) if selected_preset else "Custom settings"
    with parameter_actions:
        st.markdown(f'<div class="preset-status" role="status" aria-live="polite"><span class="status-label">Current:</span><span class="status-value" title="{escape(status, quote=True)}">{escape(status)}</span></div>', unsafe_allow_html=True)
    with st.expander("Custom presets", expanded=False):
        st.caption("Select a preset to apply it. Painted fields and guides are kept.")
        for preset_key, preset in PRESETS.items():
            preset_button, preset_help = st.columns([8, 1], gap="small")
            with preset_button:
                st.button(preset["label"], key="preset_" + preset_key,
                          type="primary" if selected_preset == preset_key and not preset_modified else "secondary",
                          on_click=apply_preset, args=(preset_key,), use_container_width=True)
            with preset_help:
                preset_description(preset["label"], preset["desc"])

    with parameter_group("Fiber", "Set the number, length and thickness of fibers."):
        parameter_slider("Fiber count", 20, 3000, step=10, key="num_fibers",
            help="Number of fibers to generate. More fibers usually make the image denser and take longer to render. Their starting positions follow the density map.")
        parameter_slider("Fiber length", 20, 400, step=5, key="spline_length",
            help="Integration length used to grow each fiber on the simulation grid. Higher values produce longer paths; boundaries can shorten them. This is not a calibrated physical length.")
        parameter_slider("Fiber thickness", 0.5, 10.0, step=0.5, key="thickness",
            help="Width used when drawing fibers into the SHG image. Higher values make brighter bands appear thicker without changing the underlying fiber paths.")

    with parameter_group("Alignment", "Control how consistently fibers point in the same direction and how closely they follow the field."):
        parameter_slider("Overall alignment", 0.0, 1.0, step=0.01, key="G_align",
            help="Controls variation in the generated direction field. Lower values produce more varied directions; higher values make directions more consistent. Painted directions can override the generated field locally.")
        parameter_slider("Follow direction field", 0.0, 1.0, step=0.01, key="L_align",
            help="Controls how strongly fibers follow the underlying direction field, and also affects field relaxation. Lower values allow more independent paths; higher values follow the field more closely.")
        parameter_slider("Broad curvature", 0.0, 1.0, step=0.01, key="G_curve",
            help="Bends the generated direction field across the whole image. Lower values favor a straighter field; higher values introduce larger curved patterns. Separate from small waves along a fiber.")

    with parameter_group("Texture", "Adjust small bends and how waviness varies across the network."):
        parameter_slider("Fiber waviness", 0.0, 1.0, step=0.01, key="L_curve",
            help="Controls local bending, sinusoidal offsets and spline smoothing. Increasing this adds more small-scale waviness; it does not set a wavelength in physical units.")
        parameter_slider("Wave variation", 0.0, 1.0, step=0.01, key="L_wave_freq",
            help="Controls the spatial field used to vary the waves along fibers. It changes the distribution and clustering of the wavelength factor, rather than specifying one wavelength for every fiber.")

    with parameter_group("Depth and focus", "Distribute fibers across depth planes and control blur away from the focal plane."):
        st.caption("A layered projection with depth-dependent blur.")
        parameter_slider("Fiber layers", 1, 20, step=1, key="n_layers",
            help="Splits the existing fibers into this many planes. It does not multiply the fiber count. Each plane is blurred according to its distance from the focal plane.")
        parameter_slider("Depth range", 0.5, 50.0, step=0.1, key="depth",
            help="Model depth range used to space the layers and calculate defocus blur. Higher values increase separation and blur away from focus. These are model units, not calibrated micrometers.")
        parameter_slider("Focus position", 0.0, 1.0, step=0.1, key="focal_plane",
            help="Position within the depth range: 0 is the front and 1 is the back. With one layer, the layer is at the front; moving focus away from 0 adds blur.")

    with parameter_group("Random seed", "Use the same seed and settings to reproduce a network. Change the seed for a different network."):
        parameter_number("Random seed", step=1, key="seed",
            help="Repeat a seed with the same fields, guides and parameters to reproduce the same generated network. Change it to generate another realization.")


# ══════════════════════════════════════════════════════════════════════════════
# MIDDLE : Render & Output
# ══════════════════════════════════════════════════════════════════════════════
with col_mid:
    with output_actions:
        render_btn = st.button("Render image", type="primary", use_container_width=True)

    output_slot = st.empty()
    if render_btn:
        output_slot.markdown('<div class="output-empty" role="status" aria-live="polite">Generating SHG image…</div>', unsafe_allow_html=True)
        with st.container():
            rng   = np.random.default_rng(int(st.session_state.seed))
            X, Y  = create_grid(CANVAS_SIZE, resolution_factor=RESOLUTION_FACTOR)
            shape = X.shape

            # well_configs = [{"x":w["x"],"y":w["y"],"w":w.get("rx",45),"h":w.get("ry",28),"angle":w.get("angle",0)}
            #                 for w in st.session_state.wells]
            # _, wells_arr, hard_mask = generate_custom_fields_from_canvas(
            #     density_configs=[], well_configs=well_configs, shape=shape)

            # D_raw = st.session_state.density_arr.reshape(CANVAS_SIZE, CANVAS_SIZE).astype(np.float64)
            # if D_raw.max() < 1e-4: D_raw = np.full((CANVAS_SIZE, CANVAS_SIZE), 0.3)
            # D = np.where(hard_mask, 0.0, D_raw)

            # # Global orientation field
            # # base_field = np.atan2(st.session_state.vec_qx_arr,st.session_state.vec_qy_arr)
            # # basex = np.array(st.session_state.vec_qx_arr)
            # # basey = np.array(st.session_state.vec_qy_arr)
            # # basemag = np.array(st.session_state.vec_mag_arr)
            # # basex = basex.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # # basey = basey.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # # basemag = basemag.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # # Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng,
            # #                                      basex=basex,
            # #                                      basey =basey,
            # #                                      basemag = basemag
            # #                                     )

            # Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng,)
            
            # print(Qx_g.shape, Qy_g.shape)
            # # Blend with user-painted vector field
            # Qx_g, Qy_g = blend_orientation(
            #     Qx_g, Qy_g,
            #     st.session_state.vec_qx_arr,
            #     st.session_state.vec_qy_arr,
            #     st.session_state.vec_mag_arr,
            # )
            
            D_raw = upsample_canvas(st.session_state.density_arr, CANVAS_SIZE, shape)
            if D_raw.max() < 1e-4:
                D_raw = np.full(shape, 0.3)

            vec_qx_up  = upsample_canvas(st.session_state.vec_qx_arr,  CANVAS_SIZE, shape)
            vec_qy_up  = upsample_canvas(st.session_state.vec_qy_arr,  CANVAS_SIZE, shape)
            vec_mag_up = upsample_canvas(st.session_state.vec_mag_arr,  CANVAS_SIZE, shape)
            # Re-normalize the upsampled Q-field (bilinear interp breaks unit length)
            vec_q_norm = np.sqrt(vec_qx_up**2 + vec_qy_up**2)
            vec_q_norm = np.where(vec_q_norm < 1e-8, 1.0, vec_q_norm)
            vec_qx_up /= vec_q_norm
            vec_qy_up /= vec_q_norm

            # ── Wells: scale pixel coords from canvas space to grid space ──────────
            scale = RESOLUTION_FACTOR
            well_configs = [
                {
                    "x":     w["x"]           * scale,
                    "y":     w["y"]           * scale,
                    "w":     w.get("rx", 45)  * scale,
                    "h":     w.get("ry", 28)  * scale,
                    "angle": w.get("angle", 0),        # angle is unitless, no scaling
                }
                for w in st.session_state.wells
            ]
            _, wells_arr, hard_mask = generate_custom_fields_from_canvas(
                density_configs=[], well_configs=well_configs, shape=shape)

            D = np.where(hard_mask, 0.0, D_raw)

            # ── Orientation field ──────────────────────────────────────────────────
            Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng)

            Qx_g, Qy_g = blend_orientation(
                Qx_g, Qy_g,
                vec_qx_up.ravel(),   # blend_orientation reshapes internally
                vec_qy_up.ravel(),
                vec_mag_up.ravel(),
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

            # wave_amplitude_px = int(st.session_state.wave_amplitude_px)
            # wave_wavelength_px = int(st.session_state.wave_wavelength_px)

            num_fibers    = int(st.session_state.num_fibers)
            spline_length = int(st.session_state.spline_length)
            seeds         = sample_seeds_from_density(D, num_fibers, L_density=0.5, rng=rng)
            aux_curve     = sample_field_at_seeds(seeds, curve_field)
            aux_conn      = sample_field_at_seeds(seeds, conn_field)
            aux_wave_freq = sample_field_at_seeds(seeds, wave_freq_field)

            splines = []
            for i, seed_pt in enumerate(seeds):
                raw = generate_fiber(Qx, Qy, seed_pt, step_size=1.0, spline_length=spline_length,
                                     L_curve=st.session_state.L_curve, susceptibility=st.session_state.L_align, rng=rng)
                off = sinusoidal_fiber_offset(
                    raw, wave_amp=aux_curve[i], wave_freq=aux_wave_freq[i], 
                    # wave_amplitude_px=wave_amplitude_px,
                    # wave_wavelength_px=wave_wavelength_px,
                    rng=rng
                )
                smoothing = 0.8 + 5.0 * (1 - st.session_state.L_curve)
                num_samples = max(100, int(4 * spline_length))
                splines.append(fit_spline(off, smoothing= smoothing, num_samples=num_samples))

            splines_np = np.array(splines, dtype=object)
            layers = np.array_split(splines_np, st.session_state.n_layers, axis=0)

            imgs = [
                rasterize_splines(H=shape[0], W=shape[1], splines=l,
                                       thickness=float(st.session_state.thickness),
                                       aux_L_conn=aux_conn, intensity_seed=int(st.session_state.seed))
                for i, l in enumerate(layers)
            ]

            raster = np.zeros_like(imgs[0])
            for i, img in enumerate(imgs):
                d_val = i*st.session_state.depth/st.session_state.n_layers
                defocus = abs(d_val - st.session_state.focal_plane*st.session_state.depth)
                sigma = float(defocus * 3)
            
                raster += gaussian_filter(img, sigma=sigma)

            st.session_state.shg_image   = raster
            st.session_state.shg_splines = splines
            st.session_state.shg_D       = D
            st.session_state.shg_Qx      = Qx
            st.session_state.shg_Qy      = Qy
            st.session_state.shg_grid_size = shape[0]
            st.session_state.active_params = dict(
                seed=int(st.session_state.seed), num_fibers=num_fibers,
                spline_length=spline_length, thickness=float(st.session_state.thickness),
                G_align=float(st.session_state.G_align), L_align=float(st.session_state.L_align),
                G_curve=float(st.session_state.G_curve), L_curve=float(st.session_state.L_curve),
                L_conn=float(st.session_state.L_conn), L_wave_freq=float(st.session_state.L_wave_freq),
                n_wells=len(st.session_state.wells), vec_painted=has_v,
            )

    with output_slot.container():
        # ── Display ────────────────────────────────────────────────────────────────
        if st.session_state.shg_image is not None:
            image_tab, paths_tab = st.tabs(["SHG image", "Fiber paths"])
            with image_tab:
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

                with st.expander("Rendered parameters"):
                    st.json(st.session_state.active_params)

            with paths_tab:
                _ss("preview_density", False)
                _ss("preview_directions", False)
                _ss("preview_splines", True)
                fig, ax = plt.subplots(figsize=(5,5), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                plot_fields(
                    ax=ax, D=st.session_state.shg_D,
                    Qx=st.session_state.shg_Qx, Qy=st.session_state.shg_Qy,
                    splines=st.session_state.shg_splines,
                    image_size=CANVAS_SIZE,
                    grid_size=st.session_state.shg_grid_size,   # add this
                    show_density=st.session_state.preview_density,
                    show_quiver=st.session_state.preview_directions,
                    show_splines=st.session_state.preview_splines,
                )
                fig.tight_layout(pad=0)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.caption("Colors distinguish paths; they do not represent a measured property.")
                # Visibility checkboxes
                cb1, cb2, cb3 = st.columns(3)
                with cb1: st.checkbox("Density", key="preview_density")
                with cb2: st.checkbox("Directions", key="preview_directions")
                with cb3: st.checkbox("Fiber paths", key="preview_splines")


                # Download orientation field PNG
                c3, c4 = st.columns(2, gap="small")
                with c3:
                    fig2, ax2 = plt.subplots(figsize=(5,5), facecolor="#0d1117")
                    ax2.set_facecolor("#0d1117")
                    plot_fields(ax=ax2, D=st.session_state.shg_D,
                                Qx=st.session_state.shg_Qx, Qy=st.session_state.shg_Qy,
                                splines=st.session_state.shg_splines, image_size=CANVAS_SIZE,
                                grid_size=st.session_state.shg_grid_size,   # add this
                                show_density=st.session_state.preview_density,
                                show_quiver=st.session_state.preview_directions,
                                show_splines=st.session_state.preview_splines)
                    fig2.tight_layout(pad=0)
                    buf2 = io.BytesIO(); fig2.savefig(buf2,format="png",bbox_inches="tight",pad_inches=0,facecolor="#0d1117")
                    plt.close(fig2); buf2.seek(0)
                    st.download_button("↓ Preview PNG", data=buf2.getvalue(),
                        file_name="fiber_paths.png", mime="image/png", use_container_width=True)
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
            <div class="output-empty">Select Render image to generate an SHG image.<br>
              An empty density map uses uniform density.
            </div>""", unsafe_allow_html=True)
        
st.markdown("""
<div style="margin-top:32px;padding-top:12px;border-top:1px solid var(--shg-border);
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
    <span style="color:var(--shg-muted);font-size:20px">
    © Bhaskar Lab · Computational Biology &amp; Machine Learning · University of Wisconsin–Madison
  </span>
  <span style="color:#cc1543;font-size:20px;font-weight:600">CBML</span>
</div>
""", unsafe_allow_html=True)