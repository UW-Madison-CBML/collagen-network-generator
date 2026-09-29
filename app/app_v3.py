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

from typing import List, Tuple, Optional, Dict, Any

from typing import Optional

import sys
from pathlib import Path

current_dir = Path(__file__).resolve().parent

project_root = current_dir.parent 

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from synthetic_code.VectorField import *
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
_ss("show_density", True)
_ss("show_vectors", True)
_ss("show_splines", True)
_ss("show_quiver",  True)
_ss("wave_amplitude_px", 2.8)
_ss("wave_wavelength_px", None)

_ss("n_layers",  1)
_ss("depth", 1.0)
_ss("focal_plane", 0.0)

_ss("dark_mode", True)

if st.session_state.dark_mode:
    BG       = "#0d1117"
    SURFACE  = "#161b27"
    PANEL    = "#1e2535"
    BORDER   = "#2d3650"
    TEXT     = "#e2e8f0"
    MUTED    = "#64748b"
else:
    BG       = "#ffffff"
    SURFACE  = "#f8fafc"
    PANEL    = "#f1f5f9"
    BORDER   = "#cbd5e1"
    TEXT     = "#0f172a"
    MUTED    = "#64748b"

ACCENT = "#cc1543"


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHG Simulator", layout="wide", initial_sidebar_state="collapsed",page_icon="assets/uw-logo-vertical-color-web-digital.png")
# st.markdown("""<style>
# html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:#0d1117!important;color:#e2e8f0!important}
# [data-testid="stAppViewBlockContainer"]{padding-top:0.8rem!important}
# #MainMenu,footer,header,[data-testid="stDeployButton"],[data-testid="stToolbar"],[data-testid="collapsedControl"]{display:none!important}
# *{font-family:Inter,system-ui,sans-serif!important}
# [data-testid="stSlider"] label{color:#94a3b8!important;font-size:12px!important}
# [data-testid="stButton"]>button{background:#1e2535!important;border:1.5px solid #2d3650!important;color:#94a3b8!important;border-radius:7px!important;transition:all .15s!important}
# [data-testid="stButton"]>button:hover{border-color:#4ade8066!important;color:#4ade80!important}
# button[kind="primary"]{background:#4ade8022!important;border-color:#4ade80!important;color:#4ade80!important;font-weight:700!important}
# button[kind="primary"]:hover{background:#4ade8033!important}
# [data-testid="stExpander"]{background:#161b27!important;border:1px solid #2d3650!important;border-radius:8px!important}
# [data-testid="stExpander"] summary{color:#94a3b8!important}
# hr{border-color:#2d3650!important;margin:0.6rem 0!important}
# [data-testid="stDownloadButton"]>button{background:#60a5fa18!important;border:1.5px solid #60a5fa!important;color:#60a5fa!important;border-radius:7px!important;width:100%!important}
# [data-testid="stNumberInput"] input{background:#1e2535!important;border:1px solid #2d3650!important;color:#e2e8f0!important;border-radius:6px!important}
# [data-testid="stCheckbox"] label{color:#94a3b8!important;font-size:12px!important}
# iframe[title="shg_canvas"]{min-height:660px!important}
# </style>""", unsafe_allow_html=True)

st.markdown(f"""
<style>
  .brand-bar {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 8px 0 10px;
    border-bottom: 1px solid #2d3650;
    margin-bottom: 10px;
  }}
  .brand-left {{
    display: flex;
    align-items: center;
    gap: 14px;
  }}
  .brand-divider {{
    width: 1px;
    height: 36px;
    background: #cc154366;
    flex-shrink: 0;
  }}
  .brand-lab {{
    display: flex;
    flex-direction: column;
    gap: 1px;
  }}
  .brand-lab-name {{
    font-size: 13px;
    font-weight: 700;
    color: #cc1543;
    letter-spacing: -0.01em;
    line-height: 1.2;
  }}
  .brand-lab-sub {{
    font-size: 10px;
    color: #94a3b8;
    font-weight: 400;
    letter-spacing: 0.02em;
  }}
  .brand-app {{
    display: flex;
    flex-direction: column;
    gap: 2px;
  }}
  .brand-app-title {{
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #f8fafc;
  }}
  .brand-app-sub {{
    font-size: 11px;
    color: #64748b;
  }}
  .brand-right img {{
    height: 40px;
    opacity: 0.9;
  }}
  /* Tint the active-preset indicator and primary button to match UW red */
  button[kind="primary"] {{
    background: #cc154322 !important;
    border-color: #cc1543 !important;
    color: #cc1543 !important;
  }}
  button[kind="primary"]:hover {{
    background: #cc154333 !important;
  }}
  /* Tint the render-button glow */
  .stButton > button:hover {{
    border-color: #cc154366 !important;
    color: #cc1543 !important;
  }}
</style>

<div class="brand-bar">
  <div class="brand-left">
    <img src="data:image/png;base64,{LOGO_B64_horizontal}" style="height:55px;opacity:0.95" alt="UW–Madison">
    <div class="brand-divider"></div>
    <div class="brand-lab">
      <div class="brand-lab-name">CBML</div>
      <div class="brand-lab-sub">Computational Biology &amp; Machine Learning</div>
      <div class="brand-lab-sub" style="color:#475569">Bhaskar Lab · UW–Madison</div>
    </div>
    <div class="brand-divider"></div>
    <div class="brand-app">
      <div class="brand-app-title" style="color:#475569">SHG Fiber Simulator</div>
      <div class="brand-app-sub">Interactive Field Editor</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# st.markdown(f"""<style>
# html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{{background:{BG}!important;color:{TEXT}!important}}
# [data-testid="stAppViewBlockContainer"]{{padding-top:0.8rem!important}}
# #MainMenu,footer,header,[data-testid="stDeployButton"],[data-testid="stToolbar"],[data-testid="collapsedControl"]{{display:none!important}}
# *{{font-family:Inter,system-ui,sans-serif!important}}
# [data-testid="stSlider"] label{{color:{MUTED}!important;font-size:12px!important}}
# [data-testid="stButton"]>button{{background:{PANEL}!important;border:1.5px solid {BORDER}!important;color:{MUTED}!important;border-radius:7px!important;transition:all .15s!important}}
# [data-testid="stButton"]>button:hover{{border-color:{ACCENT}66!important;color:{ACCENT}!important}}
# button[kind="primary"]{{background:{ACCENT}22!important;border-color:{ACCENT}!important;color:{ACCENT}!important;font-weight:700!important}}
# button[kind="primary"]:hover{{background:{ACCENT}33!important}}
# [data-testid="stExpander"]{{background:{SURFACE}!important;border:1px solid {BORDER}!important;border-radius:8px!important}}
# [data-testid="stExpander"] summary{{color:{MUTED}!important}}
# hr{{border-color:{BORDER}!important;margin:0.6rem 0!important}}
# [data-testid="stDownloadButton"]>button{{background:#60a5fa18!important;border:1.5px solid #60a5fa!important;color:#60a5fa!important;border-radius:7px!important;width:100%!important}}
# [data-testid="stNumberInput"] input{{background:{PANEL}!important;border:1px solid {BORDER}!important;color:{TEXT}!important;border-radius:6px!important}}
# [data-testid="stCheckbox"] label{{color:{MUTED}!important;font-size:12px!important}}
# iframe[title="shg_canvas"]{{min-height:660px!important}}
# .tip-wrap{{position:relative;display:inline-flex;align-items:center;gap:5px;font-size:14px;color:{TEXT};font-weight:400;letter-spacing:0;text-transform:none;margin-bottom:4px}}
# .tip{{position:relative;display:inline-flex;align-items:center;justify-content:center;width:14px;height:14px;border-radius:50%;border:1px solid {BORDER};color:{MUTED};font-size:9px;font-weight:700;cursor:default;flex-shrink:0}}
# .tip:hover::after{{content:attr(data-tip);position:absolute;right:0;top:20px;background:{PANEL};border:1px solid {BORDER};border-radius:6px;padding:6px 10px;font-size:11px;color:{TEXT};width:max-content;max-width:min(320px,90vw);white-space:normal;word-wrap:break-word;line-height:1.5;pointer-events:none;z-index:9999;font-weight:400;text-transform:none;letter-spacing:0}}
# </style>""", unsafe_allow_html=True)

# Add this once, e.g. right after your existing st.markdown("""<style>...""") call
st.markdown("""
<style>
.tip-wrap {
    position: relative;      /* establishes positioning context for the tooltip */
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 14px;
    color: #475569;
    font-weight: 400;
    letter-spacing: 0;
    text-transform: none;
    margin-bottom: 4px;
}

.tip {
    position: relative;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 14px;
    height: 14px;
    border-radius: 50%;
    border: 1px solid #475569;
    color: #94a3b8;
    font-size: 9px;
    font-weight: 700;
    cursor: default;
    flex-shrink: 0;
}

.tip:hover::after {
    content: attr(data-tip);
    position: absolute;
    right: 0;
    top: 20px;
    background: #1e2535;
    border: 1px solid #2d3650;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 11px;
    color: #e2e8f0;
    width: max-content;          /* grow to fit the text naturally */
    max-width: min(320px, 90vw); /* but never wider than 320px or 90% of viewport */
    white-space: normal;
    word-wrap: break-word;
    line-height: 1.5;
    pointer-events: none;
    z-index: 9999;
    font-weight: 400;
    text-transform: none;
    letter-spacing: 0;
}
/* Flip left when near right edge */
 .tip.flip:hover::after { left: auto; right: 18px; }
</style>
""", unsafe_allow_html=True)

st.markdown("<style>[data-testid='stSlider'] label { display:none !important; }</style>", unsafe_allow_html=True)

# Helper — replaces your _sec() calls where you want a tooltip
def _sec_tip(label, tip, flip=False):
    flip_cls = " flip" if flip else ""
    st.markdown(
        f"<div class='tip-wrap'>{label}"
        f"<span class='tip{flip_cls}' data-tip='{tip}'>?</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

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

    # Renormalize — a linear blend of unit vectors is not itself a unit vector
    norm = np.sqrt(Qx**2 + Qy**2)
    norm = np.where(norm < 1e-8, 1.0, norm)
    return Qx / norm, Qy / norm

# ── Custom orientation field plot ─────────────────────────────────────────────
def plot_fields(ax, D, Qx, Qy, splines, image_size,
                show_density=True, show_quiver=True, show_splines=True):
    ax.set_xlim(0, image_size); ax.set_ylim(image_size, 0); ax.axis("off")

    if show_density:
        ax.imshow(D, cmap="magma", origin="upper",
                  extent=[0, image_size, image_size, 0], alpha=1.0)

    if show_quiver and Qx is not None:
        shape = Qx.shape
        step = max(1, image_size // 24)
        rows = np.arange(step//2, shape[0], step)
        cols = np.arange(step//2, shape[1], step)
        R, C = np.meshgrid(rows, cols, indexing="ij")
        theta = 0.5 * np.arctan2(Qy[R, C], Qx[R, C])
        u =  np.cos(theta)
        v = -np.sin(theta)
        ax.quiver(C, R, u, v,
                  color="#f59e0b",      # amber — distinct from white splines
                  headlength=4,         # restore arrowhead
                  headaxislength=3.5,
                  headwidth=3,
                  pivot="middle",
                  scale=28,
                  alpha=0.75,
                  width=0.004)

    if show_splines and splines:
        for sp in splines:
            if len(sp) > 0:
                ax.plot(sp[:, 1], sp[:, 0], color="white", linewidth=1.0, alpha=0.75)

    return ax

# ── Header ─────────────────────────────────────────────────────────────────────
# st.markdown("""
# <div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px">
#   <div style="width:9px;height:9px;border-radius:50%;background:#cc1543;box-shadow:0 0 12px #cc154388;flex-shrink:0"></div>
#   <span style="font-size:15px;font-weight:700;letter-spacing:-0.02em">SHG Fiber Simulator</span>
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

# Toggle button — sits in top-right via Streamlit columns trick
# _, toggle_col = st.columns([0.85, 0.15])
# with toggle_col:
#     label = "☀️ Light" if st.session_state.dark_mode else "🌙 Dark"
#     if st.button(label, key="theme_toggle", use_container_width=True):
#         st.session_state.dark_mode = not st.session_state.dark_mode
#         st.rerun()

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

    #syn.generate_synthetic_shg(seed=11, G_align=0.7, G_density=0.7, G_curve=0.5, G_conn=1, L_align=0.6, L_density=0, L_conn=0.3, L_curve=0, L_intensity=0.5, spline_length=50, spline_num=1200)
    #
    "RED": {
        "label": "Type 1 (Cancerous Tending)",
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
    "YELLOW_BAD": { # syn.generate_synthetic_shg(seed=10, G_align=0.4, G_density=0.7, G_curve=0, G_conn=1, L_align=0.3, L_density=0.4, L_conn=1, L_curve=0, spline_length=60, spline_num=1200)

        "label": "Type 2 (Non-cancerous Tending)",
        "color": "#fbbf24",
        "desc":  "Low alignment, moderate density, short fibers",
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
        "label": "Type 3 (Non-cancerous Tending)",
        "color": "#fde68a",
        "desc":  "Very high density, moderate curve, many short fibers",
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
        "label": "Type 4 (Non-cancerous Tending)",
        "color": "#a78bfa",
        "desc":  "Balanced alignment and density",
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

        "label": "Type 5 (Cancerous Tending)",
        "color": "#22d3ee",
        "desc":  "",
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
    #     "desc":  "Balanced — test configuration",
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

def apply_preset(key):
    p = PRESETS[key]["params"]
    for k, v in p.items():
        st.session_state[k] = v

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — Parameters
# ══════════════════════════════════════════════════════════════════════════════
with col_right:
    def _sec(label):
        st.markdown(f"<div style='color:#010408;font-size:10px;font-weight:700;letter-spacing:.07em;"
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
    
    def _tip(label, tip):
        st.markdown(
            f"<div class='tip-wrap'>{label}"
            f"<span class='tip' data-tip='{tip}'>?</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    _sec("Fiber")
    # _sec_tip("Fiber", "Controls the number, length and thickness of individual fibers")
    _tip("Count", "Total number of fibers")
    st.session_state.num_fibers    = st.slider("Count",     20,  3000, st.session_state.num_fibers,    step=10)
    _tip("Length", "Length of each fiber")
    st.session_state.spline_length = st.slider("Length",    20,  400, st.session_state.spline_length, step=5)
    _tip("Thickness", "Thickness of each fiber")
    st.session_state.thickness     = st.slider("Thickness", 0.5, 10.0,st.session_state.thickness,    step=0.5)
    # st.session_state.wave_amplitude_px     = st.slider("Wave amplitude", 0.5, 5.0,st.session_state.wave_amplitude_px,    step=0.5)
    # st.session_state.wave_wavelength_px     = st.slider("Wave wavelength", 0.5, .0,st.session_state.wave_wavelength_px,    step=0.5)

    st.markdown("---")
    _sec("Alignment")
    # _sec_tip("Alignment", "G_align: how parallel fibers are globally. L_align: how much local neighborhood smoothing is applied. Curve parameters control bending. Local Curve determines small scale waviness.")
    _tip("Global align", "how parallel fibers are globally")
    st.session_state.G_align = st.slider("Global align", 0.0, 1.0, st.session_state.G_align, step=0.01)
    _tip("Local align", "Alignment with underyling vector field")
    st.session_state.L_align = st.slider("Local align",  0.0, 1.0, st.session_state.L_align, step=0.01)
    _tip("Global curve", "Large scale curvature")
    st.session_state.G_curve = st.slider("Global curve", 0.0, 1.0, st.session_state.G_curve, step=0.01)
    _tip("Local curve", "small scale waviness")
    st.session_state.L_curve = st.slider("Local curve",  0.0, 1.0, st.session_state.L_curve, step=0.01)
    st.markdown("---")
    _sec("Texture")
    # _sec_tip("Texture", "Wavelength: spatial frequency of local fiber waves")
    # st.session_state.L_conn      = st.slider("Connectivity", 0.0, 1.0, st.session_state.L_conn,      step=0.01)
    _tip("Wavelength", "spatial frequency of local fiber waves")
    st.session_state.L_wave_freq = st.slider("Wavelength",  0.0, 1.0, st.session_state.L_wave_freq, step=0.01)
    st.markdown("---")
    _tip("Seed", "Seed value used for psuedo random number generation")
    st.session_state.seed = st.number_input("Random seed", value=int(st.session_state.seed), step=1)
    _sec("3D Layers")
    # _sec_tip("3D layers", "Z depth is the overall depth of the sample. Layers: the number of distinct planes of fibers (default 1). Focal plane %: the position of the focal plane as a percentage of the total depth")
    _tip("Layers","The number of distinct planes of fibers that are equally spaced in depth (default 1)")
    st.session_state.n_layers = st.slider("Layers", 1,20, value=int(st.session_state.n_layers), step=1)
    _tip("Z depth","Overall depth of the sample")
    st.session_state.depth = st.slider("Z depth", 0.5,50.0, value=(st.session_state.depth), step=0.1)
    _tip("Focal plane %","The position of the focal plane as a percentage of the total depth")
    st.session_state.focal_plane = st.slider("Focal plane %", 0.0,1.0,value=(st.session_state.focal_plane), step=0.1)

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
            # base_field = np.atan2(st.session_state.vec_qx_arr,st.session_state.vec_qy_arr)
            # basex = np.array(st.session_state.vec_qx_arr)
            # basey = np.array(st.session_state.vec_qy_arr)
            # basemag = np.array(st.session_state.vec_mag_arr)
            # basex = basex.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # basey = basey.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # basemag = basemag.reshape(CANVAS_SIZE,CANVAS_SIZE)
            # Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng,
            #                                      basex=basex,
            #                                      basey =basey,
            #                                      basemag = basemag
            #                                     )

            Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng,)

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
        
st.markdown("""
<div style="margin-top:32px;padding-top:12px;border-top:1px solid #2d3650;
     display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px">
  <span style="color:#334155;font-size:10px">
    © Bhaskar Lab · Computational Biology &amp; Machine Learning · University of Wisconsin–Madison
  </span>
  <span style="color:#cc1543;font-size:10px;font-weight:600">CBML</span>
</div>
""", unsafe_allow_html=True)