"""
SHG Fiber Simulator — Redesigned app.py
Uses a custom JS canvas component for fast, responsive painting.
"""
import base64
import io
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from app.shg_backend import (
    _well_weight_and_tangent,
    create_grid,
    fit_spline,
    generate_custom_fields_from_canvas,
    generate_fiber,
    make_fiber_aux_fields,
    make_global_orientation,
    make_wave_freq_field,
    plot_density_vector_and_splines,
    rasterize_splines,
    relax,
    sample_field_at_seeds,
    sample_seeds_from_density,
    sinusoidal_fiber_offset,
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="SHG Simulator", layout="wide", initial_sidebar_state="collapsed")

# ── Global dark theme injection ────────────────────────────────────────────────
st.markdown("""<style>
html,body,[data-testid="stApp"],[data-testid="stAppViewContainer"]{background:#0d1117!important;color:#e2e8f0!important}
[data-testid="stAppViewBlockContainer"]{padding-top:0.8rem!important}
[data-testid="stSidebar"]{background:#161b27!important;border-right:1px solid #2d3650!important}
#MainMenu,footer,header,[data-testid="stDeployButton"],[data-testid="stToolbar"],[data-testid="collapsedControl"]{display:none!important}
*{font-family:Inter,system-ui,sans-serif!important}
[data-testid="stMetric"]{background:#1e2535!important;border:1px solid #2d3650!important;border-radius:8px!important;padding:10px 14px!important}
[data-testid="stMetricLabel"]{color:#64748b!important;font-size:11px!important}
[data-testid="stMetricValue"]{color:#e2e8f0!important;font-size:20px!important}
div[data-baseweb="slider"] div[data-testid="stTickBar"]{display:none}
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
/* iframe sizing */
iframe[title="shg_canvas"]{min-height:640px!important}
</style>""", unsafe_allow_html=True)

# ── Register custom component ──────────────────────────────────────────────────
_COMPONENT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canvas_component")
_canvas_fn = components.declare_component("shg_canvas", path=_COMPONENT_DIR)

def shg_canvas(mode, canvas_size, wells, density_b64="", key=None):
    return _canvas_fn(
        mode=mode,
        canvas_size=canvas_size,
        wells=json.dumps(wells),
        density_b64=density_b64,
        key=key,
        default=None,
    )

# ── Session state init ─────────────────────────────────────────────────────────
CANVAS_SIZE = 512

def _ss(k, v):
    if k not in st.session_state:
        st.session_state[k] = v

_ss("density_arr",   np.zeros(CANVAS_SIZE * CANVAS_SIZE, dtype=np.float32))
_ss("wells",         [])
_ss("active_tab",    "density")
_ss("shg_image",     None)
_ss("shg_splines",   None)
_ss("shg_D",         None)
_ss("shg_Qx",        None)
_ss("shg_Qy",        None)
_ss("active_params", {})
_ss("num_fibers",    300)
_ss("spline_length", 150)
_ss("thickness",     2.5)
_ss("G_align",       0.50)
_ss("L_align",       0.50)
_ss("G_curve",       0.30)
_ss("L_curve",       0.50)
_ss("L_conn",        0.50)
_ss("L_wave_freq",   0.25)
_ss("seed",          42)

# ── Density encode/decode ──────────────────────────────────────────────────────
def density_to_b64(arr):
    return base64.b64encode(arr.astype(np.float32).tobytes()).decode()

def b64_to_density(b64, size):
    try:
        raw = base64.b64decode(b64)
        arr = np.frombuffer(raw, dtype=np.float32).copy()
        if arr.size == size * size:
            return arr
    except Exception:
        pass
    return np.zeros(size * size, dtype=np.float32)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;align-items:center;gap:10px;padding:4px 0 10px">
  <div style="width:9px;height:9px;border-radius:50%;background:#4ade80;box-shadow:0 0 12px #4ade80;flex-shrink:0"></div>
  <span style="font-size:15px;font-weight:700;letter-spacing:-0.02em">SHG Fiber Simulator</span>
  <span style="color:#475569;font-size:12px">Interactive Field Editor</span>
</div>
<hr>
""", unsafe_allow_html=True)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_mid, col_right = st.columns([2.1, 2.1, 1.1], gap="medium")

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT — Canvas workspace
# ═══════════════════════════════════════════════════════════════════════════════
with col_left:
    # Tab switcher (uses session state flag, no st.rerun needed — component handles
    # mode internally; we just pass the current mode on next render)
    tab_c1, tab_c2 = st.columns(2, gap="small")
    with tab_c1:
        density_btn = st.button(
            "🟢  Density Map",
            use_container_width=True,
            type="primary" if st.session_state.active_tab == "density" else "secondary",
        )
    with tab_c2:
        wells_btn = st.button(
            "🔵  Orientation Wells",
            use_container_width=True,
            type="primary" if st.session_state.active_tab == "wells" else "secondary",
        )

    if density_btn and st.session_state.active_tab != "density":
        st.session_state.active_tab = "density"
        st.rerun()
    if wells_btn and st.session_state.active_tab != "wells":
        st.session_state.active_tab = "wells"
        st.rerun()

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # Encode current density for the component
    d_b64 = density_to_b64(st.session_state.density_arr)

    # Render canvas component — single stable key so it doesn't reset between renders
    result = shg_canvas(
        mode=st.session_state.active_tab,
        canvas_size=CANVAS_SIZE,
        wells=st.session_state.wells,
        density_b64=d_b64,
        key="shg_canvas",
    )

    # Handle return value from JS
    if result is not None:
        rtype = result.get("type", "")
        if rtype == "density":
            b64 = result.get("density_b64", "")
            sz  = result.get("size", CANVAS_SIZE)
            if b64:
                st.session_state.density_arr = b64_to_density(b64, sz)
        elif rtype == "wells":
            st.session_state.wells = result.get("wells", [])

    # Legend + status
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-top:6px">
      <div style="width:88px;height:6px;border-radius:3px;
          background:linear-gradient(to right,#440154,#3b528b,#21908d,#5dc963,#fde725)"></div>
      <span style="color:#475569;font-size:10px">low → high density</span>
    </div>""", unsafe_allow_html=True)

    n_wells = len(st.session_state.wells)
    has_d   = float(st.session_state.density_arr.max()) > 1e-3
    st.markdown(
        f"<div style='color:#475569;font-size:10px;margin-top:4px'>"
        f"{'Density painted · ' if has_d else 'Density empty · '}"
        f"{n_wells} well{'s' if n_wells!=1 else ''} placed</div>",
        unsafe_allow_html=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT — Parameters
# ═══════════════════════════════════════════════════════════════════════════════
with col_right:
    def _section(label):
        st.markdown(f"<div style='color:#475569;font-size:10px;font-weight:700;letter-spacing:.07em;"
                    f"text-transform:uppercase;margin:10px 0 6px'>{label}</div>", unsafe_allow_html=True)

    _section("Fiber")
    st.session_state.num_fibers    = st.slider("Count",     20,  800, st.session_state.num_fibers,    step=10)
    st.session_state.spline_length = st.slider("Length",    20,  400, st.session_state.spline_length, step=5)
    st.session_state.thickness     = st.slider("Thickness", 0.5, 10.0,st.session_state.thickness,    step=0.5)

    st.markdown("---")
    _section("Alignment")
    st.session_state.G_align = st.slider("Global align", 0.0, 1.0, st.session_state.G_align, step=0.01)
    st.session_state.L_align = st.slider("Local relax",  0.0, 1.0, st.session_state.L_align, step=0.01)
    st.session_state.G_curve = st.slider("Global curve", 0.0, 1.0, st.session_state.G_curve, step=0.01)
    st.session_state.L_curve = st.slider("Local curve",  0.0, 1.0, st.session_state.L_curve, step=0.01)

    st.markdown("---")
    _section("Texture")
    st.session_state.L_conn      = st.slider("Connectivity", 0.0, 1.0, st.session_state.L_conn,     step=0.01)
    st.session_state.L_wave_freq = st.slider("Wobble freq",  0.0, 1.0, st.session_state.L_wave_freq, step=0.01)

    st.markdown("---")
    _section("Seed")
    st.session_state.seed = st.number_input("Random seed", value=int(st.session_state.seed), step=1)

    st.markdown("---")
    # Status card
    st.markdown(f"""
    <div style="background:#1e2535;border:1px solid #2d3650;border-radius:8px;padding:10px 12px;font-size:11px">
      <div style="color:#475569;font-size:10px;font-weight:700;letter-spacing:.06em;text-transform:uppercase;margin-bottom:6px">Status</div>
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="color:#64748b">Density</span>
        <span style="color:{'#4ade80' if has_d else '#f87171'}">{'painted' if has_d else 'empty'}</span>
      </div>
      <div style="display:flex;justify-content:space-between;margin-bottom:3px">
        <span style="color:#64748b">Wells</span>
        <span style="color:#60a5fa">{n_wells}</span>
      </div>
      <div style="display:flex;justify-content:space-between">
        <span style="color:#64748b">Canvas</span>
        <span style="color:#94a3b8">{CANVAS_SIZE}²</span>
      </div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLE — Render & Output
# ═══════════════════════════════════════════════════════════════════════════════
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

            # Build well array from canvas wells
            well_configs = [
                {"x": w["x"], "y": w["y"],
                 "w": w.get("rx", 45), "h": w.get("ry", 28),
                 "angle": w.get("angle", 0)}
                for w in st.session_state.wells
            ]

            # Get hard mask from wells (density_configs=[] since we paint directly)
            _, wells_arr, hard_mask = generate_custom_fields_from_canvas(
                density_configs=[], well_configs=well_configs, shape=shape
            )

            # Use painted density; fall back to uniform if blank
            D_raw = st.session_state.density_arr.reshape(CANVAS_SIZE, CANVAS_SIZE).astype(np.float64)
            if D_raw.max() < 1e-4:
                D_raw = np.full((CANVAS_SIZE, CANVAS_SIZE), 0.3)

            # Apply well interior masks
            D = np.where(hard_mask, 0.0, D_raw)

            # Orientation field
            Qx_g, Qy_g = make_global_orientation(shape, st.session_state.G_align, st.session_state.G_curve, rng)
            if len(wells_arr) > 0:
                Qx_w, Qy_w, influence = _well_weight_and_tangent(X, Y, wells_arr, Qx_g, Qy_g)
            else:
                Qx_w = Qy_w = np.zeros(shape)
                influence    = np.zeros(shape)

            Qx, Qy = relax(Qx_g, Qy_g, Qx_g, Qy_g, Qx_w, Qy_w, influence,
                           D, st.session_state.G_align, st.session_state.L_align,
                           wells_arr, X, Y)

            curve_field, conn_field = make_fiber_aux_fields(
                shape, st.session_state.L_curve, st.session_state.L_conn, rng)
            wave_freq_field = make_wave_freq_field(shape, st.session_state.L_wave_freq, rng)

            num_fibers    = int(st.session_state.num_fibers)
            spline_length = int(st.session_state.spline_length)

            seeds         = sample_seeds_from_density(D, num_fibers, L_density=0.5, rng=rng)
            aux_curve     = sample_field_at_seeds(seeds, curve_field)
            aux_conn      = sample_field_at_seeds(seeds, conn_field)
            aux_wave_freq = sample_field_at_seeds(seeds, wave_freq_field)

            splines = []
            for i, seed_pt in enumerate(seeds):
                raw = generate_fiber(
                    Qx, Qy, seed_pt,
                    step_size=1.0,
                    spline_length=spline_length,
                    L_curve=st.session_state.L_curve,
                    susceptibility=1.0 - aux_curve[i],
                    rng=rng,
                )
                off = sinusoidal_fiber_offset(raw, wave_amp=aux_curve[i],
                                              wave_freq=aux_wave_freq[i], rng=rng)
                splines.append(fit_spline(off, num_samples=max(50, spline_length * 2)))

            raster = rasterize_splines(
                H=shape[0], W=shape[1], splines=splines,
                thickness=float(st.session_state.thickness),
                aux_L_conn=aux_conn,
                intensity_seed=int(st.session_state.seed),
            )

            st.session_state.shg_image   = raster
            st.session_state.shg_splines = splines
            st.session_state.shg_D       = D
            st.session_state.shg_Qx      = Qx
            st.session_state.shg_Qy      = Qy
            st.session_state.active_params = {
                "seed":          int(st.session_state.seed),
                "num_fibers":    num_fibers,
                "spline_length": spline_length,
                "thickness":     float(st.session_state.thickness),
                "G_align":       float(st.session_state.G_align),
                "L_align":       float(st.session_state.L_align),
                "G_curve":       float(st.session_state.G_curve),
                "L_curve":       float(st.session_state.L_curve),
                "L_conn":        float(st.session_state.L_conn),
                "L_wave_freq":   float(st.session_state.L_wave_freq),
                "n_wells":       len(st.session_state.wells),
            }

    # ── Display ────────────────────────────────────────────────────────────────
    if st.session_state.shg_image is not None:
        img_arr = st.session_state.shg_image
        st.image(img_arr, clamp=True, use_container_width=True)

        c1, c2 = st.columns(2, gap="small")
        with c1:
            norm = img_arr if img_arr.dtype == np.uint8 else (
                (img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8) * 255
            ).astype(np.uint8)
            buf = io.BytesIO()
            Image.fromarray(norm).save(buf, format="PNG")
            st.download_button(
                "↓ PNG",
                data=buf.getvalue(),
                file_name=f"shg_{st.session_state.active_params.get('seed',0)}.png",
                mime="image/png",
                use_container_width=True,
            )
        with c2:
            st.download_button(
                "↓ JSON",
                data=json.dumps({
                    "parameters": st.session_state.active_params,
                    "splines": [s.tolist() for s in (st.session_state.shg_splines or [])],
                }, indent=2),
                file_name=f"shg_splines_{st.session_state.active_params.get('seed',0)}.json",
                mime="application/json",
                use_container_width=True,
            )

        with st.expander("Parameters"):
            st.json(st.session_state.active_params)

        # Orientation field preview
        st.markdown("---")
        st.markdown("<div style='color:#475569;font-size:10px;font-weight:700;letter-spacing:.07em;"
                    "text-transform:uppercase;margin-bottom:6px'>Orientation Field</div>",
                    unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 5), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        plot_density_vector_and_splines(
            ax=ax,
            D=st.session_state.shg_D,
            Qx=st.session_state.shg_Qx,
            Qy=st.session_state.shg_Qy,
            splines=st.session_state.shg_splines,
            image_size=CANVAS_SIZE,
        )
        fig.tight_layout(pad=0)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.markdown("""
        <div style="background:#161b27;border:1px solid #2d3650;border-radius:8px;
             padding:40px 20px;text-align:center;margin-top:8px">
          <div style="font-size:32px;margin-bottom:10px;opacity:0.4">⟳</div>
          <div style="color:#64748b;font-size:12px;line-height:1.6">
            Paint a density field and<br>optionally place wells,<br>
            then click <b style="color:#4ade80">Render</b>.
          </div>
        </div>""", unsafe_allow_html=True)