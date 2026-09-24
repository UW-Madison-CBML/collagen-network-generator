import io
import json
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go

try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    CANVAS_AVAILABLE = False

from app.shg_backend import (
    create_grid,
    make_wells,
    make_density,
    make_global_orientation,
    _well_weight_and_tangent,
    relax,
    make_fiber_aux_fields,
    make_wave_freq_field,
    sample_seeds_from_density,
    sample_field_at_seeds,
    generate_fiber,
    fit_spline,
    rasterize_splines,
    generate_custom_fields_from_canvas,
    plot_density_vector_and_splines
)

st.set_page_config(page_title="Interactive Synthetic SHG Image Simulator", layout="wide")
st.title("Interactive Wells & Density Map SHG Simulator")

if not CANVAS_AVAILABLE:
    st.error("Please run `pip install streamlit-drawable-canvas` to enable interactive point/well placement!")

# =====================================================================
# SIDEBAR CONTROLS
# =====================================================================
st.sidebar.header("Mode")
mode = st.sidebar.radio("Input Mode", ["Random Generation", "Interactive GUI"])

manual_seed = st.sidebar.number_input("Base Random Seed", value=777, step=1)
image_size = st.sidebar.slider("Canvas Dimensions (px)", 256, 1024, 512, step=128)

st.sidebar.markdown("---")
st.sidebar.subheader("Fiber & Alignment Controls")
st.sidebar.caption("*(Used manually in Canvas Mode or overridden randomly in Procedural Mode)*")

num_fibers = st.sidebar.slider("Number of Fibers", 10, 1000, 500, step=20)
spline_length = st.sidebar.slider("Spline Length", 20, 400, 150, step=10)
thickness = st.sidebar.slider("Fiber Thickness (px)", 1.0, 10.0, 2.5, step=0.5)

G_align = st.sidebar.slider("Global Alignment", 0.0, 1.0, 0.5)
L_align = st.sidebar.slider("Local Relaxation (L_align)", 0.0, 1.0, 0.5)
G_curve = st.sidebar.slider("Global Curvature (G_curve)", 0.0, 1.0, 0.5)

L_curve = st.sidebar.slider("Local Fiber Curvature", 0.0, 1.0, 0.5)
L_conn = st.sidebar.slider("Local Connectivity Gaps", 0.0, 1.0, 0.5)
L_wave_freq = st.sidebar.slider("Wobble Wave Frequency", 0.0, 1.0, 0.25)

# Mesh grid setup
X, Y = create_grid(image_size, resolution_factor=1.0)
shape = X.shape

preview_rng = np.random.default_rng(manual_seed)

# =====================================================================
# MAIN WORKSPACE & PER-POINT CONTROLS
# =====================================================================

wells = np.empty((0, 5))
D = np.zeros(shape)
hard_zero_mask = np.zeros(shape, dtype=bool)

if mode == "Random Generation" or not CANVAS_AVAILABLE:
    st.sidebar.subheader("Procedural Settings")
    G_density = st.sidebar.slider("Global Density", 0.0, 1.0, 0.5)
    L_density = st.sidebar.slider("Local Density Noise", 0.0, 1.0, 0.5)
    G_conn = st.sidebar.slider("Well Connectivity", 0.0, 1.0, 0.5)

    wells, W, hard_zero_mask = make_wells(X, Y, G_conn, preview_rng)
    D = make_density(W, hard_zero_mask, G_density, L_density, preview_rng)

else:
    st.subheader("Canvas Point & Well Placement")
    
    placement_type = st.radio(
        "Placement Tool:", 
        ["🔴 Density Center Point", "🔵 Structural Orientation Well"], 
        horizontal=True
    )

    stroke_col = "#FF0000" if "Density" in placement_type else "#0000FF"

    col_canvas, col_settings = st.columns([1, 1])

    with col_canvas:
        canvas_result = st_canvas(
            fill_color=stroke_col,
            stroke_color=stroke_col,
            stroke_width=8,
            background_color="#000000",
            height=image_size,
            width=image_size,
            drawing_mode="point",
            key="shg_interactive_canvas",
        )

    raw_density_pts = []
    raw_well_pts = []

    if canvas_result is not None and canvas_result.json_data is not None:
        objects = canvas_result.json_data["objects"]
        for obj in objects:
            if obj.get("type") == "circle":
                x_px = float(obj.get("left", 0) + obj.get("radius", 0))
                y_px = float(obj.get("top", 0) + obj.get("radius", 0))
                
                color = obj.get("stroke", "").upper()
                if color == "#FF0000":
                    raw_density_pts.append({"x": x_px, "y": y_px})
                else:
                    raw_well_pts.append({"x": x_px, "y": y_px})

    density_configs = []
    well_configs = []

    with col_settings:
        tab_density, tab_wells = st.tabs(["🔴 Density Points", "🔵 Orientation Wells"])

        with tab_density:
            if not raw_density_pts:
                st.info("Click on the canvas (with 🔴 tool selected) to add density points.")
            for idx, pt in enumerate(raw_density_pts):
                with st.expander(f"🔴 Density Pt #{idx + 1} at ({int(pt['x'])}, {int(pt['y'])})", expanded=False):
                    c1, c2 = st.columns(2)
                    with c1:
                        w_val = st.slider(f"Width (Sigma X)", 5.0, 200.0, 45.0, key=f"d_w_{idx}")
                        intensity_val = st.slider(f"Intensity", 0.1, 5.0, 1.0, key=f"d_int_{idx}")
                    with c2:
                        h_val = st.slider(f"Height (Sigma Y)", 5.0, 200.0, 45.0, key=f"d_h_{idx}")
                    
                    density_configs.append({"x": pt["x"], "y": pt["y"], "w": w_val, "h": h_val, "intensity": intensity_val})

        with tab_wells:
            if not raw_well_pts:
                st.info("Click on the canvas (with 🔵 tool selected) to add structural wells.")
            for idx, pt in enumerate(raw_well_pts):
                with st.expander(f"🔵 Well #{idx + 1} at ({int(pt['x'])}, {int(pt['y'])})", expanded=True):
                    c1, c2 = st.columns(2)
                    with c1:
                        w_val = st.slider(f"Radius X (px)", 10.0, 200.0, 50.0, key=f"w_w_{idx}")
                        angle_val = st.slider(f"Rotation Angle (°)", 0.0, 180.0, 0.0, key=f"w_ang_{idx}")
                    with c2:
                        h_val = st.slider(f"Radius Y (px)", 10.0, 200.0, 50.0, key=f"w_h_{idx}")
                    
                    well_configs.append({"x": pt["x"], "y": pt["y"], "w": w_val, "h": h_val, "angle": angle_val})

    D, wells, hard_zero_mask = generate_custom_fields_from_canvas(density_configs, well_configs, shape)

# =====================================================================
# DISPLAY WORKSPACE & RENDER CONTROL
# =====================================================================

st.markdown("---")
res_col1, res_col2 = st.columns(2)

with res_col1:
    st.subheader("Generated Synthetic SHG Image")
    render_clicked = st.button("Render Synthetic SHG Image", use_container_width=True, type="primary")

if render_clicked:
    with st.spinner("Generating SHG image..."):
        if mode == "Random Generation":
            current_seed = int(np.random.randint(0, 1000000))
            rng = np.random.default_rng(current_seed)

            # Fully randomize simulation parameters
            run_params = {
                "seed": current_seed,
                "num_fibers": int(rng.integers(80, 500)),
                "spline_length": int(rng.integers(50, 250)),
                "thickness": round(float(rng.uniform(1.5, 4.5)), 2),
                "G_align": round(float(rng.uniform(0.1, 0.9)), 2),
                "L_align": round(float(rng.uniform(0.1, 0.9)), 2),
                "G_curve": round(float(rng.uniform(0.0, 0.8)), 2),
                "L_curve": round(float(rng.uniform(0.1, 0.9)), 2),
                "L_conn": round(float(rng.uniform(0.1, 0.9)), 2),
                "L_wave_freq": round(float(rng.uniform(0.1, 0.9)), 2),
                "G_density": round(float(rng.uniform(0.2, 0.8)), 2),
                "L_density": round(float(rng.uniform(0.2, 0.8)), 2),
                "G_conn": round(float(rng.uniform(0.1, 0.9)), 2),
            }

            # Regenerate fields with randomized parameters
            wells, W, hard_zero_mask = make_wells(X, Y, run_params["G_conn"], rng)
            D = make_density(W, hard_zero_mask, run_params["G_density"], run_params["L_density"], rng)

        else:
            current_seed = manual_seed
            rng = np.random.default_rng(current_seed)
            run_params = {
                "seed": current_seed,
                "num_fibers": num_fibers,
                "spline_length": spline_length,
                "thickness": thickness,
                "G_align": G_align,
                "L_align": L_align,
                "G_curve": G_curve,
                "L_curve": L_curve,
                "L_conn": L_conn,
                "L_wave_freq": L_wave_freq,
            }

        # Orientation & Relaxation
        Qx_g, Qy_g = make_global_orientation(shape, run_params["G_align"], run_params["G_curve"], rng)
        Qx_w, Qy_w, influence = _well_weight_and_tangent(X, Y, wells, Qx_g, Qy_g)
        Qx, Qy = relax(Qx_g, Qy_g, Qx_g, Qy_g, Qx_w, Qy_w, influence, D, run_params["G_align"], run_params["L_align"], wells, X, Y)

        curve_field, conn_field = make_fiber_aux_fields(shape, run_params["L_curve"], run_params["L_conn"], rng)
        wave_freq_field = make_wave_freq_field(shape, run_params["L_wave_freq"], rng)

        # Seed & streamline tracing
        seeds = sample_seeds_from_density(D, run_params["num_fibers"], L_density=0.5, rng=rng)
        aux_curve = sample_field_at_seeds(seeds, curve_field)
        aux_conn = sample_field_at_seeds(seeds, conn_field)
        aux_wave_freq = sample_field_at_seeds(seeds, wave_freq_field)

        splines = []
        for i, seed_pt in enumerate(seeds):
            raw_fiber = generate_fiber(
                Qx, Qy, seed_pt,
                step_size=1.0,
                spline_length=run_params["spline_length"],
                L_curve=run_params["L_curve"],
                susceptibility=1.0 - aux_curve[i],
                rng=rng
            )
            splines.append(fit_spline(raw_fiber, num_samples=max(50, run_params["spline_length"] * 2)))

        # Rasterize image
        raster_img = rasterize_splines(
            H=shape[0], W=shape[1],
            splines=splines,
            thickness=run_params["thickness"],
            aux_wave_amp=aux_curve,
            aux_wave_freq=aux_wave_freq,
            aux_L_conn=aux_conn,
            intensity_seed=current_seed
        )
        
        # Save state
        st.session_state["cached_shg_image"] = raster_img
        st.session_state["active_params"] = run_params
        st.session_state["cached_splines"] = splines
        st.session_state["cached_D"] = D
        st.session_state["cached_Qx"] = Qx
        st.session_state["cached_Qy"] = Qy

# =====================================================================
# RENDER OUTPUTS & DOWNLOAD CONTROLS
# =====================================================================

with res_col1:
    if "cached_shg_image" in st.session_state:
        st.image(st.session_state["cached_shg_image"], clamp=True)
        
        # --- DOWNLOAD PNG BUTTON ---
        img_arr = st.session_state["cached_shg_image"]
        
        # Normalize/convert array to uint8 uint8 png format
        if img_arr.dtype != np.uint8:
            norm_img = ((img_arr - img_arr.min()) / (img_arr.max() - img_arr.min() + 1e-8) * 255).astype(np.uint8)
        else:
            norm_img = img_arr

        pil_img = Image.fromarray(norm_img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        buffer.seek(0)

        current_seed = st.session_state.get("active_params", {}).get("seed", manual_seed)

        st.download_button(
            label="Download SHG Image (PNG)",
            data=buffer,
            file_name=f"shg_image_seed_{current_seed}.png",
            mime="image/png",
            use_container_width=True
        )

        if "active_params" in st.session_state:
            with st.expander("View Parameters for this Image"):
                st.json(st.session_state["active_params"])
    else:
        st.info("Click **Render Randomized SHG Image** to trigger SHG image generation.")

with res_col2:
    st.subheader("Density Field, Vectors & Traced Fiber Splines")
    
    # If a run was rendered, plot the active splines overlaid on the active field
    if "cached_splines" in st.session_state:
        fig, ax = plt.subplots(figsize=(6, 6))
        im = plot_density_vector_and_splines(
            ax=ax,
            D=st.session_state["cached_D"],
            Qx=st.session_state["cached_Qx"],
            Qy=st.session_state["cached_Qy"],
            splines=st.session_state["cached_splines"],
            image_size=image_size
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density Value")
        st.pyplot(fig)

        # --- DOWNLOAD SPLINES (JSON) BUTTON ---
        raw_splines = st.session_state["cached_splines"]
        
        # Convert list of NumPy spline arrays into JSON-serializable structure
        splines_export = {
            "parameters": st.session_state.get("active_params", {}),
            "splines": [sp.tolist() for sp in raw_splines]
        }
        json_data = json.dumps(splines_export, indent=2)

        st.download_button(
            label="Download Spline Paths + Metadata (JSON)",
            data=json_data,
            file_name=f"shg_splines_seed_{current_seed}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        # Initial preview before clicking render
        fig, ax = plt.subplots(figsize=(6, 6))
        Qx_g, Qy_g = make_global_orientation(shape, G_align, G_curve, preview_rng)
        Qx_w, Qy_w, influence = _well_weight_and_tangent(X, Y, wells, Qx_g, Qy_g)
        Qx_p, Qy_p = relax(Qx_g, Qy_g, Qx_g, Qy_g, Qx_w, Qy_w, influence, D, G_align, L_align, wells, X, Y)
        
        im = plot_density_vector_and_splines(
            ax=ax,
            D=D,
            Qx=Qx_p,
            Qy=Qy_p,
            splines=[],
            image_size=image_size
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Density Value")
        st.pyplot(fig)
