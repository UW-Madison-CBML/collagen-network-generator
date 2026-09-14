import numpy as np
import matplotlib.pyplot as plt

import VectorField as vecfield
import SplineSample as splinesamp
import Rasterize as raster
import Opacity as opacity

import vf_viz as viz

def generate_synthetic_shg(
    image_size=256,
    resolution_factor=4,
    raster_size=512,
    fiber_width_px=2.0,
    spline_num=400,
    spline_length=100,
    seed=1,
    G_align=0.7,
    G_density=0.7,
    G_curve=0.5,
    G_conn=0.5,
    L_align=0.6,
    L_density=0.6,
    L_curve=0.5,
    L_conn=0.3,
    # per-fiber susceptibility to the local vector field. None -> derived
    # from L_align (low L_align gives a mix of field-following and
    # straight/independent "rogue" fibers, producing crossover). Pass a
    # value in [0,1] directly to control this independently of L_align.
    L_susceptibility=None,
    # spline waviness (ground-truth wavy-fiber geometry)
    wave_amplitude_px=2.8,      # max wobble amplitude, in raster-space pixels
    wave_wavelength_px=None,    # wobble wavelength, in raster-space pixels
                                # (None -> derived from fiber_width_px)
    L_wave_freq=0.5,            # spatial clustering of wavelength variation
    show_plots=True,
    save_prefix="synthetic",
    minimal=False,
):
    rng = np.random.default_rng(seed)

    # Vector field
    X, Y = vecfield.create_grid(image_size, resolution_factor)
    # wells, W, hard_zero_mask = vecfield.make_wells(X, Y, G_conn, rng)
    # if not minimal:
    #     viz.plot_all_wells(wells=wells, X=X, Y=Y, hard_mask=hard_zero_mask)

    # Removed wells
    wells = []
    W = np.ones_like(X)
    hard_zero_mask = np.zeros(X.shape, dtype=bool)

    D = vecfield.make_density(W, hard_zero_mask, G_density, L_density, rng)
    Qx_g, Qy_g = vecfield.make_global_orientation(X.shape, G_align, G_curve, rng)
    Qx_w, Qy_w, influence = vecfield.make_well_orientation(X, Y, Qx_g, Qy_g, wells)
    Qx, Qy = vecfield.relax(
        Qx_g.copy(), Qy_g.copy(), Qx_g, Qy_g, Qx_w, Qy_w, influence, D, G_align, L_align,
        wells=wells, X=X, Y=Y,
    )
    theta = vecfield.axial_to_theta(Qx, Qy)

    aux_curve_field, aux_conn_field = vecfield.make_fiber_aux_fields(X.shape, L_curve, L_conn, rng)
    aux_wave_freq_field = vecfield.make_wave_freq_field(X.shape, L_wave_freq, rng)

    if not minimal:
        np.savez_compressed(
            f"{save_prefix}_Q_field.npz",
            Qx=Qx, Qy=Qy, D=D,
            well_influence=influence,
            aux_curve_field=aux_curve_field,
            aux_conn_field=aux_conn_field,
            aux_wave_freq_field=aux_wave_freq_field,
        )

    # Spline stage
    H, Wpx = Qx.shape
    seeds = splinesamp.sample_seeds_from_density(D, spline_num, L_density, rng)

    susceptibility_bias = L_align if L_susceptibility is None else L_susceptibility
    aux_susceptibility = splinesamp.per_spline_auxiliary_values(susceptibility_bias, len(seeds), rng)


    # fibers = [
    #     splinesamp.generate_fiber(
    #         Qx, Qy, s, step_size=1.0, spline_length=spline_length, L_curve=L_curve,
    #         susceptibility=aux_susceptibility[i], rng=rng,
    #     )
    #     for i, s in enumerate(seeds)
    # ]

    aux_L_curve, aux_L_conn = vecfield.sample_aux_at_seeds(seeds, aux_curve_field, aux_conn_field)
    aux_L_wave_freq = vecfield.sample_field_at_seeds(seeds, aux_wave_freq_field)

    # Adding sinusoidal_fiber_offset
    fibers = [
        splinesamp.sinusoidal_fiber_offset(

            splinesamp.generate_fiber(
                Qx, Qy, s, step_size=1.0, spline_length=spline_length, L_curve=L_curve,
                susceptibility=aux_susceptibility[i], rng=rng,
            ),
            wave_amp=aux_L_curve[i],
            wave_freq=aux_L_wave_freq[i],
            rng=rng,
        )
        for i, s in enumerate(seeds)

    ]


    smoothing = 0.8 + 5.0 * (1 - L_curve)
    num_samples = max(100, int(4 * spline_length))
    splines = [splinesamp.fit_spline(f, smoothing=smoothing, num_samples=num_samples) for f in fibers]

    opacity_rng = np.random.default_rng(seed + 1234)
    opacity_table = opacity.build_fiber_opacity_table(len(splines), opacity_rng)




    if not minimal:
        np.savez_compressed(
            f"{save_prefix}_splines.npz",
            H=H,
            W=Wpx,
            seeds=seeds,
            fibers=np.array(fibers, dtype=object),
            splines=np.array(splines, dtype=object),
            aux_L_curve=aux_L_curve,
            aux_L_conn=aux_L_conn,
            aux_L_wave_freq=aux_L_wave_freq,
            aux_susceptibility=aux_susceptibility,
            fiber_base=opacity_table.fiber_base,
            spline_length=spline_length,
        )

    # Rasterize -- barebones: splines to pixels, plus the wavy-fiber wobble
    # and the (mostly-subtle-by-default) brightness model.
    img = raster.rasterize_splines(
        H,
        Wpx,
        splines,
        thickness=fiber_width_px,
        oversample=4.0,
        out_H=raster_size,
        out_W=raster_size,
        wave_amplitude_px=wave_amplitude_px,
        wave_wavelength_px=wave_wavelength_px,
        aux_wave_amp=aux_L_curve,
        aux_wave_freq=aux_L_wave_freq,
        L_conn=L_conn,
        aux_L_conn=aux_L_conn,
        opacity_table=opacity_table,
    )

    if not minimal:
        plt.imsave(f"{save_prefix}_raster.png", img, cmap="gray")

    if show_plots:
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))

        # Density with vector field overlay
        ax[0].imshow(D, cmap="magma", extent=[0,1,1,0])
        M = D.shape[0]
        y = np.linspace(0, 1, M)
        x = np.linspace(0, 1, M)
        X_plot, Y_plot = np.meshgrid(x, y)
        step = 16
        # theta decodes the axial (double-angle) Qx,Qy back to a real
        # direction before plotting -- plotting Qx,Qy directly would show
        # the field rotating at 2x its true rate, which looks fine where
        # theta is slowly varying but produces spurious spirals near wells.
        ax[0].quiver(
            X_plot[::step, ::step],
            Y_plot[::step, ::step],
            np.cos(theta)[::step, ::step],
            -np.sin(theta)[::step, ::step],
            color="cyan",
            scale=35,
            alpha=0.8,
        )
        ax[0].set_title("Density + Vector Field")
        ax[0].axis("off")

        # Spline seeds with spline traces
        ax[1].imshow(D, cmap="magma", alpha=0.5)
        show_n = len(splines)
        for sp in splines[:show_n]:
            if sp is not None and len(sp) > 1:
                ax[1].plot(sp[:, 1], sp[:, 0], color="black", linewidth=0.6, alpha=0.6)
        ax[1].scatter(seeds[:, 1], seeds[:, 0], s=4, c="black", alpha=0.5)
        ax[1].set_title("Splines + Seeds")
        ax[1].axis("off")

        # Final rasterized image
        ax[2].imshow(img, cmap="gray")
        ax[2].set_title("Rasterized Image")
        ax[2].axis("off")

        plt.tight_layout()
        plt.show()

    return {
        "Qx": Qx,
        "Qy": Qy,
        "D": D,
        "theta": theta,
        "seeds": seeds,
        "fibers": fibers,
        "splines": splines,
        "image": img,
        "wells": wells,
        "well_influence": influence,
        "aux_L_curve": aux_L_curve,
        "aux_L_conn": aux_L_conn,
        "aux_L_wave_freq": aux_L_wave_freq,
        "aux_susceptibility": aux_susceptibility,
        "opacity_table": opacity_table,
        "spline_length": spline_length,
    }
