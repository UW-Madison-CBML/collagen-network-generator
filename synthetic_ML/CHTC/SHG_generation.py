import random
import pickle
from SyntheticGen import *

save_dir = "/staging/s/svaren/synthetic_ML/"
n_images = 5000
seed = 10

random.seed(seed)

for n in range(n_images):
    print(n)
    
    G_align = round(random.uniform(0.005, 1.0), 2)
    G_density = round(random.uniform(0.005, 1.0), 2)
    G_curve = round(random.uniform(0, 1.0), 2)
    G_conn = round(random.uniform(0, 1.0), 2)

    L_align = round(random.uniform(0.005, 1.0), 2)
    L_density = round(random.uniform(0.005, 1.0), 2)
    L_curve = round(random.uniform(0, 1.0), 2)
    L_conn = round(random.uniform(0, 1.0), 2)

    spline_length = int(round(random.uniform(50, 500), 1))
    spline_num = int(round(random.uniform(100, 1000), 1))

    wave_amplitude_px = int(round(random.uniform(0, 10)))
    wave_wavelength_px = int(round(random.uniform(10, 70)))

    L_wave_freq = round(random.uniform(0, 1.0), 2)

    base_noise = int(round(random.uniform(30, 1000), 1))

    meta_data = [
        G_align, G_density, G_curve, G_conn,
        L_align, L_density, L_curve, L_conn, 
        spline_length, spline_num, 
        wave_amplitude_px, wave_wavelength_px, 
        L_wave_freq, 
        base_noise
    ]

    res = generate_synthetic_shg(
        seed=seed, 
        G_align=G_align, G_density=G_density, G_curve=G_curve, G_conn=G_conn,
        L_align=L_align, L_density=L_density, L_conn=L_conn, L_curve=L_curve,
        spline_length=spline_length, spline_num=spline_num,
        wave_amplitude_px=wave_amplitude_px,      # bump for visibly wavier fibers
        wave_wavelength_px=wave_wavelength_px,    # None -> auto from fiber_width_px
        L_wave_freq=L_wave_freq,
        show_plots=False,
        minimal=True,
        base_noise=base_noise
    )

    res['meta'] = meta_data
    save = f"{wave_amplitude_px}_{wave_wavelength_px}_{spline_length}_{spline_num}_{base_noise}"

    with open(f"{save_dir}/synthetic_{save}.pkl", "wb") as f:
        pickle.dump(res, f)
