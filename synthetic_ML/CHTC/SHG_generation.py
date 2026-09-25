import random
import pickle
import argparse
from concurrent.futures import ProcessPoolExecutor
from SyntheticGen import *

def one_sample(n, seed):

    save_dir = "/staging/s/svaren/synthetic_ML/"
    print(f"Saving here: {save_dir}")

    print(f"Seed {seed}")
    random.seed(seed)

    G_align = round(random.uniform(0.005, 1.0), 2)
    G_density = round(random.uniform(0.005, 1.0), 2)
    G_curve = round(random.uniform(0, 1.0), 2)
    G_conn = round(random.uniform(0, 1.0), 2)

    L_align = round(random.uniform(0.005, 1.0), 2)
    L_density = round(random.uniform(0.005, 1.0), 2)
    L_curve = round(random.uniform(0, 1.0), 2)
    L_conn = round(random.uniform(0, 1.0), 2)

    spline_length = random.randint(50, 500)
    spline_num = random.randint(100, 1000)

    wave_amplitude_px = random.randint(0, 10)
    wave_wavelength_px = random.randint(10, 70)

    L_wave_freq = round(random.uniform(0, 1.0), 2)

    base_noise = random.randint(30, 1000)

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
    res['id'] = n
    save = f"{wave_amplitude_px}_{wave_wavelength_px}_{spline_length}_{spline_num}_{base_noise}"

    with open(f"{n}_{save_dir}/synthetic_{save}.pkl", "wb") as f:
        pickle.dump(res, f)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", required=True, type=int)
    args = p.parse_args()

    n_images = 1000
    tasks = [(n_images, args.seed) for n in range(n_images)]

    with ProcessPoolExecutor() as executor:
        for finished_n in executor.map(one_sample, tasks):
            print(f"Finished image {finished_n}")