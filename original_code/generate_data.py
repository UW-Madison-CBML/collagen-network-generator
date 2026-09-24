import argparse
import SyntheticGen as syn
import random
import numpy as np


parser = argparse.ArgumentParser(description="Process numeric arguments.")

parser.add_argument(
    "prefix", 
    help="data save prefix"
)

parser.add_argument(
    "-nr", "--nr_to_generate", 
    type=int, 
    default=1, 
    help="The number of data samples to generate."
)

parser.add_argument(
    "-np", "--np_to_generate", 
    type=int, 
    default=1, 
    help="The number of preset data samples to generate."
)

#RED: res = syn.generate_synthetic_shg(seed=10, G_align=0.7, G_density=0.7, G_curve=0.5, G_conn=1, L_align=0.6, L_density=0, L_conn=0.3, L_curve=0, L_intensity=0, spline_length=50, spline_num=1200, gamma=0.7, contrast=0.3, norm_targ=0.2)
#YELLOW BAD: res = syn.generate_synthetic_shg(seed=10, G_align=0.4, G_density=0.7, G_curve=0.5, G_conn=1, L_align=0, L_density=0, L_conn=0.3, L_curve=0, L_intensity=0, spline_length=50, spline_num=1200, gamma=1, contrast=0.3, norm_targ=0.75)
#YELLOW BAD 2: res = syn.generate_synthetic_shg(seed=10, G_align=0.2, G_density=1, G_curve=0.7, G_conn=1, L_align=0.6, L_density=0.7, L_conn=0.3, L_curve=0, L_intensity=0, spline_length=50, spline_num=2000, gamma=0, contrast=0, norm_targ=0.75)
#YELLOW GOOD: res = syn.generate_synthetic_shg(seed=10, G_align=0.2, G_density=1, G_curve=0.7, G_conn=1, L_align=0.6, L_density=0.6, L_conn=0.3, L_curve=0, L_intensity=1, spline_length=50, spline_num=3000, gamma=0, contrast=0, norm_targ=1)
#INTERESTING: res = syn.generate_synthetic_shg(seed=10, G_align=0.2, G_density=1, G_curve=0.7, G_conn=1, L_align=0.6, L_density=0.1, L_conn=0.3, L_curve=0, L_intensity=1, spline_length=50, spline_num=3000, gamma=0, contrast=0, norm_targ=1)
#CYAN: res = syn.generate_synthetic_shg(seed=10, G_align=0.9, G_density=0.7, G_curve=0.7, G_conn=1, L_align=0.6, L_density=0, L_conn=0, L_curve=1, L_intensity=0.1, spline_length=40, spline_num=3000, gamma=0, contrast=0.8, norm_targ=0.1)
#TEMP TEST ORANGE: res = syn.generate_synthetic_shg(seed=10, G_align=0.8, G_density=0.7, G_curve=0.7, G_conn=1, L_align=0.6, L_density=0.4, L_conn=1, L_curve=0, L_intensity=0, spline_length=30, spline_num=9000, gamma=0, contrast=0.7, norm_targ=0.75)

# Synthetic Params should contain all the arguments for the generation function
# including default values for optional parameters like show_plots and minimal.
DEFAULT_SYNTHETIC_PARAMS = {
    "image_size": 256,
    "resolution_factor": 4,
    "raster_size": 512,
    "fiber_width_px": 2.0,
    "spline_num": 400,
    "spline_length": 100,
    "seed": 1,
    "G_align": 0.7,
    "G_density": 0.7,
    "G_curve": 0.5,
    "G_conn": 0.5,
    "L_align": 0.6,
    "L_density": 0.6,
    "L_curve": 0.5,
    "L_conn": 0.3,
    "L_susceptibility": None,
    "wave_amplitude_px": 2.8,
    "wave_wavelength_px": None,
    "L_wave_freq": 0.5,
    "show_plots": True,
    "save_prefix": "synthetic",
    "minimal": False,
}


def build_synthetic_params(overrides=None):
    params = dict(DEFAULT_SYNTHETIC_PARAMS)
    if overrides:
        params.update(overrides)
    params["seed"] = random.randint(0, 10000);
    return params


synthetic_class_params = {
    "RED": build_synthetic_params({
        "seed": 10,
        "G_align": 0.7,
        "G_density": 0.7,
        "G_curve": 0.5,
        "G_conn": 1,
        "L_align": 0.6,
        "L_density": 0,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 1200,
        "show_plots": False,
        "minimal": True,
    }),
    "YELLOW_BAD": build_synthetic_params({
        "seed": 10,
        "G_align": 0.4,
        "G_density": 0.7,
        "G_curve": 0.5,
        "G_conn": 1,
        "L_align": 0,
        "L_density": 0,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 1200,
        "show_plots": False,
        "minimal": True,
    }),
    "YELLOW_GOOD": build_synthetic_params({
        "seed": 10,
        "G_align": 0.2,
        "G_density": 1,
        "G_curve": 0.7,
        "G_conn": 1,
        "L_align": 0.6,
        "L_density": 0.6,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 3000,
        "show_plots": False,
        "minimal": True,
    }),
    "INTERESTING": build_synthetic_params({
        "seed": 10,
        "G_align": 0.5,
        "G_density": 0.5,
        "G_curve": 0.5,
        "G_conn": 1,
        "L_align": 0.5,
        "L_density": 0.5,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 1200,
        "show_plots": False,
        "minimal": True,
    }),
    "CYAN": build_synthetic_params({
        "seed": 10,
        "G_align": 0.5,
        "G_density": 0.5,
        "G_curve": 0.5,
        "G_conn": 1,
        "L_align": 0.5,
        "L_density": 0.5,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 1200,
        "show_plots": False,
        "minimal": True,
    }),
    "TEMP TEST ORANGE": build_synthetic_params({
        "seed": 10,
        "G_align": 0.5,
        "G_density": 0.5,
        "G_curve": 0.5,
        "G_conn": 1,
        "L_align": 0.5,
        "L_density": 0.5,
        "L_conn": 0.3,
        "L_curve": 0,
        "spline_length": 50,
        "spline_num": 1200,
        "show_plots": False,
        "minimal": True,
    }),
}


def random_shg_params(save_prefix="synthetic"):
    params = build_synthetic_params({
        "seed": random.randint(0, 10000),
        "G_align": random.uniform(0, 1),
        "G_density": random.uniform(0, 1),
        "G_curve": random.uniform(0, 1),
        "G_conn": random.uniform(0.8, 1),
        "L_susceptibility": random.uniform(0, 1),
        "L_align": random.uniform(0, 1),
        "L_density": random.uniform(0, 1),
        "L_conn": random.uniform(0, 1),
        #50% as 0-0.05 local curvature or something and the other half 0-1
        "L_curve": random.uniform(0, 0.05) if random.random() < 0.5 else random.uniform(0, 1),
        "spline_length": random.randint(40, 60),
        "spline_num": random.randint(1000, 3000),
        "wave_amplitude_px": random.uniform(4, 8),
        "wave_wavelength_px": random.uniform(20, 40),
        "L_wave_freq": random.uniform(0, 1),
        "show_plots": False,
        "save_prefix": save_prefix,
        "minimal": True,
    })
    return params

args = parser.parse_args()
run_prefix = args.prefix
nr_to_generate = args.nr_to_generate
np_to_generate = args.np_to_generate

batch_n = 0;
MAX_BATCH_SIZE = 100;

def save():
    global batch_n, save_data
    if(len(save_data)>=MAX_BATCH_SIZE):
        # process the batch
        print(f"Processing batch {batch_n} of size {len(save_data)}")
        np.savez_compressed(f"sshg_{run_prefix}_{batch_n}.npz", data=save_data)
        batch_n += 1
        save_data = [];

save_data = [];
for i in range(nr_to_generate):
    args = random_shg_params(save_prefix=run_prefix)
    res = syn.generate_synthetic_shg(**args)
    datapoint = dict(img=res["image"], params=args)
    save_data.append(datapoint)
    save()

for i in range(np_to_generate):
    # iterate over class types
    for j, class_name in enumerate(synthetic_class_params.keys()):
        args = build_synthetic_params(synthetic_class_params[class_name])
        res = syn.generate_synthetic_shg(**args)
        datapoint = dict(img=res["image"], params=args)
        save_data.append(datapoint)
        save()
# save the last batch if it has any data
if len(save_data) > 0:
    print(f"Processing final batch {batch_n} of size {len(save_data)}")
    np.savez_compressed(f"sshg_{run_prefix}_{batch_n}.npz", data=save_data)