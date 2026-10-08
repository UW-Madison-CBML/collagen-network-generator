import random
import pickle
import argparse
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from app.synthetic_code.SyntheticGen import * 


def generate_perlin_noise_2d(shape, res, seed=24):
    """
    Generates a 2D array of Perlin Noise.
    
    Parameters:
        shape (tuple): The size of the output grid (height, width).
                       Must be a clean multiple of `res`.
        res (tuple): The number of periods/cells along the (y, x) axes.
        
    Returns:
        np.ndarray: A 2D array containing the noise values scaled between -1 and 1.
    """
    # Ensure grid dimensions align cleanly
    assert shape[0] % res[0] == 0, "Shape height must be a multiple of res height."
    assert shape[1] % res[1] == 0, "Shape width must be a multiple of res width."
    
    # Calculate cell sizes
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    
    # 1. Generate the grid coordinates inside each cell [0, 1)
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]].transpose(1, 2, 0) % 1
    
    # 2. Generate random gradient angles at every vertex
    np.random.seed(seed)
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    
    # Helper to tile and scale the vertex gradients across the grid pixels
    def tile_grads(slice_y, slice_x):
        return gradients[slice_y, slice_x].repeat(d[0], 0).repeat(d[1], 1)[:shape[0], :shape[1]]
    
    # 3. Compute dot products between vertex gradients and distance vectors
    dot = lambda grad, ox, oy: grad[..., 0] * ox + grad[..., 1] * oy
    
    n00 = dot(tile_grads(slice(0, -1), slice(0, -1)), grid[..., 0],     grid[..., 1])
    n10 = dot(tile_grads(slice(1, None), slice(0, -1)), grid[..., 0] - 1, grid[..., 1])
    n01 = dot(tile_grads(slice(0, -1), slice(1, None)), grid[..., 0],     grid[..., 1] - 1)
    n11 = dot(tile_grads(slice(1, None), slice(1, None)), grid[..., 0] - 1, grid[..., 1] - 1)
    
    # 4. Apply Perlin's quintic fade function for smooth transitions
    fade = lambda t: 6 * t**5 - 15 * t**4 + 10 * t**3
    t = fade(grid)
    
    # 5. Interpolate the dot products across both dimensions
    # Linear interpolation wrapper
    lerp = lambda a, b, x: a + x * (b - a)
    
    return lerp(lerp(n00, n10, t[..., 0]), lerp(n01, n11, t[..., 0]), t[..., 1])


def fractal_perlin_noise_2d(shape, res, octaves=4, persistence=0.5, lacunarity=2.0, seed=24):
    noise = np.zeros(shape)
    frequency = res
    amplitude = 1.0
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(shape, frequency, seed=seed)
        frequency = (int(frequency[0] * lacunarity), int(frequency[1] * lacunarity))
        amplitude *= persistence
    
    # normalize
    noise_min = noise.min()
    noise_max = noise.max()
    if noise_max > noise_min:
        noise = (noise - noise_min) / (noise_max - noise_min)
    return noise

def main(args):
    save_dir = args.save_dir

    n_images = args.n_images

    # random.seed(seed)

    for n in range(n_images):
        seed = random.randint(0, 1000000)
        G_align = round(random.uniform(0.005, 1.0), 2)
        G_density = round(random.uniform(0.005, 1.0), 2)
        G_curve = round(random.uniform(0, 1.0), 2)
        G_conn = round(random.uniform(0, 1.0), 2)

        L_align = round(random.uniform(0.005, 1.0), 2)
        L_density = round(random.uniform(0.005, 1.0), 2)
        L_curve = round(random.uniform(0, 1.0), 2)
        L_conn = round(random.uniform(0, 1.0), 2)

        spline_length = int(round(random.uniform(10, 400), 1))
        spline_num = int(round(random.uniform(500, 3000), 1))

        wave_amplitude_px = int(round(random.uniform(0, 10)))
        wave_wavelength_px = int(round(random.uniform(10, 70)))

        L_wave_freq = round(random.uniform(0, 1.0), 2)


        depth = random.uniform(0.0, 10.0)

        focal_depth_percentage = random.uniform(0.0, 1)

        resolution_factor = 2

        image_shape = (256*resolution_factor, 256*resolution_factor)
        frequency = (2, 2) 

        density = fractal_perlin_noise_2d(image_shape, frequency, octaves=4, seed=seed)

        # meta_data = [
        #     G_align, G_density, G_curve, G_conn,
        #     L_align, L_density, L_curve, L_conn, 
        #     spline_length, spline_num, 
        #     wave_amplitude_px, wave_wavelength_px, 
        #     L_wave_freq, 
        #     depth,
        #     focal_depth_percentage,
        #     density
        # ]
        parameters = {
            "G_align": G_align,
            "G_density": G_density,
            "G_curve": G_curve,
            "G_conn": G_conn,
            "L_align": L_align,
            "L_density": L_density,
            "L_curve": L_curve,
            "L_conn": L_conn,
            "spline_length": spline_length,
            "spline_num": spline_num,
            "wave_amplitude_px": wave_amplitude_px,
            "wave_wavelength_px": wave_wavelength_px,
            "L_wave_freq": L_wave_freq,
            "depth": depth,
            "density": density
        }

        print(parameters)

        res = generate_synthetic_shg_3d(
            resolution_factor = resolution_factor,
            seed=seed, 
            G_align=G_align, G_density=G_density, G_curve=G_curve, G_conn=G_conn,
            L_align=L_align, L_density=L_density, L_conn=L_conn, L_curve=L_curve,
            spline_length=spline_length, spline_num=int(spline_num/10), # divided by n layers
            wave_amplitude_px=wave_amplitude_px,      # bump for visibly wavier fibers
            wave_wavelength_px=wave_wavelength_px,    # None -> auto from fiber_width_px
            L_wave_freq=L_wave_freq,
            show_plots=False,
            minimal=True,
            depth=depth,
            n_layers=10,
            focal_depth=focal_depth_percentage*depth, #this parameters doesent actuall matter because sampling is uniform
            custom_density=density,
        )

        res['parameters'] = parameters
        save = f"{seed}_{n}"
        with open(f"{save_dir}/synthetic_{save}.pkl", "wb") as f:
            pickle.dump(res, f)




if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save_dir", default="/staging/b/blazimy", type=str)
    p.add_argument("--n_images", default=500, type=int)
    args = p.parse_args()

    main(args)

