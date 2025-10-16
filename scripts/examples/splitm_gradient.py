import jax
import jax.numpy as jnp
import lax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

from pathlib import Path
import os
import hashlib
from typing import Tuple
import time

from config import LOCAL_DATA_DIR

def trilinear_single(
        point: jnp.ndarray, 
        grid_vals: jnp.ndarray, 
        grid_mask: jnp.ndarray,
        default: float = 1.5, 
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    grid_shape = jnp.array(grid_mask.shape)
    floor = jnp.floor(point).astype(int)
    dec = point - floor

    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1]
    ], dtype=int)

    def interp(offset):
        c_idx = floor + offset
        is_inside = jnp.all((c_idx >= 0) & (c_idx < grid_shape))
        is_valid = is_inside & grid_mask[c_idx[0], c_idx[1], c_idx[2]]

        w = jnp.prod(jnp.where(offset == 0, 1-dec, dec))

        c = jnp.where(is_valid,
                      grid_vals[c_idx[0], c_idx[1], c_idx[2]], 
                      default)
        return c * w, is_valid, c_idx
    
    vals, c_valid, c_idx = jax.vmap(interp)(offsets)    
    return jnp.sum(vals), c_valid, c_idx

trilinear = jax.vmap(trilinear_single, in_axes=(0, None, None, None))
trilinear_comp = jax.jit(trilinear)

if __name__ == "__main__":

    # Common variables or configurations
    dat_save_dir = (LOCAL_DATA_DIR / Path(__file__).name.split(".")[0])
    os.makedirs(dat_save_dir, exist_ok=True)
    mpl.rcParams['axes.unicode_minus'] = False

    wavelength = 1.0
    n_a = 1.5


    # Object definition
    shape_obj = (128, 128, 128)

    t_pix_size_obj = (10) * wavelength
    l_pix_size_obj = t_pix_size_obj
    pix_size_obj = (l_pix_size_obj, t_pix_size_obj, t_pix_size_obj)

    z_obj_norm = jnp.arange(shape_obj[0]) / (shape_obj[0]-1)
    r2_obj_norm = ((jnp.arange(shape_obj[1])-shape_obj[1]//2) / (shape_obj[1]//2 + shape_obj[1]%2))[:,None]**2 + \
                  ((jnp.arange(shape_obj[2])-shape_obj[2]//2) / (shape_obj[2]//2 + shape_obj[2]%2))[None]**2

    inc_r = 0.5
    inc_n = 0.5
    
    const = (r2_obj_norm[None] - (1 - inc_r*z_obj_norm[:,None,None])**2) * ((1-z_obj_norm[:,None,None] + (z_obj_norm[:,None,None] / (1 - inc_r)**2)))
    mask = r2_obj_norm[None] <= (1 - inc_r*z_obj_norm[:,None,None])**2
    n_original = n_a - mask * inc_n * (1 - (r2_obj_norm[None] / (1 - inc_r*z_obj_norm[:,None,None])**2) + const)
    n_original = n_original

    extend_obj_xz = (-(shape_obj[0]//2)*pix_size_obj[0], (shape_obj[0]//2+shape_obj[0]%2)*pix_size_obj[0],
                     -(shape_obj[2]//2)*pix_size_obj[2], (shape_obj[2]//2+shape_obj[2]%2)*pix_size_obj[2])
    
    # Propagation grid definition
    shape_prop = (1280,4280,4280)

    t_pix_size_prop = (10) * wavelength
    l_pix_size_prop = (10) * wavelength
    pix_size_prop = (l_pix_size_prop, t_pix_size_prop, t_pix_size_prop)

    center_obj = (-(shape_prop[0]*pix_size_prop[0])/2 + (shape_obj[0]*pix_size_obj[0])/2,0,0)
    coord_plane = jnp.concatenate((jnp.ones(shape_prop[1:])[None]*pix_size_prop[0]*(shape_prop[0]//2),
                                  jnp.array(jnp.meshgrid(pix_size_prop[1]*(jnp.arange(shape_prop[1]) - shape_prop[1]//2),
                                         pix_size_prop[2]*(jnp.arange(shape_prop[2]) - shape_prop[2]//2))))).transpose(1,2,0)

    #start = time.perf_counter()
    #interp_plane = trilinear(coord_plane.reshape(-1,3), n_original, mask, n_a) 
    #end = time.perf_counter()
    #print(f"Time vmap: {end-start}s")
    #start = time.perf_counter()
    #interp_plane = trilinear(coord_plane.reshape(-1,3), n_original, mask, n_a) 
    #end = time.perf_counter()
    #print(f"Time vmap: {end-start}s")
    start = time.perf_counter()
    interp_plane_comp = trilinear(coord_plane.reshape(-1,3), n_original, mask, n_a) 
    end = time.perf_counter()
    print(f"Time vmap+jit: {end-start}s")
    start = time.perf_counter()
    interp_plane_comp = trilinear(coord_plane.reshape(-1,3), n_original, mask, n_a) 
    end = time.perf_counter()
    print(f"Time vmap+jit: {end-start}s")

    plt.figure("Index XZ section")
    plt.imshow(n_original[:,shape_obj[1]//2].T, extent=extend_obj_xz)
    plt.title("XZ section of the index of refraction.")
    plt.xlabel("$z(\\lambda)$")
    plt.ylabel("$x(\\lambda)$")
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    # Save parameters
    object_params = {"shape": shape_obj, "pix_size": pix_size_obj, "incr_r": inc_r, "incr_n": inc_n}
    

