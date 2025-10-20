import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from tqdm import tqdm # type: ignore

from pathlib import Path
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "1.0"
#os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
import jax
import jax.numpy as jnp
import hashlib
from typing import Callable, List, Tuple
import json
import time
from functools import partial

from config import LOCAL_DATA_DIR
from fgrinmet.splitm import prop_coord_to_obj_coord, paraxial_propagator_jax, paraxial_propagation_step_jax, paraxial_propagation_step_jax_conj, trilinear_interpolate
from fgrinmet.utils import fft_coord_jax

def save_secure_params_fields(
        params: dict,
        U_i: jnp.ndarray,
        U_o: jnp.ndarray,
        save_dir : str | Path,
        fig=None,
):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    params_bytes = json.dumps(save_params, sort_keys=True).encode("utf-8")
    arrays_bytes = jnp.asarray(U_i).tobytes() + jnp.asarray(U_o).tobytes()
    unique_hash = hashlib.sha256(params_bytes + arrays_bytes).hexdigest()

    index_path = save_dir / "index.json"
    if index_path.exists():
        with open(index_path, "r") as f:
            index_data = json.load(f)
    else:
        index_data = {}


def trilinear_single(
        point: jnp.ndarray, 
        grid_vals: jnp.ndarray, 
        grid_mask: jnp.ndarray,
        default: float = 1.5, 
    ) -> jnp.ndarray:

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
        return c * w
    
    vals = jax.vmap(interp, in_axes=0)(offsets)
    return jnp.sum(vals)

trilinear = jax.vmap((trilinear_single), in_axes=(0, None, None, None))
trilinear_comp = jax.jit(trilinear)

@partial(jax.jit, static_argnums=(9, 10))
def A_d(
    U_i: jnp.ndarray, 
    n_distr: jnp.ndarray, 
    mask: jnp.ndarray, 
    coord_plane_0: jnp.ndarray, 
    norm_vect: jnp.ndarray, 
    propagator: jnp.ndarray,
    wavelength: float, 
    n_a: float, 
    dz: float,
    Nz: int,
    d: int
) -> jnp.ndarray:
    
    def body_func(carry_in, i):
        U, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz  = carry_in 
        coord_plane = coord_plane_0 + i * norm_vect
        n_plane = trilinear_interpolate(coord_plane, n_distr, n_a, mask)

        U = paraxial_propagation_step_jax(U, n_plane, propagator, dz, n_a, wavelength)
        carry_out = (U, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz)
        return carry_out, None
    
    final_carry, _ = jax.lax.scan(body_func, (U_i, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz), jnp.arange(d, Nz, 1))
    U_o = final_carry[0]
    return U_o

@partial(jax.jit, static_argnums=(9, 10))
def A_d_conj(
    U_i: jnp.ndarray, 
    n_distr: jnp.ndarray, 
    mask: jnp.ndarray, 
    coord_plane_0: jnp.ndarray, 
    norm_vect: jnp.ndarray, 
    conj_propagator: jnp.ndarray,
    wavelength: float, 
    n_a: float, 
    dz: float,
    Nz: int,
    d: int
) -> jnp.ndarray:
    
    def body_func(carry_in, i):
        U, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz  = carry_in 
        coord_plane = coord_plane_0 + i * norm_vect
        n_plane = trilinear_interpolate(coord_plane, n_distr, n_a, mask)

        U = paraxial_propagation_step_jax_conj(U, n_plane, conj_propagator, dz, n_a, wavelength)
        carry_out = (U, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz)
        return carry_out, None
    
    final_carry, _ = jax.lax.scan(body_func, (U_i, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz), jnp.arange(Nz-1, d-1, -1))
    U_o = final_carry[0]
    return U_o

@partial(jax.jit, static_argnums=(9))
def C_d(
    U_i: jnp.ndarray, 
    n_distr: jnp.ndarray, 
    mask: jnp.ndarray, 
    coord_plane_0: jnp.ndarray, 
    norm_vect: jnp.ndarray, 
    propagator: jnp.ndarray,
    wavelength: float, 
    n_a: float, 
    dz: float,
    d: int
) -> jnp.ndarray:
    
    def body_func(carry_in, i):
        U, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz  = carry_in 
        coord_plane = coord_plane_0 + i * norm_vect
        n_plane = trilinear_interpolate(coord_plane, n_distr, n_a, mask)

        U = paraxial_propagation_step_jax(U, n_plane, propagator, dz, n_a, wavelength)
        carry_out = (U, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz)
        return carry_out, None
    
    final_carry, _ = jax.lax.scan(body_func, (U_i, n_distr, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, dz), jnp.arange(0, d, 1))
    U_o = final_carry[0]
    return U_o


@partial(jax.jit, static_argnums=(9))
def C_d_conj(
    U_i: jnp.ndarray,
    n_distr: jnp.ndarray,
    mask: jnp.ndarray,
    coord_plane_0: jnp.ndarray,
    norm_vect: jnp.ndarray,
    conj_propagator: jnp.ndarray,
    wavelength: float,
    n_a: float,
    dz: float,
    d: int
) -> jnp.ndarray:
    
    def body_func(carry_in, i):
        U, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz  = carry_in 
        coord_plane = coord_plane_0 + i * norm_vect
        n_plane = trilinear_interpolate(coord_plane, n_distr, n_a, mask)

        U = paraxial_propagation_step_jax_conj(U, n_plane, conj_propagator, dz, n_a, wavelength)
        carry_out = (U, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz)
        return carry_out, None
    
    final_carry, _ = jax.lax.scan(body_func, (U_i, n_distr, mask, coord_plane_0, norm_vect, conj_propagator, wavelength, n_a, dz), jnp.arange(d-1, -1, -1))
    U_o = final_carry[0]
    return U_o



if __name__ == "__main__":

    # Common variables or configurations
    dat_save_dir = (LOCAL_DATA_DIR / Path(__file__).name.split(".")[0])
    os.makedirs(dat_save_dir, exist_ok=True)
    mpl.rcParams['axes.unicode_minus'] = False
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    #jax.config.update("jax_enable_x64", False)

    wavelength = 1.0
    n_a = 1.5
    scale = 0.5
    pix_scale = 1
    fact_gap = 1.5


    # Object definition
    shape_obj = (int(4*128*pix_scale), int(128*pix_scale), int(128*pix_scale))

    t_pix_size_obj = (1) * wavelength * scale
    l_pix_size_obj = (0.5) * wavelength * scale
    pix_size_obj = (l_pix_size_obj, t_pix_size_obj, t_pix_size_obj)

    z_obj_norm = jnp.arange(shape_obj[0]) / (shape_obj[0]-1)
    r2_obj_norm = fact_gap*(((jnp.arange(shape_obj[1])-shape_obj[1]//2) / (shape_obj[1]//2 + shape_obj[1]%2))[:,None]**2 + \
                  ((jnp.arange(shape_obj[2])-shape_obj[2]//2) / (shape_obj[2]//2 + shape_obj[2]%2))[None]**2)

    inc_r = 0.35
    inc_n = 0.5
    
    const = (r2_obj_norm[None] - (1 - inc_r*z_obj_norm[:,None,None])**2) * ((1-z_obj_norm[:,None,None] + (z_obj_norm[:,None,None] / (1 - inc_r)**2)))
    mask = r2_obj_norm[None] <= (1 - inc_r*z_obj_norm[:,None,None])**2
    n_original = n_a - mask * inc_n * (1 - (r2_obj_norm[None] / (1 - inc_r*z_obj_norm[:,None,None])**2) + const)

    mask = r2_obj_norm[None].repeat(shape_obj[0], 0) <= 100
    n_original = n_a + mask*(inc_n /10) * (1 - (r2_obj_norm[None].repeat(shape_obj[0], 0)))# / (1 - inc_r*z_obj_norm[:,None,None])**2))
    n_avr = jnp.mean(n_original)

    extend_obj_XZ = (-(shape_obj[0]//2)*pix_size_obj[0], (shape_obj[0]//2+shape_obj[0]%2)*pix_size_obj[0],
                     -(shape_obj[2]//2)*pix_size_obj[2], (shape_obj[2]//2+shape_obj[2]%2)*pix_size_obj[2])
    
    # Propagation grid definition
    shape_prop = (int(4*128*pix_scale),int(128*pix_scale),int(128*pix_scale))

    t_pix_size_prop = (1) * wavelength * scale
    l_pix_size_prop = (0.5) * wavelength * scale
    pix_size_prop = (l_pix_size_prop, t_pix_size_prop, t_pix_size_prop)

    center_obj = (-(shape_prop[0]*pix_size_prop[0])/2 + (shape_obj[0]*pix_size_obj[0])/2,0,0)
    center_prop = (0,0,0)
    coord_plane_0 = jnp.array(center_prop)[None,None]+(jnp.concatenate((jnp.ones(shape_prop[1:])[None]*pix_size_prop[0]*(-shape_prop[0]//2),
                                  jnp.array(jnp.meshgrid(pix_size_prop[1]*(jnp.arange(shape_prop[1]) - shape_prop[1]//2),
                                  pix_size_prop[2]*(jnp.arange(shape_prop[2]) - shape_prop[2]//2))))).transpose(1,2,0))
        
    norm_vect = (jnp.array([1.,0.,0.]) * jnp.array(pix_size_prop)) / jnp.array(pix_size_obj)[None]
    coord_plane_0 = prop_coord_to_obj_coord(coord_plane_0, center_obj, center_prop, shape_obj, pix_size_obj)#.reshape(-1,3)

    extent_prop_YX = (-(shape_prop[2]//2)*pix_size_prop[2], (shape_prop[2]//2+shape_prop[2]%2)*pix_size_prop[2],
                     -(shape_prop[1]//2)*pix_size_prop[1], (shape_prop[1]//2+shape_prop[1]%2)*pix_size_prop[1])

    extent_prop_XZ = (-(shape_prop[0]//2)*pix_size_prop[0], (shape_prop[0]//2+shape_prop[0]%2)*pix_size_prop[0],
                     -(shape_prop[2]//2)*pix_size_prop[2], (shape_prop[2]//2+shape_prop[2]%2)*pix_size_prop[2])

    # Define the variables for the propagation steps
    U_i = jnp.ones(shape_prop[1:]).astype(jnp.complex128)
    U_o_original = jnp.copy(U_i)
    Fy, Fx = fft_coord_jax(shape_prop[1:], pix_size_prop[1:])
    propagator = paraxial_propagator_jax(Fy, Fx, pix_size_prop[0], n_a, wavelength)
    stored_prop = jnp.zeros(shape_prop[::-2])
    trilinear_interpolate = jax.jit(trilinear_interpolate)


    for i in tqdm(range(shape_prop[0])):
        coord_plane = coord_plane_0 + i * norm_vect
        n_plane = trilinear_interpolate(coord_plane, n_original, n_a, mask)
        U_o_original = paraxial_propagation_step_jax(U_o_original, n_plane, propagator, pix_size_prop[0], n_a, wavelength)
        stored_prop = stored_prop.at[:,i].set(jnp.abs(U_o_original[shape_prop[1]//2]))

        #if (i+1) == (shape_obj[0]//2):
        #    fig1, sub1 = plt.subplots(1, 2)
        #    im1 = sub1[0].imshow(n_original[i])
        #    plt.colorbar(im1, ax=sub1[0], shrink=0.18)
        #    sub1[0].set_title("Original n(x,y)")
        #    im2 = sub1[1].imshow(n_plane)
        #    plt.colorbar(im2, ax=sub1[1], shrink=0.18)
        #    sub1[1].set_title("Interpolated n(x,y)")


    import optax

    # Define the guess index and the initial field
    propagator_conj = jnp.conjugate(propagator)
    n_guess = np.full(shape_obj, n_avr)#jnp.copy(n_original)#jnp.full(shape_obj, n_avr)
    grad_n_guess = jnp.zeros(shape_obj)
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(n_guess)
    back_U_meas = A_d_conj(U_o_original, n_guess, mask, coord_plane_0, norm_vect, propagator_conj, wavelength, n_a, pix_size_prop[0], shape_prop[0], 0)
    U_diff = jnp.conjugate(U_i - back_U_meas)
    N = shape_prop[1] * shape_prop[2]
    n_epoch = 100

    # Test the operators
    #start = time.perf_counter()
    #U_o_original_comp = A_d(U_i, n_original, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size_prop[0], shape_prop[0], 0)
    #U_o_original_comp = A_d_conj(U_o_original_comp, n_original, mask, coord_plane_0, norm_vect, propagator_conj, wavelength, n_a, pix_size_prop[0], shape_prop[0], 0)
    #U_o_original_comp = C_d(U_o_guess, n_original, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size_prop[0], shape_prop[0])
    #U_o_original_comp = C_d_conj(U_o_original_comp, n_original, mask, coord_plane_0, norm_vect, propagator_conj, wavelength, n_a, pix_size_prop[0], shape_prop[0])
    #end = time.perf_counter()
    #print(f"Time A_d:{end-start}s")
    #fig_1, sub_1 = plt.subplots(1, 2)
    #im1 = sub_1[0].imshow(jnp.abs(U_o_original_comp))
    #plt.colorbar(im1, ax=sub_1[0], fraction=0.046, pad=0.04)
    #sub_1[0].set_title("$A_D(U_i)$(x,y)")
    #im2 = sub_1[1].imshow(jnp.abs(U_o_original))
    #plt.colorbar(im2, ax=sub_1[1], fraction=0.046, pad=0.04)
    #sub_1[1].set_title("$prop(U_i)$(x,y)")

    #plt.figure("U diff")
    #plt.imshow(jnp.real(U_diff))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.figure("U_meas_back")
    #plt.imshow(jnp.real(back_U_meas))
    #plt.colorbar(fraction=0.046, pad=0.04)
#
    #plt.figure("U_i")
    #plt.imshow(jnp.real(U_i))
    #plt.colorbar(fraction=0.046, pad=0.04)
    #plt.show()
    #print(jax.devices()[0].memory_stats())
    seq_train = tqdm(range(n_epoch), desc="Training", leave=True)
    mse_arr = jnp.zeros(n_epoch)

    for epoch in seq_train:
        U_o_guess = jnp.copy(U_i)
        
        for d in tqdm(range(shape_prop[0])):
            coord_plane = coord_plane_0 + d * norm_vect
            n_plane = trilinear_interpolate(coord_plane, n_guess, n_a, mask)
            U_diff = paraxial_propagation_step_jax_conj(U_diff, n_plane, propagator_conj, pix_size_prop[0], n_a, wavelength)
            #grad_const = 
            dU_dn_ijd = jnp.real((4j * jnp.pi * pix_size_prop[0] * n_plane / (N * wavelength * n_a)) * (U_diff * U_o_guess))
            grad_n_guess = grad_n_guess.at[d].set(dU_dn_ijd)# * mask[d])
            U_o_guess = paraxial_propagation_step_jax(U_o_guess, n_plane, propagator, pix_size_prop[0], n_a, wavelength)
        
        mse = jnp.mean(jnp.abs(U_o_guess - U_o_original)**2)
        mse_arr = mse_arr.at[epoch].set(float(mse))
        updates, opt_state = optimizer.update(grad_n_guess, opt_state, n_guess)
        n_guess = optax.apply_updates(n_guess, updates)

        seq_train.set_postfix({"MSE": float(mse)})


    fig4, sub4 = plt.subplots(1,2)
    im1 = sub4[0].imshow(jnp.abs(U_o_guess))
    plt.colorbar(im1, ax=sub4[0], fraction=0.046, pad=0.04)
    sub4[1].plot(mse_arr)

    fig5, sub5 = plt.subplots(3,1)
    z_slice, y_slice, x_slice = (shape_obj[0] // 2, shape_obj[1] // 2, shape_obj[2] // 2)
    im1 = sub5[0].imshow(n_guess[z_slice])
    plt.colorbar(im1, ax=sub5[0])
    sub5[0].set_title(f"YX slice at z_pix={z_slice}")
    sub5[0].set_xlabel("$x(\\lambda)$")
    sub5[0].set_ylabel("$y(\\lambda)$")
    im2 = sub5[1].imshow(n_guess[:,y_slice].T)
    plt.colorbar(im2, ax=sub5[1])
    sub5[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub5[1].set_xlabel("$z(\\lambda)$")
    sub5[1].set_ylabel("$x(\\lambda)$")
    im3 =sub5[2].imshow(n_guess[:,:, x_slice].T)
    plt.colorbar(im3, ax=sub5[2])
    sub5[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub5[2].set_xlabel("$z(\\lambda)$")
    sub5[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()
    

    fig2 = plt.figure("Index XZ section")
    plt.imshow(n_original[:,shape_obj[1]//2].T, extent=extend_obj_XZ)
    plt.title("XZ section of the index of refraction.")
    plt.xlabel("$z(\\lambda)$")
    plt.ylabel("$x(\\lambda)$")
    plt.colorbar()
    plt.tight_layout()

    fig3 = plt.figure("Propagation XZ section")
    plt.title("Field amplitude on XZ plane")
    plt.imshow(stored_prop, cmap="gray", aspect="equal", extent=extent_prop_XZ, norm=colors.LogNorm(vmin=stored_prop.min()+1e-3, vmax=float(stored_prop.max())))
    theta = jnp.linspace(0,2*np.pi,100, endpoint=True)
    zc = jnp.array([*extend_obj_XZ[:2], *extend_obj_XZ[-3::-1], extend_obj_XZ[0]]) + center_obj[0]
    xc = jnp.array([extend_obj_XZ[3], extend_obj_XZ[3] - inc_r * (extend_obj_XZ[3] - extend_obj_XZ[2])/2, extend_obj_XZ[2] + inc_r * (extend_obj_XZ[3] - extend_obj_XZ[2])/2, extend_obj_XZ[2], extend_obj_XZ[3]])
    plt.plot(zc, xc, "r", label="Lens")
    plt.legend()
    plt.xlabel("$z(\\lambda)$")
    plt.ylabel("$x(\\lambda)$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # Save parameters
    object_params = {"shape": shape_obj, "pix_size": pix_size_obj, "center": center_obj, "incr_r": inc_r, "incr_n": inc_n}
    grid_params = {"shape": shape_prop, "pix_size": pix_size_prop, "center": center_prop, "norm_vec": norm_vect}
    general_params = {"wavelength": wavelength, "n_a": n_a}

    save_params = {"general": general_params, "object": object_params, "grid": grid_params}
