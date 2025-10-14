import jax.numpy as jnp
import numpy as np
import jax
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from functools import partial
import matplotlib as mpl

from tqdm import tqdm 
from pathlib import Path
import json 
import os
import time

from fgrinmet.splitm import trilinear_interpolate, rotation_matrix, paraxial_propagator_jax, paraxial_propagation_step_jax, energy
from fgrinmet.utils import coord_jax, fft_coord_jax

from config import LOCAL_DATA_DIR

mpl.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # Define object parameters
    shape_obj = (128, 128, 128)
    wavelength = 1.0
    pix_size_object = 2*5*8*0.5 * wavelength
    center_object = (0.0, 0.0, 0.0)
    radius = 10*4*50.0 * wavelength
    Lzo, Lyo, Lxo = (shape_obj[0] * pix_size_object, shape_obj[1] * pix_size_object, shape_obj[2] * pix_size_object)
    w = 10*4*50.0  * wavelength
    n_a = 1.5
    dn = 0.3
    
    
    # Create the object
    Zo, Yo, Xo = jnp.meshgrid(
        (jnp.arange(shape_obj[0]) - shape_obj[0] // 2) * pix_size_object + center_object[0],
        (jnp.arange(shape_obj[1]) - shape_obj[1] // 2) * pix_size_object + center_object[1],
        (jnp.arange(shape_obj[2]) - shape_obj[2] // 2) * pix_size_object + center_object[2],
        indexing="ij")

    with jax.default_device(jax.devices("cpu")[0]):
        R = jnp.sqrt(Xo**2 + Yo**2 + Zo**2)
        mask = R<=radius
    n = (n_a + dn * (jnp.exp(-R**2/(w**2)) - jnp.exp(-radius**2/(w**2))) * mask)
    #n = n_a + 0.5 * (R <= radius)

    # Define grid parameters
    shape_grid = (1024, 5*1024, 5*1024)
    pix_size_plane = 10*0.125  * wavelength
    dz = 10*8*(0.125)  * wavelength
    center_object = (-(shape_grid[0]*dz)/2 + (shape_obj[0]*pix_size_object)/2,0,0)
    vec_plane = (0,0,0)#(0, 0, jnp.pi / 4)  # normal vector of the plane
    rot_m = rotation_matrix(*vec_plane)

    # Initialize plane of the grid
    Yi, Xi = jnp.meshgrid(
        (jnp.arange(shape_grid[1]) - shape_grid[1] // 2) * pix_size_plane,
        (jnp.arange(shape_grid[2]) - shape_grid[2] // 2) * pix_size_plane,
        indexing="ij")
    Yi, Xi = coord_jax(shape_grid[1:], pix_size_plane)
    Zi = jnp.zeros_like(Xi) - (shape_grid[0] // 2) * dz

    Fx, Fy = fft_coord_jax(shape_grid[1:], pix_size_plane)
    extent = [-pix_size_plane * shape_grid[2]/2, pix_size_plane * (1+shape_grid[2]/2),
              pix_size_plane * (1+shape_grid[1]/2), -pix_size_plane * shape_grid[1]/2]

    # Rotate the grid plane, transform to object grid coordinates and find normal vector
    coords_plane = jnp.stack([Zi.flatten(), Yi.flatten(), Xi.flatten()], axis=-1) @ rot_m.T
    Zpg0, Ypg0, Xpg0 = (((coords_plane[:, 0] - (center_object[0] - Lzo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 1] - (center_object[1] - Lyo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 2] - (center_object[2] - Lxo / 2)) / pix_size_object).reshape(shape_grid[1:]))
    
    from fgrinmet.splitm import prop_coord_to_obj_coord
    Zpg0, Ypg0, Xpg0 = prop_coord_to_obj_coord(coords_plane, center_object, (0,0,0), shape_obj, pix_size_object).T.reshape(3,*shape_grid[1:])

    n_vec = jnp.array([dz / pix_size_object, 0, 0]) @ rot_m.T  

    #trilinear_interpolate_partial = partial(trilinear_interpolate, mask=mask)
    trilinear_interpolation_jit = jax.jit(trilinear_interpolate)
    #jac_fn = jax.jit(jax.jacobian(lambda n_values, cord: trilinear_interpolation_jit(cord, n_values, outside=n_a), argnums=0))
    prop_step = jax.jit(paraxial_propagation_step_jax)
    energy = jax.jit(energy)
    
    propagator = paraxial_propagator_jax(Fy, Fx, dz, n_a, wavelength)
    U_i = jnp.ones(shape_grid[1:])
    E_0 = energy(U_i, pix_size_plane)
    E_arr = np.zeros(shape_grid[0], dtype=np.float64)
    log_E = np.zeros((shape_grid[:-1]), dtype=np.float64)
    
    for i in tqdm(range(shape_grid[0])):
        Zpg, Ypg, Xpg = Zpg0 + i * n_vec[0], Ypg0 + i * n_vec[1], Xpg0 + i * n_vec[2]

        cord_in = jnp.concatenate([Zpg[:,:,None], Ypg[:,:,None], Xpg[:,:,None]], axis=-1)


        n_plane = trilinear_interpolation_jit(cord_in, n, outside=n_a, mask=mask)
        U_o = prop_step(U_i, n_plane, propagator, dz, n_a, wavelength)
        U_i = U_o
        E_arr[i] = (energy(U_o, pix_size_plane)-E_0) / E_0
        log_E[i] = np.abs(U_o[shape_grid[1]//2])
        #target_plane = jnp.ones_like(n_plane) * n_a  # Example target plane for loss calculation
        #loss = jnp.sum((n_plane - target_plane)**2)
        #grad_n = jax.grad(lambda n_values: jnp.sum((trilinear_interpolation_jit(cord_in, n_values, outside=n_a) - target_plane)**2))(n)

        #if i in ((shape_grid[0]//5)*np.arange(1,5)):#(shape_grid[0] // 8):
        #    
        #    fig, sub = plt.subplots(1,2)
        #    im1 = sub[0].imshow(n_plane, cmap='jet', extent=extent)
        #    plt.colorbar(im1, ax=sub[0], fraction=0.046, pad=0.04)
        #    sub[0].set_xlabel("$x(\\lambda)$")
        #    sub[0].set_ylabel("$y(\\lambda)$")
        #    im2 = sub[1].imshow(jnp.abs(U_o), cmap="gray", extent=extent, norm=colors.LogNorm(vmin=jnp.abs(U_o).min(), vmax=jnp.abs(U_o).max()))
        #    plt.colorbar(im2, ax=sub[1], fraction=0.046, pad=0.04)
        #    sub[1].set_xlabel("$x(\\lambda)$")
        #    sub[1].set_ylabel("$y(\\lambda)$")
        #    sub[0].set_title(f"Refractive index at the plane in $z_g$={(i*hz)}$\\lambda$")
        #    sub[1].set_title(f"Propagated modulus of the field at the plane in $z_g$={(i*hz)}$\\lambda$")
        #    plt.tight_layout()
        #    plt.show()

    #plt.show()

    control_data = {
        "wavelength": wavelength,
        "object": {
            "shape_obj": shape_obj,
            "pix_size_object": pix_size_object,
            "center_object": center_object,
            "radius": radius,
            "w": w,
            "n_a": n_a,
            "dn":dn
        },
        "prop_grid": {
            "shape_grid": shape_grid,
            "pix_size_plane": pix_size_plane,
            "dz": dz
        },
        "prop_params": {
            "vec_plane": vec_plane
        }
    }

    dat_save_dir = (LOCAL_DATA_DIR / Path(__file__).name.split(".")[0])
    os.makedirs(dat_save_dir, exist_ok=True)    
    filename = "2mm_sphere_4"
    save = True

    fig2, sub2 = plt.subplots(1,2)
    sub2[0].set_title("Output field")
    sub2[1].set_title("Relative Energy evolution")
    im21 = sub2[0].imshow(jnp.abs(U_o), cmap="gray", extent=extent, norm=colors.LogNorm(vmin=jnp.abs(U_o).min(), vmax=jnp.abs(U_o).max()))
    plt.colorbar(im21, ax=sub2[0], fraction=0.046, pad=0.04)
    sub2[0].set_xlabel("$x(\\lambda)$")
    sub2[0].set_ylabel("$y(\\lambda)$")
    sub2[1].plot(dz*(np.arange(shape_grid[0])-shape_grid[0]//2), 100*E_arr)
    sub2[1].set_xlabel("$z(\\lambda)$")
    sub2[1].set_ylabel("Relative Energy (%)")
    plt.tight_layout()

    extent_long = [-dz * shape_grid[0]/2, dz * (1+shape_grid[0]/2),
                   -pix_size_plane * shape_grid[2]/2, pix_size_plane * (1+shape_grid[2]/2)]

    fig3 = plt.figure()
    plt.title("Field amplitude on XZ plane")
    plt.imshow(log_E.T, cmap="gray", aspect="equal", extent=extent_long, norm=colors.LogNorm(vmin=log_E.min(), vmax=log_E.max()))
    theta = jnp.linspace(0,2*np.pi,100, endpoint=True)
    xc = center_object[2] + radius*jnp.sin(theta)
    zc = center_object[0] + radius*jnp.cos(theta)
    plt.plot(zc, xc, "r", label="Sphere")
    plt.legend()
    plt.xlabel("$z(\\lambda)$")
    plt.ylabel("$x(\\lambda)$")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    
    if save:
        with open(dat_save_dir/(filename+".json"), "w") as f:
            json.dump(control_data, f, indent=4)
        fig2.savefig(dat_save_dir/("loss_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
        fig3.savefig(dat_save_dir/("zprop_fig_"+filename+".png"), dpi=300, bbox_inches="tight")

    plt.show()
    


    print("hey")
