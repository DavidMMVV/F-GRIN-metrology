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
from fgrinmet.splitm import prop_coord_to_obj_coord

from fgrinmet.splitm import trilinear_interpolate, rotation_matrix, paraxial_propagator_jax, paraxial_propagation_step_jax, energy
from fgrinmet.utils import coord_jax, fft_coord_jax

import jax
import gc

from config import LOCAL_DATA_DIR

mpl.rcParams['axes.unicode_minus'] = False

def trilinear_single(point, values, outside, mask):
    grid_shape = jnp.array(values.shape)
    floor_pt = jnp.floor(point).astype(int)
    dec = point - floor_pt

    offsets = jnp.array([
        [0,0,0],[0,0,1],[0,1,0],[0,1,1],
        [1,0,0],[1,0,1],[1,1,0],[1,1,1]
    ], dtype=int)

    def interp_one(off):
        corner = floor_pt + off  # shape (3,)
        inside = jnp.all((corner >= 0) & (corner < grid_shape))
        w = jnp.prod(jnp.where(off == 0, 1 - dec, dec))
        
        # acceder solo al índice escalar de cada dimensión
        val = jnp.where(
            inside & mask[corner[0], corner[1], corner[2]],
            values[corner[0], corner[1], corner[2]],
            outside
        )
        return w * val

    return jnp.sum(jax.vmap(interp_one)(offsets))

@partial(jax.jit, static_argnums=(5,))  # shape_grid is argument #5
def full_scan(U_i, Zpg0, Ypg0, Xpg0, n_vec, shape_grid, n, n_a, mask,
              propagator, dz, wavelength, pix_size_plane, E_0):

    def step(carry, i):
        U_i = carry

        # compute coordinates for this plane
        Zpg = Zpg0 + i * n_vec[0]
        Ypg = Ypg0 + i * n_vec[1]
        Xpg = Xpg0 + i * n_vec[2]

        coord_in = jnp.concatenate([Zpg[:,:,None], Ypg[:,:,None], Xpg[:,:,None]], axis=-1)

        # interpolation and propagation
        n_plane = trilinear_interpolation_jit(coord_in, n, n_a, mask)
        U_o = prop_step(U_i, n_plane, propagator, dz, n_a, wavelength)

        # energy metrics
        e_val = (energy(U_o, pix_size_plane) - E_0) / E_0
        log_val = jnp.abs(U_o[shape_grid[1] // 2])

        return U_o, (e_val, log_val)
        # run scan
    U_final, (E_arr, log_E) = jax.lax.scan(step, U_i, jnp.arange(shape_grid[0]))
    return U_final, E_arr, log_E
    
# Vectorización sobre todos los puntos del plano
trilinear_vmap = jax.jit(jax.vmap(trilinear_single, in_axes=(0, None, None, None)))


if __name__ == "__main__":
    trilinear_interpolation_jit = jax.jit(trilinear_interpolate)
    prop_step = jax.jit(paraxial_propagation_step_jax)
    energy = jax.jit(energy)
    num = 10
    for j in tqdm(range(num)):
        # Define object parameters
        shape_obj = (256, 256, 256)
        wavelength = 1.0
        pix_size_object = 5*8*0.5 * wavelength
        center_object = (0.0, 0.0, 0.0)
        radius = 40*50.0 * wavelength
        Lzo, Lyo, Lxo = (shape_obj[0] * pix_size_object, shape_obj[1] * pix_size_object, shape_obj[2] * pix_size_object)
        w = 40*50.0  * wavelength
        n_a = 1.5
        dn = 0.3
        
        
        # Create the object
        Zo, Yo, Xo = jnp.meshgrid(
            (jnp.arange(shape_obj[0]) - shape_obj[0] // 2) * pix_size_object + center_object[0],
            (jnp.arange(shape_obj[1]) - shape_obj[1] // 2) * pix_size_object + center_object[1],
            (jnp.arange(shape_obj[2]) - shape_obj[2] // 2) * pix_size_object + center_object[2],
            indexing="ij")

        with jax.default_device(jax.devices("cpu")[0]):
            R = jnp.sqrt(Xo**2 + Yo**2)
            mask = R<=radius
        n = (n_a + dn * (jnp.exp(-R**2/(w**2)) - jnp.exp(-radius**2/(w**2))) * mask)
        #n = n_a + 0.5 * (R <= radius)

        # Define grid parameters
        shape_grid = (int(256 * 2 * (j+1) / num), 4096, 4096)
        pix_size_plane = 10*0.125  * wavelength
        dz = 10.0 #9 * (0.125/4)  * wavelength
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
        
        #Zpg0, Ypg0, Xpg0 = prop_coord_to_obj_coord(coords_plane, center_object, (0,0,0), shape_obj, pix_size_object).T.reshape(3,*shape_grid[1:]) # TODO: fix the interpolation function

        n_vec = jnp.array([dz / pix_size_object, 0, 0]) @ rot_m.T  

        #trilinear_interpolate_partial = partial(trilinear_interpolate, mask=mask)

        
        #jac_fn = jax.jit(jax.jacobian(lambda n_values, cord: trilinear_interpolation_jit(cord, n_values, outside=n_a), argnums=0))

        propagator = paraxial_propagator_jax(Fy, Fx, dz, n_a, wavelength)
        U_i = jnp.ones(shape_grid[1:])
        E_0 = energy(U_i, pix_size_plane)
        E_arr = np.zeros(shape_grid[0], dtype=np.float64)
        log_E = np.zeros((shape_grid[:-1]), dtype=np.float64)
        
        """for i in tqdm(range(shape_grid[0]), leave=False):
            Zpg, Ypg, Xpg = Zpg0 + i * n_vec[0], Ypg0 + i * n_vec[1], Xpg0 + i * n_vec[2]

            coord_in = jnp.concatenate([Zpg[:,:,None], Ypg[:,:,None], Xpg[:,:,None]], axis=-1)

            #coord_flat = coord_in.reshape(-1,3)
            #n_plane_flat = trilinear_vmap(coord_flat, n, n_a, mask)
            #n_plane = n_plane_flat.reshape(shape_grid[1:])
            n_plane = trilinear_interpolation_jit(coord_in, n, n_a, mask)

            U_o = prop_step(U_i, n_plane, propagator, dz, n_a, wavelength)
            U_i = U_o
            E_arr[i] = (energy(U_o, pix_size_plane)-E_0) / E_0
            log_E[i] = np.abs(U_o[shape_grid[1]//2])"""
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
        # run scan
        num_planes = int(shape_grid[0])
        U_i = U_i.astype(jnp.complex128)
        U_o, E_arr, log_E = full_scan(U_i, Zpg0, Ypg0, Xpg0, n_vec, shape_grid, n, n_a, mask, propagator, dz, wavelength, pix_size_plane, E_0)
        gc.collect()
        jax.clear_caches()

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
        filename = f"2mm_cylider_z_pix_{j}"
        save = True

        fig2, sub2 = plt.subplots(1,2)
        sub2[0].set_title("Output field")
        sub2[1].set_title("Relative Energy evolution")
        im1 = sub2[0].imshow(jnp.abs(U_o), cmap="gray", extent=extent, norm=colors.LogNorm(vmin=jnp.abs(U_o).min(), vmax=jnp.abs(U_o).max()))
        plt.colorbar(im1, ax=sub2[0], fraction=0.046, pad=0.04)
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

            # Initialize plane of the grid
        Zi, Xi = coord_jax(shape_grid[::2], [dz, pix_size_plane])
        Yi = jnp.zeros_like(Xi)
        coords_plane_XZ = jnp.stack([Zi.flatten(), Yi.flatten(), Xi.flatten()], axis=-1) @ rot_m.T

        Zpg0, Ypg0, Xpg0 = (((coords_plane_XZ[:, 0] - (center_object[0] - Lzo / 2)) / pix_size_object).reshape(shape_grid[::2]), 
                        ((coords_plane_XZ[:, 1] - (center_object[1] - Lyo / 2)) / pix_size_object).reshape(shape_grid[::2]), 
                        ((coords_plane_XZ[:, 2] - (center_object[2] - Lxo / 2)) / pix_size_object).reshape(shape_grid[::2]))

        plane_XZ = jnp.concatenate([Zpg0[:,:,None], Ypg0[:,:,None], Xpg0[:,:,None]], axis=-1)
        n_XZ = trilinear_interpolation_jit(plane_XZ, n, n_a, mask)

        fig4 = plt.figure()
        plt.title("n distribution on XZ plane")
        plt.imshow(n_XZ.T, cmap="jet", aspect="equal", extent=extent_long)
        theta = jnp.linspace(0,2*np.pi,100, endpoint=True)
        xc = center_object[2] + radius*jnp.sin(theta)
        zc = center_object[0] + radius*jnp.cos(theta)
        plt.plot(zc, xc, "r", label="Sphere")
        plt.legend()
        plt.xlabel("$z(\\lambda)$")
        plt.ylabel("$x(\\lambda)$")
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.tight_layout()

        extent_ft = [-(1/(2*pix_size_plane)),(1/(2*pix_size_plane)),-(1/(2*pix_size_plane)),(1/(2*pix_size_plane))]
        fig5 = plt.figure()
        plt.title("Amplitude of the FT of the field.")
        plt.imshow(jnp.abs(jnp.fft.ifftshift(jnp.fft.fft2(jnp.fft.fftshift(U_o)))), cmap="gray", extent=extent_ft, norm=colors.LogNorm(vmin=jnp.abs(U_o).min(), vmax=jnp.abs(U_o).max()))
        plt.colorbar()
        plt.xlabel("$f_x(\\lambda^{-1})$")
        plt.ylabel("$f_y(\\lambda^{-1})$")
        
        fig6 = plt.figure()
        plt.title("Phase of the FT of the field.")
        plt.imshow(jnp.angle(jnp.fft.ifftshift(jnp.fft.fft2(jnp.fft.fftshift(U_o)))), cmap="jet", extent=extent_ft)
        plt.colorbar()
        plt.xlabel("$f_x(\\lambda^{-1})$")
        plt.ylabel("$f_y(\\lambda^{-1})$")

        if save:
            with open(dat_save_dir/(filename+".json"), "w") as f:
                json.dump(control_data, f, indent=4)
            fig2.savefig(dat_save_dir/("loss_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
            fig3.savefig(dat_save_dir/("zprop_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
            fig4.savefig(dat_save_dir/("index_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
            fig5.savefig(dat_save_dir/("FT_U_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
            fig6.savefig(dat_save_dir/("FT_phase_U_fig_"+filename+".png"), dpi=300, bbox_inches="tight")
        plt.close()
        del U_o, E_arr, log_E


