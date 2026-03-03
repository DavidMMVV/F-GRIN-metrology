import jax.numpy as jnp
import jax
from scipy.special import hermite
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
import optax
import time
from functools import partial
from jax import random

from fgrinmet.utils.operators import FT2, iFT2
from fgrinmet.splitm.interpolation import trilinear_interpolate, trilinear_interpolate_grad
from fgrinmet.utils import poly_exp, poly_adj, poly_sum

@partial(jax.jit, static_argnums=(5))
def propagate_modes(
        eps_distr: jnp.ndarray, 
        propagator: jnp.ndarray, 
        U_in: jnp.ndarray, 
        plane_rel: jnp.ndarray,
        norm_vect: jnp.ndarray, 
        nz: int,
        dz: float,
        eps_a: float, 
        l: float
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Propagate multiple input modes through a slice of thickness dz with permittivity distribution eps_distr.

    Args:
        eps_distr (jnp.ndarray): Permittivity distribution of the slice.
        propagator (jnp.ndarray): Precomputed propagator in Fourier space.
        U_in (jnp.ndarray): Input modes to be propagated, shape (n_modes, H, W).
        dz (float): Thickness of the slice.
        eps_a (float): Ambient permittivity.
        l (float): Wavelength.

    Returns:
        jnp.ndarray: Output modes after propagation, shape (n_modes, H, W).
    """
    def step(U_carry, index):
        plane = plane_rel + norm_vect[None, None] * index
        eps_plane = trilinear_interpolate(plane, eps_distr, outside = 0.0)

        #U_mid = jnp.exp(1j * jnp.pi * eps_plane[index] * dz / (l * jnp.sqrt(eps_a))) * iFT2(propagator[None] * FT2(U_carry))
        Uo = iFT2(propagator[None] * FT2(jnp.exp(1j * jnp.pi * eps_plane[None] * dz / (l * jnp.sqrt(eps_a))) * iFT2(propagator[None] * FT2(U_carry))))
        energy = jnp.abs(U_carry).sum(axis=(-2,-1))
        return Uo, energy

    U_out, energies = jax.lax.scan(step, U_in, jnp.arange(nz))
    energies = energies.transpose((1,0))
    return U_out, energies

@jax.jit #partial(jax.jit, static_argnums=(5))
def propagate_modes_poly(
        coeficients: jnp.ndarray,
        z_poly: jnp.ndarray,
        l_poly: jnp.ndarray,
        propagator: jnp.ndarray, 
        U_in: jnp.ndarray, 
        dz: float,
        eps_a: float, 
        l: float
    ) -> jnp.ndarray:

    def step(U_carry, index):
        eps_plane = (coeficients[:,:,None,None] * z_poly[:,None] * l_poly[None,:,index,None,None]).sum(axis=(0,1))
        Uo = iFT2(propagator[None] * FT2(jnp.exp(1j * jnp.pi * eps_plane[None] * dz / (l * jnp.sqrt(eps_a))) * iFT2(propagator[None] * FT2(U_carry))))
        energy = jnp.abs(U_carry).sum(axis=(-2,-1))
        return Uo, energy
    nz = l_poly.shape[1]
    U_out, energies = jax.lax.scan(step, U_in, jnp.arange(nz))
    energies = energies.transpose((1,0))
    return U_out, energies

if __name__ == "__main__":

    FT2 = jax.jit(FT2)
    iFT2 = jax.jit(iFT2)
    #propagate_modes = jax.jit(propagate_modes, static_argnums=(5))

    l = 1

    # Definition of the propagation grid
    prop_shape = jnp.array((16,8*128,8*128))
    prop_pix_sizes = jnp.array(((l/4), 8*l, 8*l))
    prop_center = jnp.array(((prop_shape[0]//2)*prop_pix_sizes[0],
                             (prop_shape[1]//2)*prop_pix_sizes[1],
                             (prop_shape[2]//2)*prop_pix_sizes[2]))
    prop_params = {
        "shape": prop_shape,
        "pix_sizes": prop_pix_sizes,
        "center": prop_center
    }
    z_p = (jnp.arange(prop_params["shape"][0]) - prop_params["shape"][0] // 2) * prop_params["pix_sizes"][0]
    y_p = (jnp.arange(prop_params["shape"][1]) - prop_params["shape"][1] // 2) * prop_params["pix_sizes"][1]
    x_p = (jnp.arange(prop_params["shape"][2]) - prop_params["shape"][2] // 2) * prop_params["pix_sizes"][2]
    plane_xy = jnp.concatenate([jnp.full(prop_params["shape"][1:], z_p[prop_params["shape"][0]//2])[None], jnp.array(jnp.meshgrid(y_p, x_p, indexing="ij"))]).transpose(1,2,0)
    plane_yz = jnp.concatenate([jnp.array(jnp.meshgrid(z_p, y_p, indexing="ij")), jnp.full(prop_params["shape"][:-1], x_p[prop_params["shape"][-1]//2])[None]]).transpose(2,1,0)
    plane_xz = jnp.concatenate([jnp.full(prop_params["shape"][::2], y_p[prop_params["shape"][1]//2])[None], jnp.array(jnp.meshgrid(z_p, x_p, indexing="ij"))])
    plane_xz = plane_xz.at[jnp.array([0,1])].set(plane_xz[jnp.array([1,0])]).transpose(2,1,0)

    ext_prop_xy = [x_p[0], x_p[-1], y_p[-1], y_p[0]]
    ext_prop_zy = [z_p[0], z_p[-1], y_p[-1], y_p[0]]
    ext_prop_zx = [z_p[0], z_p[-1], x_p[0], x_p[-1]]

    # Definition of the object grid
    obj_shape = jnp.array((16,128,128))
    obj_pix_sizes = jnp.array((l/4, 8*8*l, 8*8*l))
    obj_center = jnp.array(((obj_shape[0]//2)*obj_pix_sizes[0],
                            (obj_shape[1]//2)*obj_pix_sizes[1],
                            (obj_shape[2]//2)*obj_pix_sizes[2]))

    object_params = {
        "shape": obj_shape,
        "pix_sizes": obj_pix_sizes,
        "center": obj_center
    }

    z_o = (jnp.arange(object_params["shape"][0]) - object_params["shape"][0] // 2) * object_params["pix_sizes"][0]
    y_o = (jnp.arange(object_params["shape"][1]) - object_params["shape"][1] // 2) * object_params["pix_sizes"][1]
    x_o = (jnp.arange(object_params["shape"][2]) - object_params["shape"][2] // 2) * object_params["pix_sizes"][2]

    n_a = 1.5
    r2 = (y_o[None,:,None]**2 + x_o[None,None]**2)*jnp.ones_like(z_o)[:,None,None]
    radius = 8*16*20 * l
    n_original = n_a + (0.01*(1-(r2/radius**2)) * (r2 <= (radius**2))) #* (1+(z_o/(2*z_o.max())))[:,None, None]
    eps_a =  n_a**2
    eps_original = n_original**2 - eps_a

    ext_obj_xy = [x_o[0], x_o[-1], y_o[-1], y_o[0]]
    ext_obj_zy = [z_o[0], z_o[-1], y_o[-1], y_o[0]]
    ext_obj_zx = [z_o[0], z_o[-1], x_o[0], x_o[-1]]

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(eps_original[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow(eps_original[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow(eps_original[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Epsilon slice XY")
    sub[1].set_title("Epsilon slice ZY")
    sub[2].set_title("Epsilon slice ZX")
    plt.tight_layout()

    # Definition of the input waves
    modes = 3
    wx, wy = (8*16*20 * l, 8*16*20 * l)
    G = jnp.exp(-((x_p[None]/wx)**2 + (y_p[:,None]/wy)**2))
    Ui = jnp.array([G * hermite(i//modes)(jnp.sqrt(2)*y_p[:,None]/wy) * hermite(i%modes)(jnp.sqrt(2)*x_p[None]/wx) for i in range(modes**2)], dtype=jnp.complex128)
    Ui = Ui / Ui.max(axis=(1,2))[:,None, None]
    #Ui = jnp.ones(prop_params["shape"][1:], dtype=jnp.complex128)[None]

    fig, sub = plt.subplots(modes,modes)
    if Ui.shape[0] == 1:
        im = sub.imshow(jnp.abs(Ui[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"Intensity Mode (0,0)")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Ui.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Ui[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"Intensity Mode ({i//modes},{i%modes})")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()
    plt.show()
    plt.close("all")

    # Preparation of the variables for propagation loop
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(int(prop_params["shape"][2]), prop_params["pix_sizes"][2]))
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(int(prop_params["shape"][1]), prop_params["pix_sizes"][1]))
    propagator = jnp.exp(-1j * jnp.pi * l * prop_params["pix_sizes"][0] * (fx[None, :]**2 + fy[:, None]**2) / (2 * jnp.sqrt(eps_a))) * 1 * ((fx[None, :]**2 + fy[:, None]**2) <= (1/l)**2)

    y_plane, x_plane = jnp.meshgrid(y_p, x_p, indexing='ij')
    plane_prop = jnp.array([z_p[0] * jnp.ones_like(y_plane), y_plane, x_plane]).transpose(1,2,0)
    plane_rel = ((plane_prop + prop_params["center"][None, None] - object_params["center"][None, None]) / object_params["pix_sizes"][None, None]
                 + jnp.array([object_params["shape"][0]//2, object_params["shape"][1]//2, object_params["shape"][2]//2]))

    norm_vect = jnp.array([prop_params["pix_sizes"][0], 0, 0])
    norm_vect_rel = norm_vect / object_params["pix_sizes"]

    Uo = jnp.copy(Ui)

    # Propagation loop
    for i in tqdm(range(prop_params["shape"][0])):
        
        plane = plane_rel + norm_vect_rel[None, None] * i
        eps_slice = trilinear_interpolate(plane, eps_original, outside = 0.0)

        Uo = iFT2(propagator[None] * FT2(
                  jnp.exp(1j * jnp.pi * eps_slice[None] * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a))) * iFT2(
                  propagator[None] * FT2(Uo))))
    
    U_ro, energies = propagate_modes(eps_original, 
                                     propagator, 
                                     Ui, 
                                     plane_rel, 
                                     norm_vect_rel, 
                                     int(prop_params["shape"][0]),
                                     prop_params["pix_sizes"][0], 
                                     eps_a, 
                                     l)

    fig, sub = plt.subplots(modes,modes, sharex=True, sharey=True)
    if Ui.shape[0] == 1:
        im = sub.imshow(jnp.abs(Ui[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"Intensity Mode (0,0)")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Ui.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Ui[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"Intensity Mode ({i//modes},{i%modes})")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()

    fig, sub = plt.subplots(modes,modes, sharex=True, sharey=True)
    if Ui.shape[0] == 1:
        im = sub.imshow(jnp.abs(Uo[0]-Ui[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"Intensity Mode (0,0)")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Uo.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Uo[i]-Ui[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"Intensity Mode ({i//modes},{i%modes})")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()

    fig, sub = plt.subplots(modes,modes, sharex=True, sharey=True)
    if Ui.shape[0] == 1:
        sub.plot(z_p, energies[0])
        sub.set_title(f"Energy Mode (0,0)")
        sub.set_xlabel("$z(\\lambda)$")
        sub.set_ylabel("Energy")
    else:
        for i in range(Uo.shape[0]):
            im = sub[i//modes,i%modes].plot(z_p, energies[i])
            sub[i//modes,i%modes].set_title(f"Energy Mode ({i//modes},{i%modes})")
            sub[i//modes,i%modes].set_xlabel("$z(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("Energy")
    plt.tight_layout()
    plt.show()
    plt.close("all")
    #%%
    """
    Reconstruction
    """
    coef_shape = (16,6)
    # Initialize the variables
    Z_poly, L_poly = poly_exp(jnp.ones(coef_shape), prop_params["shape"], prop_params["pix_sizes"], radius)
    real_coef = poly_adj(eps_original, jnp.ones(coef_shape), object_params["pix_sizes"], radius)
    real_dist = poly_sum(real_coef, *poly_exp(jnp.ones(coef_shape), object_params["shape"], object_params["pix_sizes"], radius))

    coef_guess = jnp.zeros(coef_shape) # real_coef.copy()
    lr = 1e-2
    epsilon_tv = 1e-6
    alpha_tv = 1e-5
    n_iterations = 500
    tau = 100

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(coef_guess)
    
    seq_train = tqdm(range(n_iterations), desc="Training", leave=True)
    losses = []

    # Training loop
    for epoch in seq_train:
        try:
            grad = jnp.zeros_like(coef_guess)

            # Forward propagation
            U_sim, _ = propagate_modes_poly(coef_guess,
                                            Z_poly,
                                            L_poly, 
                                            propagator, 
                                            Ui, 
                                            #int(prop_params["shape"][0]),
                                            prop_params["pix_sizes"][0], 
                                            eps_a, 
                                            l)
            U_diff = U_sim - Uo
            U_diff_inv = FT2(jnp.conjugate(propagator) * iFT2(U_diff))
            U_sim_inv = FT2(jnp.conjugate(propagator) * iFT2(U_sim))

            for i in tqdm(range(prop_params["shape"][0]), leave=False):
                d_inv = prop_params["shape"][0] - 1 - i
                
                eps_slice = (Z_poly[:, None] * L_poly[None, :, d_inv, None, None]).sum(axis=(0,1))
                grad_slice = (2 * jnp.pi * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a)) * jnp.imag(U_diff_inv * jnp.conjugate(U_sim_inv))) / (jnp.prod(prop_params["shape"][1::]))
                grad_slice = grad_slice.mean(axis=0)

                n_total = jnp.prod(prop_params["shape"])


                grad += (grad_slice[None, None] * Z_poly[:, None] * L_poly[None, :, d_inv, None, None]).sum(axis=(2,3))
                grad = grad #/ n_total

                U_diff_inv = jnp.exp(-1j * jnp.pi * eps_slice[None] * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a))) * FT2(
                        (jnp.conjugate(propagator)[None]**2) * iFT2(U_diff_inv))
                
                U_sim_inv = jnp.exp(-1j * jnp.pi * eps_slice[None] * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a))) * FT2(
                        jnp.conjugate(propagator)[None]**2 * iFT2(U_sim_inv))
                    
            # TV regularization
            #grad_tv = jax.grad(tv_loss_periodic, argnums=0)(guess, epsilon_tv)
            grad_total = grad #+ alpha_tv * grad_tv * jnp.exp(-(epoch**2/tau**2))

            mse_jax = (jnp.abs(U_diff)**2).mean()

            total_err = float(mse_jax)

            updates, opt_state = optimizer.update(grad_total, opt_state)
            coef_guess = optax.apply_updates(coef_guess, updates)
            loss_mse = (jnp.abs(U_sim - Uo)**2).mean()

            seq_train.set_postfix({"Total loss": float(total_err)})
            losses.append(total_err)

            if epoch >= 2:
                if (jnp.abs(losses[-2]-losses[-1]) / losses[-1]) <= 1e-4:
                    break
        
        except KeyboardInterrupt:
            break

    import json
    from config import LOCAL_DATA_DIR

    guess_dist = poly_sum(coef_guess, *poly_exp(jnp.ones(coef_shape), object_params["shape"], object_params["pix_sizes"], radius))


    save_name = "{:03d}".format(1)

    save_path = LOCAL_DATA_DIR / "splitm_grad_ort_func" / "GRIN_lens"
    save_path.mkdir(parents=True, exist_ok=True)
    params_path = (save_path / "params")
    images_path = (save_path / "images")
    params_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    prop_params = {
        "shape": prop_shape.tolist(),
        "pix_sizes": prop_pix_sizes.tolist(),
        "center": prop_center.tolist()
    }
    object_params = {
        "shape": obj_shape.tolist(),
        "pix_sizes": obj_pix_sizes.tolist(),
        "center": obj_center.tolist()
    }
    save_dict = {
        "wavelength": l,
        "prop_params": prop_params,
        "object_params": object_params,
        "eps_a": eps_a,
        "lr": lr,
        "epsilon_tv": epsilon_tv,
        "alpha_tv": alpha_tv,
        "tau": tau,
        "modes": modes,
        "losses": losses,
        "total_error": float(jnp.abs(guess_dist - eps_original).mean()),
        "guess": coef_guess.tolist()
        }

    with open(params_path / f"{save_name}.json", "w") as f:
        json.dump(save_dict, f, indent=4)



    plt.figure()
    plt.title("Coeficients real distribution")
    plt.imshow(real_coef, extent=[0.5,coef_shape[1]+0.5, coef_shape[0]+0.5,0.5])
    plt.ylabel("Zernike order")
    plt.xlabel("Legendre order")
    plt.colorbar()
    plt.savefig(images_path / f"{save_name}_original_coef.jpg", dpi=300)

    plt.figure()
    plt.title("Coeficients real distribution")
    plt.imshow(coef_guess, extent=[0.5,coef_shape[1]+0.5, coef_shape[0]+0.5,0.5])
    plt.ylabel("Zernike order")
    plt.xlabel("Legendre order")
    plt.colorbar()
    plt.savefig(images_path / f"{save_name}_guess_coef.jpg", dpi=300)

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(eps_original[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow(eps_original[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow(eps_original[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Eps original slice XY")
    sub[1].set_title("Eps original slice ZY")
    sub[2].set_title("Eps original slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_original.jpg", dpi=300)

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(guess_dist[object_params["shape"][0]//2], extent=ext_obj_xy)
    im2 = sub[1].imshow(guess_dist[:,:,object_params["shape"][1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow(guess_dist[:,object_params["shape"][2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Eps guess slice XY")
    sub[1].set_title("Eps guess slice ZY")
    sub[2].set_title("Eps guess slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_guess.jpg", dpi=300)

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow((eps_original-guess_dist)[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow((eps_original-guess_dist)[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow((eps_original-guess_dist)[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Eps diff slice XY")
    sub[1].set_title("Eps diff slice ZY")
    sub[2].set_title("Eps diff slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_diff.jpg", dpi=300)

    plt.figure()
    plt.plot(losses, label="MSE loss")
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_losses.jpg", dpi=300)

    U_sim, _ = propagate_modes_poly(coef_guess,
                                    Z_poly,
                                    L_poly, 
                                    propagator, 
                                    Ui, 
                                    prop_params["pix_sizes"][0], 
                                    eps_a, 
                                    l)
    
    fig, sub = plt.subplots(modes,modes, sharex=True, sharey=True)
    if Ui.shape[0] == 1:
        im = sub.imshow(jnp.abs(Uo[0]-U_sim[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"$I_{{\\text{{diff}}}}^{{(0,0)}}$")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Ui.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Uo[i]-U_sim[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"$I_{{\\text{{diff}}}}^{{({i//modes},{i%modes})}}$")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_intensity_diff.jpg", dpi=300)

    fig, sub = plt.subplots(modes,modes, sharex=True, sharey=True)
    if Ui.shape[0] == 1:
        im = sub.imshow(jnp.abs(Uo[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"$I^{{(0,0)}}$")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Ui.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Uo[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"$I^{{({i//modes},{i%modes})}}$")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()
    plt.savefig(images_path / f"{save_name}_intensity.jpg", dpi=300)

    plt.show()
