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

def tv_loss_periodic(n, eps=1e-6):
    """Total variation isotropic regularizer (3D).
    n: [D, H, W]  índice de refracción reconstruido
    """
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n
    tv = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    return jnp.sum(tv)

def rbf_init(
    pix_sizes: jnp.ndarray, 
    shape: jnp.ndarray, 
    n_rbf: Tuple[int,...],
    li: jnp.ndarray,
    key: int = 1
    ) -> jnp.ndarray:

    coords = pix_sizes * ((((jnp.array(jnp.meshgrid(*[jnp.arange(i) for i in n_rbf], indexing='ij')).reshape(len(n_rbf), -1) + 1) * (shape / (1 + jnp.array(n_rbf)))[:,None]).T) - (shape[None] // 2))
    li_arr = jnp.tile(li, (coords.shape[0], 1))
    keygen = random.key(key)

    weights = random.uniform(keygen, shape=(coords.shape[0],1), minval=0.0, maxval=1.0)
    params = jnp.concatenate(
        [weights, jnp.concat(
            [jnp.concat(
                [coords[:, i, None],li_arr[:, i, None]], 
                        axis=1) for i in range(len(n_rbf))], 
            axis=1)], 
        axis = 1)
    return params

def rbf_eval(
        rbf_params: jnp.ndarray,
        coords: jnp.ndarray
    ) -> jnp.ndarray:
    result = jnp.zeros_like(coords[:,:,0])
    for i in range(len(rbf_params)):
        dims = coords.shape[-1]
        dims_exp = dims*[None]
        result += rbf_params[i,0] * jnp.exp(-((coords - rbf_params[i,1::2][*dims_exp])**2/(rbf_params[i,2::2][*dims_exp])**2).sum(axis=-1))
    return result

def rbf_grad(
        rbf_params: jnp.ndarray,
        coords: jnp.ndarray,
        ground_truth: jnp.ndarray
) -> jnp.ndarray:
    
    gradient = jnp.zeros_like(rbf_params)
    result = rbf_eval(rbf_params, coords)
    mse_err = 2 * (result - ground_truth)
    dims = coords.shape[-1]
    dims_exp = dims*[None]

    for i in range(len(rbf_params)):
        common_fact = jnp.exp(-jnp.sum((coords - rbf_params[i,1::2])**2/(rbf_params[i,2::2])**2, axis=-1))
        gradient = gradient.at[i,0].set((mse_err*common_fact).mean())
        gradient = gradient.at[i,1::2].set((mse_err[...,None] * common_fact[...,None] * 2 * rbf_params[i,0] * (coords-rbf_params[i,1::2][*dims_exp]) / (rbf_params[i,2::2][*dims_exp]**2)).mean(axis=(0,1)))
        gradient = gradient.at[i,2::2].set((mse_err[...,None] * common_fact[...,None] * 2 * rbf_params[i,0] * (coords-rbf_params[i,1::2][*dims_exp])**2 / (rbf_params[i,2::2][*dims_exp]**3)).mean(axis=(0,1)))
    return gradient

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
    R = 8*16*20 * l
    n_original = n_a + (0.1*(1-(r2/R**2)) * (r2 <= (R**2)))
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
    modes = 1
    #wx, wy = (8*16*20 * l, 8*16*20 * l)
    #G = jnp.exp(-((x_p[None]/wx)**2 + (y_p[:,None]/wy)**2))
    #Ui = jnp.array([G * hermite(i//modes)(jnp.sqrt(2)*y_p[:,None]/wy) * hermite(i%modes)(jnp.sqrt(2)*x_p[None]/wx) for i in range(modes**2)], dtype=jnp.complex128)
    #Ui = Ui / Ui.max(axis=(1,2))[:,None, None]
    Ui = jnp.ones(prop_params["shape"][1:], dtype=jnp.complex128)[None]

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
    
    start = time.perf_counter()
    U_ro, energies = propagate_modes(eps_original, 
                                     propagator, 
                                     Ui, 
                                     plane_rel, 
                                     norm_vect_rel, 
                                     int(prop_params["shape"][0]),
                                     prop_params["pix_sizes"][0], 
                                     eps_a, 
                                     l)
    end = time.perf_counter()
    print(f"Propagation time JIT: {end - start} seconds")
    
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
        im = sub.imshow(jnp.abs(Uo[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"Intensity Mode (0,0)")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Uo.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Uo[i])**2, extent=ext_prop_xy)
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
    # Initialize the variables
    guess = jnp.zeros_like(eps_original)
    eps_original = eps_original
    lr = 1e-2
    epsilon_tv = 1e-6
    alpha_tv = 1e-5
    n_iterations = 500
    tau = 100
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(guess)
    
    seq_train = tqdm(range(n_iterations), desc="Training", leave=True)
    losses = []

    # Training loop
    for epoch in seq_train:
        try:
            grad = jnp.zeros_like(guess)

            # Forward propagation
            U_sim, _ = propagate_modes(guess, 
                                       propagator, 
                                       Ui, 
                                       plane_rel, 
                                       norm_vect_rel, 
                                       int(prop_params["shape"][0]),
                                       prop_params["pix_sizes"][0], 
                                       eps_a, 
                                       l)
            U_diff = U_sim - Uo
            U_diff_inv = FT2(jnp.conjugate(propagator) * iFT2(U_diff))
            U_sim_inv = FT2(jnp.conjugate(propagator) * iFT2(U_sim))

            for i in tqdm(range(prop_params["shape"][0]), leave=False):
                d_inv = prop_params["shape"][0] - 1 - i
                plane = plane_rel + norm_vect_rel[None, None] * d_inv
                eps_slice, dec_fact, corners, mask = trilinear_interpolate_grad(plane, guess, outside = 0.0)
                grad_slice = (2 * jnp.pi * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a)) * jnp.imag(U_diff_inv * jnp.conjugate(U_sim_inv))) / (jnp.prod(prop_params["shape"][1::]))
                grad_slice = grad_slice[:,None] * dec_fact[None]

                grad = grad.at[corners[:,:,:,0], corners[:,:,:,1], corners[:,:,:,2]].add(jnp.where(mask[None], grad_slice, 0.0).mean(axis=0))

                U_diff_inv = jnp.exp(-1j * jnp.pi * eps_slice[None] * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a))) * FT2(
                        (jnp.conjugate(propagator)[None]**2) * iFT2(U_diff_inv))
                
                U_sim_inv = jnp.exp(-1j * jnp.pi * eps_slice[None] * prop_params["pix_sizes"][0] / (l * jnp.sqrt(eps_a))) * FT2(
                        jnp.conjugate(propagator)[None]**2 * iFT2(U_sim_inv))
                    
            # TV regularization
            grad_tv = jax.grad(tv_loss_periodic, argnums=0)(guess, epsilon_tv)
            grad_total = grad + alpha_tv * grad_tv * jnp.exp(-(epoch**2/tau**2))

            mse_jax = (jnp.abs(U_diff)**2).mean()
            tv_loss = tv_loss_periodic(guess, epsilon_tv)
            tv_contrib = tv_loss * alpha_tv * jnp.exp(-(epoch**2/tau**2))
            total_err = mse_jax + tv_contrib

            updates, opt_state = optimizer.update(grad_total, opt_state)
            guess = optax.apply_updates(guess, updates)
            loss_mse = (jnp.abs(U_sim - Uo)**2).mean()

            seq_train.set_postfix({"Total loss": float(total_err), "MSE": float(mse_jax), "TV": float(tv_contrib)})
            losses.append([total_err, mse_jax, tv_contrib])

        
        except KeyboardInterrupt:
            break

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(eps_original[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow(eps_original[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow(eps_original[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Epsilon real slice XY")
    sub[1].set_title("Epsilon real slice ZY")
    sub[2].set_title("Epsilon real slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(guess[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow(guess[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow(guess[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Epsilon guess slice XY")
    sub[1].set_title("Epsilon guess slice ZY")
    sub[2].set_title("Epsilon guess slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow((eps_original-guess)[eps_original.shape[0]//2], extent=ext_obj_xy, aspect=1)
    im2 = sub[1].imshow((eps_original-guess)[:,:,eps_original.shape[1]//2].T, extent=ext_obj_zy, aspect=0.005)
    im3 = sub[2].imshow((eps_original-guess)[:,eps_original.shape[2]//2].T, extent=ext_obj_zx, aspect=0.005)
    plt.colorbar(im1, ax=sub[0])
    plt.colorbar(im2, ax=sub[1])
    plt.colorbar(im3, ax=sub[2])
    sub[0].set_title("Epsilon diff slice XY")
    sub[1].set_title("Epsilon diff slice ZY")
    sub[2].set_title("Epsilon diff slice ZX")
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].set_xlabel("$z(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    sub[2].set_xlabel("$z(\\lambda)$")
    sub[2].set_ylabel("$x(\\lambda)$")
    plt.tight_layout()

    losses = jnp.array(losses).T
    labels = ["Total Loss", "MSE", "TV"]
    plt.figure()
    for i in range(3):
        plt.plot(losses[i], label=labels[i])
    plt.yscale("log")
    plt.ylabel("Loss")
    plt.xlabel("Iteration")
    plt.legend()

    U_sim, energies = propagate_modes(guess, 
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
        im = sub.imshow(jnp.abs(Uo[0]-U_sim[0])**2, extent=ext_prop_xy)
        plt.colorbar(im, ax=sub)
        sub.set_title(f"Intensity difference Mode (0,0)")
        sub.set_xlabel("$x(\\lambda)$")
        sub.set_ylabel("$y(\\lambda)$")
    else:
        for i in range(Ui.shape[0]):
            im = sub[i//modes,i%modes].imshow(jnp.abs(Uo[i]-U_sim[i])**2, extent=ext_prop_xy)
            plt.colorbar(im, ax=sub[i//modes,i%modes])
            sub[i//modes,i%modes].set_title(f"Intensity difference Mode ({i//modes},{i%modes})")
            sub[i//modes,i%modes].set_xlabel("$x(\\lambda)$")
            sub[i//modes,i%modes].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()
    plt.show()