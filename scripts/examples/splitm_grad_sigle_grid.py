import numpy as np
import os
os.environ["XLA_FLAGS"] = "--xla_gpu_autotune_level=0"
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import hermite
from dataclasses import dataclass
from functools import partial

from fgrinmet.utils.operators import FT2, iFT2
from fgrinmet.constructions.geometries import FGRINcomponent

def propagate(eps_distr: jnp.ndarray, propagator: jnp.ndarray, U_in: jnp.ndarray, dz: float, eps_a: float, l:float) -> jnp.ndarray:
    """Propagate the input field U_in through a slice of thickness dz with permittivity distribution eps_distr.

    Args:
        eps_distr (jnp.ndarray): Permittivity distribution of the slice.
        propagator (jnp.ndarray): Precomputed propagator in Fourier space.
        U_in (jnp.ndarray): Input field to be propagated.
        dz (float): Thickness of the slice.
        eps_a (float): Ambient permittivity.
        l (float): Wavelength.

    Returns:
        jnp.ndarray: Output field after propagation.
    """
    def step(U_carry, index):
        U_mid = jnp.exp(1j * jnp.pi * eps_distr[index] * dz / (l * jnp.sqrt(eps_a))) * iFT2(propagator * FT2(U_carry))
        U_out = iFT2(propagator * FT2(U_mid))

        return U_out, None

    U_out = jax.lax.scan(step, U_in, jnp.arange(eps_distr.shape[0]))
    return U_out

def mse(eps_distr: jnp.ndarray, propagator: jnp.ndarray, U_in: jnp.ndarray, U_target: jnp.ndarray, dz: float, eps_a: float, l:float) -> float:
    U_sim, _ = propagate(eps_distr, propagator, U_in, dz, eps_a, l)
    return jnp.mean(jnp.abs(U_sim - U_target)**2)
    
def tv_loss_periodic(n, eps=1e-6):
    """Total variation isotropic regularizer (3D).
    n: [D, H, W]  índice de refracción reconstruido
    """
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n
    tv = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    return jnp.sum(tv)

"""
Simulation"""
if __name__ == "__main__":
    l = 1
    epsilon_a = 1.5**2
    name = "Quadratic distribution"
    n_points = (128, 128, 128)
    pix_sizes = (l, l, l)#(6 * l / n_points[0], 10 * l / n_points[1], 10 * l / n_points[2])
    cuad_cube_dist = lambda Z, Y, X, a_0:(1*((0.25**2-(X/np.max(X)-0.5)**2-(Y/np.max(Y)-0.5)**2)))*((0.25**2)>=((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#0.5*(Z/np.max(Z)-0.5)/(1+((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#+0.5 - ((X-1000*l)**2+(Y-1000*l)**2) * (X-1000)  / 1000000000
    #cuad_cube_dist = lambda Z, Y, X, a_0:(((X/np.max(X))-0.5)*60*((0.25**2-(X/np.max(X)-0.5)**2-(Y/np.max(Y)-0.5)**2)))*((0.25**2)>=((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#0.5*(Z/np.max(Z)-0.5)/(1+((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#+0.5 - ((X-1000*l)**2+(Y-1000*l)**2) * (X-1000)  / 1000000000
    cuad_cube = FGRINcomponent(name, cuad_cube_dist, n_points, pix_sizes, a_0=jnp.array([0]))
    epsilon_original = cuad_cube.generate()

    extent_yx = cuad_cube.genextent[2] + cuad_cube.genextent[1]
    extent_xz = cuad_cube.genextent[0] + cuad_cube.genextent[2]

    fig, sub = plt.subplots(1,3,figsize=(10,5))
    im1 = sub[0].imshow((epsilon_original[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_title("XY-Original Slice")
    im2 = sub[1].imshow((epsilon_original[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_title("XZ-Original Slice")
    im3 = sub[2].imshow((epsilon_original[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub[2])
    sub[2].set_title("YZ-Original Slice")
    cuad_cube.show()
    
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[1], d=pix_sizes[1]))
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[2], d=pix_sizes[2]))
    paraxial_fact = jnp.exp(-1j * jnp.pi * l * pix_sizes[0] * (fx[None, :]**2 + fy[:, None]**2) / (2 * jnp.sqrt(epsilon_a)))# * ((fx[None, :]**2 + fy[:, None]**2) <= (1/l)**2))

    coordinates_obj = cuad_cube.coordinates1D
    Z, Y, X = jnp.meshgrid(*coordinates_obj, indexing='ij')
    

    wx, wy = (50 * l, 50 * l)
    Ui = jnp.ones(n_points[1:], dtype=jnp.complex128) #* jnp.exp(-(((X[0]-500*l)/wx)**2 + ((Y[0]-500*l)/wy)**2)) * ((((X[0]-500*l)/wx)**2 + ((Y[0]-500*l)/wy)**2) <= 1) #jnp.ones(n_points[1:])#jnp.exp(-(((X[0]-50)/wx)**2 + ((Y[0]-50)/wy)**2))
    Uo = jnp.copy(Ui)
    Uo_inv = jnp.copy(Ui)

    U_o_mid = jnp.full((n_points[0], n_points[1]), jnp.nan, dtype=jnp.complex128)
    U_o_mid = U_o_mid.at[0,:].set(Ui[n_points[1]//2,:])
    U_o_mid_inv = jnp.full((n_points[0], n_points[1]), jnp.nan, dtype=jnp.complex128)
    U_o_mid_inv = U_o_mid_inv.at[0,:].set(Ui[n_points[1]//2,:])
    energy = jnp.zeros((n_points[0]+1))
    energy = energy.at[0].set(jnp.sum(jnp.abs(Ui)**2, axis=(-2,-1)) * pix_sizes[1] * pix_sizes[2])

    for slice_idx in tqdm(range(n_points[0])):
        epsilon_slice = epsilon_original[slice_idx, :, :]
        Uo = iFT2(paraxial_fact * FT2(
                  jnp.exp(1j * jnp.pi * epsilon_slice * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                  paraxial_fact * FT2(Uo))))
        U_o_mid = U_o_mid.at[slice_idx+1,:].set(Uo[n_points[1]//2,:])
        
        epsilon_slice_inv = epsilon_original[n_points[0] - 1 - slice_idx, :, :]

        Uo_inv = iFT2((paraxial_fact**-1) * FT2(
                  jnp.exp(-1j * jnp.pi * epsilon_slice_inv * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                  (paraxial_fact**-1) * FT2(Uo_inv))))
        U_o_mid_inv = U_o_mid_inv.at[slice_idx+1,:].set(Uo_inv[n_points[1]//2,:])
        energy = energy.at[slice_idx+1].set(jnp.sum(jnp.abs(Uo)**2) * pix_sizes[1] * pix_sizes[2])
        
        """if slice_idx % 20 == 0:
            fig, sub = plt.subplots(1,3,figsize=(15,5))
            sub[0].set_title(f"Slice {slice_idx} - XZ Plane")
            im1 = sub[0].imshow(jnp.abs(U_o_mid.T), extent=extent_xz)
            plt.colorbar(im1, ax=sub[0])
            sub[1].set_title(f"Slice {slice_idx} - XY Plane")
            im2 = sub[1].imshow(jnp.abs(Uo.T), extent=extent_yx)
            plt.colorbar(im2, ax=sub[1])
            sub[2].set_title(f"Slice {slice_idx} - XY Plane FT")
            im3 = sub[2].imshow(jnp.log(jnp.abs(FT2(Uo.T))), extent=extent_yx)
            plt.colorbar(im3, ax=sub[2])"""
    
    plt.figure()
    plt.plot(energy)
    plt.ylabel("Energy")
    plt.xlabel("Propagation Slice")

    fig, sub = plt.subplots(1,2,figsize=(10,5))
    sub[0].set_title("Input Field")
    im1 = sub[0].imshow(jnp.abs(Ui), extent=extent_yx)#jnp.log(jnp.abs(FT2(Uo))))
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    im2 = sub[1].imshow(jnp.abs(Uo), extent=extent_yx)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_xlabel("$x(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")

    U_o_prop, _ = propagate(jnp.array(epsilon_original), paraxial_fact, Ui, pix_sizes[0], epsilon_a, l)
    plt.figure()
    plt.imshow(jnp.abs(U_o_prop.T), extent=extent_yx)
    plt.ylabel("$x(\\lambda)$")
    plt.xlabel("$y(\\lambda)$")
    plt.show()

    fig = plt.figure()
    plt.imshow(jnp.abs(U_o_mid.T), extent=extent_xz)
    plt.ylabel("$x(\\lambda)$")
    plt.xlabel("$z(\\lambda)$")
    plt.show()

    #%%
    """
    Reconstruction
    """
    import optax

    guess = jnp.zeros_like(epsilon_original)
    lr = 1e-1 
    epsilon_tv = 1e-6
    alpha_tv = 1e-5
    n_iterations = 250
    tau = 50
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(guess)
    U_sim = jnp.copy(Ui)
    grad = jnp.zeros_like(guess)
    seq_train = tqdm(range(n_iterations), desc="Training", leave=True)
    losses = []
    for epoch in seq_train:
        try:
            U_sim, _= propagate(jnp.array(guess), paraxial_fact, Ui, pix_sizes[0], epsilon_a, l)
            U_diff = (U_sim - Uo)
            U_diff_inv = FT2(jnp.conjugate(paraxial_fact) * iFT2(U_diff))
            U_sim_inv = FT2(jnp.conjugate(paraxial_fact) * iFT2(U_sim))
            for i in tqdm(range(n_points[0]), leave=False):
                d_inv = n_points[0] - 1 - i

                grad_slice = (2 * jnp.pi * pix_sizes[0] / (l * jnp.sqrt(epsilon_a)) * jnp.imag(U_diff_inv * jnp.conjugate(U_sim_inv))) / (n_points[1] * n_points[2])
                grad = grad.at[d_inv, :, :].set(grad_slice)

                U_diff_inv = jnp.exp(-1j * jnp.pi * guess[d_inv, :, :] * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * FT2(
                        (paraxial_fact**-2) * iFT2(U_diff_inv))
                
                U_sim_inv = jnp.exp(-1j * jnp.pi * guess[d_inv, :, :] * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * FT2(
                        paraxial_fact**-2 * iFT2(U_sim_inv))
                
            #grad_jax = jax.grad(mse, argnums=0)(guess, paraxial_fact, Ui, Uo, pix_sizes[0], epsilon_a, l)
            grad_tv = jax.grad(tv_loss_periodic, argnums=0)(guess, epsilon_tv)
            grad_total = grad + alpha_tv * grad_tv * jnp.exp(-(epoch**2/tau**2))

            mse_jax = mse(guess, paraxial_fact, Ui, Uo, pix_sizes[0], epsilon_a, l)
            tv_loss = tv_loss_periodic(guess, epsilon_tv)
            tv_contrib = tv_loss * alpha_tv * jnp.exp(-(epoch**2/tau**2))
            total_err = mse_jax + tv_contrib

            #err = jnp.linalg.norm(grad- grad_jax) / jnp.linalg.norm(grad_jax)
            #print(err)

            updates, opt_state = optimizer.update(grad_total, opt_state)
            guess = optax.apply_updates(guess, updates)
            loss_mse = (jnp.abs(U_sim - Uo)**2).mean()

            seq_train.set_postfix({"Total loss": float(total_err), "MSE": float(mse_jax), "TV": float(tv_contrib)})
            losses.append([total_err, mse_jax, tv_contrib])
        
        except KeyboardInterrupt:
            #fig, sub = plt.subplots(2,3)
            #mult = 9
            #for i in range(3):
            #    im1 = sub[0,i].imshow(grad[-1-i*mult], extent=extent_yx)
            #    plt.colorbar(im1, ax=sub[0,i])
            #    sub[0,i].set_title(f"Gradient Slice {-1-i*mult} (Manual)")
            #    im2 = sub[1,i].imshow(((grad_jax[-1-i*mult])), extent=extent_yx)
            #    plt.colorbar(im2, ax=sub[1,i])
            #    sub[1,i].set_title(f"Gradient Slice {-1-i*mult} (Jax autograd)")
            #plt.tight_layout()
            #plt.show()
            break

    fig, sub = plt.subplots(1,3,figsize=(10,5))
    im1 = sub[0].imshow(jnp.abs(epsilon_original[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_title("XY-Original Slice")
    im2 = sub[1].imshow(jnp.abs(epsilon_original[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_title("XZ-Original Slice")
    im3 = sub[2].imshow(jnp.abs(epsilon_original[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub[2])
    sub[2].set_title("YZ-Original Slice")

    fig1, sub1 = plt.subplots(1,3,figsize=(10,5))
    im1 = sub1[0].imshow((guess[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im1, ax=sub1[0])
    sub1[0].set_title("XY-Guess Slice")
    im2 = sub1[1].imshow((guess[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub1[1])
    sub1[1].set_title("XZ-Guess Slice")
    im3 = sub1[2].imshow((guess[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub1[2])
    sub1[2].set_title("YZ-Guess Slice")

    fig2, sub2 = plt.subplots(1,3,figsize=(10,5))
    im2 = sub2[0].imshow(((guess-epsilon_original)[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im2, ax=sub2[0])
    sub2[0].set_title("XY-Guess Slice")
    im2 = sub2[1].imshow(((guess-epsilon_original)[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub2[1])
    sub2[1].set_title("XZ-Guess Slice")
    im3 = sub2[2].imshow(((guess-epsilon_original)[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub2[2])
    sub2[2].set_title("YZ-Guess Slice")

    losses = jnp.array(losses).T
    labels = ["Total Loss", "MSE", "TV"]
    plt.figure()
    for i in range(3):
        plt.plot(losses[i], label=labels[i])
    plt.yscale("log")
    plt.legend()
    plt.show()

    #%%
    """
    Saving the results
    """

    import json
    from config import LOCAL_DATA_DIR

    savenamme = "{:03d}".format(3)

    save_path = LOCAL_DATA_DIR / "Single_Ui" / "scale"
    save_path.mkdir(parents=True, exist_ok=True)
    params_path = (save_path / "params")
    images_path = (save_path / "images")
    params_path.mkdir(parents=True, exist_ok=True)
    images_path.mkdir(parents=True, exist_ok=True)

    with open(params_path / f"{savenamme}.json", "w") as f:
        params = {
            "wavelength": l,
            "ambient_permittivity": float(epsilon_a),
            "lr": lr,
            "epsilon_tv": epsilon_tv,
            "alpha_tv": alpha_tv,
            "tau": tau,
            "shape": n_points,
            "pix_sizes": [float(ps) for ps in pix_sizes],
            "losses": np.asarray(losses).tolist(),
            "total_err": jnp.abs(guess - epsilon_original).sum().item(),
            "description": name
        }
        json.dump(params, f, indent=4)

    fig, sub = plt.subplots(1,2,figsize=(10,5))
    sub[0].set_title("Input Field")
    im1 = sub[0].imshow(jnp.abs(Ui), extent=extent_yx)#jnp.log(jnp.abs(FT2(Uo))))
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    im2 = sub[1].imshow(jnp.abs(Uo), extent=extent_yx)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_xlabel("$x(\\lambda)$")
    sub[1].set_ylabel("$y(\\lambda)$")
    plt.savefig(images_path / f"{savenamme}_fields_xy.png", dpi=300)

    fig = plt.figure()
    plt.imshow(jnp.abs(U_o_mid.T), extent=extent_xz)
    plt.ylabel("$x(\\lambda)$")
    plt.xlabel("$z(\\lambda)$")
    plt.savefig(images_path / f"{savenamme}_fields_xz.png", dpi=300)

    fig, sub = plt.subplots(1,3,figsize=(10,5))
    im1 = sub[0].imshow((epsilon_original[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im1, ax=sub[0])
    sub[0].set_title("XY-Original Slice")
    im2 = sub[1].imshow((epsilon_original[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub[1])
    sub[1].set_title("XZ-Original Slice")
    im3 = sub[2].imshow((epsilon_original[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub[2])
    sub[2].set_title("YZ-Original Slice")
    plt.savefig(images_path / f"{savenamme}_original_eps.png", dpi=300)

    fig1, sub1 = plt.subplots(1,3,figsize=(10,5))
    im1 = sub1[0].imshow((guess[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im1, ax=sub1[0])
    sub1[0].set_title("XY-Guess Slice")
    im2 = sub1[1].imshow((guess[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub1[1])
    sub1[1].set_title("XZ-Guess Slice")
    im3 = sub1[2].imshow((guess[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub1[2])
    sub1[2].set_title("YZ-Guess Slice")
    plt.savefig(images_path / f"{savenamme}_guess_eps.png", dpi=300)

    fig2, sub2 = plt.subplots(1,3,figsize=(10,5))
    im2 = sub2[0].imshow(((guess-epsilon_original)[n_points[0]//2].T), extent=extent_yx)
    plt.colorbar(im2, ax=sub2[0])
    sub2[0].set_title("XY-Guess Slice")
    im2 = sub2[1].imshow(((guess-epsilon_original)[:,n_points[1]//2].T), extent=extent_xz)
    plt.colorbar(im2, ax=sub2[1])
    sub2[1].set_title("XZ-Guess Slice")
    im3 = sub2[2].imshow(((guess-epsilon_original)[:,:,n_points[2]//2].T), extent=extent_xz)
    plt.colorbar(im3, ax=sub2[2])
    sub2[2].set_title("YZ-Guess Slice")
    plt.savefig(images_path / f"{savenamme}_dif_eps.png", dpi=300)

    plt.figure()
    for i in range(3):
        plt.plot(losses[i], label=labels[i])
    plt.yscale("log")
    plt.legend()
    plt.savefig(images_path / f"{savenamme}_losses.png", dpi=300)
    plt.show()