import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from fgrinmet.utils import FT2, iFT2, FT2_i, iFT2_i, fft_coord_jax
from fgrinmet.splitm import paraxial_propagator_jax

from splitm_gradient import C_d, C_d_conj

if __name__ == "__main__":
    wavelength = 1.0
    n_a = 1.5
    shape = (50,256,256)
    pix_size = (0.5 * wavelength, 0.25 * wavelength, 0.25 * wavelength)
    z = jnp.arange(shape[0]) / shape[0]
    y = (jnp.arange(shape[1]) - shape[1] // 2) / (shape[1] // 2)
    x = (jnp.arange(shape[2]) - shape[2] // 2) / (shape[2] // 2)

    r2 = y[:,None]**2 + x[None]**2
    inc_n = 1.5
    inc_r = 0.5
    R0 = 0.5
    n_original = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  inc_n * (R0-r2[None].repeat(shape[0], axis=0)), 0)#/ (1+inc_r*z[:,None, None])**2)
    mask = jnp.ones_like(n_original).astype(bool)

    plt.figure()
    plt.imshow(n_original[shape[0] // 2])
    plt.colorbar()

    Fy, Fx = fft_coord_jax(shape[1:], pix_size[1:])

    propagator = paraxial_propagator_jax(Fy, Fx, pix_size[0], n_a, wavelength)
    propagator_conj = jnp.conjugate(propagator)
    U_i = jnp.ones(shape[1:])
    U_meas = jnp.copy(U_i)

    for d in tqdm(range(shape[0])):
        U_meas = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d]**2)) * iFT2(propagator * FT2(U_meas))))

    plt.figure()
    plt.imshow(jnp.abs(U_meas))
    plt.colorbar()

    U_back = jnp.copy(U_meas)
    n_guess = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  (inc_n/2) * (R0-r2[None].repeat(shape[0], axis=0)), 0) #jnp.full(shape, n_a+0.1)
    grad_n = jnp.zeros(shape)

    for d in tqdm(range(shape[0]-1, -1, -1)):
        U_back = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * FT2_i(propagator_conj * iFT2_i(U_back))))

    n_epochs = 100
    seq_train = tqdm(range(n_epochs), desc="Training", leave=True)
    mse_arr = jnp.zeros(n_epochs)
    optimizer = optax.lion(1e-3)
    opt_state = optimizer.init(n_guess)

    for epoch in seq_train:
        U_prop = jnp.copy(U_i)
        U_corr = jnp.conjugate(U_i - U_back)

        for d in tqdm(range(shape[0]), leave=False):
            d_conj = shape[0] - d - 1

            U_corr = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d_conj]**2)) * FT2_i(propagator_conj * iFT2_i(U_corr))))
            U_prop = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_prop))))
            U_fact = U_corr * iFT2(propagator * FT2(iFT2(propagator * FT2(U_prop))))
            U_fact_conj = jnp.conjugate(U_fact)

            U_fact_sum = U_fact.mean()
            U_fact_conj_sum = U_fact_conj.mean()

            n_contrib = (2j* jnp.pi / (wavelength * n_a)) * pix_size[0] * n_guess[d] #* jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2-n_guess[d]**2))
            n_contrib_conj = jnp.conjugate(n_contrib)

            grad_plane = (n_contrib * U_fact_conj_sum + n_contrib_conj * U_fact_sum)

            grad_n = grad_n.at[d].set(jnp.real(grad_plane))

        
        mse = jnp.mean(jnp.abs(U_prop - U_meas)**2)
        mse_arr = mse_arr.at[epoch].set(float(mse))
        updates, opt_state = optimizer.update(grad_n, opt_state, n_guess)
        n_guess = optax.apply_updates(n_guess, updates)

        seq_train.set_postfix({"MSE": float(mse)})

    plt.figure()
    plt.imshow(grad_plane.real)
    plt.colorbar()
    plt.show()

    fig, sub = plt.subplots(1,2)
    im = sub[0].imshow(jnp.abs(U_prop))
    plt.colorbar(im, ax=sub[0])
    sub[1].plot(mse_arr)

    fig5, sub5 = plt.subplots(3,1)
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice])
    plt.colorbar(im1, ax=sub5[0])
    sub5[0].set_title(f"YX slice at z_pix={z_slice}")
    sub5[0].set_xlabel("$x(\\lambda)$")
    sub5[0].set_ylabel("$y(\\lambda)$")
    im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T)
    plt.colorbar(im2, ax=sub5[1])
    sub5[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub5[1].set_xlabel("$z(\\lambda)$")
    sub5[1].set_ylabel("$x(\\lambda)$")
    im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T)
    plt.colorbar(im3, ax=sub5[2])
    sub5[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub5[2].set_xlabel("$z(\\lambda)$")
    sub5[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()


plt.show()