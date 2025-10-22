import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import optax
from tqdm import tqdm

from fgrinmet.utils import FT2, iFT2, FT2_i, iFT2_i, fft_coord_jax
from fgrinmet.splitm import paraxial_propagator_jax

from splitm_gradient import C_d, C_d_conj

def propagation(
        U_i: jnp.ndarray,
        U_meas: jnp.ndarray,
        inc_n: jnp.ndarray,



) -> jnp.ndarray:
    U_sim = jnp.copy(U_i)
    n_guess = n_a + inc_n * (R0-r2[None].repeat(shape[0], axis=0))


    for d in tqdm(range(shape[0])):
        U_sim = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_sim))))

    return ((jnp.abs(U_sim - U_meas)**2)).mean()

def tv_loss(n, eps=1e-6):
    """Total variation isotropic regularizer (3D).
    n: [D, H, W]  índice de refracción reconstruido
    """
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n
    tv = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    return jnp.sum(tv)

def tv_grad(n, eps=1e-6):
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n

    denom = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    tx = dx / denom
    ty = dy / denom
    tz = dz / denom

    div = (tx - jnp.roll(tx, 1, axis=2)) \
        + (ty - jnp.roll(ty, 1, axis=1)) \
        + (tz - jnp.roll(tz, 1, axis=0))
    return -div

if __name__ == "__main__":
    mpl.rcParams['axes.unicode_minus'] = False
    wavelength = 1.0
    n_a = 1.5
    shape = (5,256,256)
    pix_size = (0.5 * wavelength, 0.25 * wavelength, 0.25 * wavelength)
    extent_xy = [-(shape[2] // 2) * pix_size[2], ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1]]
    extent_zx = [-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2], -(shape[2] // 2) * pix_size[2]]
    extent_zy = [-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1]]
    z = jnp.arange(shape[0]) / shape[0]
    y = (jnp.arange(shape[1]) - shape[1] // 2) / (shape[1] // 2)
    x = (jnp.arange(shape[2]) - shape[2] // 2) / (shape[2] // 2)

    r2 = y[:,None]**2 + x[None]**2
    inc_n = 0.5
    inc_r = 0.5
    R0 = 0.5
    n_original = n_a + inc_n * jnp.where(r2[None].repeat(shape[0], axis=0)<=R0**2, (R0**2-r2[None].repeat(shape[0], axis=0)), 0)
    #n_original = n_a + inc_n * (R0**2-r2[None].repeat(shape[0], axis=0))
    grad_map = R0 - r2
    mask = jnp.ones_like(n_original).astype(bool)

    fig1, sub1 = plt.subplots(1,3)
    fig1.suptitle("Original Index distribution")
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    #im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice])
    im1 = sub1[0].imshow(n_original[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub1[0])
    sub1[0].set_title(f"YX slice at z_pix={z_slice}")
    sub1[0].set_xlabel("$x(\\lambda)$")
    sub1[0].set_ylabel("$y(\\lambda)$")
    #im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T)
    im2 = sub1[1].imshow(n_original[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub1[1])
    sub1[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub1[1].set_xlabel("$z(\\lambda)$")
    sub1[1].set_ylabel("$x(\\lambda)$")
    #im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T)
    im3 =sub1[2].imshow(n_original[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub1[2])
    sub1[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub1[2].set_xlabel("$z(\\lambda)$")
    sub1[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()

    Fy, Fx = fft_coord_jax(shape[1:], pix_size[1:])

    propagator = paraxial_propagator_jax(Fy, Fx, pix_size[0], n_a, wavelength)
    propagator_conj = jnp.conjugate(propagator)
    U_i = jnp.ones(shape[1:])
    U_meas = jnp.copy(U_i)

    for d in tqdm(range(shape[0])):
        U_meas = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d]**2)) * iFT2(propagator * FT2(U_meas))))
    U_meas_conj = jnp.copy(U_meas)
    for d in tqdm(range(shape[0])):
        d_conj = shape[0] - d - 1
        U_meas_conj = iFT2(propagator_conj * FT2(jnp.exp((-1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d_conj]**2)) * iFT2(propagator_conj * FT2(U_meas))))

    plt.figure()
    plt.title("Amplitude propagated with the original index")
    plt.imshow(jnp.abs(U_meas))
    plt.colorbar()
    plt.figure()
    plt.title("Amplitude propagated with the original index (conjugated)")
    plt.imshow(jnp.abs(U_meas_conj))
    plt.colorbar()


    U_back = jnp.copy(U_meas)
    #n_guess = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  (inc_n/2) * (R0-r2[None].repeat(shape[0], axis=0)), 0) #jnp.full(shape, n_a+0.1)
    n_guess = jnp.ones_like(n_original)
    grad_n = jnp.zeros(shape)

    for d in tqdm(range(shape[0]-1, -1, -1)):
        U_back = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * FT2_i(propagator_conj * iFT2_i(U_back))))

    n_epochs = 5000
    seq_train = tqdm(range(n_epochs), desc="Training", leave=True)
    mse_arr = jnp.zeros(n_epochs)
    tv_arr = jnp.zeros(n_epochs)
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(n_guess)
    alpha_tv = 1e-1
    epsilon_tv = 1e-7
    N = shape[1] * shape[2]

    for epoch in seq_train:
        U_prop = jnp.copy(U_i)
        U_corr = jnp.conjugate(U_i - U_back)

        for d in tqdm(range(shape[0]), leave=False):
            #alpha_tv = alpha_tv / (epoch+1) 
            d_conj = shape[0] - d - 1

            U_corr = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d_conj]**2)) * FT2_i(propagator_conj * iFT2_i(U_corr))))
            #n_contrib = (2j* jnp.pi / (wavelength * n_a)) * pix_size[0] * n_guess[d]
            n_contrib = -(2j* jnp.pi / (wavelength * n_a)) * pix_size[0] * n_guess[d] * jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2))

            U_fact = U_corr * n_contrib * iFT2(propagator * FT2(iFT2(propagator * FT2(U_prop))))
            #U_fact_conj = jnp.conjugate(U_fact)
#
            #U_fact_sum = U_fact.mean()
            #U_fact_conj_sum = U_fact_conj.mean()
#
            #n_contrib_conj = jnp.conjugate(n_contrib)
#
            #grad_plane = (n_contrib * U_fact_conj_sum + n_contrib_conj * U_fact_sum)
            grad_plane = jnp.real(U_fact * n_contrib)

            grad_n = grad_n.at[d].set(jnp.real(grad_plane))
            #grad_n = grad_n.at[d].set(jnp.real(U_fact * n_contrib))
            U_prop = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_prop))))

        grad_n = grad_n + alpha_tv * tv_grad(n_guess, epsilon_tv)

        tv = tv_loss(n_guess, epsilon_tv)
        tv_arr = tv_arr.at[epoch].set(tv)
        mse = jnp.mean(jnp.abs(U_prop - U_meas)**2)
        mse_arr = mse_arr.at[epoch].set(float(mse))
        updates, opt_state = optimizer.update(grad_n, opt_state, n_guess)
        n_guess = optax.apply_updates(n_guess, updates)

        seq_train.set_postfix({"MSE": float(mse), "TV": float(tv)})
        if mse <= 4e-6:
            break

    plt.figure()
    plt.title(f"Gradient of the plane at z(index)={d}")
    plt.imshow(grad_plane.real)
    plt.colorbar()

    fig, sub = plt.subplots(1,2)
    plt.title("Amplitude propagated with the backpropagated index")
    sub[0].set_title("Amplitude propagated with the guessed index (conjugated)")
    im = sub[0].imshow(jnp.abs(U_prop), extent=extent_xy)
    plt.colorbar(im, ax=sub[0])
    sub[1].plot(mse_arr[:epoch])
    sub[1].set_title("Cost")
    sub[1].set_xlabel("Iteration")
    sub[1].set_ylabel("$\\int ||U_o^{sim}-U_o^{meas}||$")
    plt.tight_layout()

    plt.figure()
    plt.title("R")
    plt.plot(tv_arr)
    plt.xlabel("iter")
    plt.ylabel("TV loss")

    fig5, sub5 = plt.subplots(1,3)
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    fig5.suptitle("Backpropagated Index distribution")
    im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice], extent=extent_xy)
    #im1 = sub5[0].imshow(n_guess[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub5[0])
    sub5[0].set_title(f"YX slice at z_pix={z_slice}")
    sub5[0].set_xlabel("$x(\\lambda)$")
    sub5[0].set_ylabel("$y(\\lambda)$")
    im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T, extent=extent_zx)
    #im2 = sub5[1].imshow(n_guess[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub5[1])
    sub5[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub5[1].set_xlabel("$z(\\lambda)$")
    sub5[1].set_ylabel("$x(\\lambda)$")
    im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T, extent=extent_zy)
    #im3 =sub5[2].imshow(n_guess[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub5[2])
    sub5[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub5[2].set_xlabel("$z(\\lambda)$")
    sub5[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()


plt.show()