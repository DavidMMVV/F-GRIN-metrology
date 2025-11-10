import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import optax
from tqdm import tqdm
import pcax
from jax import random

from typing import Callable
from pathlib import Path
import os
import json

from fgrinmet.utils import FT2, iFT2, fft_coord_jax
from fgrinmet.splitm import paraxial_propagator_jax, trilinear_interpolate, paraxial_propagation_step_jax
from config import LOCAL_DATA_DIR

from splitm_gradient import C_d, C_d_conj, A_d, A_d_conj

def show_volume(volume, title, pix_size, cmap='seismic'):
    """
    Visualizes all points in a 3D volume.
    """
    volume = jnp.array(volume)
    n_layers = volume.shape[0]
    vmax = jnp.max(volume)
    vmin = jnp.min(volume)

    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True)
    axes = axes.flatten()

    for z in range(n_layers):
        axes[z].imshow(
            volume[z],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            extent=[0, volume.shape[2]*pix_size[2],
                    volume.shape[1]*pix_size[1], 0]
        )
        axes[z].set_title(f'Capa z={z} (prof={z*pix_size[0]:.1f} μm)')
        axes[z].set_xlabel('x (μm)')
        axes[z].set_ylabel('y (μm)')

    # Desactivar ejes vacíos
    #for i in range(n_layers, len(axes)):
    #    axes[i].axis('off')

    # Añadir colorbar manualmente
    # Posición [left, bottom, width, height] en coordenadas figura
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # ajusta según necesidad
    norm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    norm.set_array([])
    cbar = fig.colorbar(norm, cax=cbar_ax)
    cbar.set_label('Δ valor')

    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0,0,0.9,0.95])


def plot_complex_field(Z, extent=None, title="Complex Field", cmap="viridis"):
    if not jnp.iscomplexobj(Z):
        raise ValueError("Input array Z must contain complex numbers.")

    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(title, fontsize=14)

    # Real part
    im0 = axs[0].imshow(jnp.real(Z), extent=extent, origin='lower', cmap=cmap)
    axs[0].set_title("Real part")
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    plt.colorbar(im0, ax=axs[0])

    # Imaginary part
    im1 = axs[1].imshow(jnp.imag(Z), extent=extent, origin='lower', cmap=cmap)
    axs[1].set_title("Imaginary part")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    plt.colorbar(im1, ax=axs[1])

    plt.tight_layout()

def propagation(
        U_i: jnp.ndarray,
        U_meas: jnp.ndarray,
        inc_n: jnp.ndarray,
        R0: float,
        shape: tuple,
        pix_size: tuple,
        r2: jnp.ndarray,
        propagator: jnp.ndarray,
        wavelength: float,
        n_a: float
) -> jnp.ndarray:
    U_sim = jnp.copy(U_i)
    #n = n_a + inc_n * (R0-r2[None].repeat(shape[0], axis=0))
    n = n_a + inc_n * jnp.where((r2<=R0**2)[None].repeat(shape[0], axis=0), R0**2-r2[None].repeat(shape[0], axis=0), 0)


    for d in range(shape[0]):
        U_sim = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n[d]**2)) * iFT2(propagator * FT2(U_sim))))

    L2 = ((jnp.abs(U_sim - U_meas)**2)).sum()*pix_size[1]*pix_size[2]

    return L2


def propagation_step(U_sim, n_d, propagator, wavelength, n_a, pix_size):
    phase = jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_d**2))
    U_next = iFT2(propagator * FT2(phase * iFT2(propagator * FT2(U_sim))))
    return U_next, None

def propagation_n(
        U_i: jnp.ndarray,
        U_meas: jnp.ndarray,
        n: jnp.ndarray,
        shape: tuple,
        pix_size: tuple,
        propagator: jnp.ndarray,
        wavelength: float,
        n_a: float
) -> jnp.ndarray:
    # Run the scan along the depth axis
    U_final, _ = jax.lax.scan(
        lambda U, n_d: propagation_step(U, n_d, propagator, wavelength, n_a, pix_size),
        init=U_i,
        xs=n
    )

    L2 = jnp.sum(jnp.abs(U_final - U_meas)**2) * pix_size[1] * pix_size[2]
    return L2

def propagation_n_plus(
        U_i: jnp.ndarray,
        U_meas: jnp.ndarray,
        n: jnp.ndarray,
        shape: tuple,
        pix_size: tuple,
        propagator: jnp.ndarray,
        wavelength: float,
        n_a: float,
        n_0: float,
        d_e: float,
        Fy: jnp.ndarray,
        Fx: jnp.ndarray
) -> jnp.ndarray:
    # Run the scan along the depth axis
    U_final, _ = jax.lax.scan(
        lambda U, n_d: propagation_step(U, n_d, propagator, wavelength, n_a, pix_size),
        init=U_i,
        xs=n
    )

    U_final = iFFT2((jnp.exp(1j * wavelength * jnp.pi * d_e * (Fx**2 + Fy**2) / n_0)) * FFT2(U_final))
    L2 = jnp.sum(jnp.abs(U_final - U_meas)**2) * pix_size[1] * pix_size[2]
    return L2


def tv_loss_periodic(n, eps=1e-6):
    """Total variation isotropic regularizer (3D).
    n: [D, H, W]  índice de refracción reconstruido
    """
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n
    tv = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    return jnp.sum(tv)

def tv_loss_no_periodic(n, eps=1e-6):
    """Total variation isotropic regularizer (3D, sin condiciones periódicas).
    
    n: [D, H, W]  índice de refracción reconstruido
    """
    dx = n[:, :, 1:] - n[:, :, :-1]
    dy = n[:, 1:, :] - n[:, :-1, :]
    dz = n[1:, :, :] - n[:-1, :, :]

    dx = jnp.pad(dx, ((0, 0), (0, 0), (0, 1)))
    dy = jnp.pad(dy, ((0, 0), (0, 1), (0, 0)))
    dz = jnp.pad(dz, ((0, 1), (0, 0), (0, 0)))

    tv = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps)
    return jnp.sum(tv)

def tv_grad(n, eps=1e-6):
    dx = jnp.roll(n, -1, axis=2) - n
    dy = jnp.roll(n, -1, axis=1) - n
    dz = jnp.roll(n, -1, axis=0) - n

    denom = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps**2)
    tx = dx / denom
    ty = dy / denom
    tz = dz / denom

    div = (tx - jnp.roll(tx, 1, axis=2)) \
        + (ty - jnp.roll(ty, 1, axis=1)) \
        + (tz - jnp.roll(tz, 1, axis=0))
    return -div

if __name__ == "__main__":

    dat_save_dir = (LOCAL_DATA_DIR / Path(__file__).name.split(".")[0])
    os.makedirs(dat_save_dir, exist_ok=True)    
    save = True

    #%% Definition of parameters
    mpl.rcParams['axes.unicode_minus'] = False
    wavelength = 1.0
    n_a = 1.5
    shape = (6,256,256)
    pix_size = (0.25 * wavelength, 0.125 * wavelength, 0.125 * wavelength)
    fft_pix_size = (1 / ((shape[0] - 1) * pix_size[0]), 1 / ((shape[1] - 1) * pix_size[1]), 1 / ((shape[2] - 1) * pix_size[2]))

    extent_xy = (-(shape[2] // 2) * pix_size[2], ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1])
    extent_zx = (-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2], -(shape[2] // 2) * pix_size[2])
    extent_zy = (-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1])
    z = jnp.arange(shape[0]) / shape[0]
    y = (jnp.arange(shape[1]) - shape[1] // 2) / (shape[1] // 2)
    x = (jnp.arange(shape[2]) - shape[2] // 2) / (shape[2] // 2)

    FFT2: Callable[[jnp.ndarray], jnp.ndarray] = lambda var : (FT2(var) * pix_size[1] * pix_size[2])
    iFFT2: Callable[[jnp.ndarray], jnp.ndarray] = lambda var : (iFT2(var) / (pix_size[1] * pix_size[2]))

    FFT2 = jax.jit(FFT2)
    iFFT2 = jax.jit(iFFT2)

    r2 = y[:,None]**2 + x[None]**2
    inc_n = 0.5
    inc_r = 0.5
    R0 = 0.5
    #n_original = n_a + inc_n * jnp.where(r2[None].repeat(shape[0], axis=0)<=R0**2, (1 + (r2 / R0**2) - (2*r2**(1/2) / R0))[None].repeat(shape[0], axis=0), 0)# * x[None,None]
    #n_original = n_a + inc_n * (R0**2-r2[None].repeat(shape[0], axis=0))
    n_original = n_a + inc_n * jnp.where((r2<=R0**2)[None].repeat(shape[0], axis=0), R0**2-r2[None].repeat(shape[0], axis=0), 0)# * (-1 + shape[0] / ((1+z[:,None,None]))) / (shape[0]-1)
    #n_original = n_a + inc_n * jnp.where((r2<=R0**2)[None].repeat(shape[0], axis=0), (1+R0**2-r2)[None].repeat(shape[0], axis=0), 0)
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
    sub1[1].set_aspect(0.5)  # < 1 ensancha el eje X
    sub1[2].set_aspect(0.5)
    plt.tight_layout()

    Fy, Fx = fft_coord_jax(shape[1:], pix_size[1:])

    propagator = paraxial_propagator_jax(Fy, Fx, pix_size[0], n_a, wavelength)
    propagator_conj = jnp.conjugate(propagator)
    U_i = jnp.ones(shape[1:]).astype(jnp.complex128) * jnp.exp(1j*jnp.pi/4) #* jnp.exp(-r2/R0**2) * jnp.exp(2j * jnp.pi * (1e3/1e2) * jnp.sin(theta) * y[:,None])
    U_meas = jnp.copy(U_i)

    # Compute the measured field
    for d in tqdm(range(shape[0])):
        U_meas = iFFT2(propagator * FFT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d]**2)) * iFFT2(propagator * FFT2(U_meas))))
    
    #%% Checking the hermitian conjugate operator properies

    # Check the properties of the hermitian adjoint operator
    U_meas_conj = jnp.copy(U_i)
    for d in tqdm(range(shape[0])):
        d_conj = shape[0] - d - 1
        #U_meas_conj = FFT2(propagator_conj * iFFT2(jnp.exp((-1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d_conj]**2)) * FFT2(propagator_conj * iFFT2(U_meas_conj))))
        U_meas_conj = FFT2(propagator_conj * iFFT2(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d_conj]**2)) * FFT2(propagator_conj * iFFT2(U_meas_conj))))
    
    U_meas_conj = jnp.conjugate(U_meas_conj)
    plt.figure()
    plt.title("$|A_{d+1}C_d[U^{in}]|$")
    plt.imshow(jnp.abs(U_meas), extent=extent_xy)
    plt.xlabel("$x(\\lambda)$")
    plt.ylabel("$y(\\lambda)$")
    plt.colorbar()
    plt.figure()
    plt.title("$|A^{\\dagger}_{d+1}C^{\\dagger}_d[U^{\\text{in } \\dagger}]|$")
    plt.imshow(jnp.abs(U_meas_conj), extent=extent_xy)
    plt.xlabel("$x(\\lambda)$")
    plt.ylabel("$y(\\lambda)$")
    plt.colorbar()
    plt.figure()
    plt.title("$|A^{\\dagger}_{d+1}C^{\\dagger}_d[U^{\\text{in } \\dagger}]-(A_{d+1}C_d[U^{in}])^\\dagger|$")
    plt.imshow(jnp.abs(U_meas_conj-U_meas), extent=extent_xy)
    plt.xlabel("$x(\\lambda)$")
    plt.ylabel("$y(\\lambda)$")
    plt.colorbar()


    print(f"<AU|U> = {(U_meas * jnp.conjugate(U_i)).sum() * pix_size[1] * pix_size[2]}")
    print(f"<U|A^tU> = {(U_i * U_meas_conj).sum() * pix_size[1] * pix_size[2]}")


    coord_plane_0 = jnp.array([jnp.zeros(shape[1:]), *jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[2]), indexing="ij")]).transpose(1,2,0)
    norm_vect = jnp.array((1,0,0))
    U_test = jnp.exp(r2 / R0**2).astype(jnp.complex128)

    """import time
    from functools import partial

    A_d_comp = jax.jit(partial(A_d, mask=mask, coord_plane_0=coord_plane_0, norm_vect=norm_vect, propagator=propagator, wavelength=wavelength, n_a=n_a, dz=pix_size[0] , Nz=shape[0]), static_argnames="d")
    A_d_conj_comp = jax.jit(partial(A_d_conj, mask=mask, coord_plane_0=coord_plane_0, norm_vect=norm_vect, conj_propagator=propagator_conj, wavelength=wavelength, n_a=n_a, dz=pix_size[0], Nz=shape[0]), static_argnames="d")
    C_d_comp = jax.jit(partial(C_d, mask=mask, coord_plane_0=coord_plane_0, norm_vect=norm_vect, propagator=propagator, wavelength=wavelength, n_a=n_a, dz=pix_size[0]), static_argnames="d")
    C_d_conj_comp = jax.jit(partial(C_d_conj, mask=mask, coord_plane_0=coord_plane_0, norm_vect=norm_vect, conj_propagator=propagator_conj, wavelength=wavelength, n_a=n_a, dz=pix_size[0]), static_argnames="d")
    
    print(f"<U|U> = {(U_i * jnp.conjugate(U_i)).sum() * pix_size[1] * pix_size[2]}")
    print(f"<AU|AU> = {(A_d_comp(U_i=U_i, n_distr=n_original, d=0) * jnp.conjugate(A_d_comp(U_i=U_i, n_distr=n_original, d=0))).sum() * pix_size[1] * pix_size[2]}")
    print(f"<AU|U> = {(A_d_comp(U_i=U_i, n_distr=n_original, d=0) * jnp.conjugate(U_i)).sum() * pix_size[1] * pix_size[2]}")
    print(f"<U|A^tU> = {(U_i * jnp.conjugate(A_d_conj_comp(U_i=U_i, n_distr=n_original, d=0))).sum() * pix_size[1] * pix_size[2]}")
    print(f"<U|A^tU> = {(U_i * A_d_comp(U_i=jnp.conjugate(U_i), n_distr=n_original, d=0)).sum() * pix_size[1] * pix_size[2]}")

    n_guess = jnp.ones(shape)

    print("<Uo|Ad(Uo-Umeas)>")

    start = time.perf_counter()
    U_A_d = A_d_comp(U_i=U_test, n_distr=n_original, d=0)
    end = time.perf_counter()
    print(f"U_A_d time: {end-start}s")
    start = time.perf_counter()
    U_A_d_conj = A_d_conj_comp(U_i=U_A_d, n_distr=n_original, d=0)
    end = time.perf_counter()
    print(f"U_A_d_conj time: {end-start}s")
    plot_complex_field(U_A_d_conj, extent=extent_xy, title="Complex initial field", cmap="viridis")
    print(f"MSE: {jnp.abs(jnp.conj(U_A_d)-U_A_d_conj).sum()}")
    start = time.perf_counter()
    U_C_d = C_d_comp(U_i=U_test, n_distr=n_original, d=shape[0])
    end = time.perf_counter()
    print(f"U_C_d time: {end-start}s")
    start = time.perf_counter()
    U_C_d_conj = C_d_conj_comp(U_i=U_test, n_distr=n_original, d=shape[0])
    end = time.perf_counter()
    print(f"U_C_d_conj time: {end-start}s")
    print(f"MSE: {jnp.abs(jnp.conj(U_C_d)-U_C_d_conj).sum()}")
    plt.close()

    #plot_complex_field(U_i, extent=extent_xy, title="Complex Input Field", cmap="viridis")
    U_control = jnp.copy(U_meas)
    for d in tqdm(range(shape[0])):
        d_conj = shape[0] - d - 1
        U_control = FT2(propagator_conj * iFT2(jnp.exp(-(1j*jnp.pi*pix_size[0] / (wavelength * n_a)) * (n_a**2 - n_original[d_conj]**2)) * FT2(propagator_conj * iFT2(U_control))))
    #plot_complex_field(U_control, extent=extent_xy, title="Complex Input Field Backpropagated", cmap="viridis")
    #plot_complex_field(U_control-U_i, extent=extent_xy, title="Complex Input Field diff", cmap="viridis")
    #plt.show()

    n_original_uniform = jnp.ones_like(n_original) * n_a
    U_control = jnp.copy(U_i)
    U_for = jnp.zeros((shape[0] + 1, *shape[1:]), dtype=jnp.complex128)
    U_back = jnp.zeros((shape[0] + 1, *shape[1:]), dtype=jnp.complex128)
    U_for = U_for.at[0].set(U_control)
    for d in tqdm(range(shape[0])):
        U_control = iFT2(propagator * FT2(jnp.exp((1j*jnp.pi*pix_size[0] / (wavelength * n_a)) * (n_a**2 - n_original[d]**2)) * iFT2(propagator * FT2(U_control))))
        U_for = U_for.at[d+1].set(U_control)
    #plot_complex_field(U_control - U_meas, extent=extent_xy, title=f"Complex Control output field in for d={d}", cmap="viridis")
    U_back = U_back.at[shape[0]].set(U_control)
    for d in tqdm(range(shape[0])):
        d_conj = shape[0] - d - 1
        U_control = FT2(propagator_conj * iFT2(jnp.exp(-(1j*jnp.pi*pix_size[0] / (wavelength * n_a)) * (n_a**2 - n_original[d_conj]**2)) * FT2(propagator_conj * iFT2(U_control))))
        U_back = U_back.at[d_conj].set(U_control)
        #plot_complex_field(U_control, extent=extent_xy, title=f"Complex Control field in back d={d_conj}", cmap="viridis")
    #plot_complex_field(U_control - U_i, extent=extent_xy, title=f"Complex Control input field in for d={d_conj}", cmap="viridis")
    #[plot_complex_field(U_back[d]-U_for[d], extent=extent_xy, title=f"Complex Control field diff={d}", cmap="viridis") for d in range(shape[0]+1)]
    plt.show()"""


    #%% Reconstruction only increase of n

    """inc_n_guess = jnp.array([0.])
    opt = optax.adam(1e-2)
    opt_st = opt.init(inc_n_guess)
    n_estim = 100
    seq_tim = tqdm(range(n_estim))
    for tim in seq_tim:
        mse, gradient = jax.value_and_grad(propagation, argnums=(2,3))(U_i, U_meas, inc_n_guess, R0, shape, pix_size, r2, propagator, wavelength, n_a)
        updates, opt_st = opt.update(gradient[0], opt_st, inc_n_guess)
        inc_n_guess = jnp.array(optax.apply_updates(inc_n_guess, updates))
        seq_tim.set_postfix({"MSE": float(mse), "inc_n_guess": float(inc_n_guess[0])})

    n_guess = n_a + inc_n_guess * jnp.where((r2<=R0**2)[None].repeat(shape[0], axis=0), R0**2-r2[None].repeat(shape[0], axis=0), 0)

    fig1, sub1 = plt.subplots(1,3)
    fig1.suptitle("Original Index distribution")
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    #im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice])
    im1 = sub1[0].imshow(n_guess[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub1[0])
    sub1[0].set_title(f"YX slice at z_pix={z_slice}")
    sub1[0].set_xlabel("$x(\\lambda)$")
    sub1[0].set_ylabel("$y(\\lambda)$")
    #im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T)
    im2 = sub1[1].imshow(n_guess[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub1[1])
    sub1[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub1[1].set_xlabel("$z(\\lambda)$")
    sub1[1].set_ylabel("$x(\\lambda)$")
    #im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T)
    im3 =sub1[2].imshow(n_guess[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub1[2])
    sub1[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub1[2].set_xlabel("$z(\\lambda)$")
    sub1[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()

    U_A_d = A_d(jnp.ones_like(U_i).astype(jnp.complex128), n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)

    plot_complex_field(U_meas, extent=extent_xy, title="Complex Field Measured", cmap="viridis")    
    plot_complex_field(U_A_d, extent=extent_xy, title="Complex Field Simulated", cmap="viridis")
    print(mse, gradient)
    plt.show()"""
    #%% Reconstruction pixel by pixel autodiff

    """n_guess = jnp.full_like(n_original, n_a)    
    opt = optax.adam(1e-2)
    opt_st = opt.init(n_guess)
    n_estim = 100
    seq_tim = tqdm(range(n_estim))
    for tim in seq_tim:
        mse, gradient = jax.value_and_grad(propagation_n, argnums=(2,))(U_i, U_meas, n_guess, shape, pix_size, propagator, wavelength, n_a)
        updates, opt_st = opt.update(gradient[0], opt_st, n_guess)
        n_guess = jnp.array(optax.apply_updates(n_guess, updates))
        seq_tim.set_postfix({"MSE": float(mse), "n_guess": float(n_guess[shape[0] // 2, shape[1] // 2, shape[2] // 2])})

    fig1, sub1 = plt.subplots(1,3)
    fig1.suptitle("Original Index distribution")
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    #im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice])
    im1 = sub1[0].imshow(n_guess[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub1[0])
    sub1[0].set_title(f"YX slice at z_pix={z_slice}")
    sub1[0].set_xlabel("$x(\\lambda)$")
    sub1[0].set_ylabel("$y(\\lambda)$")
    #im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T)
    im2 = sub1[1].imshow(n_guess[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub1[1])
    sub1[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub1[1].set_xlabel("$z(\\lambda)$")
    sub1[1].set_ylabel("$x(\\lambda)$")
    #im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T)
    im3 =sub1[2].imshow(n_guess[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub1[2])
    sub1[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub1[2].set_xlabel("$z(\\lambda)$")
    sub1[2].set_ylabel("$y(\\lambda)$")
    plt.tight_layout()

    U_A_d = A_d(jnp.ones_like(U_i).astype(jnp.complex128), n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)

    plot_complex_field(U_meas, extent=extent_xy, title="Complex Field Measured", cmap="viridis")    
    plot_complex_field(U_A_d, extent=extent_xy, title="Complex Field Simulated", cmap="viridis")
    print(mse, gradient)
    plt.show()"""


    #%% Reconstruction pixel by pixel manually
    #n_guess = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  (inc_n/2) * (R0-r2[None].repeat(shape[0], axis=0)), 0) #jnp.full(shape, n_a+0.1)
 
    """n_guess = jnp.full_like(n_original, n_a)
    mse_corr, gradient_corr = jax.value_and_grad(propagation_n, argnums=(2,))(U_i, U_meas, n_guess, shape, pix_size, propagator, wavelength, n_a)
    gradient_corr = gradient_corr[0]
    U_sim = A_d(U_i.astype(jnp.complex128), n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)
    grad_n = jnp.zeros(shape)
    dif_Us = jnp.conj(U_sim-U_meas)
    #plot_complex_field(dif_Us, extent=extent_xy, title="diff_Us", cmap="viridis")

    for d in tqdm(range(shape[0])):
        U_mult_plane = C_d(U_i.astype(jnp.complex128), n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], d)
        U_mult_plane = iFT2(propagator * FT2(
            (-2j* jnp.pi / (wavelength * n_a)) * pix_size[0] * n_guess[d] * 
            jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * 
            iFT2(propagator * FT2(U_mult_plane))))
        U_mult_plane = A_d(U_i.astype(jnp.complex128), n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], d)
        #plot_complex_field(dif_Us, extent=extent_xy, title=f"U_mult {d}", cmap="viridis")
        grad_n = grad_n.at[d].set(2*jnp.real(dif_Us * U_mult_plane))

    dif_grad = gradient_corr-grad_n
    show_volume(grad_n, "L2 grad", pix_size, cmap='viridis') # $\\nabla \\mathcal{L^2}$
    show_volume(gradient_corr, "L2 grad correct", pix_size, cmap='viridis') # $\\nabla \\mathcal{L^2}$
    show_volume(dif_grad, "Difference between L2 grads", pix_size, cmap='viridis')
    plt.show()"""

    #n_guess = jnp.full(shape, n_original.mean()).astype(jnp.float64)
    n_guess = jnp.full_like(n_original, n_a)
    print(n_guess[0,0,0])
    grad_n = jnp.zeros(shape)

    n_epochs = 10000
    seq_train = tqdm(range(n_epochs), desc="Training", leave=True)
    mse_arr = jnp.zeros(n_epochs)
    tv_arr = jnp.zeros(n_epochs)
    total_err_arr = jnp.zeros(n_epochs)
    lr = 5e-3
    optimizer_name = "adam"
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(n_guess)
    alpha_tv = 10e-3
    epsilon_tv = 1e-9
    tau = 200
    N = shape[1] * shape[2]
    control_data = {"simulation": {"shape": shape, "pix_size": pix_size}, 
                    "optimization": {"optimizer": optimizer_name, "step_i": lr, "alpha_tv": alpha_tv, "tau": tau, "esilon_tv": epsilon_tv}}
    n_0 = 1.5
    d_e = 30 * wavelength

    U_meas_plus = iFFT2((jnp.exp(1j * wavelength * jnp.pi * d_e * (Fx**2 + Fy**2) / n_0)) * FFT2(U_meas))
    many = n_epochs
    for epoch in seq_train:
        try:
            """U_prop = jnp.copy(U_i)
            U_back = jnp.copy(U_meas)
            for d in tqdm(range(shape[0]), leave=False): #tqdm(range(shape[0]-1, -1, -1), leave=False):
                U_back = FT2(propagator_conj * iFT2(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * FT2(propagator_conj * iFT2(U_back))))
            #U_back = jnp.conjugate(U_back)
            if epoch == -10:
                plot_complex_field(U_meas, extent=extent_xy, title="Complex Field Measured", cmap="viridis")
                plot_complex_field(U_back, extent=extent_xy, title="Complex Field backpropagated", cmap="viridis")
                plot_complex_field(U_back-U_meas, extent=extent_xy, title="Complex Field backpropagatdiff", cmap="viridis")

            for d in tqdm(range(shape[0]), leave=False):
                d_conj = shape[0] - d - 1

                U_corr = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_i - U_back))))

                #U_corr = FT2(propagator_conj * iFT2(U_corr))

                n_contrib = -(4 * jnp.pi / (wavelength * n_a)) * n_guess[d] * jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_prop))

                U_prop = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_prop))))

                grad_plane = jnp.imag(n_contrib * jnp.conjugate(U_corr)) * pix_size[1]*pix_size[2]

                grad_n = grad_n.at[d].set(grad_plane)"""
        
            #jax.profiler.start_trace(LOCAL_DATA_DIR)
            #mse, gradient_corr = jax.value_and_grad(propagation_n_plus, argnums=(2,))(U_i, U_meas_plus, n_guess, shape, pix_size, propagator, wavelength, n_a, n_0, d_e, Fy, Fx)
            mse, gradient_corr = jax.value_and_grad(propagation_n, argnums=(2,))(U_i, U_meas, n_guess, shape, pix_size, propagator, wavelength, n_a)
            #jax.profiler.stop_trace()
            #print("Trace saved to ", LOCAL_DATA_DIR)
            #print(jax.make_jaxpr(propagation_n)(U_i, U_meas, n_guess, shape, pix_size, propagator, wavelength, n_a))
            
            gradient_corr = gradient_corr[0]
            grad_n = gradient_corr

            tv = tv_loss_periodic(n_guess, epsilon_tv)
            tv_arr = tv_arr.at[epoch].set(tv)
            #mse = jnp.sum(jnp.abs(U_prop - U_meas)**2)*(pix_size[1] * pix_size[2])
            #mse = mse_corr
            mse_arr = mse_arr.at[epoch].set(float(mse))
            tv_contrib = tv * alpha_tv * jnp.exp(-(epoch**2/tau**2))
            total_err = mse + tv_contrib #* (mse_old-total_err)
            total_err_arr = total_err_arr.at[epoch].set(total_err)
            grad_tv = jax.grad(tv_loss_periodic, argnums=0)(n_guess, epsilon_tv)#tv_grad(n_guess, epsilon_tv)
            #if (epoch) % 100 == 0:
            #    dif_grad = gradient_corr-grad_n
            #    show_volume(grad_n, "L2 grad", pix_size, cmap='viridis') # $\\nabla \\mathcal{L^2}$
            #    show_volume(gradient_corr, "L2 grad correct", pix_size, cmap='viridis') # $\\nabla \\mathcal{L^2}$
            #    plt.show()
                #show_volume(dif_grad, "Difference between L2 grads", pix_size, cmap='viridis')
                #plt.show()
            #    show_volume(-grad_tv, "$\\nabla \\text{TV}$", pix_size, cmap="viridis")
            grad_n = grad_n + alpha_tv * grad_tv * jnp.exp(-(epoch**2/tau**2))# * (mse_old-total_err)
            #if (epoch) % 100 == 0:
            #    show_volume(-grad_n,  "Total gradient",  pix_size, cmap="viridis")
            #    show_volume(n_guess - (grad_n * int(epoch) * lr), "comparison",  pix_size, cmap="viridis")#jnp.full_like(n_guess, n_a) )
            #    plt.show()


            updates, opt_state = optimizer.update(grad_n, opt_state, n_guess)
            #show_volume((grad_n * lr), "Estimated change",  pix_size, cmap="viridis")#jnp.full_like(n_guess, n_a) )
            #show_volume(updates, "Real change",  pix_size, cmap="viridis")#jnp.full_like(n_guess, n_a) )            
            #show_volume(n_guess-(grad_n * lr),  "n_guess_man_updated",  pix_size, cmap="viridis")
            n_guess = jnp.array(optax.apply_updates(n_guess, updates))
            #show_volume(n_guess,  "n_guess",  pix_size, cmap="viridis")
            #plt.show()

            seq_train.set_postfix({"Total": float(total_err), "MSE": float(mse), "TV": float(tv)})
            many -= 1
            if (jnp.log10(total_err/mse)) < -3 or (jnp.log10(mse) <= -14) or many <= 0:
                option = input("Do you want to continue(y/n)? (total=MSE)") == "n"
                if option:
                    break
                else:
                    many = int(input("how many times?"))
        except KeyboardInterrupt:
            break
    
    U_prop = jnp.copy(U_i)
    U_prop = A_d(U_prop, n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)
    
    #plot_complex_field(U_meas, extent=extent_xy, title="Complex Field", cmap="viridis")
    #plot_complex_field(U_prop, extent=extent_xy, title="Complex Field", cmap="viridis")

    fig, sub = plt.subplots(1,2)
    sub[0].set_title("$|| U_{sim} - U_{meas} ||$")
    im = sub[0].imshow(jnp.abs(U_prop - U_meas), extent=extent_xy)
    plt.colorbar(im, ax=sub[0])
    sub[0].set_xlabel("$x(\\lambda)$")
    sub[0].set_ylabel("$y(\\lambda)$")
    sub[1].plot(total_err_arr[:epoch], label="Total cost")
    sub[1].plot(mse_arr[:epoch], label="MSE loss")
    sub[1].plot((tv_arr[:epoch]* alpha_tv * jnp.exp(-(jnp.arange(epoch)**2/tau**2))), label="TV loss")
    sub[1].legend()
    sub[1].set_title("Total cost function")
    sub[1].set_xlabel("Iteration")
    sub[1].set_ylabel("$\\mathcal{C}(n)$")#"$\\int ||U_o^{sim}-U_o^{meas}||$")
    sub[1].set_yscale("log")
    fig.tight_layout()

    fig2, sub2 = plt.subplots(2,1)
    sub2[0].set_title("MSE")
    sub2[0].plot(mse_arr[:epoch])
    sub2[0].set_xlabel("Iteration")
    sub2[0].set_ylabel("MSE loss")
    sub2[0].set_yscale("log")
    sub2[1].set_title("TV regularizer")
    sub2[1].plot(tv_arr[:epoch])
    sub2[1].set_yscale("log")
    sub2[1].set_xlabel("Iteration")
    sub2[1].set_ylabel("TV loss")
    fig2.tight_layout()

    fig5, sub5 = plt.subplots(1,3)
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    fig5.suptitle("Backpropagated Index distribution")
    #im1 = sub5[0].imshow(n_guess[z_slice] - n_original[z_slice], extent=extent_xy)
    im1 = sub5[0].imshow(n_guess[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub5[0])
    sub5[0].set_title(f"YX slice at z_pix={z_slice}")
    sub5[0].set_xlabel("$x(\\lambda)$")
    sub5[0].set_ylabel("$y(\\lambda)$")
    #im2 = sub5[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T, extent=extent_zx)
    im2 = sub5[1].imshow(n_guess[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub5[1])
    sub5[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub5[1].set_xlabel("$z(\\lambda)$")
    sub5[1].set_ylabel("$x(\\lambda)$")
    #im3 =sub5[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T, extent=extent_zy)
    im3 =sub5[2].imshow(n_guess[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub5[2])
    sub5[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub5[2].set_xlabel("$z(\\lambda)$")
    sub5[2].set_ylabel("$y(\\lambda)$")
    sub5[1].set_aspect(0.5)  # < 1 ensancha el eje X
    sub5[2].set_aspect(0.5)
    plt.tight_layout()

    fig6, sub6 = plt.subplots(1,3)
    z_slice, y_slice, x_slice = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    fig6.suptitle("$n_{back}-n_{original}$")
    im1 = sub6[0].imshow(n_guess[z_slice] - n_original[z_slice], extent=extent_xy)
    plt.colorbar(im1, ax=sub6[0])
    sub6[0].set_title(f"YX slice at z_pix={z_slice}")
    sub6[0].set_xlabel("$x(\\lambda)$")
    sub6[0].set_ylabel("$y(\\lambda)$")
    im2 = sub6[1].imshow(n_guess[:,y_slice].T - n_original[:,y_slice].T, extent=extent_zx)
    plt.colorbar(im2, ax=sub6[1])
    sub6[1].set_title(f"XZ slice at y_pix={y_slice}")
    sub6[1].set_xlabel("$z(\\lambda)$")
    sub6[1].set_ylabel("$x(\\lambda)$")
    im3 =sub6[2].imshow(n_guess[:,:, x_slice].T - n_original[:,:, x_slice].T, extent=extent_zy)
    plt.colorbar(im3, ax=sub6[2])
    sub6[2].set_title(f"Yz slice at x_pix={x_slice}")
    sub6[2].set_xlabel("$z(\\lambda)$")
    sub6[2].set_ylabel("$y(\\lambda)$")
    sub6[1].set_aspect(0.5)  # < 1 ensancha el eje X
    sub6[2].set_aspect(0.5)
    plt.tight_layout()

    if save:
        filename = "6"
        with open(dat_save_dir/(filename+".json"), "w") as f:
            json.dump(control_data, f, indent=4)
        freedom = "logr"
        fig1.savefig(dat_save_dir/(f"original_distribution_{freedom}_freedom"+filename+".png"), dpi=300, bbox_inches="tight")
        fig5.savefig(dat_save_dir/(f"retrieved_distribution_{freedom}_freedom"+filename+".png"), dpi=300, bbox_inches="tight")
        fig6.savefig(dat_save_dir/(f"diff_distribution_{freedom}_freedom"+filename+".png"), dpi=300, bbox_inches="tight")
        fig.savefig(dat_save_dir/(f"Total_loss_{freedom}_freedom"+filename+".png"), dpi=300, bbox_inches="tight")
        fig2.savefig(dat_save_dir/(f"Losses_{freedom}_freedom"+filename+".png"), dpi=300, bbox_inches="tight")

    plt.show()