import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib as mpl
import optax
from tqdm import tqdm

from fgrinmet.utils import FT2, iFT2, FT2_i, iFT2_i, fft_coord_jax
from fgrinmet.splitm import paraxial_propagator_jax

from splitm_gradient import C_d, C_d_conj, A_d, A_d_conj

def show_volume(volume, title, pix_size, cmap='seismic'):
    """
    Visualizes all points in a 3D volume.
    """
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
    for i in range(n_layers, len(axes)):
        axes[i].axis('off')

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

    denom = jnp.sqrt(dx**2 + dy**2 + dz**2 + eps**2)
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
    pix_size = (0.25 * wavelength, 0.125 * wavelength, 0.125 * wavelength)
    extent_xy = (-(shape[2] // 2) * pix_size[2], ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1])
    extent_zx = (-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[2] // 2) + (shape[2] % 2)) * pix_size[2], -(shape[2] // 2) * pix_size[2])
    extent_zy = (-(shape[0] // 2) * pix_size[0], ((shape[0] // 2) + (shape[0] % 2)) * pix_size[0],
                 ((shape[1] // 2) + (shape[1] % 2)) * pix_size[1], -(shape[1] // 2) * pix_size[1])
    z = jnp.arange(shape[0]) / shape[0]
    y = (jnp.arange(shape[1]) - shape[1] // 2) / (shape[1] // 2)
    x = (jnp.arange(shape[2]) - shape[2] // 2) / (shape[2] // 2)

    r2 = y[:,None]**2 + x[None]**2
    inc_n = 0.5
    inc_r = 0.5
    R0 = 0.5
    #n_original = n_a + inc_n * jnp.where(r2[None].repeat(shape[0], axis=0)<=R0**2, (1 + (r2 / R0**2) - (2*r2**(1/2) / R0))[None].repeat(shape[0], axis=0), 0)# * x[None,None]
    #n_original = n_a + inc_n * (R0**2-r2[None].repeat(shape[0], axis=0))
    n_original = n_a + inc_n * jnp.where((r2<=R0**2)[None].repeat(shape[0], axis=0), R0**2-r2[None].repeat(shape[0], axis=0), 0)
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
    plt.tight_layout()

    Fy, Fx = fft_coord_jax(shape[1:], pix_size[1:])

    propagator = paraxial_propagator_jax(Fy, Fx, pix_size[0], n_a, wavelength)
    propagator_conj = jnp.conjugate(propagator)
    U_i = jnp.ones(shape[1:])
    U_meas = jnp.copy(U_i)

    # Compute the measured field
    for d in tqdm(range(shape[0])):
        U_meas = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d]**2)) * iFT2(propagator * FT2(U_meas))))
    

    # Check the properties of the hermitian adjoint operator
    U_meas_conj = jnp.conjugate(jnp.copy(U_i))
    for d in tqdm(range(shape[0])):
        d_conj = shape[0] - d - 1
        U_meas_conj = FT2_i(propagator_conj * iFT2_i(jnp.exp((-1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d_conj]**2)) * FT2_i(propagator_conj * iFT2_i(U_meas_conj))))

    plt.figure()
    plt.title("$|A_{d+1}C_d[U^{in}]|$")
    plt.imshow(jnp.abs(U_meas), extent=extent_xy)
    plt.xlabel("x(\\lambda)")
    plt.ylabel("y(\\lambda)")
    plt.colorbar()
    plt.figure()
    plt.title("$|A^{\\dagger}_{d+1}C^{\\dagger}_d[U^{\\text{in } \\dagger}]|$")
    plt.imshow(jnp.abs(U_meas_conj), extent=extent_xy)
    plt.xlabel("x(\\lambda)")
    plt.ylabel("y(\\lambda)")
    plt.colorbar()
    plt.figure()
    plt.title("$|A^{\\dagger}_{d+1}C^{\\dagger}_d[U^{\\text{in } \\dagger}]-(A_{d+1}C_d[U^{in}])^\\dagger|$")
    plt.imshow(jnp.abs(U_meas_conj-jnp.conj(U_meas)), extent=extent_xy)
    plt.xlabel("x(\\lambda)")
    plt.ylabel("y(\\lambda)")
    plt.colorbar()

    coord_plane_0 = jnp.array([jnp.zeros(shape[1:]),
                               *jnp.meshgrid(jnp.arange(shape[1]), jnp.arange(shape[2]))]).transpose(1,2,0)
    norm_vect = jnp.array((1,0,0))
    U_test = jnp.exp(r2 / R0**2).astype(jnp.complex128)

    import time
    start = time.perf_counter()
    U_A_d = A_d(U_test, n_original, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)
    end = time.perf_counter()
    print(f"U_A_d time: {end-start}s")
    start = time.perf_counter()
    U_A_d_conj = A_d_conj(U_test, n_original, mask, coord_plane_0, norm_vect, propagator_conj, wavelength, n_a, pix_size[0], shape[0], 0)
    end = time.perf_counter()
    print(f"U_A_d_conj time: {end-start}s")
    print(f"MSE: {jnp.abs(jnp.conj(U_A_d)-U_A_d_conj).sum()}")
    start = time.perf_counter()
    U_C_d = C_d(U_test, n_original, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0])
    end = time.perf_counter()
    print(f"U_C_d time: {end-start}s")
    start = time.perf_counter()
    U_C_d_conj = C_d_conj(U_test, n_original, mask, coord_plane_0, norm_vect, propagator_conj, wavelength, n_a, pix_size[0], shape[0])
    end = time.perf_counter()
    print(f"U_C_d_conj time: {end-start}s")
    print(f"MSE: {jnp.abs(jnp.conj(U_C_d)-U_C_d_conj).sum()}")

    inc_n_guess = jnp.array([0.])
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
    plt.show()

    #n_guess = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  (inc_n/2) * (R0-r2[None].repeat(shape[0], axis=0)), 0) #jnp.full(shape, n_a+0.1)
    n_guess = jnp.full_like(n_original, n_a)
    #n_guess = jnp.full(shape, n_original.mean()).astype(jnp.float64)
    print(n_guess[0,0,0])
    grad_n = jnp.zeros(shape)

    n_epochs = 10000
    seq_train = tqdm(range(n_epochs), desc="Training", leave=True)
    mse_arr = jnp.zeros(n_epochs)
    tv_arr = jnp.zeros(n_epochs)
    total_err_arr = jnp.zeros(n_epochs)
    lr = 1e-3
    optimizer = optax.amsgrad(lr)
    opt_state = optimizer.init(n_guess)
    alpha_tv = 1e-1
    epsilon_tv = 1e-9
    tau = 20000
    N = shape[1] * shape[2]
    mse_old = 0
    total_err = 1

    
    for epoch in seq_train:
        try:
            U_prop = jnp.ones(shape[1:]).astype(jnp.complex128)
            U_back = jnp.copy(U_meas)
            for d in tqdm(range(shape[0]-1, -1, -1), leave=False):
                U_back = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * FT2_i(propagator_conj * iFT2_i(U_back))))
        
            U_corr = jnp.conjugate(jnp.ones(shape[1:]) - U_back)

            for d in tqdm(range(shape[0]), leave=False):
                d_conj = shape[0] - d - 1

                U_corr = FT2_i(propagator_conj * iFT2_i(jnp.exp(-(1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d_conj]**2)) * FT2_i(propagator_conj * iFT2_i(U_corr))))

                n_contrib = -(2j* jnp.pi / (wavelength * n_a)) * pix_size[0] * n_guess[d] * jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2))

                U_fact = n_contrib * iFT2(propagator * FT2(U_prop)) * FT2_i(propagator * iFT2_i((U_corr)))

                U_prop = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_guess[d]**2)) * iFT2(propagator * FT2(U_prop))))

                grad_plane = 2*jnp.real(U_fact * n_contrib)

                grad_n = grad_n.at[d].set(grad_plane)

            tv = tv_loss(n_guess, epsilon_tv)
            tv_arr = tv_arr.at[epoch].set(tv)
            mse = jnp.sum(jnp.abs(U_prop - U_meas)**2)*(pix_size[1] * pix_size[2])
            mse_arr = mse_arr.at[epoch].set(float(mse))
            total_err = mse + tv * alpha_tv * jnp.exp(-(epoch**2/tau**2)) #* (mse_old-total_err)
            mse_old = total_err
            total_err_arr = total_err_arr.at[epoch].set(total_err)
            grad_tv = jax.grad(tv_loss, argnums=0)(n_guess, epsilon_tv)#tv_grad(n_guess, epsilon_tv)
            #if (epoch) % 100 == 0:
            #    show_volume(-grad_n, "$\\nabla \\mathcal{L^2}$", pix_size, cmap='viridis')
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
            n_guess = optax.apply_updates(n_guess, updates)
            #show_volume(n_guess,  "n_guess",  pix_size, cmap="viridis")
            #plt.show()

            seq_train.set_postfix({"Total": float(total_err), "MSE": float(mse), "TV": float(tv)})
            if mse <= 4e-6:
                break
        except KeyboardInterrupt:
            U_prop = jnp.ones(shape[1:]).astype(jnp.complex128)
            U_prop = A_d(U_prop, n_guess, mask, coord_plane_0, norm_vect, propagator, wavelength, n_a, pix_size[0], shape[0], 0)
            break
    
    
    plot_complex_field(U_meas, extent=extent_xy, title="Complex Field", cmap="viridis")
    plot_complex_field(U_prop, extent=extent_xy, title="Complex Field", cmap="viridis")

    fig, sub = plt.subplots(1,2)
    sub[0].set_title("Amplitude propagated with the guessed index")
    im = sub[0].imshow(jnp.abs(U_prop), extent=extent_xy)
    plt.colorbar(im, ax=sub[0])
    sub[1].plot(total_err_arr[:epoch])
    sub[1].set_title("Total cost function")
    sub[1].set_xlabel("Iteration")
    sub[1].set_ylabel("$\\mathcal{C}(n)$")#"$\\int ||U_o^{sim}-U_o^{meas}||$")
    plt.tight_layout()

    fig2, sub2 = plt.subplots(2,1)
    sub2[0].set_title("MSE")
    sub2[0].plot(mse_arr[:epoch])
    sub2[0].set_xlabel("Iteration")
    sub2[0].set_ylabel("MSE loss")
    sub2[1].set_title("TV regularizer")
    sub2[1].plot(tv_arr[:epoch])
    sub2[1].set_xlabel("Iteration")
    sub2[1].set_ylabel("TV loss")

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
    plt.tight_layout()


plt.show()