import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from tqdm import tqdm

from fgrinmet.utils.operators import FT2, iFT2
from fgrinmet.constructions.geometries import FGRINcomponent
#def step_prop(carry, slice):
#    paraxial_fact, epsilon_cube, Ui = carry
#carry_out, outputs = jax.lax.scan(step_prop, carry_in, jnp.arange(10))



def propagate(object_grid: FGRINcomponent, propagation_grid: ):
    
"""
Simulation"""
if __name__ == "__main__":
    l = 1
    epsilon_a = 1.5**2
    name = "Quadratic distribution"
    n_points = (10, 500, 500)
    pix_sizes = (10 * l / n_points[0], 100 * l / n_points[1], 100 * l / n_points[2])
    cuad_cube_dist = lambda Z, Y, X, a_0: 3*((0.25**2-(X/np.max(X)-0.5)**2-(Y/np.max(Y)-0.5)**2))*((0.25**2)>=((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#0.5*(Z/np.max(Z)-0.5)/(1+((X/np.max(X)-0.5)**2+(Y/np.max(Y)-0.5)**2))#+0.5 - ((X-1000*l)**2+(Y-1000*l)**2) * (X-1000)  / 1000000000
    cuad_cube = FGRINcomponent(name, cuad_cube_dist, n_points, pix_sizes, a_0=jnp.array([0]))
    epsilon_original = cuad_cube.generate() - epsilon_a
    cuad_cube.show()
    
    fy = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[1], d=pix_sizes[1]))
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[2], d=pix_sizes[2]))
    paraxial_fact = jnp.exp(-1j * jnp.pi * l * pix_sizes[0] * (fx[None, :]**2 + fy[:, None]**2) / (2 * jnp.sqrt(epsilon_a))) * ((fx[None, :]**2 + fy[:, None]**2) <= (1/l)**2)

    coordinates_obj = cuad_cube.coordinates1D
    Z, Y, X = jnp.meshgrid(*coordinates_obj, indexing='ij')
    
    extent_yx = cuad_cube.genextent[2] + cuad_cube.genextent[1]
    extent_xz = cuad_cube.genextent[0] + cuad_cube.genextent[2]

    wx, wy = (50 * l, 50 * l)
    Ui = jnp.ones(n_points[1:]) #* jnp.exp(-(((X[0]-500*l)/wx)**2 + ((Y[0]-500*l)/wy)**2)) * ((((X[0]-500*l)/wx)**2 + ((Y[0]-500*l)/wy)**2) <= 1) #jnp.ones(n_points[1:])#jnp.exp(-(((X[0]-50)/wx)**2 + ((Y[0]-50)/wy)**2))
    Uo = jnp.copy(Ui)
    Uo_inv = jnp.copy(Ui)

    U_o_mid = jnp.full((n_points[0], n_points[1]), jnp.nan)
    U_o_mid = U_o_mid.at[0,:].set(Ui[n_points[1]//2,:])
    U_o_mid_inv = jnp.full((n_points[0], n_points[1]), jnp.nan)
    U_o_mid_inv = U_o_mid_inv.at[0,:].set(Ui[n_points[1]//2,:])

    for slice_idx in tqdm(range(n_points[0])):
        epsilon_slice = epsilon_original[slice_idx, :, :]
        Uo = iFT2(paraxial_fact * FT2(
                  jnp.exp(1j * jnp.pi * epsilon_slice * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                  paraxial_fact * FT2(Uo))))
        U_o_mid = U_o_mid.at[slice_idx,:].set(Uo[n_points[1]//2,:])
        
        epsilon_slice_inv = epsilon_original[n_points[0] - 1 - slice_idx, :, :]

        Uo_inv = iFT2((paraxial_fact**-1) * FT2(
                  jnp.exp(-1j * jnp.pi * epsilon_slice_inv * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                  (paraxial_fact**-1) * FT2(Uo_inv))))
        U_o_mid_inv = U_o_mid_inv.at[slice_idx,:].set(Uo_inv[n_points[1]//2,:])

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
            plt.colorbar(im3, ax=sub[2])

            fig2, sub2 = plt.subplots(1,3,figsize=(15,5))
            sub2[0].set_title(f"Slice {slice_idx} - XZ Plane")
            im1 = sub2[0].imshow(jnp.abs(U_o_mid_inv.T), extent=extent_xz)
            plt.colorbar(im1, ax=sub2[0])
            sub2[1].set_title(f"Slice {slice_idx} - XY Plane")
            im2 = sub2[1].imshow(jnp.abs(Uo_inv.T), extent=extent_yx)
            plt.colorbar(im2, ax=sub2[1])
            sub2[2].set_title(f"Slice {slice_idx} - XY Plane FT")
            im3 = sub2[2].imshow(jnp.log(jnp.abs(FT2(Uo_inv.T))), extent=extent_yx)
            plt.colorbar(im3, ax=sub2[2])
            plt.show()"""
    
    fig, sub = plt.subplots(1,2,figsize=(10,5))
    sub[0].set_title("Input Field")
    im1 = sub[0].imshow(Ui, extent=extent_yx)#jnp.log(jnp.abs(FT2(Uo))))
    plt.colorbar(im1, ax=sub[0])
    im2 = sub[1].imshow(jnp.abs(Uo), extent=extent_yx)
    plt.colorbar(im2, ax=sub[1])
    plt.show()

    fig = plt.figure()
    plt.imshow(jnp.abs(U_o_mid.T), extent=extent_xz)
    plt.show()

    #%%
    """
    Reconstruction
    """
    import optax

    guess = jnp.zeros_like(epsilon_original)
    lr = 1e-2
    n_iterations = 100
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(guess)
    U_sim = jnp.copy(Ui)
    grad = jnp.zeros_like(guess)
    seq_train = tqdm(range(n_iterations), desc="Training", leave=True)

    for iteration in seq_train:
        for i in tqdm(range(n_points[0]), leave=False):
            U_sim = iFT2(paraxial_fact * FT2(
                    jnp.exp(1j * jnp.pi * guess[i, :, :] * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                    paraxial_fact * FT2(U_sim))))
        U_diff = (U_sim - Uo)
        U_sim_inv = jnp.copy(U_sim)
        for i in tqdm(range(n_points[0]), leave=False):
            d_inv = n_points[0] - 1 - i

            U_diff = iFT2((paraxial_fact**-1) * FT2(
                    jnp.exp(-1j * jnp.pi * guess[d_inv, :, :] * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                    (paraxial_fact**-1) * FT2(U_diff))))
            
            U_sim_inv = iFT2(paraxial_fact**-1 * FT2(
                    jnp.exp(-1j * jnp.pi * guess[d_inv, :, :] * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * iFT2(
                    paraxial_fact**-1 * FT2(U_sim_inv))))
            
            grad_slice = (2 * jnp.pi * pix_sizes[0] / (l * jnp.sqrt(epsilon_a))) * jnp.imag(jnp.conjugate(U_diff) * U_sim_inv)
            
            grad = grad.at[d_inv, :, :].set(grad_slice)
        
        updates, opt_state = optimizer.update(grad, opt_state)
        guess = optax.apply_updates(guess, updates)
        loss_mse = (jnp.abs(U_sim - Uo)**2).sum()

        seq_train.set_postfix({"Total": float(loss_mse), "MSE": float(loss_mse), "TV": float(loss_mse)})