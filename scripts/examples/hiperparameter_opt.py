import json
import numpy as np
import os
import re
import matplotlib.pyplot as plt

from config import LOCAL_DATA_DIR


if __name__ == "__main__":
    #%%
    """
    Learning rates and tau
    """
    lr_dir = LOCAL_DATA_DIR / "Single_Ui"
    lr_dirs = [lr_dir / d for d in os.listdir(lr_dir) if re.match("\\d{1}_lr", d)]
    params_files = [[d / "params" / f for f in os.listdir(d / "params")] for d in lr_dirs]

    params = []
    for row_files in params_files:
        row_params = []
        for f in row_files:
            with open(f, "r") as file:
                param_dict = json.load(file)
            row_params.append(param_dict)
        params.append(row_params)
    
    print(params[0][0].keys())
    taus = []
    lrs = []
    mse = []
    errors = []
    for row in params:
        taus_row = []
        lrs_row = []
        mse_row = []
        errors_row = []

        for param in row:
            taus_row.append(param["tau"])
            lrs_row.append(param["lr"])
            mse_row.append(param["losses"][0][-1])
            errors_row.append(param["total_err"])

        taus.append(taus_row)
        lrs.append(lrs_row)
        mse.append(mse_row)
        errors.append(errors_row)

    taus = np.array(taus)
    lrs = np.array(lrs)
    mse = np.array(mse)
    errors = np.array(errors)

    plt.figure()
    cont = plt.tricontourf(np.log10(lrs.flatten()), np.log10(taus.flatten()), np.log10(mse.flatten()), levels=40, cmap="viridis")
    pl = plt.scatter(np.log10(lrs.flatten()), np.log10(taus.flatten()), c=np.log10(mse.flatten()), cmap=cont.cmap, norm=cont.norm, edgecolors="k", s=40)
    plt.colorbar(cont, label="$log_{10}(MSE)$")
    plt.xlabel("$log_{10}(lr)$")
    plt.ylabel("$log_{10}(tau)$")
    plt.savefig(lr_dir / "mse_lr_tau.jpg", dpi=300)

    plt.figure()
    cont = plt.tricontourf(np.log10(lrs.flatten()), np.log10(taus.flatten()), np.log10(errors.flatten()), levels=40, cmap="viridis")
    pl = plt.scatter(np.log10(lrs.flatten()), np.log10(taus.flatten()), c=np.log10(errors.flatten()), cmap=cont.cmap, norm=cont.norm, edgecolors="k", s=40)
    plt.colorbar(cont, label="$log_{10}(Error)$")
    plt.xlabel("$log_{10}(lr)$")
    plt.ylabel("$log_{10}(tau)$")
    plt.savefig(lr_dir / "error_lr_tau.jpg", dpi=300)

    plt.show()

    #%%
    """
    MSE mapping
    """
    """from splitm_grad_new import propagate
    import jax.numpy as jnp
    from jax import random

    l = 1
    epsilon_a = 1.5**2
    n_points = (128, 128, 128)
    pix_sizes = (0.125*l/4, 0.125*l, 0.125*l)
    x = (jnp.arange(n_points[2]) - n_points[2] // 2) * pix_sizes[2]
    y = (jnp.arange(n_points[1]) - n_points[1] // 2) * pix_sizes[1]
    z = (jnp.arange(n_points[0]) - n_points[0] // 2) * pix_sizes[0]
    x_norm = x - x.min()
    x_norm = x_norm / x_norm.max() - 0.5
    y_norm = y - y.min()
    y_norm = y_norm / y_norm.max() - 0.5
    z_norm = z - z.min()
    z_norm = z_norm / z_norm.max() - 0.5

    eps_original = 6*((0.25**2 - x_norm[None, None]**2 - y_norm[None,:,None]**2) * (0.25**2 <= (x_norm[None, None]**2 + y_norm[None,:,None]**2))) * jnp.ones_like(z_norm)[:,None,None]

    fy = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[1], d=pix_sizes[1]))
    fx = jnp.fft.fftshift(jnp.fft.fftfreq(n_points[2], d=pix_sizes[2]))
    paraxial_fact = jnp.exp(-1j * jnp.pi * l * pix_sizes[0] * (fx[None, :]**2 + fy[:, None]**2) / (2 * jnp.sqrt(epsilon_a)))

    U_i = jnp.ones(n_points[1:], dtype=jnp.complex128)

    a_arr = jnp.linspace(-4,4,21)
    b_arr = jnp.linspace(-4,4,21)
    U_original, _ = propagate(eps_original, paraxial_fact, U_i, pix_sizes[0], epsilon_a, l)
    mse_arr = jnp.zeros((len(a_arr), len(b_arr)))

    for i, a in enumerate(a_arr):
        for j, b in enumerate(b_arr):
            eps_mod = 6*(((1+a)*0.25**2 - x_norm[None, None]**2 - y_norm[None,:,None]**2) * ((1+a)*0.25**2 <= (x_norm[None, None]**2 + y_norm[None,:,None]**2))) * jnp.ones_like(z_norm)[:,None,None]
            eps_mod = eps_mod + b * 0.5 * random.normal(random.PRNGKey(0), eps_mod.shape)
            U_s, _ = propagate(eps_mod, paraxial_fact, U_i, pix_sizes[0], epsilon_a, l)
            mse = jnp.mean(jnp.abs(U_s - U_original)**2)
            mse_arr = mse_arr.at[i,j].set(mse)
            print(f"a={a}, b={b}, mse={mse}")
        print(f"{i}/{len(a_arr)} done")

    
    plt.figure()
    plt.imshow(mse_arr, extent = [-4,4,4,-4])
    plt.colorbar(label="MSE")
    plt.xlabel("b")
    plt.ylabel("a")
    plt.savefig(lr_dir / "mse_a_b.jpg", dpi=300)
    plt.show()"""