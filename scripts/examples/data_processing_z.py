from config import LOCAL_DATA_DIR
import json
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    z_pix_size_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "z_pix_size_plane_wave" / "params"
    z_npix_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "z_npix_plane_wave" / "params" # group 1 dims
    savedir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "z_analysis"
    savedir.mkdir(parents=True, exist_ok=True)
    z_pix_size_files = [z_pix_size_dir / d for d in os.listdir(z_pix_size_dir)]
    z_npix_files = [z_npix_dir / d for d in os.listdir(z_npix_dir)]

    z_pix_size_params = []
    for f in z_pix_size_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        z_pix_size_params.append(param_dict)
    
    z_npix_params = []
    for f in z_npix_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        z_npix_params.append(param_dict)

    z_pix_size = [z_pix_size_params[i]["prop_params"]["pix_sizes"][0] for i in range(len(z_pix_size_params))]
    z_npix_size = [z_npix_params[i]["prop_params"]["pix_sizes"][0] for i in range(len(z_npix_params))]
    z_pix_size_error = [z_pix_size_params[i]["total_error"] for i in range(len(z_pix_size_params))]
    z_npix_error = [z_npix_params[i]["total_error"] for i in range(len(z_npix_params))]

    z_pix_size_npoints = [np.prod(z_pix_size_params[i]["prop_params"]["shape"]) for i in range(len(z_pix_size_params))]
    z_npix_npoints = [np.prod(z_npix_params[i]["prop_params"]["shape"]) for i in range(len(z_npix_params))]
    z_pix_size_lerror = [z_pix_size_error[i][-1] for i in range(len(z_pix_size_params))]
    z_npix_lerror = [z_npix_error[i][-1] for i in range(len(z_npix_params))]

    plt.figure()
    for i in range(len(z_pix_size_params)):
        plt.plot(z_pix_size_error[i], label=f"Group 4: {z_pix_size[i]:.2f} $\\lambda$")
    for i in range(len(z_npix_params)):
        plt.plot(z_npix_error[i], label=f"Group 3: {z_npix_size[i]:.2f} $\\lambda$")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "total_error.jpg", dpi=300)

    plt.figure()
    for i in range(len(z_pix_size_params)):
        plt.plot(z_pix_size_error[i] / z_pix_size_npoints[i], label=f"Group 4: {z_pix_size[i]:.2f} $\\lambda$")
    for i in range(len(z_npix_params)):
        plt.plot(z_npix_error[i] / z_npix_npoints[i], label=f"Group 3: {z_npix_size[i]:.2f} $\\lambda$")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "mean_error.jpg", dpi=300)

    plt.figure()
    plt.plot(z_pix_size, z_pix_size_lerror, "o-", label="Group 4: Npix constant")
    plt.plot(z_npix_size, z_npix_lerror, "s-", label="Group 3: Dim constant")
    plt.xlabel("Pixel Size")
    plt.ylabel("Final Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "last_error.jpg", dpi=300)

    z_pix_size_lerror = [z_pix_size_error[i][-1] / z_pix_size_npoints[i] for i in range(len(z_pix_size_params))]
    z_npix_lerror = [z_npix_error[i][-1] / z_npix_npoints[i] for i in range(len(z_npix_params))]

    plt.figure()
    plt.plot(z_pix_size, z_pix_size_lerror, "o-", label="Group 4: Npix constant")
    plt.plot(z_npix_size, z_npix_lerror, "s-", label="Group 3: Dim constant")
    plt.xlabel("Pixel Size")
    plt.ylabel("Final Mean Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "last_mean_error.jpg", dpi=300)

    plt.show()