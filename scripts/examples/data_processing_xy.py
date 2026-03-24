from config import LOCAL_DATA_DIR
import json
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    xy_dims_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "xy_dims_plane_wave" / "params"
    xy_npix_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "xy_npix_plane_wave" / "params"
    savedir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "xy_analysis"
    savedir.mkdir(parents=True, exist_ok=True)
    xy_dims_files = [xy_dims_dir / d for d in os.listdir(xy_dims_dir)]
    xy_npix_files = [xy_npix_dir / d for d in os.listdir(xy_npix_dir)]

    xy_dims_params = []
    for f in xy_dims_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        xy_dims_params.append(param_dict)
    
    xy_npix_params = []
    for f in xy_npix_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        xy_npix_params.append(param_dict)

    xy_dims_range = [xy_dims_params[i]["prop_params"]["pix_sizes"][-1] * xy_dims_params[i]["prop_params"]["shape"][-1] for i in range(len(xy_dims_params))]
    xy_npix_range = [xy_npix_params[i]["prop_params"]["pix_sizes"][-1] * xy_npix_params[i]["prop_params"]["shape"][-1] for i in range(len(xy_npix_params))]
    xy_dims_error = [xy_dims_params[i]["total_error"] for i in range(len(xy_dims_params))]
    xy_npix_error = [xy_npix_params[i]["total_error"] for i in range(len(xy_npix_params))]

    xy_dims_npoints = [np.prod(xy_dims_params[i]["prop_params"]["shape"]) for i in range(len(xy_dims_params))]
    xy_npix_npoints = [np.prod(xy_npix_params[i]["prop_params"]["shape"]) for i in range(len(xy_npix_params))]
    xy_dims_lerror = [xy_dims_error[i][-1]/xy_dims_npoints[i] for i in range(len(xy_dims_params))]
    xy_npix_lerror = [xy_npix_error[i][-1]/xy_npix_npoints[i] for i in range(len(xy_npix_params))]

    plt.figure()
    for i in range(len(xy_dims_params)):
        plt.plot(xy_dims_error[i], label=f"Group 2: {xy_dims_range[i]:.2f} $\\lambda$")
    for i in range(len(xy_npix_params)):
        plt.plot(xy_npix_error[i], label=f"Group 1: {xy_npix_range[i]:.2f} $\\lambda$")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "total_error.jpg", dpi=300)

    plt.figure()
    for i in range(len(xy_dims_params)):
        plt.plot(xy_dims_error[i] / xy_dims_npoints[i], label=f"Group 2: {xy_dims_range[i]:.2f} $\\lambda$")
    for i in range(len(xy_npix_params)):
        plt.plot(xy_npix_error[i] / xy_npix_npoints[i], label=f"Group 1: {xy_npix_range[i]:.2f} $\\lambda$")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "mean_error.jpg", dpi=300)

    plt.figure()
    plt.plot(xy_dims_range, xy_dims_lerror, "o-", label="Group 2: Varying Dimensions")
    plt.plot(xy_npix_range, xy_npix_lerror, "s-", label="Group 1: Varying Pixels")

    plt.xlabel("Range")
    plt.ylabel("Final Mean Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "last_error.jpg", dpi=300)

    plt.show()
    print(xy_dims_range)
    print(xy_npix_range)