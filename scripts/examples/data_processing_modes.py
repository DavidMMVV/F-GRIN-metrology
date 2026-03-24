from config import LOCAL_DATA_DIR
import json
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    modes_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "modes" / "params"
    modes_z_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "modes_z" / "params"
    savedir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "modes_analysis"
    savedir.mkdir(parents=True, exist_ok=True)
    modes_files = [modes_dir / d for d in os.listdir(modes_dir)]
    modes_z_files = [modes_z_dir / d for d in os.listdir(modes_z_dir)]

    modes_params = []
    for f in modes_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        modes_params.append(param_dict)
    
    modes_z_params = []
    for f in modes_z_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        modes_z_params.append(param_dict)
    
    modes_losses = np.array([param["losses"][1] for param in modes_params])
    modes_z_losses = np.array([param["losses"][1] for param in modes_z_params])
    modes_total_err = np.array([param["total_error"] for param in modes_params])
    modes_z_total_err = np.array([param["total_error"] for param in modes_z_params])
    modes_name = [f"{param['modes']} " + ("mode" if param['modes'] == 1 else "modes") for param in (modes_params)]
    modes_name[-1] = "Plane wave" 
    print(modes_losses.shape, modes_z_losses.shape, modes_total_err.shape, modes_z_total_err.shape)

    plt.figure()
    for i in range(len(modes_params)):
        plt.plot(modes_losses[i], label=modes_name[i])
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "modes_losses.jpg", dpi=300)

    plt.figure()
    for i in range(len(modes_z_params)):
        plt.plot(modes_z_losses[i], label=modes_name[i])
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "modes_z_losses.jpg", dpi=300)

    plt.figure()
    for i in range(len(modes_params)):
        plt.plot(modes_total_err[i], label=modes_name[i])
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "modes_total_err.jpg", dpi=300)

    plt.figure()
    for i in range(len(modes_z_params)):
        plt.plot(modes_z_total_err[i], label=modes_name[i])
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend()
    plt.savefig(savedir / "modes_z_total_err.jpg", dpi=300)
    plt.show()