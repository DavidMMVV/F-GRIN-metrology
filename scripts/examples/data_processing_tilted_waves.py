from config import LOCAL_DATA_DIR
import json
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.stats import linregress

if __name__ == "__main__":
    tilted_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "tilted_gaussian_25_modes" / "params"
    tilted_z_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "tilted_gaussian_z_25_modes" / "params"
    tilted_z_obj_shape_dir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "tilted_gaussian_z_25_obj_shape" / "params"
    savedir = LOCAL_DATA_DIR / "splitm_grad_multimode" / "tilted_gaussian_analysis"
    savedir.mkdir(parents=True, exist_ok=True)
    tilted_files = [tilted_dir / d for d in os.listdir(tilted_dir)]
    tilted_z_files = [tilted_z_dir / d for d in os.listdir(tilted_z_dir)]
    tilted_z_obj_shape_files = [tilted_z_obj_shape_dir / d for d in os.listdir(tilted_z_obj_shape_dir)]

    tilted_params = []
    for f in tilted_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        tilted_params.append(param_dict)
    
    tilted_z_params = []
    for f in tilted_z_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        tilted_z_params.append(param_dict)

    tilted_z_obj_shape_params = []
    for f in tilted_z_obj_shape_files:
        with open(f, "r") as file:
            param_dict = json.load(file)
        tilted_z_obj_shape_params.append(param_dict)    
    
    tilted_mse = ([param["losses"][1] for param in tilted_params])
    tilted_z_mse = ([param["losses"][1] for param in tilted_z_params])
    tilted_z_obj_shape_mse = ([param["losses"][1] for param in tilted_z_obj_shape_params])
    tilted_total_err = ([param["total_error"] for param in tilted_params])
    tilted_z_total_err = ([param["total_error"] for param in tilted_z_params])
    tilted_z_obj_shape_total_err = ([param["total_error"] for param in tilted_z_obj_shape_params])

    tilted_names = [f"{param['modes']} " + ("wave" if param['modes'] == 1 else "waves") for param in (tilted_params)]
    tilted_z_names = [f"{param['modes']} " + ("wave" if param['modes'] == 1 else "waves") for param in (tilted_z_params)]
    tilted_z_obj_shape_names = [param["object_params"]["shape"][1] for param in tilted_z_obj_shape_params]
    
    plt.figure("total_error_tilted")
    for i in range(len(tilted_total_err)):
        plt.plot(tilted_total_err[i], label=tilted_names[i])
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Total_error_tilted.jpg", dpi=300)

    y_tilted_total_err = [params["total_error"][-1] for params in tilted_params]
    x_tilted_total_err = [params["modes"] for params in tilted_params]

    plt.figure("last_total_error_tilted")
    plt.scatter(x_tilted_total_err, y_tilted_total_err)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of Modes")
    plt.ylabel("Last Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Last_total_error_tilted.jpg", dpi=300)

    plt.figure("total_error_tilted_z")
    for i in range(len(tilted_z_total_err)):
        plt.plot(tilted_z_total_err[i], label=tilted_z_names[i])
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Total_error_tilted_z.jpg", dpi=300)

    y_tilted_z_total_err = [params["total_error"][-1] for params in tilted_z_params]
    x_tilted_z_total_err = [params["modes"] for params in tilted_z_params]

    plt.figure("last_total_error_tilted_z")
    plt.scatter(x_tilted_z_total_err, y_tilted_z_total_err)
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Number of Modes")
    plt.ylabel("Last Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Last_total_error_tilted_z.jpg", dpi=300)


    plt.figure("total_error_tilted_z_obj_shape")
    for i in range(len(tilted_z_obj_shape_total_err)):
        plt.plot(tilted_z_obj_shape_total_err[i], label=f"{tilted_z_obj_shape_names[i]}")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Total_error_tilted_z_obj_shape.jpg", dpi=300)

    #y_tilted_z_obj_shape_total_err = [params["total_error"][-1] / np.prod(params["object_params"]["shape"]) for params in tilted_z_obj_shape_params]
    y_tilted_z_obj_shape_total_err = np.array([params["total_error"][-1] for params in tilted_z_obj_shape_params])
    x_tilted_z_obj_shape_total_err = np.array([params["object_params"]["shape"][1] for params in tilted_z_obj_shape_params])

    x_log = np.log(x_tilted_z_obj_shape_total_err)
    y_log = np.log(y_tilted_z_obj_shape_total_err)
    reg_vals = linregress(x_log, y_log)
    print(dir(reg_vals))
    r_value = reg_vals.rvalue
    x_adj = np.exp(np.linspace(min(x_log), max(x_log), 100))
    y_adj = np.exp(reg_vals.intercept) * x_adj**reg_vals.slope
    y_min_adj = np.exp(reg_vals.intercept-reg_vals.intercept_stderr) * x_adj**(reg_vals.slope-reg_vals.stderr)
    y_max_adj = np.exp(reg_vals.intercept+reg_vals.intercept_stderr) * x_adj**(reg_vals.slope+reg_vals.stderr)

    plt.figure("last_total_error_tilted_z_obj_shape")
    plt.scatter(x_tilted_z_obj_shape_total_err, y_tilted_z_obj_shape_total_err)
    plt.plot(x_adj,y_adj,color="tab:orange")
    plt.fill_between(x_adj, y_min_adj, y_max_adj, alpha=0.3, color="tab:orange")
    plt.text(200, 100000, 
             f"$y = e^{{{reg_vals.intercept:.1f}\\pm{reg_vals.intercept_stderr:.1f}}}$" \
             f"$x^{{{reg_vals.slope:.2f}\pm{reg_vals.stderr:.2f}}}$ \n" \
             f"$r^2 = {r_value**2:.3f}$")
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Pixels in y direction")
    plt.ylabel("Last Total Error")
    plt.tight_layout()
    plt.savefig(savedir / "Last_total_error_tilted_z_obj_shape.jpg", dpi=300)

    plt.figure("mse_tilted")
    for i in range(len(tilted_total_err)):
        plt.plot(tilted_mse[i], label=tilted_names[i])
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")  
    plt.tight_layout()
    plt.savefig(savedir / "MSE_tilted.jpg", dpi=300) 

    plt.figure("mse_tilted_z")
    for i in range(len(tilted_z_total_err)):
        plt.plot(tilted_z_mse[i], label=tilted_z_names[i])
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")   
    plt.tight_layout()
    plt.savefig(savedir / "MSE_tilted_z.jpg", dpi=300)

    plt.figure("mse_tilted_z_obj_shape")
    for i in range(len(tilted_z_obj_shape_total_err)):
        plt.plot(tilted_z_obj_shape_mse[i], label=f"{tilted_z_obj_shape_names[i]}")
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")   
    plt.tight_layout()
    plt.savefig(savedir / "MSE_tilted_z_obj_shape.jpg", dpi=300)