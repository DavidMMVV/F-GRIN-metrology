import torch
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from tqdm import tqdm 

import json
from typing import Tuple, Dict, Any

from fgrinmet.splitm import propagate_paraxial
from fgrinmet.utils import GaussianBeamR, GaussianBeamZ, Beam
from fgrinmet.utils import coord_pytorch
from fgrinmet import DEVICE_TORCH

BEAM_MAP = {
    "GaussianBeamZ": GaussianBeamZ,
    "GaussianBeamR": GaussianBeamR
}

def get_prop_data_beams(sim_shape: tuple[int,int,int] | list[int], 
                        wavelength: float, 
                        tpix_size: float, 
                        dz: float, 
                        beam_params: dict,
                        na: float,
                        n_samples: int = 100) -> Tuple[Figure, Dict[str, Any]]:

    pix_sizes = [dz, tpix_size, tpix_size]
    Lz = dz*(sim_shape[0])
    
    Y, X = coord_pytorch(sim_shape[1:], tpix_size)
    Z = torch.arange(sim_shape[0], device=DEVICE_TORCH) * dz


    beam = BEAM_MAP[beam_params["type"]](y=Y, 
                                         x=X, 
                                         **{k: v for k,v in beam_params.items() if k!= "type"}, 
                                         wavelength=wavelength)
    
    n_vol = torch.as_tensor(beam.get_n()[None].repeat((sim_shape[0],1,1)) 
             if isinstance(beam, GaussianBeamR) 
             else beam.get_n(Z)[:,None, None].repeat((1, sim_shape[1], sim_shape[2])), dtype=torch.float64, device=DEVICE_TORCH)

    i_sampled = torch.linspace(0, sim_shape[0]-1, steps=n_samples, device=DEVICE_TORCH).long()
    i_prev = 0
    Ui = beam.get_E(0)
    L_arr_mod = torch.zeros(i_sampled.shape, dtype=torch.float64, device=DEVICE_TORCH)
    L_arr_phase = torch.zeros(i_sampled.shape, dtype=torch.float64, device=DEVICE_TORCH)

    for n, i in enumerate(i_sampled[1:]):
        Uo = propagate_paraxial(Ui, n_vol[i_prev:i], pix_sizes, 0, na, wavelength)
        Ui = Uo.clone()
        i_prev = i
        Uo_an = beam.get_E(float(i * dz))
        L_mod = ((Uo_an.abs() - Uo.abs())**2).sum() / Uo.numel()
        L_ph = ((Uo_an.angle() - Uo.angle())**2).sum() / Uo.numel()
        L_arr_mod[n+1] = L_mod
        L_arr_phase[n+1] = L_ph

    #Uo = propagate_paraxial(Ui, n_vol, pix_sizes, 0, na, wavelength)

    extent = [X[0,0].detach().cpu().numpy(), X[-1,-1].detach().cpu().numpy(), 
              Y[0,0].detach().cpu().numpy(), Y[-1,-1].detach().cpu().numpy()]
    titles = ["Refraction index", "MSE function evolution", 
              "Analytic beam", "Computed beam", 
              "Amplitude difference", "Phase difference"]
    fig, sub = plt.subplots(3, 2, constrained_layout=True)

    if isinstance(beam, GaussianBeamR):
        fig.suptitle("Comparison for $n(r_\\bot)$", fontsize=16)
        figures = [n_vol[0].detach().cpu(), (i_sampled.detach().cpu().numpy(), L_arr_mod.detach().cpu().numpy(), L_arr_phase.detach().cpu().numpy()), 
                   Uo_an.abs().detach().cpu(), Uo.abs().detach().cpu(),
                   ((Uo.abs() - Uo_an.abs())).detach().cpu(),
                   ((Uo.angle() - Uo_an.angle())).detach().cpu()]
        for i, ax in enumerate(sub.flatten()):
            if isinstance(figures[i], tuple):
                ax.plot(*figures[i][:2], label="Module MSE")
                ax.plot(*figures[i][::2], label="Phase MSE")
                ax.legend()

            else:
                if i != 2:
                    ax.sharex(sub[1,0])
                    ax.sharey(sub[1,0])
                rep = ax.imshow(figures[i], extent=extent, origin="lower")
                plt.colorbar(rep, ax=ax)
                ax.set_xlabel("$x(\\lambda)$")
                ax.set_ylabel("$y(\\lambda)$")

            ax.set_title(titles[i])

    elif isinstance(beam, GaussianBeamZ):
        fig.suptitle("Comparison for $n(z)$", fontsize=16)
        figures = [(torch.arange(sim_shape[0])*dz, n_vol[:,0,0].detach().cpu()),
                   (i_sampled.detach().cpu().numpy()*dz, L_arr_mod.detach().cpu().numpy(), L_arr_phase.detach().cpu().numpy()), 
                   Uo_an.abs().detach().cpu(), Uo.abs().detach().cpu(),
                   ((Uo.abs() - Uo_an.abs())).detach().cpu(),
                   ((Uo.angle() - Uo_an.angle())).detach().cpu()]
        
        for i, ax in enumerate(sub.flatten()):
            if i == 0:
                ax.plot(*figures[i])
                ax.set_xlabel("$z(\\lambda)$")
                ax.set_ylabel("$n(z)$")
            elif i == 1:
                ax.plot(*figures[i][:2], label="Module MSE")
                ax.plot(*figures[i][::2], label="Phase MSE")
                ax.legend()
                ax.set_xlabel("$z(\\lambda)$")
                ax.set_ylabel("$MSE$")
            else:
                if i != 2:
                    ax.sharex(sub[1,0])
                    ax.sharey(sub[1,0])
                rep = ax.imshow(figures[i], extent=extent, origin="lower")
                plt.colorbar(rep, ax=ax)
                ax.set_xlabel("$x(\\lambda)$")
                ax.set_ylabel("$y(\\lambda)$")
            ax.set_title(titles[i])
    else:
        raise TypeError("An error with the beam type has ocurred!")
    
    data_to_return = {
        "sim_shape": sim_shape,
        "wavelength": wavelength,
        "tpix_size": tpix_size,
        "dz": dz,
        "beam_params": beam_params,
        "na": na,
        "n_samples": n_samples,
        "L_arr_mod": L_arr_mod.detach().cpu(),
        "L_arr_phase": L_arr_phase.detach().cpu(),
    }
    
    return fig, data_to_return
    
def load_beam_params_from_json(json_path: str):
    """
    Reads a json and returns the data needed for my main function
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    
    # Convertimos listas a tuplas o tensores si es necesario
    sim_shape = tuple(data["sim_shape"])
    wavelength = data["wavelength"]
    tpix_size = data["tpix_size"]
    dz = data["dz"]
    na = data["na"]
    n_samples = data.get("n_samples", 100)
    beam_params = data["beam_params"]
    
    return{"sim_shape": sim_shape, "wavelength": wavelength, "tpix_size": tpix_size, 
           "dz": dz, "beam_params": beam_params, "na": na, "n_samples": n_samples}

def save_figure_and_data(fig, data_dict: dict, prefix: str):
    """
    Save output figure and data used for the simulation.
    """

    fig.savefig(f"{prefix}_figure.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Convert tensors to lists
    json_data = {}
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            json_data[k] = v.tolist()
        else:
            json_data[k] = v

    # Save JSON
    with open(f"{prefix}_data.json", "w") as f:
        json.dump(json_data, f, indent=4)

if __name__ == "__main__":
    from config import DATA_DIR
    from pathlib import Path

    out_dir = (((Path(DATA_DIR) / "local") / "splitm_test") / "gaussz") / "test"
    out_dir.mkdir(parents = True, exist_ok = True)

    sim_shape = (2048,512,512)
    wavelength = 650e-9
    tpix_size = 1e-6 / wavelength
    dz = 1e-6 / wavelength
    na = 1.5
    A = 1.0
    pix_sizes = [dz, tpix_size, tpix_size]
    Lz = dz*sim_shape[0]
    Lr = tpix_size * float(sim_shape[1] if sim_shape[1]<=sim_shape[2] else sim_shape[2])
    a_r = 0.2*Lr # Beam waist
    b_r = 2*torch.pi / (100*Lz) # Beam spacial frequency

    a_z = -0.1/(Lz**2) # Variation of refraction index
    b_z = 1.5 # Base refraction index
    s_z = 1 / (0.2*Lr)**2 # Inverse beam waist squared
    n_samples = 50

    wavelength /= wavelength

    beam_dicts = [{"sim_shape": sim_shape, "wavelength": wavelength, "tpix_size": tpix_size, "dz": dz,
            "beam_params": {"type": "GaussianBeamZ", "A": A,
             "s": s_z, "a": a_z, "b": b_z, "na": na},
             "na": na, "n_samples": n_samples},
           {"sim_shape": sim_shape, "wavelength": wavelength, "tpix_size": tpix_size, "dz": dz,
            "beam_params": {"type": "GaussianBeamR", "A": A,
             "a": a_r, "b": b_r, "na": na},
            "na": na, "n_samples": n_samples}
            ]

    #k = 60
    #beam_dicts = [{"sim_shape": sim_shape, "wavelength": wavelength, "tpix_size": tpix_size*2 * i / k, "dz": dz,
    #        "beam_params": {"type": "GaussianBeamZ", "A": A,
    #         "s": s_z, "a": a_z, "b": b_z, "na": na},
    #        "na": na, "n_samples": n_samples} for i in range(1,k)]

    for i in tqdm(range(len(beam_dicts))):
        params = get_prop_data_beams(**beam_dicts[i])
        out_filename = str(out_dir / str(i))
        save_figure_and_data(*params, out_filename)