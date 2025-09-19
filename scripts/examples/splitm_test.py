import torch
import matplotlib.pyplot as plt

import time

from fgrinmet.splitm import propagate_paraxial
from fgrinmet.utils import GaussianBeam 
from fgrinmet.utils import coord_pytorch
from fgrinmet import DEVICE



if __name__ == "__main__":

    device = DEVICE
    sim_shape = (512,512,512)
    wavelength = 650e-9
    tpix_size = 1e-6 / wavelength
    dz = 1e-6 / wavelength
    pix_sizes = [dz, tpix_size, tpix_size]
    
    Y, X = coord_pytorch(sim_shape[1:], tpix_size)
    X, Y = (X, Y)

    na = 1.5 # Average index of refraction
    a = 1e-3 / wavelength # Beam waist
    b = 2*torch.pi / (1000e-3 / wavelength) # Beam spacial frequency
    g_beam = GaussianBeam(y=Y, x=X, a=a, b=b, na=na, wavelength=wavelength/wavelength)

    n_vol = g_beam.get_n()[None].repeat(sim_shape[0],1,1).requires_grad_() #type: ignore
    Ui = g_beam.get_E(0)

    start = time.perf_counter()
    Uo = propagate_paraxial(Ui, n_vol, pix_sizes, 0, na, wavelength/wavelength)
    end = time.perf_counter()
    print(f"Not compiled: {end-start}s")

    n_samples = 100
    Lz = dz*sim_shape[0]
    i_sampled = torch.linspace(0, sim_shape[0]-1, steps=n_samples, device=device).long()
    Uos = g_beam.get_E(0)
    i_prev = 0

    L_arr = torch.empty(i_sampled.shape, dtype=torch.float64, device=device)
    var_E_arr = torch.empty(i_sampled.shape[0], dtype=torch.float64, device=device)

    L_arr[0] = (Ui.abs()**2 - Uos.abs()**2).sum() / Uos.numel()
    var_E_arr[0] = 0

    for n, i in enumerate(i_sampled[1:]):
        Uos = propagate_paraxial(Uos, n_vol[i_prev:i], pix_sizes, 0, na, wavelength/wavelength)
        i_prev = i
        Uos_an = g_beam.get_E(float(i * dz))
        L = (Uos_an.abs()**2 - Uos.abs()**2).sum() / Uos.numel()
        L_arr[n+1] = L

    fig, sub = plt.subplots(2, 2)#, figsize=(15,15))

    figures = [n_vol[0].detach().cpu(), (i_sampled.detach().cpu().numpy(), L_arr.detach().cpu().numpy()), 
               Uo.real.detach().cpu(), (Uo - g_beam.get_E(Lz)).angle().detach().cpu()]
    extent = [X[0,0].detach().cpu().numpy(), X[-1,-1].detach().cpu().numpy(), 
              Y[0,0].detach().cpu().numpy(), Y[-1,-1].detach().cpu().numpy()]

    for i, ax in enumerate(sub.flatten()):
        if i == 1:
            ax.plot(*figures[i])
        else:
            rep = ax.imshow(figures[i], extent=extent, origin="lower")
            plt.colorbar(rep, ax=ax)

    plt.tight_layout()
    plt.show()