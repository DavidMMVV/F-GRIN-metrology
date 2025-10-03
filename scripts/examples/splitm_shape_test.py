import torch 
import matplotlib.pyplot as plt

from typing import Tuple

from fgrinmet import DEVICE_TORCH
from fgrinmet.splitm.propagation import propagate_paraxial_sta_check
from fgrinmet.utils import coordinates


def spherical_mask(coordinates: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], center: Tuple[float, float, float], radius: float, device=DEVICE_TORCH):
    """Generates a spherical mask.
    Args:
        coordinates (Tuple[torch.Tensor, torch.Tensor, torch.Tensor]): Tuple of 3D tensors with the coordinates.
        center (Tuple[float, float, float]): Center of the sphere.
        radius (float): Radius of the sphere.
        device (torch.device, optional): Device where the tensors are allocated. Defaults to DEVICE.
    Returns:
        mask (torch.Tensor): 3D tensor with the spherical mask.
    """
    assert len(coordinates) == 3, "Shape must be 3D"
    return coordinates[0].to(device=device, dtype=torch.float32)**2 \
         + coordinates[1].to(device=device, dtype=torch.float32)**2 \
         + coordinates[2].to(device=device, dtype=torch.float32)**2 <= radius**2

def unwrap_phase(p, discont=torch.pi):
    """N-D phase unwrap similar to numpy.unwrap, applied along all axes."""
    for dim in range(p.ndim):
        dp = torch.diff(p, dim=dim)
        dp_mod = (dp + torch.pi) % (2*torch.pi) - torch.pi
        dp_mod[(dp_mod == -torch.pi) & (dp > 0)] = torch.pi
        correction = dp_mod - dp
        correction = torch.cumsum(correction, dim=dim)
        shape = list(p.shape)
        shape[dim] = 1
        correction = torch.cat([torch.zeros(shape, device=p.device), correction], dim=dim)
        p = p + correction
    return p

if __name__ == "__main__":
    # Simulation parameters
    sim_shape = (2048, 600, 600) #512, 512)
    wavelength = 532e-9
    tpix_size = 150e-9 / wavelength
    dz = 25e-9 / wavelength
    pix_size = (dz, tpix_size, tpix_size)
    na = 1.5
    w = 20e-6 / wavelength
    radius = 20e-6 / wavelength
    wavelength = 1.0
    dn = 0.3

    Ui = torch.ones(sim_shape[1:], device=DEVICE_TORCH)
    n_vol = torch.zeros(sim_shape, device=DEVICE_TORCH) + na
    Z,Y,X = coordinates.coord_pytorch(sim_shape, pix_size, device=DEVICE_TORCH) 

    spherical_tuple = spherical_mask((X,Y,Z), center=(0,0,0), radius=radius, device=DEVICE_TORCH)
    const = torch.exp(torch.tensor(-(radius**2) / w**2, 
                                device=DEVICE_TORCH, 
                                dtype=X.dtype))

    n_vol[spherical_tuple] = (
        torch.exp(-(X[spherical_tuple]**2 + Y[spherical_tuple]**2 + Z[spherical_tuple]**2) / w**2).float()
        - const) * dn + na
    Uo, L_mod = propagate_paraxial_sta_check(Ui, n_vol, pix_size, 0, na, wavelength)

    extent = [X.min().item(), X.max().item(), Y.min().item(), Y.max().item()]

    titles = ["$n(z=0, y, x)$", "Energy variation","Amplitude", "Phase"]
    fig, sub = plt.subplots(2, 2, figsize=(15, 10))
    im1 = sub[0, 0].imshow(n_vol[n_vol.shape[0]//2].cpu().numpy(), cmap='jet', extent=extent)
    plt.colorbar(im1, ax=sub[0, 0])
    im2 = sub[0, 1].plot(torch.arange(sim_shape[0])*dz, L_mod.cpu().numpy(), label="$\\int|U|^2 dS$")
    sub[0, 1].legend()
    im3 = sub[1, 0].imshow(Uo.abs().cpu().numpy(), cmap='jet', extent=extent)
    plt.colorbar(im3, ax=sub[1, 0])
    im4 = sub[1, 1].imshow(unwrap_phase(Uo.angle()).cpu().numpy(), cmap='jet', extent=extent)
    plt.colorbar(im4, ax=sub[1, 1])

    for i, (ax, title) in enumerate(zip(sub.flatten(), titles)):
        ax.set_title(title)
        if i != 1:
            ax.set_xlabel("$x(\\lambda)$")
            ax.set_ylabel("$y(\\lambda)$")
        else:
            ax.set_ylabel("Energy")
            ax.set_xlabel("$z(\\lambda)$")
    plt.tight_layout()
    plt.show()