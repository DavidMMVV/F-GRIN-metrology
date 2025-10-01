import torch
import numpy as np

from typing import List, Tuple, Union

from fgrinmet.globvar import DEVICE_TORCH
from fgrinmet.utils import coord_pytorch, fft_coord_pytorch, FT2, iFT2

def propagate_paraxial(
        Ui: torch.Tensor,  # 2D tensor with shape (H, W) 
        n_vol: torch.Tensor, # 3D tensor with shape (D, H, W) 
        pix_size: float | List[float] | Tuple[float,...] = 1.0,
        axis: int = 0,
        na: float = 1.5,
        wavelength: float = 1.0
    ) -> torch.Tensor:
    """Propagates a beam over a square prism volume considering paraxial approximation.

    Args:
        Ui (torch.Tensor): Input field with dimensions of the transversal plane in the media.
        n_vol (torch.Tensor): Distribution of index of refraction in the media.
        pix_size (float | ShapeLike): Size of the pixel over each dimension. If it
        a cubic pixel of this size will be considered.
        axis (int, optional): Axis along which the propagation is performed. Defaults to 0
        na (float, optional): Average index of refraction of the media. Defaults to 1.5.
        wavelength (float, optional): Wavelength of the light. Defaults to 1.0.
9.
    Returns:
        Uo(torch.Tensor): Output field. 
    """
    shape = np.array(n_vol.shape)
    dtype = n_vol.dtype
    device = n_vol.device

    is_taxes = np.arange(len(shape)) != axis
    tshape = list(shape[is_taxes])
    Nz = shape[axis]

    tpix_size = [pix_size[i] for i in range(len(shape)) if is_taxes[i]] if isinstance(pix_size, list | tuple) else pix_size
    Fy, Fx = fft_coord_pytorch(tshape, tpix_size, dtype, device) 
    dz = float(pix_size[axis]) if isinstance(pix_size, list | tuple) else pix_size
    
    parax_prop = torch.exp((1j * torch.pi * wavelength * dz / (2 * na)) * (Fx**2 + Fy**2))

    # Compute the propagation
    Uo = Ui.clone()
    for i in range(Nz):
        Uo = iFT2(parax_prop * FT2(torch.exp((1j * torch.pi * dz /(na * wavelength))*(na**2-n_vol[i]**2)) *
                                    iFT2(parax_prop * FT2(Uo))))

    return Uo

# TODO: Implement in jax
def propagate_paraxialjax():
    pass

def propagate_paraxial_sta_check(
        Ui: torch.Tensor,  # 2D tensor with shape (H, W) 
        n_vol: torch.Tensor, # 3D tensor with shape (D, H, W) 
        pix_size: float | List[float] | Tuple[float,...] = 1.0,
        axis: int = 0,
        na: float = 1.5,
        wavelength: float = 645e-9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagates a beam over a square prism volume considering paraxial approximation.

    Args:
        Ui (torch.Tensor): Input field with dimensions of the transversal plane in the media.
        n_vol (torch.Tensor): Distribution of index of refraction in the media.
        pix_size (float | ShapeLike): Size of the pixel over each dimension. If it is a float
        a cubic pixel of this size will be considered.
        axis (int, optional): Axis along which the propagation is performed. Defaults to 0.
        na (float, optional): Average index of refraction of the media. Defaults to 1.5.
        wavelength (float, optional): Wavelength of the light. Defaults to 645e-9.

    Returns:
        Uo(torch.Tensor): Output field.
    """
    shape = np.array(n_vol.shape)
    dtype = n_vol.dtype
    device = n_vol.device

    is_taxes = np.arange(len(shape)) != axis
    tshape = list(shape[is_taxes])
    Nz = shape[axis]

    tpix_size = [pix_size[i] for i in range(len(shape)) if is_taxes[i]] if isinstance(pix_size, list | tuple) else pix_size
    Fy, Fx = fft_coord_pytorch(tshape, tpix_size, dtype, device) 
    dz = float(pix_size[axis]) if isinstance(pix_size, list | tuple) else pix_size
    
    parax_prop = torch.exp((1j * torch.pi * wavelength * dz / (2 * na)) * (Fx**2 + Fy**2))

    # Compute the propagation
    Uo = Ui.clone()
    mod_0 = (Ui.abs()**2).sum()
    L_mod = torch.zeros(Nz, dtype=mod_0.dtype, device=device)
    for i in range(Nz):
        Uo = iFT2(parax_prop * FT2(torch.exp((1j * torch.pi * dz /(na * wavelength))*(na**2-n_vol[i]**2)) *
                                    iFT2(parax_prop * FT2(Uo))))
        L_mod[i] = (Uo.abs()**2).sum() - mod_0
        
    return Uo, L_mod

#def propagate_asm_sta_check(
#        Ui: torch.Tensor,  # 2D tensor with shape (H, W) 
#        n_vol: torch.Tensor, # 3D tensor with shape (D, H, W) 
#        pix_size: float | List[float] | Tuple[float,...] = 1.0,
#        axis: int = 0,
#        na: float = 1.5,
#        wavelength: float = 645e-9
#    ) -> Tuple[torch.Tensor, torch.Tensor]:
#    """Propagates a beam over a square prism volume considering paraxial approximation.
#
#    Args:
#        Ui (torch.Tensor): Input field with dimensions of the transversal plane in the media.
#        n_vol (torch.Tensor): Distribution of index of refraction in the media.
#        pix_size (float | ShapeLike): Size of the pixel over each dimension. If it is a float
#        a cubic pixel of this size will be considered.
#        axis (int, optional): Axis along which the propagation is performed. Defaults to 0.
#        na (float, optional): Average index of refraction of the media. Defaults to 1.5.
#        wavelength (float, optional): Wavelength of the light. Defaults to 645e-9.
#
#    Returns:
#        Uo(torch.Tensor): Output field.
#    """
#    shape = np.array(n_vol.shape)
#    dtype = n_vol.dtype
#    device = n_vol.device
#
#    is_taxes = np.arange(len(shape)) != axis
#    tshape = list(shape[is_taxes])
#    Nz = shape[axis]
#
#    tpix_size = [pix_size[i] for i in range(len(shape)) if is_taxes[i]] if isinstance(pix_size, list | tuple) else pix_size
#    Fy, Fx = fft_coord_pytorch(tshape, tpix_size, dtype, device) 
#    dz = float(pix_size[axis]) if isinstance(pix_size, list | tuple) else pix_size
#    
#    parax_prop = torch.exp((1j * torch.pi * wavelength * dz / (2 * na)) * (Fx**2 + Fy**2))
#
#    # Compute the propagation
#    Uo = Ui.clone()
#    mod_0 = (Ui.abs()**2).sum()
#    L_mod = torch.zeros(Nz, dtype=mod_0.dtype, device=device)
#    for i in range(Nz):
#        Uo = iFT2(parax_prop * FT2(torch.exp((1j * torch.pi * dz /(na * wavelength))*(na**2-n_vol[i]**2)) *
#                                    iFT2(parax_prop * FT2(Uo))))
#        L_mod[i] = (Uo.abs()**2).sum() - mod_0
#    return Uo, L_mod
