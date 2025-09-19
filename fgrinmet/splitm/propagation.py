import torch
import numpy as np

from typing import List, Tuple, Union

from ..globvar import DEVICE
from ..utils.coordinates import coord_pytorch, fft_coord_pytorch, fft_coord_pytorch_jit
from ..utils.operators import FT2, iFT2

def propagate_paraxial(
        Ui: torch.Tensor,  # 2D tensor with shape (H, W) 
        n_vol: torch.Tensor, # 3D tensor with shape (D, H, W) 
        pix_size: float | List[float] | Tuple[float,...] = 1.0,
        axis: int = 0,
        na: float = 1.5,
        wavelength: float = 645e-9
    ) -> torch.Tensor:
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

    Fy, Fx = fft_coord_pytorch(tshape, pix_size, dtype, device)
    dz = float(pix_size[axis]) if isinstance(pix_size, list | tuple) else pix_size
    
    # Compute the free space paraxial propagator
    C = 1j*torch.pi * wavelength * dz / (2 * na)
    parax_prop = torch.exp(C * (Fx**2 + Fy**2))

    # Compute the propagation
    Uo = Ui.clone()
    for i in range(Nz):
        Uo = iFT2(parax_prop * FT2(torch.exp(-2*C*(n_vol[i]**2-na**2)) * iFT2(FT2(Uo))))

    return Uo

def propagate_paraxial_jit(
        Ui: torch.Tensor,
        n_vol: torch.Tensor,
        pix_size: Union[float, List[float]] = 1.0,
        axis: int = 0,
        na: float = 1.5,
        wavelength: float = 1.0
    ) -> torch.Tensor:
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
    shape = n_vol.shape
    device = n_vol.device
    dtype = n_vol.dtype

    is_taxes: List[int] = []
    for i in range(len(shape)):
        if i != axis:
            is_taxes.append(i)

    tshape: List[int] = []
    for i in is_taxes:
        tshape.append(shape[i])

    Nz = shape[axis]

    Fy, Fx = fft_coord_pytorch_jit(tshape, pix_size, dtype, device)
    if isinstance(pix_size, list):
        dz = float(pix_size[axis])
    else:
        dz = pix_size

    
    # Compute the free space paraxial propagator
    C_real = torch.tensor(wavelength * dz / (2 * na), dtype=Fx.dtype, device=Fx.device)
    C = torch.complex(C_real*0, C_real)
    parax_prop = torch.exp(C * (Fx**2 + Fy**2))

    # Compute the propagation
    Uo = Ui.clone()
    for i in range(Nz):
        Uo = iFT2(parax_prop * FT2(torch.exp(-2*C*(n_vol[i]**2-na**2)) * iFT2(FT2(Uo))))

    return Uo