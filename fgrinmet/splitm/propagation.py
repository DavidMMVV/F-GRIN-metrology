import torch
import numpy as np
import jax.numpy as jnp
from jax import jit

from typing import List, Tuple, Optional

from ..utils import coord_jax, fft_coord_jax, coord_pytorch, fft_coord_pytorch, FT2, iFT2
from .interpolation import trilinear_interpolate

# TODO: Implement in jax
@jit
def propagate_paraxial_jax(
        Ui: jnp.ndarray,
        n_vals: jnp.ndarray,
        prop_shape: List[int] | Tuple[int,int,int],
        obj_shape: List[int] | Tuple[int,int,int],
        init_plane_coord: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None,
        prop_pix_size: float | List[float] | Tuple[float,float,float] = 1.0,
        obj_pix_size: float | List[float] | Tuple[float,float,float] = 1.0,
        na: float = 1.5,
        wavelength: float = 645e-9
        ) -> jnp.ndarray:
    """Propagate a beam over a rectangular prism volume considering paraxial aproximation. the axis along which the field is propagatd is the first (axis=0).

    Args:
        Ui (jnp.ndarray): Input field with dimensions of the transversal plane in the media. 2D tensor with shape (Hg, Wg).
        n_vol(jnp.ndarray): Distribution of the index of refraction. 3D tensor with shape (Do, Ho, Wo).
        mask (Optional[jnp.ndarray], optional): A mask to specify valid points with shape (Do, Ho, Wo) which fulfills that N = mask.sum(). Defaults to None.
        init_plane_coord (Optional[jnp.ndarray], optional): Initial plane coordinates. Defaults to None.
        prop_pix_size(jnp.ndarray): Pixel size of the propagation grid. Defaults to 1.0.
        obj_pix_size(jnp.ndarray): Pixel size of the object. Defaults to 1.0.
        na (float, optional): Average index of refraction. Defaults to 1.5.
        wavelength (float, optional): Wavelength of the light in the vacuum. Defaults to 645e-9.

    Returns:
        Uo(jnp.ndarray): Output field.
    """
    def step(carry_in, iterable):
        Ui, prop_plane_in, na, mask, z_vec, prop_fact = carry_in
        i = iterable
        plane_coord = prop_plane_in + z_vec[None, None]

        n_plane = trilinear_interpolate(plane_coord, n_vals, na, mask)

        Uo = paraxial_propagation_step_jax(Ui, n_plane, prop_fact, dz, na, wavelength)
        carry_out = (Uo, plane_coord, na, mask, z_vec, prop_fact)

        return carry_out, None #stored_values


    t_prop_pix_size = prop_pix_size[1:] if isinstance(prop_pix_size, list | tuple) else prop_pix_size
    dz = prop_pix_size[0] if isinstance(prop_pix_size, tuple | list) else prop_pix_size
    t_obj_pix_size = obj_pix_size[1:] if isinstance(obj_pix_size, list | tuple) else obj_pix_size

    z_dir = jnp.cross(init_plane_coord[0,1] - init_plane_coord[0,0], 
                      init_plane_coord[1,0] - init_plane_coord[0,0])
    z_vec = z_dir * dz / jnp.sqrt((z_dir**2).sum())
    
    Fy, Fx = fft_coord_jax(prop_shape[1:], t_prop_pix_size)

    prop_fact = paraxial_propagator_jax(Fy, Fx, dz, na, wavelength)
    carry_in = (Ui, init_plane_coord, na, mask, z_vec, prop_fact, dz, na, wavelength)

    for i in range(prop_shape[0]):
        step
        #trilinear_interpolate(plane_coords, n_vol, na, mask)

    return jnp.ones_like(Ui)

def paraxial_propagator_jax(
        Fy: jnp.ndarray,
        Fx: jnp.ndarray,
        dz: float = 1.0,
        na: float = 1.5,
        wavelength: float = 645e-9
    ) -> jnp.ndarray:
    """Computes the propagation term in free space considering paraxial approximation.

    Args:
        Fy (jnp.ndarray): Fourier frequency coordinate in Y direction.
        Fx (jnp.ndarray): Fourier frequency coordinate in X direction.
        dz (float, optional): Step length. Defaults to 1.0.
        na (float, optional): Average index of refraction of the media. Defaults to 1.5.
        wavelength (float, optional): Wavelength of the light in vacuum. Defaults to 645e-9.

    Returns:
        propagator (jnp.ndarray): Paraxial propagator factor.
    """
    return jnp.exp((1j * torch.pi * wavelength * dz / (2 * na)) * (Fx**2 + Fy**2))

def paraxial_propagation_step_jax(
        Ui: jnp.ndarray,
        n_plane: jnp.ndarray,
        prop_factor: jnp.ndarray,
        dz: float = 1.0,
        na: float = 1.5,
        wavelength: float = 645e-9
    ) -> jnp.ndarray:
    """Computes a single propagaion step in paraxial approximation.

    Args:
        Ui (jnp.ndarray): Input field with dimensions of the transversal plane in the media. 2D tensor with shape (Hg, Wg).
        n_plane(jnp.ndarray): Values of the index of refraction in the slice.
        prop_factor (jnp.ndarray): Free space propagation factor.
        dz (float, optional): Step length. Defaults to 1.0.
        na (float, optional): Average index of refraction of the media. Defaults to 1.5.
        wavelength (float, optional): Wavelength of the light in vacuum. Defaults to 645e-9.

    Returns:
        Uo(jnp.ndarray): Output field from the slice.
    """

    return iFT2(prop_factor * FT2(jnp.exp((1j * torch.pi * dz /(na * wavelength))*(na**2-n_plane**2)) *
                            iFT2(prop_factor * FT2(Ui))))

def energy(
        field: jnp.ndarray, 
        t_pix_size: float | List[float] | Tuple[float,float]):
    """Computes the energy of the field in a certain plane.

    Args:
        field (jnp.ndarray): Field distribution in the plane.
        t_pix_size (float | List[float] | Tuple[float,float]): Pizel sizes in the plane.

    Returns:
        Energy(float): Energy of the field in the plane.
    """

    return ((jnp.abs(field)**2).sum())*(np.prod(t_pix_size) if isinstance(t_pix_size, list | tuple) else t_pix_size**2)

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

def propagate_paraxial_sta_check(
        Ui: torch.Tensor,  # 2D tensor with shape (H, W) 
        n_vol: torch.Tensor, # 3D tensor with shape (D, H, W) 
        pix_size: float | List[float] | Tuple[float,...] = 1.0,
        axis: int = 0,
        na: float = 1.5,
        wavelength: float = 645e-9
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Propagates a beam over a rectangular prism volume considering paraxial approximation.

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
        L_mod[i] = ((Uo.abs()**2).sum() - mod_0) * (tpix_size**2 if isinstance(tpix_size, (int, float)) else float(np.prod(tpix_size))) / mod_0
        
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
