from typing import overload
import torch
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod

NumericArray = float | np.ndarray | torch.Tensor

class Beam(ABC):
    """
    Abstract base class for all beams.
    """
    def __init__(self, x: NumericArray, y: NumericArray):
        self.x = x
        self.y = y

    @abstractmethod
    def get_n(self, *args, **kwargs) -> NumericArray:
        """Return the refractive index distribution."""
        ...

    @abstractmethod
    def get_E(self, *args, **kwargs) -> NumericArray:
        """Return the electric field at position z."""
        ...


@dataclass
class GaussianBeamZ(Beam):
    y: NumericArray
    x: NumericArray
    A: float
    s: float
    a: float
    b: float
    na: float
    wavelength: float
    """_summary_
    """

    def get_n(self, z: NumericArray) -> NumericArray:
        return self.a * z**2 + self.b

    def get_E(self, z: float):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            device, dtype = self.x.device, self.x.dtype
            const = torch.tensor(1 / (1 - 1j * self.wavelength * z * self.s / (torch.pi * self.na)), 
                                dtype=float_to_complex_dtype(dtype), device=device)
            phase_phi = torch.tensor((1j * (torch.pi * self.na / self.wavelength) * z * 
                                    (1 - (self.a**2 * z**4 / 5 + 2*self.a*self.b*z**2 / 3 + self.b**2) / self.na**2)), dtype=float_to_complex_dtype(dtype), device=device)
            return self.A * const * torch.exp(-const * self.s * (self.x**2 + self.y**2)) * torch.exp(phase_phi)
        else:
            const_np = 1 / (1 - 1j * self.wavelength * z * self.s / (np.pi * self.na))
            phase_phi_np = (1j * (np.pi * self.na / self.wavelength) * z * 
                                        (1 - (self.a**2 * z**4 / 5 + 2*self.a*self.b*z**2 / 3 + self.b**2) / self.na**2))
            return self.A * const_np * np.exp(-const_np * self.s * (self.x**2 + self.y**2)) * np.exp(phase_phi_np)
        
@dataclass
class GaussianBeamR(Beam):
    y: NumericArray
    x: NumericArray
    A: float
    a: float
    b: float
    na: float
    wavelength: float

    def get_n(self):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            return torch.sqrt(
                self.na**2 + (self.wavelength**2 / (2*(torch.pi**2)*(self.a**4))) *
                (self.a**2 - 2*(self.x**2 + self.y**2)) - (self.wavelength * self.b / torch.pi))
        else:
            return np.sqrt(
                self.na**2 + (self.wavelength**2 / (2*(np.pi**2)*(self.a**4))) *
                (self.a**2 - 2*(self.x**2 + self.y**2)) - (self.wavelength * self.b / np.pi))

    def get_E(self, z):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            device, dtype = self.x.device, self.x.dtype
            phase = torch.tensor(1j * self.b * z, dtype=float_to_complex_dtype(dtype), device=device)
            return torch.exp(-(self.x**2 + self.y**2) / self.a**2) * torch.exp(phase)
        else:
            return np.exp(-(self.x**2 + self.y**2) / self.a**2) * np.exp(1j * self.b * z)
        
        
def float_to_complex_dtype(dtype: torch.dtype) -> torch.dtype:
    """
    Convert a PyTorch real dtype to the corresponding complex dtype.
    
    float32 -> complex64
    float64 -> complex128
    """
    if dtype == torch.float32:
        return torch.complex64
    elif dtype == torch.float64:
        return torch.complex128
    else:
        raise TypeError(f"Unsupported dtype: {dtype}")