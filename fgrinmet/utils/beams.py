import torch
import numpy as np

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

NumericArray = float | np.ndarray | torch.Tensor

class Beam(ABC):
    """
    Abstract base class for all beams.
    """

    def __init__(self, x: NumericArray, y: NumericArray):
        self.x = x
        self.y = y

    @abstractmethod
    def get_n(self) -> Any:
        """Return the refractive index distribution."""
        pass

    @abstractmethod
    def get_E(self, z: float) -> Any :
        """Return the electric field at position z."""
        pass

@dataclass
class GaussianBeam(Beam):
    y: NumericArray
    x: NumericArray
    a: float
    b: float
    na: float
    wavelength: float

    def get_n(self):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            return torch.sqrt(
                self.na**2 + (self.wavelength / (torch.pi**2))**2 *
                ((self.a**2 - (self.x**2 + self.y**2)) / self.a**4 - (torch.pi * self.b / self.wavelength))
            )
        else:
            return np.sqrt(
                self.na**2 + (self.wavelength / (np.pi**2))**2 *
                ((self.a**2 - (self.x**2 + self.y**2)) / self.a**4 - (np.pi * self.b / self.wavelength))
            )

    def get_E(self, z):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            device, dtype = self.x.device, self.x.dtype
            phase = torch.tensor(1j * self.b * z, dtype=float_to_complex_dtype(dtype), device=device)
            return torch.exp((self.x**2 + self.y**2) / self.a**2) * torch.exp(phase)
        else:
            return np.exp((self.x**2 + self.y**2) / self.a**2) * np.exp(1j * self.b * z)
        
@dataclass
class ExpBeam(Beam):
    y: NumericArray
    x: NumericArray
    a: float
    b: float
    na: float
    wavelength: float

    def get_n(self):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            return torch.sqrt(
                self.na**2 + (self.wavelength / (torch.pi**2))**2 *
                ((self.a**2 - (self.x**2 + self.y**2)) / self.a**4 - (torch.pi * self.b / self.wavelength))
            )
        else:
            return np.sqrt(
                self.na**2 + (self.wavelength / (np.pi**2))**2 *
                ((self.a**2 - (self.x**2 + self.y**2)) / self.a**4 - (np.pi * self.b / self.wavelength))
            )

    def get_E(self, z):
        if isinstance(self.x, torch.Tensor) and isinstance(self.y, torch.Tensor):
            device, dtype = self.x.device, self.x.dtype
            phase = torch.tensor(1j * self.b * z, dtype=float_to_complex_dtype(dtype), device=device)
            return torch.exp((self.x**2 + self.y**2) / self.a**2) * torch.exp(phase)
        else:
            return np.exp((self.x**2 + self.y**2) / self.a**2) * np.exp(1j * self.b * z)
        
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
    
