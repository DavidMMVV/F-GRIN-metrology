import torch

from typing import List, Tuple, Union

from fgrinmet.globvar import DEVICE

def coord_pytorch(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
        dtype: torch.dtype = torch.float64,
        device: torch.device = DEVICE,
) -> tuple[torch.Tensor, ...]:
    """Generates the grid with coordinates for real space as a pytorch tensor.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float64.
        device (torch.device, optional): Device in which the tensors are stored. Defaults to DEVICE.

    Returns:
        Xi (tuple[torch.Tensor, ...]): Tuple of each coordinate of the grid
    """

    xi = [(torch.arange(int(shape[i]), dtype=dtype, device=device) - shape[i] / 2) *
                                 (pix_size[i] if isinstance(pix_size, list | tuple) else pix_size)
                                 for i in range(len(shape))]
    Xi = torch.meshgrid(*xi, indexing='ij')
    return Xi


def fft_coord_pytorch(
        shape: List[int] | Tuple[int,...],
        pix_size: float | list[float] | Tuple[float,...],
        dtype: torch.dtype = torch.float64,
        device: torch.device = DEVICE,
) -> tuple[torch.Tensor, ...]:
    """Generates the grid with coordinates for fourier space as a pytorch tensor.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float64.
        device (torch.device, optional): Device in which the tensors are stored. Defaults to DEVICE.

    Returns:
        Fi (tuple[torch.Tensor, ...]): Tuple of each coordinate of the grid.
    """
    fi = [torch.fft.fftshift(torch.fft.fftfreq(shape[i], 
                                               d=(pix_size[i] if isinstance(pix_size, list | tuple) else pix_size), 
                                               dtype=dtype).to(device)) 
                                               for i in range(len(shape))]
    Fi = torch.meshgrid(*fi, indexing='ij')
    return Fi

def fft_coord_pytorch_jit(
        shape: List[int],
        pix_size: Union[float, List[float]],
        dtype: torch.dtype = torch.float64,
        device: torch.device = torch.device('cpu')
    ) -> tuple[torch.Tensor, torch.Tensor]:

    """Generates the grid with coordinates for fourier space as a pytorch tensor optimized for compilation.

    Args:
        shape (List[int] | Tuple[int,...]): Shape of the grid.
        pix_size (float | list[float] | Tuple[float,...]): Pixel sizes over each dimension. If 
        it is a float, takes square pixels
        dtype (torch.dtype, optional): Data type of the output tensors. Defaults to torch.float64.
        device (torch.device, optional): Device in which the tensors are stored. Defaults to DEVICE.

    Returns:
        Fi (tuple[torch.Tensor, ...]): Tuple of each coordinate of the grid.
    """
    fi: List[torch.Tensor] = []
    for i in range(2):
        if isinstance(pix_size, list):
            d = float(pix_size[i])
        else:
            d = pix_size
        freq = torch.fft.fftfreq(shape[i], d=d, dtype=dtype).to(device)
        freq = torch.fft.fftshift(freq)
        fi.append(freq)
    Fy, Fx = torch.meshgrid(fi[0], fi[1], indexing="ij")
    return Fy, Fx