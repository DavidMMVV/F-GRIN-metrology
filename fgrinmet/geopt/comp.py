from typing import overload
import numpy as np
from numba import njit # type: ignore

def define_steps(
        dim: int, 
        steps: int = 1
                 ) -> tuple[np.ndarray, np.ndarray]:
    """
    Define the steps to follow and the coordinate in which each step is performed.

    Args:
        dim (int): Number of dimensions of the problem array.
        steps (int, optional): Number of steps. Defaults to 1.

    Returns:
        tuple[np.ndarray | None, np.ndarray | None]: _description_
    """
    if steps == 1:
        core = np.eye(dim, dtype=int).repeat(2, axis=0)
        core[::2] *= -1
        axes_num = np.arange(dim).repeat(2, axis=0)
        return core, axes_num
    else:
        return np.array([[np.nan,np.nan]]), np.array([[False,False]])



