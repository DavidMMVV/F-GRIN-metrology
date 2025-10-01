import numpy as np
import torch
import jax
import jax.numpy as jnp

from typing import Optional, Tuple


def eikonal_solver():
    pass

def solver_2nd_order_eikonal():
    pass

def solver_1st_order_transport():
    pass

def solver_2nd_order_transport():
    pass

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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    a0 = torch.tensor([0,1,2,3,4], dtype=torch.float64, requires_grad=True)
    b  = torch.tensor([0,1,2,3,4], dtype=torch.float64, requires_grad=True)
    c  = torch.tensor([0,1,2,3,4], dtype=torch.float64, requires_grad=True)
    a = a0
    for i in range(1, a.shape[0]-1):
        a_new = a.clone()
        a_new[i] = a[i-1]*b[i+1] + a[i+1]*b[i-1]
        a = a_new

    u = a.sum()**2
    u.backward()

    print(u)
    print(a.grad)