import numpy as np
from numba import njit #type:ignore
import torch

from typing import Optional, Tuple

from .comp import define_steps 

@njit
def solver_1st_order_eikonal():
    pass

@njit
def solver_2nd_order_eikonal():
    pass

@njit
def solver_1st_order_transport():
    pass

@njit
def solver_2nd_order_transport():
    pass

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