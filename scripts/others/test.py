import time
import numpy as np

# ============================================================
# Configuración
# ============================================================
np.random.seed(0)
N = 512  # tamaño del cubo
K = 5   # tamaño del kernel
x = np.random.rand(N, N, N).astype(np.float32)
kernel = np.random.rand(K, K, K).astype(np.float32)

# ============================================================
# Numba
# ============================================================
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def conv3d_numba(x, k):
    nx, ny, nz = x.shape
    kk = k.shape[0]
    outx, outy, outz = nx-kk+1, ny-kk+1, nz-kk+1
    out = np.zeros((outx,outy,outz), dtype=np.float32)
    for i in prange(outx):
        for j in range(outy):
            for l in range(outz):
                acc = 0.0
                for ii in range(kk):
                    for jj in range(kk):
                        for ll in range(kk):
                            acc += x[i+ii,j+jj,l+ll] * k[ii,jj,ll]
                out[i,j,l] = acc
    return out

# ============================================================
# JAX
# ============================================================
import jax
import jax.numpy as jnp

def conv3d_jax(x, k):
    x = jnp.array(x)
    k = jnp.array(k)
    return jax.lax.conv_general_dilated(
        x[None, None, :, :, :],   # [batch, in_channels, D,H,W]
        k[None, None, :, :, :],   # [out_channels, in_channels, kd,kh,kw]
        (1,1,1),                  # stride
        "VALID"
    )

# ============================================================
# PyTorch
# ============================================================
import torch
import torch.nn.functional as F

def conv3d_torch(x, k, device="cpu"):
    xt = torch.tensor(x, device=device).unsqueeze(0).unsqueeze(0)
    kt = torch.tensor(k, device=device).unsqueeze(0).unsqueeze(0)
    return F.conv3d(xt, kt).squeeze()

# ============================================================
# Benchmark Helper
# ============================================================
def benchmark(func, *args, repeat=5, warmup=True, warmup_runs=1):
    if warmup:
        for _ in range(warmup_runs):
            func(*args)
    times = []
    for _ in range(repeat):
        t0 = time.time()
        func(*args)
        t1 = time.time()
        times.append(t1 - t0)
    return np.mean(times), np.std(times)

# ============================================================
# Ejecución
# ============================================================
print("Benchmark convolución 3D ({}^3 tensor, kernel {}^3):".format(N,K))

# Numba
mean, std = benchmark(conv3d_numba, x, kernel)
print(f"Numba:   {mean:.4f} ± {std:.4f} s")

# JAX
conv3d_jit = jax.jit(conv3d_jax)
mean, std = benchmark(conv3d_jit, x, kernel, warmup_runs=5)
print(f"JAX (CPU/GPU): {mean:.4f} ± {std:.4f} s")

# Torch
device = "cuda" if torch.cuda.is_available() else "cpu"
mean, std = benchmark(lambda a,b: conv3d_torch(a,b,device), x, kernel)
print(f"Torch ({device}): {mean:.4f} ± {std:.4f} s")
