import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
from tqdm import tqdm

from fgrinmet.utils import FT2, iFT2, FT2_i, iFT2_i, fft_coord_jax
from fgrinmet.splitm import paraxial_propagator_jax

from splitm_gradient import C_d, C_d_conj
wavelength = 1.0
n_a = 1.5
shape = (30,256,256)
pix_size = (0.5 * wavelength, 0.25 * wavelength, 0.25 * wavelength)
z = jnp.arange(shape[0]) / shape[0]
y = (jnp.arange(shape[1]) - shape[1] // 2) / (shape[1] // 2)
x = (jnp.arange(shape[2]) - shape[2] // 2) / (shape[2] // 2)

r2 = y[:,None]**2 + x[None]**2
inc_n = 1.5
inc_r = 0.5
R0 = 0.5
n_original = n_a + jnp.where(r2[None].repeat(shape[0], axis=0) <= R0,  inc_n * (R0-r2[None].repeat(shape[0], axis=0)), 0)#/ (1+inc_r*z[:,None, None])**2)
mask = jnp.ones_like(n_original).astype(bool)

plt.figure()
plt.imshow(n_original[shape[0] // 2])
plt.colorbar()

Fy, Fx = fft_coord_jax(shape[1:], pix_size[1:])

propagator = paraxial_propagator_jax(Fy, Fx, pix_size[0], n_a, wavelength)
propagator_conj = jnp.conjugate(propagator)
U_i = jnp.ones(shape[1:])
U_meas = jnp.copy(U_i)

for d in tqdm(range(shape[0])):
    U_meas = iFT2(propagator * FT2(jnp.exp((1j * jnp.pi / (wavelength * n_a)) * pix_size[0] * (n_a**2 - n_original[d]**2)) * iFT2(propagator * FT2(U_meas))))

plt.figure()
plt.imshow(jnp.abs(U_meas))
plt.colorbar()

plt.show()