import jax
import jax.numpy as jnp
import numpy as np

# enable float64 in JAX
jax.config.update("jax_enable_x64", True)

b_0 = np.array([1.0, 2.0, 3.0], dtype=jnp.complex128)
a = jnp.arange(9, dtype=jnp.float64).reshape((3, 3))
b = np.array([4.0+1j, 5.0+3j, 6.0+3j], dtype=jnp.complex128)

def f(b0, b, pos):
    a = jnp.arange(9, dtype=jnp.complex128).reshape((3, 3))
    a = jnp.exp(1j * (jnp.pi/2) * a.at[pos].set(b0))
    return (b @ a).sum()

def f_hand(b0, b, pos):
    a = jnp.arange(len(b_0)*len(b), dtype=jnp.complex128).reshape((len(b_0), len(b)))
    for i in range(len(b_0)):
        a = a.at[pos, i].set(jnp.exp(1j * (jnp.pi/2) * b0[i]))
    
    return (b @ a).sum()
def f_cond(b0, b, mask):
    a = jnp.arange(len(b_0)*len(b), dtype=jnp.complex128).reshape((len(b_0), len(b)))
    a = jnp.where(mask, jnp.exp(1j * (jnp.pi/2) * b0).reshape(-1,1), a)
    return (b @ a).sum()

print(jax.make_jaxpr(f)(b_0, b, 1))
print(jax.value_and_grad(f, argnums=0, holomorphic=True)(b_0, b,1))
b_0 = np.array([1.0, 2.0, 3.0], dtype=jnp.complex128)
b = np.array([4.0+1j, 5.0+3j, 6.0+3j], dtype=jnp.complex128)
print(jax.value_and_grad(f_hand, argnums=0, holomorphic=True)(b_0, b, 1))
b_0 = np.array([1.0, 2.0, 3.0], dtype=jnp.complex128)
print(jax.value_and_grad(f_cond, argnums=0, holomorphic=True)(b_0, b, 1))

print(a[20])

import jax
import jax.numpy as jnp

def f_scan(carry, z):
    u0, a, b = carry
    uf = a * z + b * u0
    new_carry = (uf, a, b)
    return new_carry, None  # no guardamos ningún valor intermedio

u0 = 0
a = 1
b = 2
carry_init = (u0, a, b)

final_carry, _ = jax.lax.scan(f_scan, carry_init, jnp.arange(10))

print("final_carry:", final_carry)


bo = jnp.array([[[True, False, True],
           [False, True, False],
           [True, False, True]], [[True, False, True],
           [False, True, False],
           [True, False, True]]])

print(bo.prod(axis=1).astype(bool))

try: jax_device = jax.devices("tpu") 
except: 
    try: jax_device = jax.devices("gpu")
    except: jax_device = jax.devices("cpu")

print(jax_device)

y = jnp.array([0,1,-1])
x = jnp.array([0,1,1])
z = jnp.cross(y,x)
print(z / np.sqrt((z**2).sum()))

from fgrinmet.utils import coord_jax, FT2, iFT2
import matplotlib.pyplot as plt
Y, X = coord_jax((100,100), 1)
res = 1 * (jnp.sqrt(Y**2 + X**2) <= 25)
res_ft = iFT2(FT2(res))
plt.figure()
plt.imshow(jnp.abs(res_ft))
plt.show()