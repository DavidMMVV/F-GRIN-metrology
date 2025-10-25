import jax
import jax.numpy as jnp
import numpy as np
import time

# enable float64 in JAX
jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
slit = (jnp.abs(jnp.arange(-100, 101,1))>=10)*(jnp.abs(jnp.arange(-100, 101,1))>=20)

plt.figure()
plt.plot(jnp.fft.fftshift(jnp.fft.fftfreq(201)), jnp.abs(jnp.fft.fftshift(jnp.fft.fft(slit))))
plt.figure()
plt.plot(jnp.arange(-100, 101,1), slit)
plt.show()

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
res = z / np.sqrt((z**2).sum())
print(res)

def f_field(U, mask):
    ind = jnp.ones_like(mask).at[mask].set(U)
    exit = ind * 1.5
    return exit

print(__file__)
#out = jax.checkpoint(f_field, )

import optax
import jax
import jax.numpy as jnp
def f(x): return jnp.sum(x ** 2)  # simple quadratic function
solver = optax.adam(learning_rate=0.003)
params = jnp.array([1., 2., 3.])
print('Objective function: ', f(params))
opt_state = solver.init(params)
print(f"Opt_state: {opt_state}")
for _ in range(5):
    grad = jax.grad(f)(params)
    updates, opt_state = solver.update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    print('Objective function: {:.2E}'.format(f(params)))
    print(f"Opt_state: {opt_state}")

print(jax.local_device_count())

fun = jax.vmap(lambda col1, mat2 : (jax.vmap(lambda col1, col2: (col1*col2).sum(), in_axes=(None,1), out_axes=(0))(col1, mat2)), in_axes=(1, None), out_axes=(0))
vec_fun = lambda mat1, mat2 : mat1.T@mat2
a = jnp.arange(10000*10000).reshape(10000,10000)
b = jnp.arange(10000*9000).reshape(10000,9000)
c = jnp.arange(10000*10000).reshape(10000,10000)
d = jnp.arange(10000*9000).reshape(10000,9000)
#
#start = time.perf_counter()
#res1 = fun(a, b)
#end = time.perf_counter()
#print(f"Time vmap: {end-start}s")

#start = time.perf_counter()
#res1 = fun(a, b)
#end = time.perf_counter()
#print(f"Time vmap: {end-start}s")
#
#start = time.perf_counter()
#res1 = fun(a, b)
#end = time.perf_counter()
#print(f"Time vmap: {end-start}s")
#
comp_fun = jax.jit(fun)
comp_vec_fun = jax.jit(vec_fun)

start = time.perf_counter()
res1 = comp_fun(a, b)
end = time.perf_counter()
print(f"Time vmap+jit: {end-start}s")

start = time.perf_counter()
res1 = comp_fun(a, b)
end = time.perf_counter()
print(f"Time vmap+jit: {end-start}s")

start = time.perf_counter()
res1 = comp_fun(d, c)
end = time.perf_counter()
print(f"Time vmap+jit: {end-start}s")

start = time.perf_counter()
res1 = comp_fun(d, c)
end = time.perf_counter()
print(f"Time vmap+jit: {end-start}s")

start = time.perf_counter()
res2 = comp_vec_fun(a,b)
end = time.perf_counter()
print(f"Time indexing jit: {end-start}s")

start = time.perf_counter()
res2 = comp_vec_fun(a,b)
end = time.perf_counter()
print(f"Time indexing jit: {end-start}s")

start = time.perf_counter()
res2 = comp_vec_fun(d,c)
end = time.perf_counter()
print(f"Time indexing jit: {end-start}s")


start = time.perf_counter()
res2 = comp_vec_fun(d,c)
end = time.perf_counter()
print(f"Time indexing jit: {end-start}s")
#print(res1)
#print(res2)

