import jax.numpy as jnp


a = jnp.arange(56).reshape(8,7)
print(a)
print(a.at[jnp.array([0,1])].set(a[jnp.array([1,0])]))