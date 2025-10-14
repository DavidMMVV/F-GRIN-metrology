
import time
import jax
import jax.numpy as jnp
from fgrinmet.splitm import rotation_matrix
from tqdm import tqdm # type: ignore

from typing import Optional

def compute_planes_scan(Zpg0, Ypg0, Xpg0, n_vec, n, n_a, num_planes):
    def body_fun(carry, z):
        Zpg0, Ypg0, Xpg0, n_vec, n, n_a = carry
        Zpg = Zpg0 + z * n_vec[0]
        Ypg = Ypg0 + z * n_vec[1]
        Xpg = Xpg0 + z * n_vec[2]
        cord_in = jnp.stack([Zpg, Ypg, Xpg], axis=0)
        plane = trilinear_interpolation(cord_in, n, outside=n_a)
        carry_out = (Zpg0, Ypg0, Xpg0, n_vec, n, n_a)
        result_sum = plane.sum()
        return carry_out, result_sum
    _, planes = jax.lax.scan(body_fun, (Zpg0, Ypg0, Xpg0, n_vec, n, n_a), jnp.arange(num_planes))
    return planes

if __name__ == "__main__":
    # Define object parameters
    shape_obj = (512, 512, 512)
    pix_size_object = 0.5
    center = (0.0, 0.0, 0.0)
    radius = 50.0
    Lzo, Lyo, Lxo = (shape_obj[0] * pix_size_object, shape_obj[1] * pix_size_object, shape_obj[2] * pix_size_object)
    w = 50.0
    n_a = 1.5
    
    # Create the object
    Zo, Yo, Xo = jnp.meshgrid(
        (jnp.arange(shape_obj[0]) - shape_obj[0] / 2) * pix_size_object + center[0],
        (jnp.arange(shape_obj[1]) - shape_obj[1] / 2) * pix_size_object + center[1],
        (jnp.arange(shape_obj[2]) - shape_obj[2] / 2) * pix_size_object + center[2],
        indexing="ij")
    
    with jax.default_device(jax.devices("cpu")[0]):
        R = jnp.sqrt(Xo**2 + Yo**2 + Zo**2)
    n = n_a + 0.5 * (jnp.exp(-R**2/(w**2)) - jnp.exp(-radius**2/(w**2))) * (R <= radius)

    # Define grid parameters
    shape_grid = (1024*32, 1024, 1024)
    pix_size_plane = 0.125
    hz = 0.125/32
    vec_plane = (0, 0, jnp.pi / 4)  # normal vector of the plane
    rot_m = rotation_matrix(*vec_plane)

    # Initialize plane of the grid
    Yi, Xi = jnp.meshgrid(
        (jnp.arange(shape_grid[1]) - shape_grid[1] // 2) * pix_size_plane,
        (jnp.arange(shape_grid[2]) - shape_grid[2] // 2) * pix_size_plane,
        indexing="ij")
    Zi = jnp.zeros_like(Xi) - (shape_grid[0] / 2) * hz

    # Rotate the grid plane, transform to object grid coordinates and find normal vector
    coords_plane = jnp.stack([Zi.flatten(), Yi.flatten(), Xi.flatten()], axis=-1) @ rot_m.T
    Zpg0, Ypg0, Xpg0 = (((coords_plane[:, 0] - (center[0] - Lzo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 1] - (center[1] - Lyo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 2] - (center[2] - Lxo / 2)) / pix_size_object).reshape(shape_grid[1:]))

    n_vec = jnp.array([hz / pix_size_object, 0, 0]) @ rot_m.T  

    trilinear_interpolation_jit = jax.jit(trilinear_interpolation)
    jac_fn = jax.jit(jax.jacobian(lambda n_values, cord: trilinear_interpolation_jit(cord, n_values, outside=n_a), argnums=0))
    #jit_compute_planes = jax.jit(compute_planes)
    num_planes = int(shape_grid[0])
    #compute_planes_jit = jax.jit(compute_planes_scan)
    start = time.perf_counter()
    planes = compute_planes_scan(Zpg0, Ypg0, Xpg0, n_vec, n, n_a, num_planes)
    end = time.perf_counter()
    print(f"Compiling time: {end-start}s")

    start = time.perf_counter()
    planes = compute_planes_scan(Zpg0, Ypg0, Xpg0, n_vec, n, n_a, num_planes)
    end = time.perf_counter()
    print(f"Fast time: {end-start}s")

    for i in tqdm(range(shape_grid[0])):

        Zpg, Ypg, Xpg = Zpg0 + i * n_vec[0], Ypg0 + i * n_vec[1], Xpg0 + i * n_vec[2]

        cord_in = jnp.concatenate([Zpg[None], Ypg[None], Xpg[None]], axis=0)

        n_plane = trilinear_interpolation_jit(cord_in, n, outside=n_a)
        #target_plane = jnp.ones_like(n_plane) * n_a  # Example target plane for loss calculation
        #loss = jnp.sum((n_plane - target_plane)**2)
        #grad_n = jax.grad(lambda n_values: jnp.sum((trilinear_interpolation_jit(cord_in, n_values, outside=n_a) - target_plane)**2))(n)

        if i == (shape_grid[0] // 2):
            import matplotlib.pyplot as plt
            plt.imshow(n_plane, cmap='jet')
            plt.colorbar()
            plt.title(f"Refractive index at the center plane $z_g$={(i*hz)}$\\lambda$")

    plt.show() # type: ignore
    print("hey")
