
import time
import jax
import jax.numpy as jnp
from fgrinmet.splitm import rotation_matrix
from tqdm import tqdm

from typing import Optional
from functools import partial

def trilinear_interpolation(
        points: jnp.ndarray,
        values: jnp.ndarray,
        outside: float = 1.0,
        mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    
    """Perform trilinear interpolation on a 3D grid.

    Args:
        points (jnp.ndarray): The 3D coordinates to interpolate at. It should be of shape (3, (N, ...)) where N is the number of points in this specific dimension.
        values (jnp.ndarray): The values to interpolate. If mask is None it should be of shape (D, H, W). If mask is given it should be a 1-D array of shape (N).
        outside (float, optional): The value to return for points outside the grid and masked points. Defaults to 1.0.
        mask (Optional[jnp.ndarray], optional): A mask to specify valid points with shape (D, H, W) which fulfills that N = mask.sum(). Defaults to None.

    Returns:
        jnp.ndarray: The interpolated values at the specified points.
    """
    
    if mask is not None:
        values_grid = jnp.ones_like(mask, dtype=values.dtype).at[mask].set(values)
        grid_shape = jnp.array(values.shape)
    else:
        mask = jnp.ones(values.shape, dtype=bool)
        values_grid = values
        grid_shape = jnp.array(values.shape)
    dim_exp = tuple((points.ndim-1)*[None])

    # generate the offset for the 8 corners
    offsets = jnp.array([[0,0,0],
                         [0,0,1],
                         [0,1,0],
                         [0,1,1],
                         [1,0,0],
                         [1,0,1],
                         [1,1,0],
                         [1,1,1]], dtype=jnp.int32)
    
    floor_points = jnp.floor(points).astype(jnp.int32)
    dec_points = (offsets[::-1, :, *dim_exp] + (-1)**(offsets[::-1, :, *dim_exp]) * (points - floor_points)[None]).prod(axis=1) # Obtain the factor dependent on distance of the triliniar expresion

    corners = floor_points[None] + offsets[:, :, *dim_exp]
    condition = (((corners >= 0).prod(axis=1).astype(bool)) &
                 ((corners < grid_shape[None, :, *dim_exp]).prod(axis=1).astype(bool)) & 
                 (mask[corners[:,0], corners[:,1], corners[:,2]]))
    corners_val = jnp.where(condition, values_grid[corners[:,0], corners[:,1], corners[:,2]], outside)

    c = (corners_val * dec_points).sum(axis=0)

    # apply outside only once
    return c

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
    shape_obj = (64, 64, 64)
    pix_size_object = 1
    center = (0.0, 0.0, 0.0)
    radius = 30.0
    Lzo, Lyo, Lxo = (shape_obj[0] * pix_size_object, shape_obj[1] * pix_size_object, shape_obj[2] * pix_size_object)
    w = 30.0
    n_a = 1.5
    
    # Create the object
    Zo, Yo, Xo = jnp.meshgrid(
        (jnp.arange(shape_obj[0]) - shape_obj[0] / 2) * pix_size_object + center[0],
        (jnp.arange(shape_obj[1]) - shape_obj[1] / 2) * pix_size_object + center[1],
        (jnp.arange(shape_obj[2]) - shape_obj[2] / 2) * pix_size_object + center[2],
        indexing="ij")
    R = jnp.sqrt(Xo**2 + Yo**2 + Zo**2)
    n = n_a + 0.5 * (jnp.exp(-R**2/(w**2)) - jnp.exp(-radius**2/(w**2))) * (R <= radius)

    # Define grid parameters
    shape_grid = (1024, 1024, 1024)
    pix_size_plane = 0.125
    hz = 0.125
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

        if i == shape_grid[0] // 3:
            import matplotlib.pyplot as plt
            plt.imshow(n_plane, cmap='jet')
            plt.colorbar()
            plt.title("Refractive index at the center plane")

    plt.show()
