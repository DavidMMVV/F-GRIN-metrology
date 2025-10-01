
import jax
import jax.numpy as jnp
from fgrinmet.splitm import rotation_matrix
from tqdm import tqdm

from typing import Optional


def trilinear_interpolation(points: jnp.ndarray,
                            values: jnp.ndarray,
                            outside: float = 1.0,
                            mask: Optional[jnp.ndarray] = None) -> jnp.ndarray:
    D, H, W = values.shape

    # integer base indices
    Z0 = jnp.floor(points[0]).astype(jnp.int32)
    Y0 = jnp.floor(points[1]).astype(jnp.int32)
    X0 = jnp.floor(points[2]).astype(jnp.int32)

    Z1, Y1, X1 = Z0 + 1, Y0 + 1, X0 + 1

    # fractional part
    z = points[0] - Z0
    y = points[1] - Y0
    x = points[2] - X0

    # inside mask (boolean)
    inside = (
        (Z0 >= 0) & (Z1 < D) &
        (Y0 >= 0) & (Y1 < H) &
        (X0 >= 0) & (X1 < W)
    )

    # clip once (avoids out-of-bounds)
    Z0c, Y0c, X0c = jnp.clip(Z0, 0, D-1), jnp.clip(Y0, 0, H-1), jnp.clip(X0, 0, W-1)
    Z1c, Y1c, X1c = jnp.clip(Z1, 0, D-1), jnp.clip(Y1, 0, H-1), jnp.clip(X1, 0, W-1)

    # gather 8 corners
    c000 = values[Z0c, Y0c, X0c]
    c001 = values[Z0c, Y0c, X1c]
    c010 = values[Z0c, Y1c, X0c]
    c011 = values[Z0c, Y1c, X1c]
    c100 = values[Z1c, Y0c, X0c]
    c101 = values[Z1c, Y0c, X1c]
    c110 = values[Z1c, Y1c, X0c]
    c111 = values[Z1c, Y1c, X1c]

    # interpolation
    c = (
        c000 * (1 - z) * (1 - y) * (1 - x) +
        c001 * (1 - z) * (1 - y) * x +
        c010 * (1 - z) * y * (1 - x) +
        c011 * (1 - z) * y * x +
        c100 * z * (1 - y) * (1 - x) +
        c101 * z * (1 - y) * x +
        c110 * z * y * (1 - x) +
        c111 * z * y * x
    )

    # apply mask if given
    if mask is not None:
        valid = mask[Z0c, Y0c, X0c]
        inside = inside & valid

    # apply outside only once
    return jnp.where(inside, c, outside)


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
    shape_grid = (1024, 512, 512)
    pix_size_plane = 0.25
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
    print("here")

    trilinear_interpolation_jit = jax.jit(trilinear_interpolation)
    jac_fn = jax.jit(jax.jacobian(lambda n_values, cord: trilinear_interpolation_jit(cord, n_values, outside=n_a), argnums=0))

    for i in tqdm(range(shape_grid[0])):

        Zpg, Ypg, Xpg = Zpg0 + i * n_vec[0], Ypg0 + i * n_vec[1], Xpg0 + i * n_vec[2]

        cord_in = jnp.concatenate([Zpg[None], Ypg[None], Xpg[None]], axis=0)

        n_plane = trilinear_interpolation(cord_in, n, outside=n_a)
        #target_plane = jnp.ones_like(n_plane) * n_a  # Example target plane for loss calculation
        #loss = jnp.sum((n_plane - target_plane)**2)
        #grad_n = jax.grad(lambda n_values: jnp.sum((trilinear_interpolation_jit(cord_in, n_values, outside=n_a) - target_plane)**2))(n)

        #if i == shape_grid[0] // 3:
        #    import matplotlib.pyplot as plt
        #    plt.imshow(n_plane, cmap='jet')
        #    plt.colorbar()
        #    plt.title("Refractive index at the center plane")
        #    plt.show()
