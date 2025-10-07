import jax.numpy as jnp
import jax

from tqdm import tqdm 
import time

from fgrinmet.splitm import trilinear_interpolate, rotation_matrix, paraxial_propagator_jax, paraxial_propagation_step_jax
from fgrinmet.utils import coord_jax, fft_coord_jax

if __name__ == "__main__":
    # Define object parameters
    shape_obj = (512, 512, 512)
    wavelength = 1.0
    pix_size_object = 0.5 * wavelength
    center = (0.0, 0.0, 0.0)
    radius = 50.0 * wavelength
    Lzo, Lyo, Lxo = (shape_obj[0] * pix_size_object, shape_obj[1] * pix_size_object, shape_obj[2] * pix_size_object)
    w = 50.0  * wavelength
    n_a = 1.5
    
    
    # Create the object
    Zo, Yo, Xo = jnp.meshgrid(
        (jnp.arange(shape_obj[0]) - shape_obj[0] // 2) * pix_size_object + center[0],
        (jnp.arange(shape_obj[1]) - shape_obj[1] // 2) * pix_size_object + center[1],
        (jnp.arange(shape_obj[2]) - shape_obj[2] // 2) * pix_size_object + center[2],
        indexing="ij")
    
    with jax.default_device(jax.devices("cpu")[0]):
        R = jnp.sqrt(Xo**2 + Yo**2 + Zo**2)
    n = n_a + 0.5 * (jnp.exp(-R**2/(w**2)) - jnp.exp(-radius**2/(w**2))) * (R <= radius)

    # Define grid parameters
    shape_grid = (1024*32, 1024, 1024)
    pix_size_plane = 0.125  * wavelength
    hz = (0.125 /32)  * wavelength
    vec_plane = (0, 0, jnp.pi / 4)  # normal vector of the plane
    rot_m = rotation_matrix(*vec_plane)

    # Initialize plane of the grid
    Yi, Xi = jnp.meshgrid(
        (jnp.arange(shape_grid[1]) - shape_grid[1] // 2) * pix_size_plane,
        (jnp.arange(shape_grid[2]) - shape_grid[2] // 2) * pix_size_plane,
        indexing="ij")
    Yi, Xi = coord_jax(shape_grid[1:], pix_size_plane)
    Zi = jnp.zeros_like(Xi) - (shape_grid[0] // 2) * hz

    Fx, Fy = fft_coord_jax(shape_grid[1:], pix_size_plane)
   

    # Rotate the grid plane, transform to object grid coordinates and find normal vector
    coords_plane = jnp.stack([Zi.flatten(), Yi.flatten(), Xi.flatten()], axis=-1) @ rot_m.T
    Zpg0, Ypg0, Xpg0 = (((coords_plane[:, 0] - (center[0] - Lzo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 1] - (center[1] - Lyo / 2)) / pix_size_object).reshape(shape_grid[1:]), 
                     ((coords_plane[:, 2] - (center[2] - Lxo / 2)) / pix_size_object).reshape(shape_grid[1:]))
    
    from fgrinmet.splitm import prop_coord_to_obj_coord
    Zpg0, Ypg0, Xpg0 = prop_coord_to_obj_coord(coords_plane, center, center, shape_obj, pix_size_object).T.reshape(3,*shape_grid[1:])

    n_vec = jnp.array([hz / pix_size_object, 0, 0]) @ rot_m.T  

    trilinear_interpolation_jit = jax.jit(trilinear_interpolate)
    jac_fn = jax.jit(jax.jacobian(lambda n_values, cord: trilinear_interpolation_jit(cord, n_values, outside=n_a), argnums=0))
    prop_step = jax.jit(paraxial_propagation_step_jax)
    
    propagator = paraxial_propagator_jax(Fy, Fx, hz, n_a, wavelength)
    U_i = jnp.ones(shape_grid[1:])


    for i in tqdm(range(shape_grid[0])):

        Zpg, Ypg, Xpg = Zpg0 + i * n_vec[0], Ypg0 + i * n_vec[1], Xpg0 + i * n_vec[2]

        cord_in = jnp.concatenate([Zpg[None], Ypg[None], Xpg[None]], axis=0)

        n_plane = trilinear_interpolation_jit(cord_in, n, outside=n_a)
        U_o = prop_step(U_i, n_plane, propagator, hz, n_a, wavelength)
        U_i = U_o
        #target_plane = jnp.ones_like(n_plane) * n_a  # Example target plane for loss calculation
        #loss = jnp.sum((n_plane - target_plane)**2)
        #grad_n = jax.grad(lambda n_values: jnp.sum((trilinear_interpolation_jit(cord_in, n_values, outside=n_a) - target_plane)**2))(n)

        if i == (shape_grid[0] // 8):
            import matplotlib.pyplot as plt
            fig, sub = plt.subplots(1,2)
            im1 = sub[0].imshow(n_plane, cmap='jet')
            plt.colorbar(im1, ax=sub[0], fraction=0.046, pad=0.04)
            im2 = sub[1].imshow(jnp.abs(U_o), cmap="jet")
            plt.colorbar(im2, ax=sub[1], fraction=0.046, pad=0.04)
            sub[0].set_title(f"Refractive index at the plane in $z_g$={(i*hz)}$\\lambda$")
            sub[1].set_title(f"Propagated modulus of the field at the plane in $z_g$={(i*hz)}$\\lambda$")

    #plt.show()
    plt.figure()
    plt.title("Output field")
    plt.imshow(jnp.abs(U_o), cmap="jet")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.show()
    print("hey")
