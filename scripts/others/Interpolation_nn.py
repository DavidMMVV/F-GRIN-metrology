import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import flax.linen as nn
from flax.training import train_state
from fgrinmet.splitm import rotation_matrix

# ---------------------------
# Red neuronal
# ---------------------------
class Net(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.swish(nn.Dense(128)(x))
        x = nn.swish(nn.Dense(128)(x))
        x = nn.swish(nn.Dense(128)(x))
        return nn.Dense(1)(x).squeeze(-1)

# ---------------------------
# Utilities entrenamiento
# ---------------------------
def create_train_state(rng, model, learning_rate):
    params = model.init(rng, jnp.ones((1, 3)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

@jax.jit
def train_step(state, x_batch, y_batch):
    def loss_fn(params):
        preds = state.apply_fn(params, x_batch)
        return jnp.mean((preds - y_batch) ** 2)
    grads = jax.grad(loss_fn)(state.params)
    return state.apply_gradients(grads=grads)

# Evaluación en chunks
def eval_in_chunks(apply_fn, params, x_all, chunk_size=4096):
    outs = []
    for i in range(0, x_all.shape[0], chunk_size):
        xchunk = x_all[i:i+chunk_size]
        outs.append(apply_fn(params, xchunk))
    return jnp.concatenate(outs, axis=0)

# Gradiente de la red respecto a la entrada
def grad_n_wrt_x(apply_fn, params, x):
    # grad respecto a la entrada x
    return jax.grad(lambda xi: apply_fn(params, xi))(x)

from jax.scipy.interpolate import RegularGridInterpolator

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # parámetros
    shape = (256, 256)
    pix_size = 0.25
    radius = 20.0
    w = 20.0
    n_a = 1.5

    # generar esfera (puntos de soporte)
    R = jnp.linspace(0, radius, 10)
    theta = jnp.linspace(0, jnp.pi, 20)
    phi = jnp.linspace(0, 2*jnp.pi, 20)
    R, theta, phi = jnp.meshgrid(R, theta, phi, indexing="ij")
    coords = jnp.stack([
        (R * jnp.cos(theta)).flatten(),
        (R * jnp.sin(theta) * jnp.sin(phi)).flatten(),
        (R * jnp.sin(theta) * jnp.cos(phi)).flatten()
    ], axis=-1)
    n = n_a + 0.5 * (jnp.exp(-R.flatten()**2/(2*w**2)) - jnp.exp(-radius**2/(2*w**2)))
    n = n.reshape(-1)

    # rejilla del plano rotada
    X, Y = jnp.meshgrid(
        (jnp.arange(shape[1]) - shape[1] // 2) * pix_size,
        (jnp.arange(shape[0]) - shape[0] // 2) * pix_size,
        indexing="ij"
    )
    rot_m = rotation_matrix(0, 0, jnp.pi / 4)
    coords_grid = jnp.stack([jnp.zeros(X.size), Y.flatten(), X.flatten()], axis=-1) @ rot_m.T

    # ---------------------------
    # Normalización
    # ---------------------------
    mean_coords = coords.mean(axis=0)
    std_coords = coords.std(axis=0) + 1e-9
    coords_norm = (coords - mean_coords) / std_coords
    coords_grid_norm = (coords_grid - mean_coords) / std_coords

    y = (n - n_a)
    scale_y = jnp.std(y) + 1e-9
    y_norm = y / scale_y

    # ---------------------------
    # Crear y entrenar modelo
    # ---------------------------
    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    model = Net()
    state = create_train_state(init_rng, model, learning_rate=5e-4)

    batch_size = 256
    num_epochs = 2000
    N = coords_norm.shape[0]

    print("Training samples:", N, "Batch size:", batch_size)

    for epoch in range(1, num_epochs+1):
        rng, perm_rng = jax.random.split(rng)
        perm = jax.random.permutation(perm_rng, N)
        for i in range(0, N, batch_size):
            batch_idx = perm[i:i+batch_size]
            xb = coords_norm[batch_idx]
            yb = y_norm[batch_idx]
            state = train_step(state, xb, yb)
        if (epoch % 100) == 0 or epoch == 1:
            train_loss = eval_in_chunks(state.apply_fn, state.params, coords_norm, chunk_size=512)
            print(f"Epoch {epoch:4d}  train_loss={jnp.mean((train_loss - y_norm)**2):.6e}")

    # ---------------------------
    # Evaluación en la rejilla
    # ---------------------------
    n_grid_norm = eval_in_chunks(state.apply_fn, state.params, coords_grid_norm, chunk_size=512)
    n_grid = n_grid_norm * scale_y + n_a
    radii_grid = jnp.linalg.norm(coords_grid, axis=-1)
    n_grid = jnp.where(radii_grid <= radius, n_grid, n_a)
    n_plane = n_grid.reshape(shape)

    # ---------------------------
    # Gradiente respecto a coordenadas físicas
    # ---------------------------
    grads_norm = jax.vmap(lambda xi: grad_n_wrt_x(state.apply_fn, state.params, xi))(coords_grid_norm)

    grads_phys = grads_norm * (scale_y / std_coords[None, :])
    grad_mag = jnp.linalg.norm(grads_phys, axis=-1).reshape(shape)

    # ---------------------------
    # Interpolación lineal (1er orden) con RegularGridInterpolator
    # ---------------------------

    # Para usar RegularGridInterpolator necesitamos que los datos estén en grilla regular.
    # Como generaste coords en (R,θ,φ) con meshgrid, podemos interpolar en esas coordenadas.
    R_axis = jnp.linspace(0, radius, 10)
    theta_axis = jnp.linspace(0, jnp.pi, 20)
    phi_axis = jnp.linspace(0, 2*jnp.pi, 20)

    # Los valores de n en esa grilla regular
    n_values = n.reshape(len(R_axis), len(theta_axis), len(phi_axis))

    # Creamos el interpolador en espacio (R,θ,φ)
    interp_func = RegularGridInterpolator(
        (R_axis, theta_axis, phi_axis),
        n_values,
        method="linear",
        bounds_error=False,
        fill_value=n_a
    )

    # Ahora necesitamos pasar las coordenadas (X,Y,Z) de la rejilla al sistema (R,θ,φ)
    def cartesian_to_spherical(xyz):
        x, y, z = xyz[...,0], xyz[...,1], xyz[...,2]
        R = jnp.sqrt(x**2 + y**2 + z**2)
        theta = jnp.arccos(jnp.clip(x / (R + 1e-9), -1.0, 1.0))  # polar desde X
        phi = jnp.arctan2(y, z) % (2*jnp.pi)
        return jnp.stack([R, theta, phi], axis=-1)

    coords_grid_sph = cartesian_to_spherical(coords_grid)

    # Evaluamos interpolador
    n_interp_flat = interp_func(coords_grid_sph)
    n_interp = n_interp_flat.reshape(shape)
    # ---------------------------
    # Plots
    # ---------------------------
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.imshow(n_plane, origin='lower',
           extent=[-shape[1]*pix_size/2, shape[1]*pix_size/2,
                   -shape[0]*pix_size/2, shape[0]*pix_size/2])
    plt.colorbar(label='n (NN)')
    plt.title("NN projection")
    plt.subplot(1,2,2)
    plt.imshow(n_interp, origin='lower',
           extent=[-shape[1]*pix_size/2, shape[1]*pix_size/2,
                   -shape[0]*pix_size/2, shape[0]*pix_size/2])
    plt.colorbar(label='n (interp)')
    plt.title("Linear Interpolation Projection")
    plt.show()

    # Gradiente magnitude plot
    plt.figure(figsize=(6,5))
    plt.imshow(grad_mag, origin='lower',
           extent=[-shape[1]*pix_size/2, shape[1]*pix_size/2,
                   -shape[0]*pix_size/2, shape[0]*pix_size/2])
    plt.colorbar(label='|∇n|')
    plt.title("Magnitude of the gradient ∇n (NN)")
    plt.show()
