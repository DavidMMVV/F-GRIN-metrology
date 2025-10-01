import jax.numpy as jnp

def rotation_matrix(theta: float, phi: float, psi: float) -> jnp.ndarray:
    """Generates the rotation matrix for given Euler angles (in radians).
    Args:
        theta (float): Rotation angle around z axis.
        phi (float): Rotation angle around y axis.
        psi (float): Rotation angle around x axis.
    Returns:
        R (jnp.ndarray): 3x3 rotation matrix.
    """
    R_z = jnp.array([[1, 0, 0],
                     [0, jnp.cos(theta), -jnp.sin(theta)],
                     [0, jnp.sin(theta), jnp.cos(theta)]])
    
    R_y = jnp.array([[jnp.cos(phi), 0, jnp.sin(phi)],
                     [0, 1, 0],
                     [-jnp.sin(phi), 0, jnp.cos(phi)]])
    
    R_x = jnp.array([[jnp.cos(psi), -jnp.sin(psi), 0],
                     [jnp.sin(psi), jnp.cos(psi), 0],
                     [0, 0, 1]])
    
    R = jnp.dot(R_z, jnp.dot(R_y, R_x))
    return R