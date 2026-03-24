import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Define the medium
# ------------------------------

def n_squared(r):
    x, y, z = r
    return 2.5 - 0.1*(x**2 + y**2)

def grad_n_squared(r):
    x, y, z = r
    dnx = -0.2*x
    dny = -0.2*y
    dnz = 0.0
    return np.array([dnx, dny, dnz]) / 2.0


# ------------------------------
# Runge-Kutta 4 for ray tracing
# ------------------------------

def trace_ray(R0, T0, dt, steps):
    
    R = np.array(R0, dtype=float)
    T = np.array(T0, dtype=float)
    
    trajectory = [R.copy()]
    
    for _ in range(steps):
        
        D = lambda r: grad_n_squared(r)
        
        A = dt * D(R)
        B = dt * D(R + 0.5*dt*T + 0.125*A)
        C = dt * D(R + dt*T + 0.5*B)
        
        R = R + dt*T + (A + 2*B)/6.0
        T = T + (A + 4*B + C)/6.0
        
        trajectory.append(R.copy())
    
    return np.array(trajectory)

import jax.numpy as jnp
modes = 1
angles_ext = [-45, 45, -45, 45]
y_angles, x_angles = jnp.meshgrid(jnp.linspace(angles_ext[0], angles_ext[1], modes), jnp.linspace(angles_ext[2], angles_ext[3], modes))
angles = jnp.array((y_angles.flatten(), x_angles.flatten())).T
print(angles)

R0 = [0.2, 0.0, 0.0]
T0 = [0.0, 0.0, 1.0]   # initial direction
dt = 0.05
steps = 200

traj = trace_ray(R0, T0, dt, steps)

plt.plot(traj[:,2], traj[:,0])
plt.xlabel("z")
plt.ylabel("x")
plt.title("Propagación del rayo en medio GRIN")
plt.grid()
plt.show()