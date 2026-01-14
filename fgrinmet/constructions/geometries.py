import jax.numpy as jnp
import numpy as np
from typing import Callable
import matplotlib.pyplot as plt

class FGRINcomponent:
    """
    Class for the geometries of the FGRIN media given a distribution function with (Z,Y,X) coordinates.
    """
    def __init__(self, name: str, distribution: Callable, n_points: tuple, pix_sizes: tuple, origin: tuple = (0,0,0), a_0: jnp.ndarray = jnp.array([0])) -> None:
        self.name = name
        self.distribution = distribution
        self.n_points = n_points
        self.pixel_sizes = pix_sizes
        self.origin = origin
        self.coordinates1D = [np.arange(N) * pix - ro for N, pix, ro in zip(n_points, pix_sizes, origin)]
        self.genextent = [(self.coordinates1D[i][0], self.coordinates1D[i][-1]) for i in range(3)]
        self.a_0 = a_0

    def generate(self):
        coordinates = np.meshgrid(*self.coordinates1D, indexing='ij')
        return self.distribution(*coordinates, self.a_0)
    
    def rescale(self, pix_sizes: tuple):
        self.pixel_sizes = pix_sizes
        self.coordinates1D = [np.arange(N) * pix - ro for N, pix, ro in zip(self.n_points, pix_sizes, self.origin)]

    def change_origin(self, origin: tuple):
        self.origin = origin
        self.coordinates1D = [np.arange(N) * pix - ro for N, pix, ro in zip(self.n_points, self.pixel_sizes, origin)]

    def change_distribution(self, distribution: Callable):
        self.distribution = distribution

    def show(self):
        cube = self.generate()
        mid_slices = tuple(N // 2 for N in self.n_points)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        Z, Y, X = np.meshgrid(
            self.coordinates1D[0],
            self.coordinates1D[1],
            self.coordinates1D[2],
            indexing='ij'
        )

        levels = np.linspace(cube.min(), cube.max(), 10)

        # XY slice (Z = Zmin)
        ax.contourf(
            X[-1, :, :],
            Y[-1, :, :],
            cube[-1, :, :],
            zdir='z',
            offset=Z.max(),
            levels=levels,
            cmap='viridis'
        )

        # XZ slice (Y = Ymin)
        ax.contourf(
            X[:, 0, :],
            cube[:, 0, :],
            Z[:, 0, :],
            zdir='y',
            offset=Y.min(),
            levels=levels,
            cmap='viridis'
        )

        # YZ slice (X = Xmax)
        C = ax.contourf(
            cube[:, :, -1],
            Y[:, :, -1],
            Z[:, :, -1],
            zdir='x',
            offset=X.max(),
            levels=levels,
            cmap='viridis'
        )

        ax.set(
            xlim=[X.min(), X.max()],
            ylim=[Y.min(), Y.max()],
            zlim=[Z.min(), Z.max()]
        )

        ax.set_xlabel('$X (\\lambda)$')
        ax.set_ylabel('$Y (\\lambda)$')
        ax.set_zlabel('$Z (\\lambda)$')
        
        # Set limits of the plot from coord limits
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])

        # Plot edges
        ax.plot([xmax, xmax], [ymin, ymax], [zmax,zmax], color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmin, xmax], [ymin, ymin], [zmax,zmax], color='0.4', linewidth=1, zorder=1e3)
        ax.plot([xmax, xmax], [ymin, ymin], [zmin, zmax], color='0.4', linewidth=1, zorder=1e3)

        ax.view_init(elev=35, azim=-45)
        ax.set_box_aspect(None, zoom=1)

        fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1)
        fig.tight_layout()
        plt.show()

if __name__ == "__main__":
    l = 1
    name = "Quadratic distribution"
    n_points = (100, 100, 100)
    pix_sizes = (1 * l / n_points[0], 1 * l / n_points[1], 1 * l / n_points[2])
    cuad_cube_dist = lambda Z, Y, X, a_0: 1.5 + ((X-0.5)**2+(Y-0.5)**2+Z) / 10
    a = cuad_cube_dist(1,2,3, jnp.array([0]))
    cuad_cube = FGRINcomponent(name, cuad_cube_dist, n_points, pix_sizes, a_0=jnp.array([0]))
    cube = cuad_cube.generate()

    cuad_cube.show()