import jax.numpy as jnp

from typing import Tuple, Optional, List, Literal

def prop_coord_to_obj_coord(
        coords: jnp.ndarray,
        obj_origin: List[float] | Tuple[float,...],
        prop_origin: List[float] | Tuple[float,...],
        obj_shape: List[int] | Tuple[int,...],
        obj_pix_size: float | List[float] | Tuple[float,...] = 1.0,
) -> jnp.ndarray:
    """Computes the coordinates of the propagation grid points with respect to the upper up left corner of the object.

    Args:
        coords (jnp.ndarray): Coordinates to transform.
        obj_origin (List[float] | Tuple[int]): Position of the center of the object.
        prop_origin (List[float] | Tuple[int]): Position of the center of the propagation grid. 
        obj_shape (List[float] | Tuple[int]): Shape of the object.
        obj_pix_size (float | List[float] | Tuple[float,...], optional): Pixel sizes of the object. Defaults to 1.0.

    Returns:
        jnp.ndarray: _description_
    """
    dims_exp = tuple((coords.ndim - 1) * [None])
    return ((coords - jnp.array(obj_origin)[*dims_exp] + jnp.array(prop_origin)[*dims_exp]) / jnp.array(obj_pix_size)[*dims_exp]) + jnp.array(obj_shape)[*dims_exp] // 2


def trilinear_interpolate(
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
        grid_shape = jnp.array(mask.shape)
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
    
    # Compute the floor and the decimal factor dependent on distance of the triliniar expresion
    floor_points = jnp.floor(points).astype(jnp.int32)
    dec_fact = (offsets[::-1, :, *dim_exp] + (-1)**(offsets[::-1, :, *dim_exp]) * (points - floor_points)[None]).prod(axis=1)

    # Compute the values in the corners
    corners = floor_points[None] + offsets[:, :, *dim_exp]
    condition = (((corners >= 0).prod(axis=1).astype(bool)) &
                 ((corners < grid_shape[None, :, *dim_exp]).prod(axis=1).astype(bool)) & 
                 (mask[corners[:,0], corners[:,1], corners[:,2]]))
    corners_val = jnp.where(condition, values_grid[corners[:,0], corners[:,1], corners[:,2]], outside)

    # Find the interpolated values
    c = (corners_val * dec_fact).sum(axis=0)
    return c
