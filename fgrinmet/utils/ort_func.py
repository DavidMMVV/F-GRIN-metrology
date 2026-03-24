import jax.numpy as jnp
from scipy.special import legendre
from scipy.special import factorial
from typing import Tuple


def rad_poly(m: int, 
             n: int, 
             rho: jnp.ndarray) -> jnp.ndarray:
    """
    This function computes the zernike radial polynomial. 
    the definition has been obtained from https://en.wikipedia.org/wiki/Zernike_polynomials.

    Args:
        m (int): Column index of the Zernike polynomial. Must be lower or equal to n.
        n (int): Row index of the Zernike polynomial.
        rho (torch.Tensor): Radial coordinates defined between 0 and 1.

    Returns:
        Rmn (torch.Tensor): Computed radial Zernike polynomial.
    """
    
    if ((n - m) % 2) == 0:
        
        Rmn = jnp.sum(jnp.array([((-1)**k * factorial(n-k) * rho**(n-2*k)) /
                      (factorial(k) * factorial(((n + m) / 2) - k) * factorial(((n - m) / 2) - k))
                       for k in range(1 + (n - jnp.abs(m)) // 2)]), axis=0)
    else:
        Rmn = jnp.zeros((rho).shape)
    return Rmn

def ANSI_to_mn(index: int) -> tuple[int, int]:
    """
    Returns the m and n coefficients corresponding to the OSA/ANSI index
    defined on https://en.wikipedia.org/wiki/Zernike_polynomials.

    Args:
        index (int): ANSI index

    Returns:
        (m,n)(tuple[int, int]): 
            m (int): Column index of the Zernike polynomial. Must be lower or equal to n.
            n (int): Row index of the Zernike polynomial.
    """

    n_raw = (jnp.sqrt(1 + 8 * index) - 1) / 2
    n = int(n_raw)
    m_raw = 2*index-n*(n+2)
    m = int(m_raw)
    return m, n

def z_poly(coeficients: jnp.ndarray,
           shape: jnp.ndarray,
           pix_size: jnp.ndarray,
           radius: float) -> jnp.ndarray:
    """
    This function computes the Zernike polynomials with their corresponding weights.

    Args:
        coeficients (jnp.ndarray): Array giving the coeficients in order according to OSA/ANSI format.
        shape (jnp.ndarray): Shape of the grid in pixels.
        pix_size (jnp.ndarray): Size of the pixels in the grid in physical units.
        radius (float): Radius of the circle in physical units.

    Returns:
        Zpol (jnp.ndarray): The correspondig zernike polynomials with their corresponding weights. 
    """
    
    y = jnp.linspace(1, -1, int(shape[0]))
    x = jnp.linspace(-1, 1, int(shape[1]))
    Y, X = jnp.meshgrid(y, x, indexing="ij")
    rad_y = radius * 2 / (shape[0]*pix_size[0])
    rad_x = radius * 2 / (shape[1]*pix_size[1])
    Y_norm, X_norm = ((Y/rad_y), (X/rad_x))

    rho = jnp.sqrt(X_norm**2 + Y_norm**2)
    phi = jnp.atan2(Y_norm, X_norm)

    Zpol = jnp.zeros((len(coeficients), shape[0], shape[1]))
    for index, coef in enumerate(coeficients):
        m, n = ANSI_to_mn(int(index))
        R = rad_poly(m, n, rho) * ((X_norm**2 + Y_norm**2) <= 1)
        A = (jnp.cos(jnp.abs(m) * phi) if m >= 0 else jnp.sin(jnp.abs(m) * phi))
        Zpol = Zpol.at[index].set(coef * R * A)
    return Zpol

def l_poly(
        coeficients: jnp.ndarray,
        shape: int
) -> jnp.ndarray:
    """
    Function whih returns all the legendre polynomials

    Args:
        coeficients (jnp.ndarray): Coeficients of the legendre polynomials in order.
        shape (int): Number of pixels in the z direction.

    Returns:
        Lpoly(jnp.ndarray): The corresponding legendre polynomials with their corresponding weights. 
    """
    z = jnp.linspace(-1,1, shape)
    Lpoly = jnp.array([coef * legendre(i)(z) for i, coef in enumerate(coeficients)])

    return Lpoly

def poly_exp(
        coeficients: jnp.ndarray,
        shape: jnp.ndarray,
        pix_size: jnp.ndarray,
        radius: float
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Function which gives the distributions of the orthogonal polynomials (Zernike and Legendre).

    Args:   
        coeficients (jnp.ndarray): Coeficients of the polynomials in order.
        shape (jnp.ndarray): Shape of the grid in pixels.
        pix_size (jnp.ndarray): Size of the pixels in the grid in physical units.
        radius (float): Radius of the circle in physical units.
    Returns:
        Zpol (jnp.ndarray): The correspondig zernike polynomials with their corresponding weights. 
        Lpol (jnp.ndarray): The corresponding legendre polynomials with their corresponding weights.
    """

    Zpol = z_poly(jnp.ones(coeficients.shape[0]), shape[1:], pix_size[1:], radius)
    Lpol = l_poly(jnp.ones(coeficients.shape[1]), shape[0])
    return Zpol, Lpol
    
"""def poly_adj(
        distribution: jnp.ndarray,
        coeficients: jnp.ndarray,
        pix_size: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    shape = distribution.shape
    coeficients_out = jnp.zeros_like(coeficients)

    Zpol, Lpol = poly_exp(jnp.ones_like(coeficients), shape, pix_size, radius)

    for i, slice in enumerate(distribution):
        coeficients_out += (slice[None,None] * (Zpol[:, None] * Lpol[None, :, i, None, None])).sum(axis=(2,3))

    return coeficients_out"""

def poly_adj(
        distribution: jnp.ndarray,
        coeficients: jnp.ndarray,
        pix_size: jnp.ndarray,
        radius: float
) -> jnp.ndarray:
    """
    Function which adjust a certain 3D distribution along othogonal polynomial basis.

    Args:
        distribution (jnp.ndarray): 3D distribution to be adjusted.
        coeficients (jnp.ndarray): Coeficients of the polynomians along which to express the ditribution.
        pix_size (jnp.ndarray): Size of the pixels in the grid in physical units.
        radius (float): Radius of the circle in physical units.

    Returns:
        coeficients_out (jnp.ndarray): Output coeficients of the corresponding distribution.
    """
    
    dist_shape = jnp.array(distribution.shape)
    coef_shape = coeficients.shape

    mask = (distribution != 0)
    Zpol, Lpol = poly_exp(jnp.ones_like(coeficients), dist_shape, pix_size, radius)

    y = distribution[mask]
    M = jnp.zeros((y.shape[0],  coef_shape[0] * coef_shape[1]))
    for i in range(coef_shape[0] * coef_shape[1]):
        zer = Zpol[i//coef_shape[1]]
        leg = Lpol[i%coef_shape[1]]
        M = M.at[:,i].set((zer[None] * leg[:,None,None])[mask])
    coeficients_out, _, _,_ = jnp.linalg.lstsq(M, y, rcond=None)

    return coeficients_out.reshape(coef_shape)

def poly_sum(coeficients: jnp.ndarray,
             Zpol: jnp.ndarray,
             Lpol: jnp.ndarray) -> jnp.ndarray:
    """
    This function returns the distribution given by the coeficients of each polynomial.

    Args:
        coeficients (jnp.ndarray): Coeficients of the polynomians.
        Zpol (jnp.ndarray): Zernike polynomials.
        Lpol (jnp.ndarray): Legendre polynomials.

    Returns:
        out (jnp.ndarray): The sum of the polynomials with the corresponding coeficients. 
        The shape of the output is the same as the shape of the distribution to be adjusted.
    """

    out = jnp.zeros((Lpol.shape[1], Zpol.shape[1], Zpol.shape[2]))
    for i in range(Lpol.shape[1]):
        out = out.at[i].set((coeficients[:,:,None, None] * Lpol[None,:,i,None,None] * Zpol[:,None]).sum(axis=(0,1)))
    return out

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from coordinates import coord_jax
    coeficients = jnp.ones(16*6).reshape(16,6)
    shape = jnp.array([128,128,128])
    pix_size = jnp.array([200 / shape[0], 200 / shape[1], 200 / shape[2]])
    extent = [-(pix_size[2]*(shape[2]//2)), (pix_size[2]*(1+shape[2]//2)), -(pix_size[1]*(shape[1]//2)), (pix_size[1]*(1+shape[1]//2))]
    radius = 100
    zernikes, leg = poly_exp(coeficients, shape, pix_size, radius)
    z = jnp.linspace(-1,1,shape[0])
    
    fig, sub = plt.subplots(4,4, sharex=True, sharey=True)
    for i, dis in enumerate(zernikes):
        im = sub[i//4, i%4].imshow(dis, cmap="seismic", extent=extent)
        #im = sub[i//4, i%4].imshow((jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(jnp.exp(2j*jnp.pi*dis)*(dis!=0))))), cmap="seismic", extent=extent)
        plt.colorbar(im, ax=sub[i//4, i%4])
        sub[i//4, i%4].set_title(f"$Z_{{{i}}}$")
        if i % 4 == 0:
            sub[i//4, i%4].set_ylabel("y")
        if i // 4 == 3:
            sub[i//4, i%4].set_xlabel("x")
    plt.tight_layout()

    plt.figure()
    for i, Lpoly in enumerate(leg):
        plt.plot(z, Lpoly, label=f"$\\mathcal{{P}}_{{{i}}}$")
    plt.legend()
    plt.xlabel("z")
    plt.ylabel("P(z)")
    plt.tight_layout()

    Z,Y,X = jnp.meshgrid(*[
        (jnp.arange(shape[0]) - shape[0] // 2) * pix_size[0],
        (jnp.arange(shape[1]) - shape[1] // 2) * pix_size[1],
        (jnp.arange(shape[2]) - shape[2] // 2) * pix_size[2]], indexing='ij')
    
    R = jnp.sqrt(X**2 + Y**2)
    distribution = (1-(R/radius)**2) * (R<=radius)
    ind_adj = poly_adj(distribution, jnp.ones((16,6)), pix_size, radius)

    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(distribution[shape[0]//2])
    plt.colorbar(im1, ax=sub[0], shrink=0.23)
    sub[0].set_xlabel("x")
    sub[0].set_ylabel("y")
    im2 = sub[1].imshow(distribution[:,shape[1]//2].T)
    plt.colorbar(im2, ax=sub[1], shrink=0.23)
    sub[1].set_xlabel("z")
    sub[1].set_ylabel("x")

    im3 = sub[2].imshow(distribution[:,:,shape[2]//2].T)
    plt.colorbar(im3, ax=sub[2], shrink=0.23)
    sub[2].set_xlabel("z")
    sub[2].set_ylabel("y")
    plt.tight_layout()

    plt.figure()
    plt.imshow(ind_adj, cmap="jet")
    plt.colorbar()
    plt.xlabel("Legendre order")
    plt.ylabel("Zernike order")

    rec = poly_sum(ind_adj, zernikes, leg)
    fig, sub = plt.subplots(1,3)
    im1 = sub[0].imshow(rec[shape[0]//2])
    plt.colorbar(im1, ax=sub[0], shrink=0.23)
    im2 = sub[1].imshow(rec[:,shape[1]//2])
    plt.colorbar(im2, ax=sub[1], shrink=0.23)
    im3 = sub[2].imshow(rec[:,:,shape[2]//2])
    plt.colorbar(im3, ax=sub[2], shrink=0.23)
    plt.tight_layout()

    plt.show()

