import jax.numpy as jnp
from scipy.special import legendre
from scipy.special import factorial


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
    This function computes the combination of the Zernike polynomials with their corresponding weights.

    Args:
        coeficients (dict[int, float]): Dictionary giving the number of the Zernike polynomial in OSA/ANSI
        format with its corresponding weight.
        H (int): Height of the image in pixels.
        W (int): Width of the image in pixels.

    Returns:
        Zpol (jnp.ndarray): The profile resulting from the parameters of the coefficients for different
        Zernike polynomials. 
    """
    
    y = jnp.linspace(1, -1, shape[0]) 
    x = jnp.linspace(-1, 1, shape[1])  
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
        shape: jnp.ndarray
):
    pass
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    coeficients = jnp.ones(16)
    shape = jnp.array([1024,1024])
    pix_size = jnp.array([200 / shape[0], 200 / shape[0]])
    extent = [-(pix_size[1]*(shape[1]//2)), (pix_size[1]*(1+shape[1]//2)), -(pix_size[0]*(shape[0]//2)), (pix_size[0]*(1+shape[0]//2))]
    radius = 5
    zernikes = z_poly(coeficients, shape, pix_size, radius)

    z = jnp.linspace(-1,1,100)
    print(legendre(3)(1))
    
    fig, sub = plt.subplots(4,4, sharex=True, sharey=True)
    for i, dis in enumerate(zernikes):
        #im = sub[i//4, i%4].imshow(dis, cmap="seismic", extent=extent)
        im = sub[i//4, i%4].imshow((jnp.abs(jnp.fft.fftshift(jnp.fft.fft2(jnp.exp(2j*jnp.pi*dis)*(dis!=0))))), cmap="seismic", extent=extent)
        plt.colorbar(im, ax=sub[i//4, i%4])
        sub[i//4, i%4].set_title(f"$Z_{{{i}}}$")
    plt.tight_layout()
    plt.show()
