import numpy as np
import math

def rad_poly(m: int, 
             n: int, 
             rho: np.ndarray) -> np.ndarray:
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
        Rmn = np.sum([((-1)**k * math.factorial(n-k) * rho**(n-2*k)) /
                      (math.factorial(k) * math.factorial((n + m) // 2 - k) * math.factorial((n - m) // 2 - k))
                       for k in range((n-np.abs(m)) // 2 + 1)], axis=0)
    else:
        Rmn = np.zeros((rho).shape)
    return Rmn

def z_poly(coeficients: dict[str, float] | dict[str, int],
           H: int,
           W: int) -> np.ndarray:
    """
    This function computes the combination of the Zernike polynomials with their corresponding weights.

    Args:
        coeficients (dict[int, float]): Dictionary giving the number of the Zernike polynomial in OSA/ANSI
        format with its corresponding weight.
        H (int): Height of the image in pixels.
        W (int): Width of the image in pixels.

    Returns:
        Zpol (np.ndarray): The profile resulting from the parameters of the coefficients for different
        Zernike polynomials. 
    """
    
    y = np.linspace(1, -1, H) 
    x = np.linspace(-1, 1, W)  
    Y, X = np.meshgrid(y, x, indexing='ij')

    rho = (X**2 + Y**2) * ((X**2 + Y**2) <= 1)
    phi = np.atan2(Y, X)

    # Check if keys are integers
    key_int= all(isinstance(k, str) and k.isdigit() for k in coeficients.keys())
    
    if isinstance(coeficients, dict) and key_int:
        Zpol = np.zeros([H,W])
        for index, coef in coeficients.items():
            m, n = ANSI_to_mn(int(index))
            R = rad_poly(m, n, rho)
            A = (np.cos(np.abs(m) * phi) if m >= 0 else np.sin(np.abs(m) * phi))
            Zpol += coef * R * A
    else:
        raise(TypeError("Coeficients is not a dictionary with integer keys and float values."))

    return Zpol

def ANSI_to_mn(index: int) -> tuple[int, int]:
    """
    Returns the m and n coefficients corresponding to the OSA/ANSI index
    defined on https://en.wikipedia.org/wiki/Zernike_polynomials.

    Args:
        index (int): _description_

    Returns:
        (m,n)(tuple[int, int]): 
            m (int): Column index of the Zernike polynomial. Must be lower or equal to n.
            n (int): Row index of the Zernike polynomial.
    """

    n_raw = (math.sqrt(1 + 8 * index) - 1) / 2
    n = int(n_raw)
    m_raw = (n_raw * (n_raw + 1) / 2) - (n * (n+1) / 2)
    m = -n + 2 * int(m_raw)
    return m, n