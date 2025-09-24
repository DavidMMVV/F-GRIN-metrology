import torch
import torch.fft
import matplotlib.pyplot as plt

# ----------------------------
# 1. Parámetros
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

nx, ny = 256, 256          # resolución en XY
dx = dy = 1e-6             # tamaño de voxel
wavelength = 0.5e-6        # 500 nm
k0 = 2*torch.pi / wavelength
n0 = 1.0                   # índice de fondo
N_BORN = 10                # órdenes de Born
use_pade = True             # usar Padé approximant

# ----------------------------
# 2. Cilindro circular en XY
# ----------------------------
x = torch.linspace(-nx//2*dx, nx//2*dx, nx, device=device)
y = torch.linspace(-ny//2*dy, ny//2*dy, ny, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')

radius = 20e-6
cylinder = ((X**2 + Y**2) < radius**2).float()
n_cylinder = n0 + 1 * torch.ones_like(cylinder)#0.2*cylinder       # delta n = 0.2
delta_n = n_cylinder - n0

# ----------------------------
# 3. Onda plana de entrada en X=-L
# ----------------------------
E0 = torch.ones(ny, nx, device=device, dtype=torch.complex64)  # plano XY

# ----------------------------
# 4. Green's function 2D (XY) para propagación a lo largo de x)
# ----------------------------
def greens_function_2d(nx, ny, dx, dy, wavelength, dx_step):
    fx = torch.fft.fftfreq(nx, dx, device=device)
    fy = torch.fft.fftfreq(ny, dy, device=device)
    FX, FY = torch.meshgrid(fx, fy, indexing='ij')
    k = 2*torch.pi / wavelength
    H = torch.exp(1j * k * dx_step * torch.sqrt(1 - (wavelength*FX)**2 - (wavelength*FY)**2))
    return H

G = greens_function_2d(nx, ny, dx, dy, wavelength, dx)

# ----------------------------
# 5. Lippmann-Schwinger 2D con Born + Padé
# ----------------------------
def lippmann_schwinger_2d(E0, delta_n, dx_step, N_BORN=5, use_pade=True):
    ny, nx = delta_n.shape
    E = E0.clone()
    delta_eps = (2*n0*delta_n).to(torch.complex64)
    E_prev = E.clone()
    
    for order in range(N_BORN):
        S = delta_eps * E_prev
        S_fft = torch.fft.fft2(S)
        conv = torch.fft.ifft2(S_fft * G)
        if use_pade:
            E = E0 / (1 - conv)   # Padé approximant
        else:
            E = E0 + conv         # Born series
        E_prev = E.clone()
    return E

E_total = lippmann_schwinger_2d(E0, delta_n, dx, N_BORN, use_pade)

# ----------------------------
# 6. Intensidad en plano final (X=+L)
# ----------------------------
I = torch.abs(E_total)**2

# ----------------------------
# 7. Visualización del scattering
# ----------------------------
plt.figure(figsize=(6,5))
plt.imshow(I.cpu(), extent=[-nx*dx/2*1e6, nx*dx/2*1e6, -ny*dy/2*1e6, ny*dy/2*1e6],
           cmap='inferno', origin='lower')
plt.xlabel('x [µm]')
plt.ylabel('y [µm]')
plt.title('Scattering de un cilindro infinito (Padé Born Series)')
plt.colorbar(label='Intensidad [a.u.]')
plt.show()
