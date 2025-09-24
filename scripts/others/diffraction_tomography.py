import torch
import torch.nn.functional as F
import numpy as np

# Configuración de GPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------------------
# 1. Parámetros y volumen inicial
# ------------------------------
volume_size = 128  # Tamaño del volumen 3D
projections = 100  # Número de proyecciones simuladas

# Volumen inicial (aleatorio o con alguna estimación inicial)
volume = torch.randn((1, 1, volume_size, volume_size, volume_size), device=device, requires_grad=True)

# Ángulos de proyección simulados
angles = torch.linspace(0, np.pi, projections, device=device)

# ------------------------------
# 2. Función de proyección (Radon 3D aproximado)
# ------------------------------
def project_3d(volume, angle):
    """
    Proyección 3D simple usando rotación y suma en un eje.
    """
    # Rotar el volumen en el eje Z
    grid = create_rotation_grid(volume.shape[-3:], angle)
    rotated = F.grid_sample(volume, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    # Sumar sobre eje Z para simular proyección
    projection = rotated.sum(dim=2)
    return projection

def create_rotation_grid(shape, angle):
    """
    Crea una malla de rotación 3D para grid_sample.
    """
    D, H, W = shape
    z, y, x = torch.meshgrid(
        torch.linspace(-1, 1, D, device=device),
        torch.linspace(-1, 1, H, device=device),
        torch.linspace(-1, 1, W, device=device),
        indexing='ij'
    )
    # Rotación en el plano XY
    x_rot = x * torch.cos(angle) - y * torch.sin(angle)
    y_rot = x * torch.sin(angle) + y * torch.cos(angle)
    z_rot = z
    grid = torch.stack((x_rot, y_rot, z_rot), dim=-1).unsqueeze(0)
    return grid

# ------------------------------
# 3. Interpolación de Fourier 3D
# ------------------------------
def fourier_interpolate(volume, new_size):
    """
    Interpolación de Fourier 3D para cambio de resolución.
    """
    # FFT del volumen
    vol_f = torch.fft.fftn(volume, dim=(-3,-2,-1))
    # Cambio de tamaño mediante padding o recorte
    pad_sizes = [(0, new_size - s) if new_size > s else (0, 0) for s in volume.shape[-3:]]
    vol_f_resized = F.pad(vol_f, sum(pad_sizes[::-1], ()))  # Pad requiere tupla larga
    # IFFT para volver al espacio real
    volume_resized = torch.fft.ifftn(vol_f_resized, dim=(-3,-2,-1)).real
    return volume_resized

# ------------------------------
# 4. Simulación de proyecciones observadas (para test)
# ------------------------------
with torch.no_grad():
    true_volume = torch.zeros((1,1,volume_size,volume_size,volume_size), device=device)
    true_volume[:, :, volume_size//4:3*volume_size//4, volume_size//4:3*volume_size//4, volume_size//4:3*volume_size//4] = 1.0
    projections_real = torch.stack([project_3d(true_volume, angle) for angle in angles])

# ------------------------------
# 5. Optimización por retropropagación
# ------------------------------
optimizer = torch.optim.Adam([volume], lr=0.05)
num_iters = 200

for i in range(num_iters):
    optimizer.zero_grad()
    
    projections_pred = torch.stack([project_3d(volume, angle) for angle in angles])
    
    # Pérdida L2
    loss = F.mse_loss(projections_pred, projections_real)
    
    loss.backward()
    optimizer.step()
    
    if i % 20 == 0:
        print(f"Iter {i}, Loss: {loss.item():.6f}")

# ------------------------------
# 6. Reconstrucción final y visualización
# ------------------------------
volume_reconstructed = fourier_interpolate(volume.detach(), volume_size)
print("Reconstrucción completada. Volumen shape:", volume_reconstructed.shape)
import matplotlib.pyplot as plt

# Función para mostrar cortes del volumen
def show_slices(volume, title="Slices"):
    volume = volume.squeeze().cpu().numpy()  # Convertir a numpy
    D, H, W = volume.shape
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(volume[D//2, :, :], cmap='gray')
    axes[0].set_title('Slice XY (medio)')
    axes[1].imshow(volume[:, H//2, :], cmap='gray')
    axes[1].set_title('Slice XZ (medio)')
    axes[2].imshow(volume[:, :, W//2], cmap='gray')
    axes[2].set_title('Slice YZ (medio)')
    fig.suptitle(title)
    plt.show()

# Mostrar volumen verdadero
show_slices(true_volume, title="Volumen original")

# Mostrar volumen reconstruido
show_slices(volume_reconstructed, title="Volumen reconstruido")

