from tkinter import NO
import skfmm
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from constructions import zpol
from fgrinmet.constructions.zpol import z_poly
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os

sim_shape = np.array([3507, 2481])
pix_size = 1

U_i = np.zeros(sim_shape)
U_i[-1] = 1
phi_i = np.full(sim_shape, np.inf, dtype=np.float64)
phi_i[-1] = 0


vcoma_params = {"5": 10, "4":10, "7":0, "12":0, "11":10} 
coma = z_poly(vcoma_params, 100 ,sim_shape[1])
coma =np.abs(coma - (coma.min()))**(1/4)

n = 1.5+((np.arange(sim_shape[0])-sim_shape[0]/2)/sim_shape[0]/2)[:,None] * (coma[None,49,:])


phi = skfmm.travel_time(phi_i, 1/n, pix_size)

# target resolution
height, width = 3507, 2481

# create figure with the right pixel resolution
dpi = 100
fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
ax = fig.add_axes([0, 0, 1, 1])  # full canvas, no margins

# plot colormap
ax.imshow(1/n, cmap="magma", origin="lower")
ax.axis("off")

# add isolines in white
cs = ax.contour(phi, levels=20, colors="#FFD966", linewidths=3)
#ax.clabel(cs, inline=True, fontsize=8, fmt="%.1f", colors="white")

# ensure save directory exists
save_dir = os.path.join("data", "images")
os.makedirs(save_dir, exist_ok=True)

# save figure with exact resolution, no borders
save_path = os.path.join(save_dir, "phi_with_isolines.jpg")
fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0, format="jpg")

plt.close(fig)

print(f"Image saved at: {save_path} with resolution {width}x{height}")