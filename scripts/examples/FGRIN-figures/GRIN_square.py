import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from GRIN_sphere import add_surface

if __name__ == "__main__":
    pio.renderers.default = "browser"

    #----------------------------        
    # Top surface
    #----------------------------    
    y1, x1 = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100), indexing="ij")

    z1 =  0.5 * np.ones_like(x1)
    c1 = 1 + (z1) + (y1-1)# + x1**2
    #----------------------------
    # Side surface
    #----------------------------
    z2, y2 = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-1, 1, 100), indexing="ij")

    x2 = 1 * np.ones_like(z2)
    c2 = 1 + (z2) + (y2-1)# + x2**2

    #----------------------------
    # Other side surface
    #----------------------------
    z3, x3 = np.meshgrid(np.linspace(-0.5, 0.5, 100), np.linspace(-1, 1, 100), indexing="ij")

    y3 = 1 * np.ones_like(z3)
    c3 = 1 + z3 + (y3-1)# + x3**2

    # Create figure
    fig = go.Figure()
    add_surface(fig, x1, y1, z1, c1, name="Top Surface")
    add_surface(fig, x2, y2, z2, c2, name="Side Surface")  
    add_surface(fig, x3, y3, z3, c3, name="Other side Surface")   

    # Normalizar colorbar globalmente
    c_global = np.concatenate([c1.ravel(), c2.ravel(), c3.ravel()])
    fig.update_traces(cmin=c_global.min(), cmax=c_global.max())

    # Layout
    fig.update_layout(
        title="Axial GRIN",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(visible=False),  # oculta eje X y su grid
            yaxis=dict(visible=False),  # oculta eje Y y su grid
            zaxis=dict(visible=False),  # oculta eje Z y su grid
            aspectmode='data'
        ),
        width=900,
        height=800
    )

    fig.update_layout(
        scene_camera=dict(
            eye=dict(x=1.2, y=1.2, z=0.8)  # camera coordinates
        )
    )

    fig.show()