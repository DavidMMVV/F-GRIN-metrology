import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from GRIN_sphere import add_surface

if __name__ == "__main__":
    pio.renderers.default = "browser"

    #----------------------------        
    # Top surface
    #----------------------------    
    phi = np.linspace(0, 2*np.pi, 100)
    r = np.linspace(0, 1, 50)

    x1 = np.outer(r, np.cos(phi))
    y1 = np.outer(r, np.sin(phi))
    z1 =  0.5 * np.ones_like(x1)
    c1 = z1**3 * np.outer(np.ones_like(r), np.sin(phi*3)) * (2 - (x1**2 + y1**2))

    #----------------------------
    # Side surface
    #----------------------------
    phi = np.linspace(0, 2*np.pi, 100)
    z = np.linspace(-0.5, 0.5, 50)

    x2 = np.outer(np.ones_like(z), np.cos(phi))
    y2 = np.outer(np.ones_like(z), np.sin(phi))
    z2 = np.outer(z, np.ones_like(phi))
    c2 = z2**3 * np.outer(np.ones_like(z), np.sin(phi*3)) * (2 - (x2**2 + y2**2))

    # Create figure
    fig = go.Figure()
    add_surface(fig, x1, y1, z1, c1, name="Top Surface")
    add_surface(fig, x2, y2, z2, c2, name="Side Surface")   

    # Normalizar colorbar globalmente
    c_global = np.concatenate([c1.ravel(), c2.ravel()])
    fig.update_traces(cmin=c_global.min(), cmax=c_global.max())

    # Layout
    fig.update_layout(
        title="GRIN lens",
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            xaxis=dict(visible=False), 
            yaxis=dict(visible=False),
            zaxis=dict(visible=False), 
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