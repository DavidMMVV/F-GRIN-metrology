import numpy as np
import plotly.graph_objects as go
import plotly.io as pio


def add_surface(fig, x, y, z, c, name):
    """Function to add a surface to the figure."""
    fig.add_trace(go.Surface(
        x=x, y=y, z=z,
        surfacecolor=c,
        colorscale='Jet',
        showscale=True,
        colorbar=dict(title="Index"),
        opacity=1, 
        name=name
    ))

if __name__ == "__main__":
    pio.renderers.default = "browser"

    # ----------------------------
    # Surface 1: big sphere
    # ----------------------------
    u1 = np.linspace(0.5*np.pi, 2*np.pi, 75)
    v1 = np.linspace(0, np.pi, 100)
    U1, V1 = np.meshgrid(u1, v1)

    x1 = np.cos(U1) * np.sin(V1)
    y1 = np.sin(U1) * np.sin(V1)
    z1 = np.cos(V1)
    c1 = np.sin(3*V1) * np.cos(U1) * (x1**2+y1**2+z1**2)**3

    # ----------------------------
    # Surface 2: spherical cut
    # ----------------------------
    u2 = np.linspace(0, 0.5*np.pi, 25)
    v2 = np.linspace(np.pi/2, np.pi, 50)
    U2, V2 = np.meshgrid(u2, v2)

    x2 = np.cos(U2) * np.sin(V2)
    y2 = np.sin(U2) * np.sin(V2)
    z2 = np.cos(V2)
    c2 = np.sin(3*V2) * np.cos(U2) *(x2**2+y2**2+z2**2)**3

    # ----------------------------
    # Surface 3: inner plane
    # ----------------------------
    u3 = np.linspace(0, 0.5*np.pi, 25)
    r3 = np.linspace(0, 1, 50)
    U3, R3 = np.meshgrid(u3, r3)

    x3 = R3 * np.cos(U3)
    y3 = R3 * np.sin(U3)
    z3 = np.zeros_like(x3)
    c3 = np.sin(3*np.pi / 2) * np.cos(U3) * (x3**2+y3**2+z3**2)**3

    # ----------------------------
    # Surface 4: lateral cut
    # ----------------------------
    r4 = np.linspace(0, 1, 50)
    v4 = np.linspace(0, np.pi/2, 50)
    R4, V4 = np.meshgrid(r4, v4)

    x4 = R4 * np.sin(V4) * np.cos(0)
    y4 = R4 * np.sin(V4) * np.sin(0)
    z4 = R4 * np.cos(V4)
    c4 = np.sin(3*V4) * np.cos(0) * (x4**2+y4**2+z4**2)**3

    # ----------------------------
    # Surface 5: lateral cut
    # ----------------------------
    r5 = np.linspace(0, 1, 50)
    v5 = np.linspace(0, np.pi/2, 50)
    R5, V5 = np.meshgrid(r5, v5)

    x5 = R5 * np.sin(V5) * np.cos(0.5 * np.pi)
    y5 = R5 * np.sin(V5) * np.sin(0.5 * np.pi)
    z5 = R5 * np.cos(V5)
    c5 = np.sin(3*V5) * np.cos(0.5*np.pi) * (x5**2+y5**2+z5**2)**3

    # ----------------------------
    # Create figure
    # ----------------------------
    fig = go.Figure()

    # Añadir todas las superficies
    add_surface(fig, x1, y1, z1, c1, "Surface 1")
    add_surface(fig, x2, y2, z2, c2, "Surface 2")
    add_surface(fig, x3, y3, z3, c3, "Surface 3")
    add_surface(fig, x4, y4, z4, c4, "Surface 4")
    add_surface(fig, x5, y5, z5, c5, "Surface 5")

    # Normalizar colorbar globalmente
    c_global = np.concatenate([c1.ravel(), c2.ravel(), c3.ravel(), c4.ravel(), c5.ravel()])
    fig.update_traces(cmin=c_global.min(), cmax=c_global.max())

    # Layout
    fig.update_layout(
        title="$r \\phi \\theta$-FGRIN",
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
