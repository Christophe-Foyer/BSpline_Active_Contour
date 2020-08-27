"""
Author: Christophe Foyer
    
Description:
    A collection of functions for plotting purposes.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from matplotlib import animation

import plotly.graph_objects as go
from plotly.offline import plot as offline_plot

import pyvista as pv

# %%
def makemesh(surf, meshtype="StructuredGrid", points='evalpts'):
    """
    Makes pyvista meshes from NURBS-Python surfaces.

    Parameters
    ----------
    surf : Surface
        A NURBS-Python Surface.
    meshtype : string, optional
        Options are "PolyData" or "StructuredGrid". 
        The default is "StructuredGrid".
    points : string, optional
        DESCRIPTION. The default is 'evalpts'.

    Returns
    -------
    mesh : pyvista.PolyData or pyvista.Grid derived classes

    """
            
    if points == 'evalpts':
        pts = np.array(surf.evalpts)
        pts_square = pts.reshape((*surf.data['sample_size'], -1))
    elif points == 'ctrlpts':
        pts = np.array(surf.ctrlpts)
        pts_square = pts.reshape((*surf.data['size'], -1))
    elif points == 'control_points':
        pts_square = surf.control_points_np[:,:,:3]
        # pts_square = np.swapaxes(pts_square, 0, 1)
        # pts_square = np.swapaxes(pts_square, 1, 2)
    
    if meshtype.lower() == 'PolyData'.lower():
        mesh = pv.PolyData(pts_square)
    
    if meshtype.lower() == 'StructuredGrid'.lower():
        mesh = pv.StructuredGrid()
        mesh.points = pts
        mesh.dimensions = [*np.flip(surf.data['sample_size']),
                           1]
    return mesh

# %%
def animate(c_list, timespan=2, length=None, html=False):
    """
    An animation class from assignment 3

    Animates the first "length" elements in c_list
    If not specified uses the entire list

    Animation spans 2 seconds with 100ms blank between loops
    """

    if length is None or length > len(c_list):
        length = len(c_list)

    fig = plt.figure()

    ims = []
    for i in range(1, length, int(length*0.5/100 + 1)):
        a = np.array(c_list[i])
        im = plt.imshow(a, extent=[-1, 1, -1, 1], origin='upper')
        ims.append([im])

    ani = animation.ArtistAnimation(fig, ims,
                                    interval=timespan*1000/length,
                                    blit=True,
                                    repeat_delay=100)
    plt.colorbar()

    try:
        html = HTML(ani.to_html5_video())
        return html
    except Exception as e:
        print("Error:", Exception(e))

# %%
def animate_3d(image, nb_frames=100, plot=False):
    if nb_frames > image.shape[2]:
        nb_frames = image.shape[2]

    vol = image
    volume = vol.T
    r, c = volume[0].shape

    # Define frames
    fig = go.Figure(frames=[go.Frame(data=go.Surface(
        z=((nb_frames-1)/10 - k * 0.1) * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames - 1 - k]),
        cmin=image.min(), cmax=image.max()
        ),
        name=str(k)  # you need to name the frame
        )
        for k in range(nb_frames)])

    # Add data to be displayed before animation starts
    fig.add_trace(go.Surface(
        z=(nb_frames-1)/10 * np.ones((r, c)),
        surfacecolor=np.flipud(volume[nb_frames - 1]),
        colorscale='Gray',
        cmin=image.min(), cmax=image.max(),
        colorbar=dict(thickness=20, ticklen=4)
        ))

    def frame_args(duration):
        return {
                "frame": {"duration": duration},
                "mode": "immediate",
                "fromcurrent": True,
                "transition": {"duration": duration, "easing": "linear"},
            }

    sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

    # Layout
    fig.update_layout(
             title='Slices in volumetric data',
             width=600,
             height=600,
             scene=dict(
                        zaxis=dict(range=[-0.1, nb_frames/10],
                                   autorange=False),
                        aspectratio=dict(x=1, y=1, z=1),
                        ),
             updatemenus=[
                {
                    "buttons": [
                        {
                            "args": [None, frame_args(50)],
                            "label": "&#9654;",  # play symbol
                            "method": "animate",
                        },
                        {
                            "args": [[None], frame_args(0)],
                            "label": "&#9724;",  # pause symbol
                            "method": "animate",
                        },
                    ],
                    "direction": "left",
                    "pad": {"r": 10, "t": 70},
                    "type": "buttons",
                    "x": 0.1,
                    "y": 0,
                }
             ],
             sliders=sliders
    )

    if plot:
        fig.show()
        offline_plot(fig)

    return fig