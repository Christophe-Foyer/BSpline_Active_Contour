"""
Author: Christophe Foyer

Description:
    This is an example for 3D gradient vector flow.
"""


# %% Imports
from nurbs_active_contour.utils.GVF import GVF3D
from nurbs_active_contour.utils.image import ImageSequence
import pyvista as pv
import numpy as np


file = "./testfiles/aortic_cross_section.gif"
imseq = ImageSequence(file)
imseq = imseq.gradient()
imseq.change_resolution((100, 100, 100))

[u,v,w] = GVF3D(imseq.array, 0.02, 1000, verbose=True)

# %% Plot vectors
grid = pv.UniformGrid()
grid.dimensions = imseq.array.shape
grid.point_arrays["values"] = imseq.array.flatten(order="F")
grid['vectors'] = np.stack([u,v,w]).reshape(3, -1).T

arrows = grid.glyph(orient='vectors', scale=True, factor=1E-1,)

# Display the arrows
plotter = pv.BackgroundPlotter()

# plotter.add_mesh(grid, cmap="bone", opacity=0.5)

plotter.add_mesh(arrows, cmap="YlOrRd")

# plotter.add_mesh_clip_plane(grid)

plotter.show_grid()
plotter.show()

# %% Plot xyz components
u_seq = ImageSequence(array=u)
v_seq = ImageSequence(array=v) 
w_seq = ImageSequence(array=w)

u_seq.plot(plane_widget=True)
v_seq.plot(plane_widget=True)
w_seq.plot(plane_widget=True)