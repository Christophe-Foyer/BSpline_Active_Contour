"""
Author: Christophe Foyer

Description:
    This is an example script demonstrating hybrid 2D to 3D optimization on a 
    real dataset.
"""


# %% Imports
import numpy as np
from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.hybrid_dimension_snakes import HybridSnakes

# Import the data
file = "testfiles/spline_tube_dataset_noisy.gif"
imseq = ImageSequence(file)
imseq.change_resolution((200, 200, 200))

# Set input spline geometry
y = [30, 50, 60, 90]
x = [30, 60, 50, 40]
z = [0, 30, 80, 40]
array = np.stack([x, y, z])*2

# Options
kwargs = {
    'plane_extent':       (100, 100),
    'plane_shape':        (100, 100),
    'n':                  10,
    'init_spline_size':   10
    }

# Create a simple hybrid solver
hs = HybridSnakes(imseq, array,
                  snake_type = 'active_contour',
                  **kwargs
                  )

snakes = hs.optimize_slices()
hs.plot_snake(2)
hs.plot(plane_widget=True)

# Create a bspline hybrid solver
hs_bspl = HybridSnakes(imseq, array,
                  snake_type = 'bspline',
                  **kwargs
                  )

snakes = hs_bspl.optimize_slices()
hs_bspl.plot_snake(2)
hs_bspl.plot(plane_widget=True)
