"""
Author: Christophe Foyer

Description:
    This is an example script demonstrating hybrid 2D to 3D optimization on an 
    artificial dataset.
"""


# %% Imports
import numpy as np
from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.hybrid_dimension_snakes import HybridSnakes

# Import the data
file = "testfiles/aortic_cross_section.gif"
imseq = ImageSequence(file)
imseq.change_resolution((100, 100, 100))

# Set input spline geometry
x = [50, 50, 50]
y = [50, 50, 50]
z = np.linspace(0, 99, 3)
array = np.stack([x, y, z])

# Create a simple hybrid solver
hs = HybridSnakes(
    imseq, array,                  
    plane_extent = (100, 100),
    plane_shape = (100, 100),
    n=6,
    snake_type = 'active_contour',
    active_contour_options = {
        "alpha": -0.015,
        "beta": 100,
        "gamma": 1E-2,
        "w_edge": 1E6,
        "max_iterations": 10000,
        "boundary_condition": "periodic",
        "convergence": 0.01,
    }
    )

snakes = hs.optimize_slices()
hs.plot_snake(2)
hs.plot()

# Create a bspline hybrid solver
hs_bspl = HybridSnakes(imseq, array,                  
                  plane_extent = (100, 100),
                  plane_shape = (100, 100),
                  n=6,
                  snake_type = 'bspline'
                  )

snakes = hs_bspl.optimize_slices()
hs_bspl.plot_snake(2)
hs_bspl.plot()