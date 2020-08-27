"""
Author: Christophe Foyer

Description:
    This is an example script demonstrating 3D optimization of pre-optimized
    BSpline Surfaces using a hybrid 2D to 3D solver on a real dataset.
"""


# %% Imports
import numpy as np
from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.hybrid_dimension_snakes import HybridSnakes
from nurbs_active_contour.optimize.solver3d import Solver3D

# Import the data
file = "testfiles/aortic_cross_section.gif"
imseq = ImageSequence(file)
imseq.change_resolution((100, 100, 100))

# Set input spline geometry
x = [50, 50, 50]
y = [50, 50, 50]
z = np.linspace(0, 99, 3)
array = np.stack([x, y, z])

# %% Hybrid bspline snake

# Create a bspline hybrid solver
hs_bspl = HybridSnakes(imseq, array,                  
                  plane_extent = (100, 100),
                  plane_shape = (100, 100),
                  n=6,
                  snake_type = 'bspline'
                  )

# %% First optimize coarsly along slices
snakes = hs_bspl.optimize_slices()
surf = hs_bspl._build_surface()

# %% Now optimize a bit more
solver = Solver3D(surf, imseq, Lambda=[1E-5, 1E-5], GVF_iter=1000)

solver.optimize(options={'maxiter': 20})

solver.plot(plot_gradient=True)
