"""
Author: Christophe Foyer

Description:
    This is an example script demonstrating 3D optimization of pre-optimized
    BSpline Surfaces using a hybrid 2D to 3D solver on an artificial dataset.
"""


# %% Imports
import numpy as np
from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.hybrid_dimension_snakes import HybridSnakes
from nurbs_active_contour.optimize.solver3d import Solver3D

# Import the data
file = "testfiles/spline_tube_dataset_noisy.gif"
imseq = ImageSequence(file)
imseq.change_resolution((200, 200, 200))

# Set input spline geometry
y = [30, 50, 60, 90]
x = [30, 60, 50, 40]
z = [0, 30, 80, 40]
array = np.stack([x, y, z])*2

# %% Hybrid Snake
# Options
kwargs = {
    'plane_extent':       (100, 100),
    'plane_shape':        (100, 100),
    'n':                  10,
    'init_spline_size':   10
    }

# Create a bspline hybrid solver
hs_bspl = HybridSnakes(imseq, array,
                  snake_type = 'bspline',
                  **kwargs
                  )

snakes = hs_bspl.optimize_slices()
# hs_bspl.plot_snake(2)
# hs_bspl.plot(plane_widget=True)
surf = hs_bspl._build_surface()

# %% Now optimize a bit more

# CG
solver_cg = Solver3D(surf, imseq, Lambda=[1E-5, 1E-5],
                  GVF_iter=100, GVF_mu=0.01,
                  method='CG')
solver_cg.optimize(options={'maxiter': 100})
solver_cg.plot(plot_gradient=True)
solver_cg.plot_residual()

# BFGS
solver_bfgs = Solver3D(surf, imseq, Lambda=[1E-5, 1E-5],
                  GVF_iter=100, GVF_mu=0.01,
                  method='BFGS')
solver_bfgs.optimize(options={'maxiter': 100})
solver_bfgs.plot(plot_gradient=True)
solver_bfgs.plot_residual()

# Nelder_Mead
solver_nm = Solver3D(surf, imseq, Lambda=[1E-5, 1E-5],
                  GVF_iter=100, GVF_mu=0.01,
                  method='Nelder_Mead')
solver_nm.optimize(options={'maxiter': 100})
solver_nm.plot(plot_gradient=True)
solver_nm.plot_residual()
