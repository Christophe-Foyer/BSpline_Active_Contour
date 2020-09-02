"""
author: Christophe Foyer

Description:
    An example of the unfinished force analog model running through a dataset
"""

# %% Imports
from nurbs_active_contour.geometry.nurbs_presets import Cylinder
from nurbs_active_contour.optimize.dev.force_model_solver import \
    ForceModelSolver
from nurbs_active_contour.utils.image import ImageSequence

# %% Setup
# surf = Cylinder([5, 5, 2], offsets = [60, 50, -1], num_z=5)
surf = Cylinder([5, 5, 2], offsets = [60, 50, -1], num_z=5)
surf.delta = (0.1, 0.02)

file = "testfiles/aortic_cross_section.gif"
imseq = ImageSequence(file)

imseq.change_resolution((100, 100, 10))

# %% Solver
solver = ForceModelSolver(surf, imseq, 
                   GVF_iter=20,
                   gradient_multipliers=[-100, -100, -100],
                   elasticity=[1, 1, 0],
                   uv_elastic=[1E-5, 1E-5],
                    # max_elastic_force=[1, 1, 1],
                   lock_v_ends=[False, False, True],
                   # normalize_elastic=False
                   )

# %% Plot
solver.plot(plot_forces=True, scale_vectors=True,
            scale_factor=10, forces='gradient',
            plane_widget=True,
            plot_image=False)

solver.plot(plot_forces=True, scale_vectors=True,
            scale_factor=1, forces='elastic',
            plot_image=False)

# solver.plot(plot_forces=True, scale_vectors=True,
#             scale_factor=1, forces='all',
#             plot_image=False)

solver.plot(plot_forces=True, scale_vectors=True,
            scale_factor=1, forces='ctrlpts',
            plot_image=False)

# %% Optimize
# solver.plot(plot_initial=True,
#             plane_widget=True,
#             plot_forces=True, scale_factor=10,)

solver.optimize(num_iter=int(1E2), step_size=0.01, verbose=True)

solver.plot(plot_initial=True,
            plane_widget=False,
            plot_forces=False, scale_factor=1,)

# %%
solver.optimize(num_iter=int(1E3), step_size=0.1, verbose=True)

solver.plot(plot_initial=True,
            plane_widget=False,
            plot_forces=True, scale_factor=0.1,)
