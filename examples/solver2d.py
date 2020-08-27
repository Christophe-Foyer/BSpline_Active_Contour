"""
Author: Christophe Foyer

Description:
    This is an example script demonstrating 2D BSpline Active Contours
"""


# %% Imports
import numpy as np

from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve
from nurbs_active_contour.utils.utilities import generate_knots
from nurbs_active_contour.optimize.solver2d import Solver2D
from nurbs_active_contour.utils.image import ImageSequence

from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap as cm
from matplotlib import cm as cm2

import pyvista as pv

# %% Solve a 2D problem

# Import an image
file = "../examples/testfiles/aortic_cross_section.gif"
image = np.array(ImageSequence(file).return_image(1))

# Create a B-Spline curve instance
curve = ClosedRationalQuadraticCurve()

# Add some control points
points = (np.array([[0, 0], [0, 0.5],
                    [0, 1], [0.5, 1],
                    [1, 1], [1, 0.5],
                    [1, 0], [0.5, 0]]) * 0.5 + 0.25) \
    * (np.sqrt(image.size) - np.array([1, 1]))

curve.ctrlpts = points

# Auto-generate knot vector
curve.knotvector = generate_knots(curve)

# Set evaluation delta
curve.delta = 0.01

# Evaluate curve
curve.evaluate()

# Create solver instance
solver = Solver2D(curve, image, method="CG", Lambda=[1E-2, 1E-5])

curve_sol = solver.optimize(
        # bounds=np.tile([0, 99], (np.array(points).size, 1)),
        options={'maxiter': 100, 'disp': True, 'gtol': 1E-3},
        tol=1E-5
        )

# %% Plots


# Plot the control point polygon and the evaluated curve
inputcurve = np.array(curve.evalpts)
inputctrlpts = np.array(curve.ctrlpts)

solution = np.array(curve_sol.evalpts)
solutionctrlpts = np.array(curve_sol.ctrlpts)

# 2D Plot
plt.figure()
plt.imshow(image)
plt.colorbar()
pts = np.round(curve.evalpts).astype(int)
plt.scatter(inputcurve[:, 0], inputcurve[:, 1], s=20,
            c=solver.image[pts[:, 0], pts[:, 1]], cmap='RdPu')
plt.scatter(inputctrlpts[:, 0], inputctrlpts[:, 1], c='r')
plt.plot(inputcurve[:, 0], inputcurve[:, 1], '--r', label="Input curve")

pts = np.round(curve_sol.evalpts).astype(int)
plt.scatter(solution[:, 0], solution[:, 1], s=20,
            c=solver.image[pts[:, 0], pts[:, 1]], cmap='coolwarm')
plt.scatter(solutionctrlpts[:, 0], solutionctrlpts[:, 1], c='b')
plt.plot(solution[:, 0], solution[:, 1], 'b', label="Output curve")
plt.legend()

# 3D plot
grad = solver.gradient
grad_idx = np.indices(grad.shape)

ic_z = solver.interpolate_gradients(inputcurve) + 0.01
ic_ctrlpts_z = solver.interpolate_gradients(inputctrlpts[:, :2])

sol_z = solver.interpolate_gradients(solution) + 0.01
sol_ctrlpts_z = solver.interpolate_gradients(solutionctrlpts[:, :2])

X, Y, Z = (*grad_idx, grad)

def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int_)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int_)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int_)
    poly.lines = cells
    return poly

zscale = 200

line_ic = lines_from_points(np.stack([inputcurve[:, 0],
                                      inputcurve[:, 1],
                                      ic_z.flatten()*zscale]).T)

line_sol = lines_from_points(np.stack([solution[:, 0],
                                       solution[:, 1],
                                       sol_z.flatten()*zscale]).T)
grid = pv.StructuredGrid(X, Y, Z*zscale)
grid['values'] = Z.flatten()

plotter = pv.BackgroundPlotter()
plotter.add_mesh(line_ic, line_width=10, color="r")
plotter.add_mesh(line_sol, line_width=10, color="b")
plotter.add_mesh(grid, scalars='values',
                 cmap=plt.cm.get_cmap("viridis"))
