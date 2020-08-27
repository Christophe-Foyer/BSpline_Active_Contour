"""
Author: Christophe Foyer

Description:
    A mildly messy script that compares performance between scipy active
    contours and my implementation using bsplines
"""

import numpy as np

import matplotlib.pyplot as plt

from skimage.color import rgb2gray
from skimage.segmentation import active_contour

from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.solver2d import Solver2D
from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve

import scipy.interpolate as interpolate

import time

# %% functions
class Timer:    
    def __init__(self, name=None):
        self.name = name
    
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.interval = self.end - self.start
        print(
                "Runtime:  " + str(self.interval) + " seconds"
                + (" for " + self.name if self.name
                   else "")
              )
        
def _interpolate_bspline(init, return_spline=True):
        """
        Interpolate a BSpline from a set of points
        """
        k=2
        kv = np.linspace(0, 1, num_ctrlpts+2*k)
        
        tck, u = interpolate.splprep([init[:, 1], init[:, 0]], 
                                     task=-1, t=kv, s=0, k=k, per=1)
            
        t, c, k = tck
        c = np.array(c).T
        
        if return_spline:
            spline = interpolate.BSpline(t, c, k, extrapolate='periodic')
            return spline
        else:
            return t, c, k

# %%
file = 'testfiles/aortic_cross_section.gif'
imseq = ImageSequence(file)

img = np.array(imseq.return_image(0))
img = rgb2gray(img)

# %% Parameters
maxiter = 40
num_ctrlpts = 20  # for easy comparison, we convert trad to bspline

# Something to have the same basis
s = np.linspace(0, 2*np.pi, 200)
r = 50 + 20*np.sin(s)
c = 50 + 20*np.cos(s)
init = np.array([r, c]).T

# %% Traditional

with Timer('trad') as t:
    init2 = init
    # DIY callback loop
    ctrlpts_trad = []
    for iternum in range(maxiter):
        snake = active_contour(img,
                               init2, alpha=-0.015, beta=10, gamma=0.001,
                               w_edge=1E6, 
                               max_iterations=1,  # external iteration control
                               coordinates='rc')
        init2 = snake
        
        #Convert to spline
        t, cpts, k = _interpolate_bspline(snake, return_spline=False)
        ctrlpts_trad.append(cpts)

spline = interpolate.BSpline(t, cpts, k, extrapolate='periodic')
xx = np.linspace(0, 1, 1000)
out = spline(xx)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
plt.plot(r, c, '--r')
plt.plot(cpts[:, 0], cpts[:, 1], '--g')
plt.plot(out[:, 0], out[:, 1])
plt.title('Scikit')

# %% Bspline

t, cpts, k = _interpolate_bspline(init, False)
            
curve = ClosedRationalQuadraticCurve()
curve.ctrlpts = cpts
curve.knotvector = t

# Set evaluation delta
curve.delta = 0.01
curve.evaluate()

bsnake_opts = {'maxiter': 50, 'disp': True, 'gtol': 1E-2}
bsnake_opts_tol = 1E-5

with Timer('gvf') as t:
    solver = Solver2D(curve, img, method="CG", Lambda=[0, 0])

with Timer('bspline') as t:
    curve_sol = solver.optimize(
                        options=bsnake_opts,
                        tol=bsnake_opts_tol,)
cpts = np.array(curve_sol.ctrlpts)[:, :2]
snake = np.array(curve_sol.evalpts)

spline = _interpolate_bspline(snake)
xx = np.linspace(0, 1, 1000)
out = spline(xx)
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(img, cmap=plt.cm.gray)
plt.plot(r, c, '--r')
plt.plot(cpts[:, 0], cpts[:, 1], '--g')
plt.plot(out[:, 1], out[:, 0])
plt.title('bspline')

# %% Compare costs

def plot_costs(cpts_list):
    plt.figure()
    
    costs = []
    for cpts in cpts_list:
        costs.append(solver.cost_function(np.array(cpts.flatten())))
    
    costs = np.array(costs)
    iters = np.linspace(1, len(costs), len(costs))
    plt.plot(iters, costs)
    plt.title("Cost vs. iterations (scikit) - Min = " 
              + str(np.round(min(costs), 3)))
    plt.xlabel("iteration")
    plt.ylabel("Cost")
    
plot_costs(ctrlpts_trad)

solver.plot_residual()
plt.title("Cost vs. iterations (bspline) - Min = " 
          + str(np.round(min(solver.cost_list), 3)))
