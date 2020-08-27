"""
Author: Christophe Foyer
    
Description:
    This file introduces a solver class for fitting curves to geometries found
    in  2D images.
"""

import numpy as np

from skimage.filters import gaussian

from scipy.interpolate import RegularGridInterpolator

from nurbs_active_contour.optimize.solvers import GeneralSolver, CallbackList, \
    _default_method
from nurbs_active_contour.utils.GVF.GVF2D import GVF
from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve
from nurbs_active_contour.utils.utilities import generate_knots

# %% solver 2D

class Solver2D(GeneralSolver):
    """
    A solver for the 2D BSpline Snakes problem.
    """
    
    # parameters
    gaussian_smoothing = 3
    GVF_iter = 1000
    GVF_mu = 0.001
    
    fill_value = 0

    def __init__(self, curve, image, method=_default_method,
                 interpolation='linear', Lambda=[1, 1]):
        super().__init__(self.cost_function, method=method)

        self.init_curve = curve
        self.image = np.array(image)

        self._interpolation_type = interpolation
        
        assert len(Lambda) == 2
        self.Lambda = Lambda

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, value):
        self._image = value
        self._generate_gradient()
        self._generate_interpolated()
        
    def _generate_gradient(self):
        image = np.array(self.image)
        
        image = gaussian(image, self.gaussian_smoothing)
        
        out = GVF(image, self.GVF_mu, self.GVF_iter)
        
        self.gradient = -np.linalg.norm(out, axis=0)

    def _generate_interpolated(self):
        image = self.gradient

        values = image
        points = [np.linspace(0, ind-1, ind) for ind in values.shape]

        self.interpolate_gradients = \
            RegularGridInterpolator(points, values,
                                    bounds_error=False,
                                    fill_value=self.fill_value)
            
    def cost_function(self, x):
        """
        Evaluate the cost based on the formula 2 in B-spline snakes
        """
        
        if type(x) == np.ndarray:
            points = np.array(x).reshape((-1, 2))
        else:
            points = np.array(x._value).reshape((-1, 2))
        points = np.vstack([points, points[:2, :]])
        
        
        # WORKS WITH AUTOGRAD and quite a bit faster
        evalpts = self.init_curve.evaluator.function(points)

        # Calculate the cost based on the image (or rather image derivative)
        # brightness
        cost = np.sum(self.interpolate_gradients(evalpts))
        
        # This is a numerical approximation, for better performance/accuracy
        # make the code compatible with automatic differentiation
        cost += self.Lambda[0] * np.sum(np.diff(evalpts, n=1, axis=0)**2)
        cost += self.Lambda[1] * np.sum(np.diff(evalpts, n=2, axis=0)**2)
        # print(cost)

        return cost

    def optimize(self, save_steps=True, **kwargs):

        ctrlpts = np.array(self.init_curve.ctrlpts)[:-2, :2]
        x0 = ctrlpts.flatten()
        self.ctrlpts_shape = ctrlpts.shape

        if save_steps:
            self.iter_list = CallbackList([])
            
            def callback():
                self.iter_list.callback()
                print('\rMinimization iteration: ' + str(len(self.iter_list)),
                      end='', flush=True)
            
            kwargs['callback'] = callback
        
        print("Optimizing snake...")
        parameters = super().optimize(x0, **kwargs)
        
        self.optimization_output = parameters
        
        curve_sol = ClosedRationalQuadraticCurve()
        curve_sol.ctrlpts = parameters.x.reshape((-1, 2))
        curve_sol.knotvector = generate_knots(curve_sol)
        curve_sol.delta = 0.01
        curve_sol.evaluate()

        return curve_sol