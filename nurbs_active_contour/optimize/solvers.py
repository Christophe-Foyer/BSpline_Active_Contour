"""
Created on Fri Mar 20 11:03:30 2020

Author: Christophe Foyer
    
Description:
    This file introduces base solver classes and tools for these classes.
    It also contains a crude attempt at multiresolution analysis.
"""

import numpy as np

from scipy.optimize import minimize
from scipy.interpolate import griddata
from scipy.interpolate import RegularGridInterpolator
from scipy import ndimage

from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve

from nurbs_active_contour.utils.image import ImageSequence
from numpy import diff

import matplotlib.pyplot as plt
import pyvista as pv

from inspect import signature

# %% Defaults
_default_method = "CG"


# %% Classes

class CallbackList(list):
    """
    A class for scipy optimize callbacks based on the built-in list class.
    """
    def callback(self, x):
        self.append(x.tolist())


class Function:
    """
    A function class that stores kwargs as attributes.
    Used here to mimic MATLAB struct behavior.
    """
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class GeneralSolver:
    """
    A parent class used for different solver implementations.
    """

    iter_list = []    

    def __init__(self, cost_function, method=_default_method):

        assert len(signature(cost_function).parameters) == 1, \
            "Cost function should only take 1 argument (list of parameters)"

        self.cost_function = cost_function
        self.method = method
        
    def cost_function(self, x):
        # Placeholder function
        return 0

    def optimize(self, x0, **kwargs):
        
        if kwargs.get('save_steps', True):
            self.iter_list = CallbackList([])
            kwargs['callback'] = self.iter_list.callback
            
        self.method = kwargs.pop('method', self.method)

        # This was originally meant to facilitate using autograd
        F = Function(f=self.cost_function)

        self.F = F

        return minimize(F.f, x0,
                        method=self.method,
                        **kwargs)

    def plot_residual(self):
        assert len(self.iter_list) > 0, "No data to plot."
        
        plt.figure()
        
        if hasattr(self, 'cost_list'):
                costs = self.cost_list
        else:
            costs = self.calculate_costs()
        
        costs = np.array(costs)
        iters = np.linspace(1, len(costs), len(costs))
        plt.plot(iters, costs)
        plt.title("Cost vs. iterations")
        plt.xlabel("iteration")
        plt.ylabel("Cost")
        
    def calculate_costs(self):
        costs = []
        for iteration in self.iter_list:
            costs.append(self.cost_function(np.array(iteration)))
            
        self.cost_list = costs
        return costs                                 
    

class MultiRes_Solver3D_Wrapper():
    
    def __init__(self, init_surf, image, scaling=2, maxiter=3, **kwargs):
        """
        Multires solver3D init method.

        Parameters
        ----------
        image : nurbs_active_contour.utils.image.ImageSequence
            An image sequence of the data to fit to.
        initcurve : nurbs_active_contour.geometry.surface.RationalQuadraticSurface
            Initial surface which will be modified to fit.
        scaling : float, optional
            DESCRIPTION. The default is 2.
        maxiter : int, optional
            DESCRIPTION. The default is 3.
        **kwargs : optional
            Additional arguments which will be passed on to the solver.

        Returns
        -------
        None.

        """
        assert isinstance(image, ImageSequence)
        
        self.image = image
        self.scaling=scaling
        self.init_surf = init_surf
        
        self.imagelist = [image]
        _, scaled = image.multiresolution(maxiter, scaling)
        for scaled_image in scaled:
            self.imagelist.append(scaled_image)
            
        #kwargs that will be passed to the solver
        self.init_kwargs = kwargs
    
    def fit(self, **kwargs):
        self.solvers = []
        self.surfaces = []
        
        # Downscale surface
        surf = self._scale_surf(self.init_surf,
                                1/self.scaling**(len(self.imagelist)-1))
        self.surfaces.append(surf)
        
        for i in range(len(self.imagelist)):
            image = self.imagelist[-(i+1)]
    
            solver = Solver3D(surf, image, **self.init_kwargs)
            self.solvers.append(solver)
            
            solver.optimize(**kwargs)
            
            outputsurf = solver.lastsurf
            
            self.surfaces.append(outputsurf)
            surf = self._scale_surf(outputsurf, self.scaling)
            
        return outputsurf
            
    def _scale_surf(self, surf, scaling):
        ctrlpts = np.array(surf.ctrlpts)
        
        # scale surface on x/y
        ctrlpts = ctrlpts*np.array([scaling,scaling,1])
        
        # Rebuild surface
        ctrlpts = np.hstack([ctrlpts, np.ones((ctrlpts.shape[0], 1))])
        surf_out = RationalQuadraticSurface()
        surf_out.delta = surf.delta
        surf_out.set_ctrlpts(ctrlpts, *surf.data['size'],
                             closed_u=surf.closed_u, 
                             closed_v=surf.closed_v)
        surf_out.knotvector_u = surf.knotvector_u
        surf_out.knotvector_v = surf.knotvector_v
        
        return surf_out
        
    def plot_residual(self):
        costs = []
        assert hasattr(self, 'solvers')
        
        cuts = [0]
        
        for solver in self.solvers:
            if hasattr(solver, 'cost_list'):
                cost = solver.cost_list
            else:
                cost = solver.calculate_costs()
                
            costs = np.hstack([costs, cost])
            cuts.append(len(costs))
            
        plt.figure()
        plt.plot(np.array(costs))
        for cut in cuts:
            plt.axvline(x=cut, color='red')
        plt.title("Cost vs. iterations")
        plt.xlabel("iteration")
        plt.ylabel("Cost")
                