"""
Author: Christophe Foyer
    
Description:
    This file introduces a solver class for fitting surface to geometries found
    in set of images describing a 3D space.
"""

import numpy as np

from scipy.interpolate import RegularGridInterpolator

import pyvista as pv

from nurbs_active_contour.geometry.surface import RationalQuadraticSurface
from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.optimize.solvers import GeneralSolver, \
    _default_method
from nurbs_active_contour.utils.GVF import GVF3D


# %% 3D solver
class Solver3D(GeneralSolver):
    """
    A solver for the 3D BSpline surfaces problem.
    """
    
    # Parameters:
    GVF_mu = 0.001
    GVF_iter = 100
    
    fill_value=0
    
    def __init__(self, surf, image_sequence, method=_default_method,
                 Lambda=1, verbose=True, **kwargs):
        super().__init__(self.cost_function, method=method)
        
        # Set any parameters to object attributes (makes changing above easy)
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        assert isinstance(image_sequence, ImageSequence), \
            "image_sequence is of type: " + str(type(image_sequence))
        
        self.verbose = verbose
        
        self.init_surf = surf
        self.image_sequence = image_sequence

        if ((type(Lambda) == list) or (type(Lambda) == tuple)):
            self.Lambda = Lambda
        else:
            self.Lambda = [Lambda, Lambda]
        
    @property
    def image_sequence(self):
        return self._image_sequence

    @image_sequence.setter
    def image_sequence(self, value):
        self._image_sequence = value
        
        self._generate_image_gradients()
        self._generate_interpolated()
        
    def _generate_image_gradients(self):
        """
        Generate image gradients
        """
        #self.image_gradients = self._image_sequence.gradient()
        out = GVF3D(self.image_sequence.array, self.GVF_mu, self.GVF_iter,
                    verbose = self.verbose)
        
        gradient = ImageSequence(array=-np.linalg.norm(out, axis=0))
        self.image_gradients = gradient
        
    def _generate_interpolated(self):
        """
        Generate the gradient interpolation handle.
        This gives the gradient function the necessary smoothness
        for the optimization to run (function is lipshitz continuous).
        """
        
        values = self.image_gradients.array
        points = [np.linspace(0, ind-1, ind) for ind in values.shape]
        
        # Assume no gradient outside bounds (fill value)
        interpolator = RegularGridInterpolator(points, values,
                                               bounds_error=False,
                                               fill_value=self.fill_value
                                               )
        
        self.interpolate_gradients = interpolator
    
    def cost_function(self, x):
        """
        Evaluate the cost based on the formula 2 in B-spline snakes
        """
        
        surf = RationalQuadraticSurface()
        self.surface = surf
        ctrlpts = np.array(x.data).reshape(self.ctrlpts_shape)
        surf.set_ctrlpts(ctrlpts,
                         closed_u=self.init_surf.closed_u, 
                         closed_v=self.init_surf.closed_v,
                         gen_knots=True)
        
        surf.delta = self.init_surf.delta
        
        evalpts = np.array(surf.evalpts)

        # Calculate the cost based on the image (or rather image derivative)
        # brightness
        cost = np.sum(self.interpolate_gradients(evalpts))
        
        evalpts_square = evalpts.reshape((*surf.data['sample_size'], -1))

        # Curvature:
        def deriv(axis, order=2):
            assert axis in [0, 1]
            pts = evalpts_square
            
            if axis == 0:
                pts = np.hstack([pts, 
                                 pts[:, :1].reshape(-1, 1, pts.shape[-1])])
            if axis == 1:
                pts = np.vstack([pts, 
                                 pts[:1, :].reshape(1, -1, pts.shape[-1])])
            
            k = np.diff(pts, n=2, axis=axis)
            k = np.linalg.norm(k, axis=2)
            
            return k
        
        cost += self.Lambda[0] * np.sum(deriv(axis=0)**2)
        cost += self.Lambda[1] * np.sum(deriv(axis=1)**2)
        
        cost += self.Lambda[0] * np.sum(deriv(0, 1)**2)
        cost += self.Lambda[1] * np.sum(deriv(1, 1)**2)

        return cost
    
    def optimize(self, **kwargs):
        
        self.costs = []

        ctrlpts = self.init_surf.control_points_np
        self.ctrlpts_shape = ctrlpts.shape
        x0 = ctrlpts.flatten()

        parameters = super().optimize(x0, **kwargs)
        self.optimization_output = parameters

        return parameters.x
    
    def plot(self, plane_widget=True, plot_gradient=False):
        if not plot_gradient:
            gridvalues = self.image_sequence.array
        else:
            gridvalues = self.image_gradients.array
        surf = self.init_surf
        outputsurf = self.surface
        image = pv.UniformGrid()
        image.dimensions = gridvalues.shape
        image.point_arrays["values"] = gridvalues.flatten(order="F")
        plotter = pv.BackgroundPlotter()
        plotter.add_mesh(image, cmap="bone", opacity=0.5)
        
        def makemesh(surf, meshtype="StructuredGrid", points='evalpts'):
            
            if points == 'evalpts':
                pts = np.array(surf.evalpts)
                pts_square = pts.reshape((*surf.data['sample_size'], -1))
            elif points == 'ctrlpts':
                pts = np.array(surf.ctrlpts)
                pts_square = pts.reshape((*surf.data['size'], -1))
            
            if meshtype.lower() == 'PolyData'.lower():
                mesh = pv.PolyData(pts_square)
            
            if meshtype.lower() == 'StructuredGrid'.lower():
                mesh = pv.StructuredGrid()
                mesh.points = pts
                mesh.dimensions = [*np.flip(surf.data['sample_size']),
                                   1]
            return mesh
        
        plotter.add_mesh(makemesh(surf), color='blue')
        plotter.add_mesh(makemesh(surf, 'PolyData'), color='blue')
        
        plotter.add_mesh(makemesh(outputsurf), color='red')
        plotter.add_mesh(makemesh(outputsurf, 'PolyData'), color='red')
        
        plotter.add_mesh(makemesh(surf, 'PolyData', 'ctrlpts'),
                         color='cyan')
        plotter.add_mesh(makemesh(outputsurf, 'PolyData', 'ctrlpts'),
                         color='orange')      

        if plane_widget:
            plotter.add_mesh_clip_plane(image)
            
        plotter.show_grid()
        plotter.show()

        return plotter 