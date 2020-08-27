"""
Author: Christophe Foyer

Description:
    This script extends and redefines the base NURBS-Python surface class to be 
    locked to degree 2 and forces a periodic constraint similar to that found
    in scipy's definition of a BSpline curve, on one or both of the principle 
    direction of the surface.
"""

from geomdl.NURBS import Surface
from nurbs_active_contour.geometry.evaluators import RQSurfEvaluator
import numpy as np

class RationalQuadraticSurface(Surface):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        ## Evaluator not implemented
        # self.evaluator = RQSurfEvaluator()
        
        self._degree = [2, 2]
        # self.closed_u = kwargs.get('closed_u', False)
        # self.closed_v = kwargs.get('closed_v', False)

    @property
    def degree(self):
        return self._degree

    @degree.setter
    def degree(self, value):
        print("Degrees are locked to 2.")
        
    @property
    def degree_u(self):
        return self._degree[0]

    @degree_u.setter
    def degree_u(self, value):
        print("Degree is locked to 2.")
        
    @property
    def degree_v(self):
        return self._degree[1]

    @degree_v.setter
    def degree_v(self, value):
        print("Degree is locked to 2.")
        
    closed_u = None
    closed_v = None
        
    def set_ctrlpts(self, value, n_u=None, n_v=None, gen_knots=False,
                    closed_u=False, closed_v=False, **kwargs):
        """
        Format for surfaces is:
            ([x, y, z, w], n_u, n_v)
        n_u and n_v are optional if the list is a shaped numpy array
        """
        ctrlpts = np.array(value)
        ctrlpts_np = ctrlpts
        
        if n_u and n_v:
            # Trying to fix refining support
            ctrlpts_np = ctrlpts_np.reshape(n_u, n_v, 4)
        
        if self.closed_u == None:
            self.closed_u = closed_u
        elif self.closed_u and not closed_u:
            ctrlpts_np = ctrlpts_np[1:-1, :]
        
        if self.closed_v == None:
            self.closed_v = closed_v
        elif self.closed_v and not closed_v:
            # a bit hackish
            ctrlpts_np = ctrlpts_np[:, 1:-1]
        
        # Get the controlpoints
        self.control_points_np = ctrlpts_np

        # Stitch for each direction if needed
        if closed_u:
            # Stitch it
            stitch = np.mean([ctrlpts[0,:],ctrlpts[-1,:]], axis=0)
            ctrlpts = np.concatenate([stitch.reshape(1, -1, 4),
                                      ctrlpts,
                                      stitch.reshape(1, -1, 4)],
                                     axis=0)
        
        if closed_v:
            # Stitch it
            stitch = np.mean([ctrlpts[:,0],ctrlpts[:,-1]], axis=0)
            ctrlpts = np.concatenate([stitch.reshape(-1, 1, 4),
                                      ctrlpts,
                                      stitch.reshape(-1, 1, 4)],
                                     axis=1)
        
        if n_u == None:
            n_u = ctrlpts.shape[0]
        if n_v == None:
            n_v = ctrlpts.shape[1]
            
        # Make it fit the python-nurbs format
        ctrlpts = ctrlpts.reshape((-1, 4)).tolist()

        super().set_ctrlpts(ctrlpts, n_u, n_v, **kwargs)
        
        # generates uniform knotvectors
        if gen_knots:
            self.knotvector_u = [0, 0,
                                 *np.linspace(0, 1, self.data['size'][0]-1),
                                 1, 1]
            self.knotvector_v = [0, 0,
                                 *np.linspace(0, 1, self.data['size'][1]-1),
                                 1, 1]
