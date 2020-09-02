"""
Author: Christophe Foyer
    
Description:
    This file introduces a solver class for fitting cylinders to geometries
    found in a set of images.
    The solver solves a sequence of 2D problems and stitches the output back
    together as a BSpline surface.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyvista as pv

import vg
from pytransform3d.rotations import matrix_from_axis_angle

from skimage.segmentation import active_contour
from skimage.filters import gaussian

import scipy.interpolate as interpolate
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import splprep, splev
from scipy.spatial.transform import Rotation as Rot

from nurbs_active_contour.utils.image import ImageSequence
from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve
from nurbs_active_contour.geometry.surface import RationalQuadraticSurface
from nurbs_active_contour.utils.plotting import makemesh
from nurbs_active_contour.optimize.solver2d import Solver2D

import warnings

# %% Hybrid Snake

class HybridSnakes():
    
    ### Parameters
    # These can be accessed by passing through the init or changing
    # after initialization
    
    verbose=True
    
    # interpolation
    fill_value = 0
    
    # slicing
    plane_shape = (100, 100)
    plane_extent = (100, 100)
    
    # input snake geometry
    init_spline_size = 20
    num_points = 50
    
    # snake type
    snake_type = 'active_contour'
    gaussian_smoothing = 3
    
    # active contour parameters
    active_contour_options = {
        "alpha": -0.015,
        "beta": 100,
        "gamma": 1E-2,
        "w_edge": 1E6,
        "max_iterations": 10000,
        "boundary_condition": "periodic",
        "convergence": 0.01,
    }
    
    # BSpline Snakes
    bsnake_opts = {'maxiter': 50, 'disp': True, 'gtol': 1E-2}
    bsnake_opts_tol = 1E-5
    bsnake_num_ctrlpts = 10
    bsnake_lambda = [1E-2, 1E-5]
    
    ### Init
    
    def __init__(self, image, *args, n=10, **kwargs):
        # Set any parameters to object attributes (makes changing above easy)
        for key, val in kwargs.items():
            setattr(self, key, val)
            
        assert len(args) == 3 or \
            (len(args)==1 and type(args[0])==np.ndarray 
             and 3 in args[0].shape and len(args[0].shape)==2), \
                "Inputs should be x, y, z or single 2D xyz ndarray"
            
        # convert arguments to x,y,z
        if len(args) == 3:
            x, y, z = args
        elif len(args) == 1:
            if args[0].shape[0] == 3:
                x, y, z = args[0] 
            elif args[0].shape[1] == 3:
                x, y, z = args[0].T
        
        self.spline_coords = (x, y, z)
        
        assert isinstance(image, ImageSequence)
        self.image_sequence = image
        
        self.spline = self._make_spline(x, y, z, n, **kwargs)
       
    ### Properties
    
    @property
    def image_sequence(self):
        return self._image_sequence

    @image_sequence.setter
    def image_sequence(self, value):
        self._image_sequence = value  # Set value
        
        self._generate_interpolated()
        
    ### Private methods
    
    def _make_spline(self, x, y, z, n, spline_degree=2, **kwargs):
        """
        Builds the spline from simple coordinates and number of evalpoints.
        """
        tck, u = splprep([x, y, z], s=0, k=spline_degree)
        
        evalpts = np.linspace(0, 1, n)
        
        pts = np.array(splev(evalpts, tck))
        der = np.array(splev(evalpts, tck, der=1))
        
        return {"pts": pts, "der": der, "n": n}
        
    def _generate_interpolated(self):
        """
        Generate the image interpolation handle.
        """
        
        values = self.image_sequence.array
        points = [np.linspace(0, ind-1, ind) for ind in values.shape]
        
        # Assume no gradient outside bounds (fill value)
        interpolator = RegularGridInterpolator(points, values,
                                               bounds_error=False,
                                               fill_value=self.fill_value
                                               )
        
        self.interpolator = interpolator
    
    def _planes(self):
        """
        Generates planes from HybridSolver parameters at each evalpoint of the
        input spline. These are for slicing for the 2D problem.
        """
        
        xrange, yrange = self.plane_extent
        nx, ny = self.plane_shape
        
        x = np.linspace(xrange/2, -xrange/2, nx)
        y = np.linspace(yrange/2, -yrange/2, ny)
        mesh = np.array(np.meshgrid(x, y))
        mesh = mesh.reshape(2, -1).T
        mesh = np.concatenate([mesh, np.zeros((mesh.shape[0], 1))], axis=1)
        
        pts = self.spline['pts']
        der = self.spline['der']
        n = self.spline['n']
        
        points = []
        for i in range(n):
            points_slice = self._rotmat(der[:, i], mesh)
            points_slice = points_slice + pts[:, i]
            points.append(points_slice)
            
        points = np.stack(points)
        
        return points
    
    def _rotmat(self, vector, points):
        """
        Rotates a 3xn array of 3D coordinates from the +z normal to an
        arbitrary new normal vector.
        """
        
        vector = vg.normalize(vector)
        axis = vg.perpendicular(vg.basis.z, vector)
        angle = vg.angle(vg.basis.z, vector, units='rad')
        
        a = np.hstack((axis, (angle,)))
        R = matrix_from_axis_angle(a)
        
        r = Rot.from_matrix(R)
        rotmat = r.apply(points)
        
        return rotmat
    
    def _flat_to_plane(self, points, i):
        """
        Converts a 2D plane to a 3D plane whose origin and normal vector
        are the same as the i'th evaluated point of the input spline.
        """
        
        points = np.array(points)
        points = points - np.array(self.plane_extent)/2
        
        points = np.concatenate([points,
            np.zeros((points.shape[0], 1))], axis=1)
        
        points = self._rotmat(self.spline['der'][:, i], points)
        points = points + self.spline['pts'][:, i]
        
        return points
    
    def _interpolate_planes(self):
        """
        Interpolate slices cutting through the hybridsolver's image_sequence
        data.
        """
        
        planes = self._planes()
        nx, ny = self.plane_shape
        
        interpolated = []
        for plane in planes:
            p = plane.reshape((*(nx, ny), -1))
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                interpolated.append(self.interpolator(p))
            
        self.interpolated_planes = interpolated
            
        return interpolated
    
    def _interpolate_bspline(self, init, return_spline=True):
        """
        Interpolate a BSpline from a set of points
        """
        k=2
        kv = np.linspace(0, 1, self.bsnake_num_ctrlpts+2*k)
        
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            tck, u = interpolate.splprep([init[:, 1], init[:, 0]], 
                                         task=-1, t=kv, s=0, k=k, per=1)
            
        t, c, k = tck
        c = np.array(c).T
        
        if return_spline:
            spline = interpolate.BSpline(t, c, k, extrapolate='periodic')
            return spline
        else:
            return t, c, k
    
    def _2D_snake(self, image):
        """
        This method fits either scikit-image.active_contour or 
        nurbs_active_contour.optimize.solver2d.Solver2D to the input image.

        Parameters
        ----------
        image : nurbs_active_contour.utils.image.ImageSequence

        Returns
        -------
        dict
            Optimization input/output and solvers.

        """
        
        image = gaussian(image, self.gaussian_smoothing)
        
        s = np.linspace(0, 2*np.pi, self.num_points)
        r = image.shape[0]/2 + self.init_spline_size*np.sin(s)
        c = image.shape[1]/2 + self.init_spline_size*np.cos(s)
        init = np.array([r, c]).T
        
        curve_sol = None
        solver = None
        
        bspl_in = self._interpolate_bspline(init)
        
        if self.snake_type == 'active_contour':
            
            snake = active_contour(image,
                                   init,
                                   coordinates='rc',
                                   **self.active_contour_options,
                                   )
            
        elif self.snake_type == 'bspline':
            t, c, k = self._interpolate_bspline(init, False)
            
            curve = ClosedRationalQuadraticCurve()
            curve.ctrlpts = c
            curve.knotvector = t
            
            # Set evaluation delta
            curve.delta = 0.01
            curve.evaluate()
            
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                solver = Solver2D(curve, image, method="CG", 
                                  Lambda=self.bsnake_lambda)
                curve_sol = solver.optimize(
                    options=self.bsnake_opts,
                    tol=self.bsnake_opts_tol,
                    )
            snake = np.array(curve_sol.evalpts)
        else:
            raise Exception("Unrecognised snake type: " + self.snake_type)
        
        # This should probably not be reinterpolated for the BSpline solver
        bspl_out = self._interpolate_bspline(snake)
        
        return {'output': snake, 'init': init, 
                "curve": curve_sol, "solver": solver,
                "bspl_in": bspl_in, "bspl_out": bspl_out}
    
    def _build_surface(self, geometry='output'):
        """
        Builds a NURBS-Python surface from 

        Parameters
        ----------
        geometry : TYPE, optional
            Choose between 'init' and output 'geometry'. 
            The default is 'output'.

        Returns
        -------
        surf : nurbs_active_contour.geometry.surface.RationalQuadraticSurface

        """
        
        points = []
        for i in range(len(self.snakes)):
            if self.snake_type == 'active_contour' and geometry == 'output':
                pts = self.snakes[i]['bspl_out'].c
            elif geometry == 'output':
                pts = np.array(self.snakes[i]['curve'].ctrlpts)[:, :2]
            elif geometry == 'init':
                pts = self.snakes[i]['bspl_in'].c
            else:
                raise Exception('Geometry must be output/init, not ' 
                                + str(geometry))
            points_slice = self._flat_to_plane(pts, i)
            points.append(points_slice)
            
        points = np.stack(points)
        
        points = np.concatenate([points, np.ones((*points.shape[:2], 1))],
                                axis=2)
        
        surf = RationalQuadraticSurface()
        surf.set_ctrlpts(points,
                         closed_v = True,
                         gen_knots=True)
        return surf
            
    ### Public methods
    
    def plot_spline(self):
        x, y, z = self.spline_coords
        spline = self.spline
        pts = spline['pts']
        der = spline['der']
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.plot(pts[0], pts[1], pts[2])
        ax.quiver(*pts, *der, length=0.05)
        ax.scatter(x, y, z)
        
        planes = self._planes()
        ax.scatter(planes[:, :, 0], planes[:, :, 1], planes[:, :, 2])
    
    def plot_snake(self, idx):
        assert idx < len(self.interpolated_planes)
        
        img = self.interpolated_planes[idx]
        snake = self.snakes[idx]['output']
        init = self.snakes[idx]['init']
        
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.imshow(img, cmap=plt.cm.gray)
        ax.plot(init[:, 1], init[:, 0], '--r', lw=3)
        ax.plot(snake[:, 1], snake[:, 0], '-b', lw=3)
        ax.set_xticks([]), ax.set_yticks([])
        ax.axis([0, img.shape[1], img.shape[0], 0])
        
        plt.show()
        
    def plot(self, 
             plot_gradient=False,
             plot_image=True, 
             plotter=None,
             plot_snakes=True,
             plot_init=False,
             plane_widget=False,
             plot_surface=True,
             plot_init_surf=True,):
        
        if not plot_gradient:
            gridvalues = self.image_sequence.array
        else:
            gradients = [self.snakes[i]['solver'].gradient
                         for i in range(len(self.snakes))]
            gridvalues = np.stack(gradients, axis=-1)
        
        if not plotter:
            plotter = pv.BackgroundPlotter()
        
        if plot_image:
            image = pv.UniformGrid()
            image.dimensions = gridvalues.shape
            image.point_arrays["values"] = gridvalues.flatten(order="F")
            plotter.add_mesh(image, cmap="bone", opacity=0.5)
            
        if plot_snakes:
            for i, snake in enumerate(self.snakes):
                points = np.stack([snake['output'][:, 1],
                                   snake['output'][:, 0]],
                                  axis=-1)
                points = self._flat_to_plane(points, i)
                point_cloud = pv.PolyData(points)
                plotter.add_mesh(point_cloud)
                
                if plot_init:
                    points = self._flat_to_plane(snake['init'], i)
                    point_cloud = pv.PolyData(points)
                    plotter.add_mesh(point_cloud)
                    
        if plane_widget and plot_image:
            plotter.add_mesh_clip_plane(image)
            
        if plot_surface:
            surf = self._build_surface()
            plotter.add_mesh(makemesh(surf), color='blue')
            plotter.add_mesh(makemesh(surf, 'PolyData'), color='blue')
            plotter.add_mesh(makemesh(surf, 'PolyData', 'ctrlpts'),
                             color='cyan')
            
        if plot_init_surf:
            surf = self._build_surface('init')
            plotter.add_mesh(makemesh(surf), color='red')
            plotter.add_mesh(makemesh(surf, 'PolyData'), color='red')
            plotter.add_mesh(makemesh(surf, 'PolyData', 'ctrlpts'),
                             color='orange')
                    
        plotter.show_grid()
        plotter.show()

        return plotter
    
    def optimize_slices(self, return_surface=False):
        """
        Sequentially solve all the 2D problems we have set up.
        
        TODO: add non-verbose option
        """
        iplanes = self._interpolate_planes()
        
        # This could very easily be done in parallel:
        self.snakes = []
        for i, iplane in enumerate(iplanes): 
            if verbose:
                print('\rOptimizing snakes: ' + str(i + 1) + \
                      '/' + str(len(iplanes)),
                      end='', flush=True)
                    
            self.snakes.append(self._2D_snake(iplane))
        
        if verbose: 
            print('')
            
        self.surface = self._build_surface()
        
        if return_surface == True:
            return self.surface
        else:
            return self.snakes
            

# %% Main

if __name__ == "__main__":
    # x = [0, 1, 2, 3, 6]
    # y = [0, 2, 5, 6, 2]
    # z = [0, 3, 5, 7, 10]
    # array = np.stack([x, y, z])*5 + 20
    
    x = [50, 50, 50]
    y = [50, 50, 50]
    z = np.linspace(0, 99, 3)
    array = np.stack([x, y, z])
    
    file = "../../examples/testfiles/aortic_cross_section.gif"
    imseq = ImageSequence(file)
    imseq.change_resolution((100, 100, 100))
    
    hs_bspl = HybridSnakes(imseq, array,                  
                      plane_extent = (100, 100),
                      plane_shape = (100, 100),
                      n=6,
                      snake_type = 'bspline'
                      )
    
    snakes = hs_bspl.optimize_slices()
    hs_bspl.plot_snake(4)
    hs_bspl.plot()
    
    # Active contour
    hs = HybridSnakes(imseq, array,                  
                      plane_extent = (100, 100),
                      plane_shape = (100, 100),
                      n=6,
                      snake_type = 'active_contour'
                      )
    
    snakes = hs.optimize_slices()
    hs.plot_snake(4)
    hs.plot()
    
    hs.plot_spline()