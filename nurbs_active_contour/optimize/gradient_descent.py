from nurbs_active_contour.utils.GVF import GVF3D
from nurbs_active_contour.utils.image import ImageSequence

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import copy

import pyvista as pv
from nurbs_active_contour.utils.plotting import makemesh

# %% Gradient Descent - Force Analog Model

class GD_solver():
    
    ### Default Parameters
    
    # Optimization values
    fill_value = 0      # fill value for interpolation
    num_iter = None     # number of iterations
    step_size = None    # step size
    
    # Gradient Vector Flow
    GVF_iter = 1000      # GVF algorithm default iter
    GVF_mu = 0.0001       # GVF algorithm default mu
    GVF_verbose=True    # GVF function verbosity
    
    # Physical analogs
    gradient_multipliers = [1, 1, 1]  # slope cartesian multiplier
    stiffness = [1, 1, 1]             # bending cartesian multiplier
    elasticity = [1, 1, 1]            # stretching cartesian multiplier
    
    # Force bounds - clips xyz components
    max_stiffness_force = [np.inf, np.inf, np.inf]
    max_elastic_force = [np.inf, np.inf, np.inf]  
    # This is not great since it depends on number of points
    
    # Normalize forces
    normalize_elastic = True
    normalize_stiffness = False
    
    # u_v direction multipliers
    uv_stiffness = [1, 1]
    uv_elastic = [1, 1]
    
    # Locks
    
    # Might implement lock to plan later if time allows
    lock_v_ends = [False, False, False]  # x,y,z
    lock_u_ends = [False, False, False]  # x,y,z
    
    ### Storage
    
    surface_list = []
    step_count = 0
    
    ### Init
    
    def __init__(self, surface, image, **kwargs):
        # Set any parameters to object attributes (makes changing above easy)
        for key, val in kwargs.items():
            setattr(self, key, val)
        
        self.image_sequence = image
        self.init_surface = surface
        self.surface = surface
       
    ### Properties
    
    @property
    def image_sequence(self):
        return self._image_sequence

    @image_sequence.setter
    def image_sequence(self, value):
        self._image_sequence = value  # Set value
        
        self._generate_image_gradients()  # Calculate gradients
        
        self._generate_interpolated_gradients()  # Provide necessary smoothness
        
    @property
    def surface(self):
        return self._surface

    @surface.setter
    def surface(self, value):
        self._surface = copy.deepcopy(value)
        
        self._gen_ctrlpts_map()
        
    ### Private Methods
        
    def _generate_image_gradients(self):
        """
        Provides GVF gradients in the x, y, and z directions
        """
        
        # Maybe should give access to parameters
        v = GVF3D(self.image_sequence.gradient().array, 
                  self.GVF_mu, self.GVF_iter, self.GVF_verbose)
        self.image_gradients = [ImageSequence(array=direction)
                                for direction in v]
        
    def _generate_interpolated_gradients(self):
        """
        Generate the gradient interpolation handle.
        This gives the gradient function the necessary smoothness
        for the optimization to run (function is lipshitz continuous).
        """
        
        # for each direction (x, y, z)
        self.interpolated_gradients = []
        for direction in self.image_gradients:
            values = direction.array
            points = [np.linspace(0, ind-1, ind) for ind in values.shape]
            
            # Assume no gradient outside bounds (fill value)
            interpolator = RegularGridInterpolator(points, values,
                                                   bounds_error=False,
                                                   fill_value=self.fill_value
                                                   )
            self.interpolated_gradients.append(interpolator)
            
    def _get_eval_points(self, **kwargs):
        """
        Retrieve wanted evaluated points from the surface
        """
        
        evalpts = np.array(
            self.surface.evalpts).reshape(
                (*self.surface.data['sample_size'], -1))
        
        return evalpts
    
    def _gen_ctrlpts_map(self):
        
        # Determine how to split the data
        splits = [None, None]
        
        surf = self.surface
        
        degree_u = surf.degree[1]
        if surf.closed_u:
            splits[0] = surf.knotvector_u[degree_u+1:-degree_u]
        else:
            # This is not perfect but will do (last value >1)
            # splits[0] = ((np.indices((surf.data['size'][0]-degree_u+1,))+0.5) \
            #     /(surf.data['size'][0])).flatten()
            n = surf.control_points_np.shape[0]
            delta = 1/(n+(n-2))
            cuts = [delta]
            for i in range(n-2):
                cuts.append(cuts[-1]+2*delta)
            cuts.append(1)
            splits[0] = cuts
            
        degree_v = surf.degree[1]
        if surf.closed_v:
            splits[1] = surf.knotvector_v[degree_v+1:-degree_v]
        else:
            # This is not perfect but will do (last value >1)
            # splits[1] = ((np.indices((surf.data['size'][1]-degree_v+1,))+0.5) \
            #     /(surf.data['size'][1])).flatten()
            n = surf.control_points_np.shape[1]
            delta = 1/(n+(n-2))
            cuts = [delta]
            for i in range(n-2):
                cuts.append(cuts[-1]+2*delta)
            cuts.append(1)
            splits[1] = cuts
                
        splits = [np.clip([0, *splits[0]], 0, 1),
                  np.clip([0, *splits[1]], 0, 1)]
        self._splits = splits
        
        # Split the data
        evalpts = self._get_eval_points()
        
        indices = np.indices(evalpts.shape[:2])
        
        assignments = np.zeros(self.surface.control_points_np.shape[:2],
                               dtype=object)
        
        cuts_u = np.ceil(splits[0]*surf.data['sample_size'][0]).astype(int)
        cuts_v = np.ceil(splits[1]*surf.data['sample_size'][1]).astype(int)
        
        self._cuts_u = cuts_u
        self._cuts_v = cuts_v
        # print(indices.shape)
        # print(cuts_u, cuts_v)
        
        for i in range(1, len(cuts_u)):
            for j in range(1, len(cuts_v)):
                mincut_u = cuts_u[i-1]
                maxcut_u = cuts_u[i]
                mincut_v = cuts_v[j-1]
                maxcut_v = cuts_v[j]
            
                # print(mincut_u, maxcut_u, mincut_v, maxcut_v)
                
                # print(i, j)
            
                assignments[i-1, j-1] = \
                    (indices[0, mincut_u:maxcut_u, mincut_v:maxcut_v], 
                     indices[1, mincut_u:maxcut_u, mincut_v:maxcut_v])
            
        # This should never change. Ideally only call once
        self._ctrlpts_map = assignments
        
        return assignments
    
    def _assign_to_ctrlpts(self, forces):
        
        ctrlpts_forces = np.zeros((forces.shape[0],
                                   *self.surface.control_points_np.shape[:2]))
        
        for i in range(self._ctrlpts_map.shape[0]):
            for j in range(self._ctrlpts_map.shape[1]):
                for k in range(forces.shape[0]):
                    force = np.mean(forces[k][self._ctrlpts_map[i, j]])
                    ctrlpts_forces[k, i, j] = force
                    
        return ctrlpts_forces
    
    def _apply_locks(self, ctrlpts_forces):
        
        # Apply end locks
        for i in range(3):
            if self.lock_v_ends[i]:
                ctrlpts_forces[i, [0, -1], :] = 0
            if self.lock_u_ends[i]:
                ctrlpts_forces[i, :, [0, -1]] = 0
                
        return ctrlpts_forces
    
    def _normalize(self, force):
        # The lazy way of keeping ~zeros is to add a tiny number...
        return force / (np.linalg.norm(force, axis=0) + 1e-16)
    
    ### Public methods
    
    def step(self):
        """
        Take a time step. (Forward Euler)
        """
        # Get control points (shaped properly)
        ctrlpts = self.surface.control_points_np
        
        # Force on ctrlpts
        forces = self.calculate_ctrlpts_forces()
        
        # Remove unwanted components
        forces = self._apply_locks(forces)
        
        forces = np.swapaxes(forces, 0, 1)
        forces = np.swapaxes(forces, 1, 2)
        
        # Forward euler
        ctrlpts[:,:,:3] = ctrlpts[:,:,:3] + self.step_size * forces
        
        self.surface.set_ctrlpts(
            ctrlpts,
            closed_v = self.surface.closed_v,
            closed_u = self.surface.closed_u,
            gen_knots=True
            )
        
        # Increment counter
        self.step_count += 1
        
    def calculate_ctrlpts_forces(self):
        forces = self._assign_to_ctrlpts(self.calculate_forces())
        
        return forces
        
    def calculate_forces(self):
        
        gradient_force = self.calc_gradient_force()
        stiffness_force = self.calc_stiffness_force()
        elastic_force = self.calc_elastic_force()
        
        forces = gradient_force + stiffness_force + elastic_force
        
        return forces
        
    def calc_gradient_force(self):
        """
        Calculate the forces incured from gradients/GVF
        """
        mul = self.gradient_multipliers
        
        # Get the gradients for evaluated points
        points = self._get_eval_points()
        grad_x = self.interpolated_gradients[0](points) * mul[0]
        grad_y = self.interpolated_gradients[1](points) * mul[1]
        grad_z = self.interpolated_gradients[2](points) * mul[2]
        
        forces = np.stack([grad_x, grad_y, grad_z])
        
        return forces
    
    def calc_stiffness_force(self):
        """
        Calculate the forces incured from curvature/bending
        """
        
        return 0
    
        ### Force Bounds
        # stiffness disabled because = 0 for now
        # stiffness_force = np.clip(stiffness_force,
        #                           -np.array(self.max_stiffness_force),
        #                           self.max_stiffness_force)
        
        ### Normalize
        # if self.normalize_stiffness:
        #     self._normalize(stiffness_force)
    
    def calc_elastic_force(self):
        """
        Calculate the forces incured from tension/compression
        """
        
        evalpts = self._get_eval_points()
        
        if self.surface.closed_u:
            evalpts = np.concatenate(
                [evalpts[-2, :].reshape(1, -1, 3),
                 evalpts, 
                 evalpts[1, :].reshape(1, -1, 3)],
                axis=0)
        if self.surface.closed_v:
            evalpts = np.concatenate(
                [evalpts[:, -2].reshape(-1, 1, 3),
                 evalpts,
                 evalpts[:, 1].reshape(-1, 1, 3)],
                axis=1)
        
        u_mul = self.uv_elastic[0]
        v_mul = self.uv_elastic[1]
        
        forces = np.zeros_like(self._get_eval_points())
        if self.surface.closed_u:
            forces += u_mul * \
                (2 * evalpts[1:-1, :] - evalpts[:-2, :] - evalpts[2:, :])
        else:
            if self.surface.closed_v:
                forces[1:, :] += u_mul*\
                    (evalpts[:-1, 1:-1] - evalpts[1:, 1:-1])
                forces[:-1, :] += u_mul*\
                    (evalpts[1:, 1:-1] - evalpts[:-1, 1:-1])
            else:
                forces[1:, :] += u_mul*\
                    (evalpts[:-1, :] - evalpts[1:, :])
                forces[:-1, :] += u_mul*\
                    (evalpts[1:, :] - evalpts[:-1, :])
            
        if self.surface.closed_v:
            forces += v_mul*\
                (2 * evalpts[:, 1:-1] - evalpts[:, :-2] - evalpts[:, 2:])
        else:
            if self.surface.closed_u:
                forces[:, 1:] += v_mul*\
                    (evalpts[1:-1, :-1] - evalpts[1:-1, 1:])
                forces[:, :-1] += v_mul*\
                    (evalpts[1:-1, 1:] - evalpts[1:-1, :-1])
            else:
                forces[:, 1:] += v_mul*\
                    (evalpts[:, :-1] - evalpts[:, 1:])
                forces[:, :-1] += v_mul*\
                    (evalpts[:, 1:] - evalpts[:, :-1])
            
        forces = np.stack([forces[:,:,0],
                           forces[:,:,1],
                           forces[:,:,2]])
        
        ### Normalize
        if self.normalize_elastic:
            forces = self._normalize(forces)
        
        
        ### Add factors
        mul = self.elasticity
        forces = forces * np.array(mul).reshape(3,1,1)
        
        ### Force bounds
        forces = np.clip(forces.reshape(-1, 3), 
                                -np.array(self.max_elastic_force),
                                self.max_elastic_force
                                ).reshape(forces.shape)
        
        return forces
        
    def optimize(self, verbose=False, save_surfs=True,
                 plotter=None, **kwargs):
        
        self.step_size = kwargs.pop('step_size', self.step_size)
        assert self.step_size != None, 'step size can not be None'
        if type(self.step_size) != list:
            self.step_size = [self.step_size]*3
        
        self.num_iter = kwargs.pop('num_iter', self.num_iter)
        assert self.num_iter != None, 'num_iter can not be None'
        assert type(self.num_iter) == int, 'num_iter must be an integer'
        
        for i in range(self.num_iter):
            if verbose:
                print('\rOptimization iter: ' + str(i + 1) + \
                      '/' + str(self.num_iter),
                      end='', flush=True)
                
            self.step()
            
            if save_surfs:  
                self.surface_list.append(copy.deepcopy(self.surface))
            if plotter:
                pass
            
        if verbose: 
            print('')
            
        return self.surface
    
    def plot(self, plane_widget=False, 
             plot_gradient=False, gradients='all',
             plot_initial=False, 
             plot_forces=False, scale_vectors=False,
             scale_factor=1, forces='all', 
             plot_image=True,
             plotter=None, 
             surface=None):
        
        if not plot_gradient:
            gridvalues = self.image_sequence.array
        else:
            if gradients=='x':
                gridvalues = self.image_gradients[0].array
            elif gradients=='y':
                gridvalues = self.image_gradients[1].array
            elif gradients=='z':
                gridvalues = self.image_gradients[2].array
            else:
                gridvalues = np.mean(np.stack(
                    [x.array for x in solver.image_gradients]),
                    axis=0)
            
        if not plotter:
            plotter = pv.BackgroundPlotter()
            
        image = None
        if plot_image:
            image = pv.UniformGrid()
            image.dimensions = gridvalues.shape
            image.point_arrays["values"] = gridvalues.flatten(order="F")
            plotter.add_mesh(image, cmap="bone", opacity=0.5)
        
        if plot_initial:
            surf = self.init_surface
            plotter.add_mesh(makemesh(surf), color='blue')
            plotter.add_mesh(makemesh(surf, 'PolyData'), color='blue')
            plotter.add_mesh(makemesh(surf, 'PolyData', 'ctrlpts'),
                             color='cyan')
        
        if not surface:
            outputsurf = self.surface
        else:
            outputsurf = surface
        surface_mesh = makemesh(outputsurf) 
        
        if plot_forces and forces != 'ctrlpts':
            if 'gradient' in forces:
                f = self.calc_gradient_force()
            elif 'elastic' in forces:
                f = self.calc_elastic_force()
            elif 'stiffness' in forces:
                f = self.calc_stiffness_force()
            else:
                f = self.calculate_forces()
            
            surface_mesh['vectors'] = f.reshape(3, -1).T
            arrows = surface_mesh.glyph(orient='vectors', 
                                        scale=scale_vectors,
                                        factor=scale_factor,)
            plotter.add_mesh(arrows, cmap="YlOrRd")
            
        if plot_forces and 'ctrlpts' in forces or 'all' in forces:
            f = self.calculate_ctrlpts_forces()
            f = self._apply_locks(f)
            ctrlpts_mesh = makemesh(outputsurf, 'PolyData', 'control_points')
            ctrlpts_mesh['vectors'] = f.reshape(3, -1).T
            arrows = ctrlpts_mesh.glyph(orient='vectors', 
                                        scale=scale_vectors,
                                        factor=scale_factor,)
            plotter.add_mesh(arrows, cmap="YlOrRd")
            
        plotter.add_mesh(surface_mesh, color='red')
        plotter.add_mesh(makemesh(outputsurf, 'PolyData'), color='red')
        plotter.add_mesh(makemesh(outputsurf, 'PolyData', 'ctrlpts'), color='orange')    

        if plane_widget and image:
            plotter.add_mesh_clip_plane(image)
            
        plotter.show_grid()
        plotter.show()

        return plotter 
    
    def animate(self):
        pass
    
# %% --- Main ---
if __name__ == "__main__":
    # %% Setup
    # from circle import surf
    
    # surf.set_ctrlpts(surf.control_points * np.array([98, 98, 1, 1]),
    #                   closed_v = True, gen_knots=True)
    
    from nurbs_active_contour.geometry.nurbs_presets import Cylinder
    # surf = Cylinder([5, 5, 2], offsets = [60, 50, -1], num_z=5)
    surf = Cylinder([5, 5, 2], offsets = [60, 50, -1], num_z=5)
    surf.delta = (0.1, 0.02)
    
    # surf = Cylinder([20, 20, 5], offsets = [300, 300, 0], num_z=5)
    
    # from geomdl import operations
    # surf = operations.refine_knotvector(surf, [0, 1])
    
    file = "../../examples/testfiles/aortic_cross_section.gif"
    # file = "../../examples/testfiles/MRI_SUB1_subset/SUB1_subset/"
    # file = "../../examples/testfiles/CARDIAC_CT_ANGIO_(_Retro)_11/"
    imseq = ImageSequence(file)
    
    imseq.change_resolution((100, 100, 10))
    
    # %% Solver
    solver = GD_solver(surf, imseq, 
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
    