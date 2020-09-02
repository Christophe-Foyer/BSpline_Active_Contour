from nurbs_active_contour.geometry.surface import RationalQuadraticSurface
import numpy as np

class Cylinder(RationalQuadraticSurface):
    
    _points = np.array([
        [[0, 0, 1],
         [0, 1, 1],
         [1, 1, 1],
         [1, 0, 1]],
        ])
    _points = np.concatenate([_points, np.ones((*_points.shape[:2], 1))],
                             axis=2)
    
    def __init__(self, factors=[1, 1, 1], offsets=[0, 0, 0],
                 num_z = 3, **kwargs):
        
        assert len(factors) == 3
        assert len(offsets) == 3
        assert num_z >= 3
        
        _points = self._points
        points = np.concatenate([_points]*num_z, axis=0)
        points[:, :, 2] = points[:, :, 2] + np.indices((num_z,)).reshape(-1, 1)
        points = points * np.array([*factors, 1]) + np.array([*offsets, 0])
        
        super().__init__(**kwargs)
        
        self.set_ctrlpts(points,
                         closed_v = True,
                         gen_knots=True)
        
        self.closed_v = True    