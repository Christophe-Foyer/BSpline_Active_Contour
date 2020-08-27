# import autograd.numpy as np
from nurbs_active_contour.geometry.bspline import ClosedRationalQuadraticCurve
from nurbs_active_contour.geometry.surface import RationalQuadraticSurface
import numpy as np


def generate_knots(geom, closed=True):

    if isinstance(geom, ClosedRationalQuadraticCurve):
        curve = geom
        
        assert curve.degree == 2
        degree = 2
        length = len(curve.ctrlpts)
        kv = np.linspace(0, 1, length-1)
    
        if closed:
            start = kv[2:2+degree] - 1/(length-2)*(2+degree)
            end = kv[:2] + 1 + 1/(length-2)
    
            return np.hstack([start, kv, end])
    
        else:
            start = np.zeros(degree)
            end = np.ones(degree)
    
            return np.hstack([start, kv, end])
        
    elif isinstance(geom, RationalQuadraticSurface):
        surface = geom
        
        assert surface.degree == [2, 2]
        degree_u = 2
        degree_v = 2
        length_u, length_v = surface.data['size']
        
        # u knotvector
        kv_u = np.linspace(0, 1, length_u-1)
        # if surface.closed_u:
        #     start = kv_u[2:2+degree_u] - 1/(length_u-2)*(2+degree_u)
        #     end = kv_u[:2] + 1 + 1/(length_u-2)
    
        #     kv_u = np.hstack([start, kv_u, end])
    
        # else:
        start = np.zeros(degree_u)
        end = np.ones(degree_u)

        kv_u = np.hstack([start, kv_u, end])
            
        # v knotvector
        kv_v = np.linspace(0, 1, length_v-1)
        # if surface.closed_v:
            # start = kv_v[2:2+degree_v] - 1/(length_v-2)*(2+degree_v)
            # end = kv_v[:2] + 1 + 1/(length_v-2)
    
            # kv_v = np.hstack([start, kv_v, end])
    
        # else:
        start = np.zeros(degree_v)
        end = np.ones(degree_v)

        kv_v = np.hstack([start, kv_v, end])
        
        return kv_u, kv_v
        
        
def make_closed_surf(nx, ny, dist, height):
    
    def cylinder_pnt(angle, dist, z):
        x = np.cos(angle)
        y = np.sin(angle)
        # if abs(angle) < np.sqrt(2)/2:
        #     w = np.linalg.norm([1, y])/np.linalg.norm([x, y])
        # else:
        #     w = np.linalg.norm([x, 1])/np.linalg.norm([x, y])
        w = max(abs(x), abs(y))
        # w = 1
        
        z = z*w
        #return (x, y, z, w)
        return [round(x, 4)*dist, round(y, 4)*dist,
                round(z, 4), round(w, 4)]

    ctrlpts = np.vstack([[cylinder_pnt(a, dist, z) 
                          for a in np.linspace(0, 2*np.pi, ny)]
                         for z in np.linspace(0, height, nx)]).tolist()
    return ctrlpts


def cylinder_knotvector(geom):
    assert isinstance(geom, RationalQuadraticSurface)
    surface = geom
    
    degree_u = 2
    degree_v = 2
    
    length_u, length_v = surface.data['size']

    if surface.closed_u:
        assert (length_u-1) % 2 == 0, "n_v should be odd"
        
        num = int((length_u-1)/2)-1
        kv_u = np.linspace(0+1/(num+1), 1-1/(num+1), num)
        kv_u = np.rot90(np.stack([kv_u]*2), 3).flatten()
        kv_u = np.hstack([np.zeros(degree_u+1), kv_u, np.ones(degree_u+1)])
    else:
        kv_u = np.linspace(0, 1, length_u-1)
        kv_u = np.hstack([np.zeros(degree_u), kv_u, np.ones(degree_u)])
    
    if surface.closed_v:
        
        assert (length_v-1) % 2 == 0, "n_v should be odd"
        
        num = int((length_v-1)/2)-1
        kv_v = np.linspace(0+1/(num+1), 1-1/(num+1), num)
        kv_v = np.rot90(np.stack([kv_v]*2), 3).flatten()
        kv_v = np.hstack([np.zeros(degree_v+1), kv_v, np.ones(degree_v+1)])
        
    else:
        kv_v = np.linspace(0, 1, length_v-1)
        kv_v = np.hstack([np.zeros(degree_v), kv_v, np.ones(degree_v)])
        
    
    return kv_u, kv_v