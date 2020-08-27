import numpy as np
#import numba

# @numba.njit
def EnforceMirrorBoundary(f):
    """
    % This function enforces the mirror boundary conditions
    % on the 3D input image f. The values of all voxels at 
    % the boundary is set to the values of the voxels 2 steps 
    % inward
    """
    
    [N, M, O] = np.shape(f);

    xi = np.arange(1, M-2);
    yi = np.arange(1, N-2);
    zi = np.arange(1, O-2);

    # Corners
    f[[0, N-1], [0, M-1], [0, O-1]] = f[[2, N-3], [2, M-3], [2, O-3]]

    # Edges
    f[np.ix_([0, N-1], [0, M-1], zi)] = \
        f[np.ix_([2, N-3], [2, M-3], zi)]
        
    f[np.ix_(yi, [0, M-1], [0, O-1])] = \
        f[np.ix_(yi, [2, M-3], [2, O-3])]
    f[np.ix_([0, N-1], xi, [0, O-1])] = \
        f[np.ix_([2, N-3], xi, [2, O-3])]

    # Faces
    f[np.ix_([0, N-1], xi, zi)] = \
        f[np.ix_([2, N-3], xi, zi)];
    f[np.ix_(yi, [0, M-1], zi)] = \
        f[np.ix_(yi, [2, M-3], zi)];
    f[np.ix_(yi, xi, [0, O-1])] = \
        f[np.ix_(yi, xi, [2, O-3])];   
    
    return f 