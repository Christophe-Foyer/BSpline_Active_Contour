import numpy as np
from scipy.ndimage.filters import laplace as del2

def GVF(f, mu, ITER, verbose=True, pad_image=10):
    """
    %GVF Compute gradient vector flow.
    %   [u,v] = GVF(f, mu, ITER) computes the
    %   GVF of an edge map f.  mu is the GVF regularization coefficient
    %   and ITER is the number of iterations that will be computed.  
    
    %   Chenyang Xu and Jerry L. Prince 6/17/97
    %   Copyright (c) 1996-99 by Chenyang Xu and Jerry L. Prince
    %   Image Analysis and Communications Lab, Johns Hopkins University
    
    %   modified on 9/9/99 by Chenyang Xu
    %   MATLAB do not deal their boundary condition for gradient and del2 
    %   consistently between MATLAB 4.2 and MATLAB 5. Hence I modify
    %   the function to take care of this issue by the code itself.
    %   Also, in the previous version, the input "f" is assumed to have been
    %   normalized to the range [0,1] before the function is called. 
    %   In this version, "f" is normalized inside the function to avoid 
    %   potential error of inputing an unnormalized "f".
    
    Based on the MATLAB code by Xu and Prince.
    
    Not sure why the image needs padding on the +x and +y but seems to fix
    weird behavior somewhat. -CF
    """
    
    if pad_image >= True:
        shape = f.shape
        
        imgpad = np.zeros((np.array(f.shape)+3))
        imgpad[:shape[0], :shape[1]] = f
        f = imgpad
    
    [m,n] = f.shape
    fmin  = np.min(f[:, :]);
    fmax  = np.max(f[:, :]);
    f = (f-fmin)/(fmax-fmin);  #% Normalize f to the range [0,1]
    
    f = BoundMirrorExpand(f);  #% Take care of boundary condition
    [fx,fy] = np.gradient(f);     #% Calculate the gradient of the edge map
    u = fx; v = fy;            #% Initialize GVF to the gradient
    SqrMagf = fx*fx + fy*fy; #% Squared magnitude of the gradient field
    
    if verbose:
        print('')
    
    #% Iteratively solve for the GVF u,v
    for i in range(ITER):
      u = BoundMirrorEnsure(u);
      v = BoundMirrorEnsure(v);
      u = u + mu*4*del2(u) - SqrMagf*(u-fx);
      v = v + mu*4*del2(v) - SqrMagf*(v-fy);
      if verbose:
          print('\rGVF iter: ' + str(i+1) + '/' + str(ITER),
                end='', flush=True)
    
    if verbose:
        print('')
        
    u = BoundMirrorShrink(u);
    v = BoundMirrorShrink(v);
    
    u = u.T
    v = v.T
    
    if pad_image >= True:
        u = u[:shape[0], :shape[1]]
        v = v[:shape[0], :shape[1]]
    
    return [u,v]

def BoundMirrorEnsure(A):
    """
    % Ensure mirror boundary condition          %
    % The number of rows and columns of A must be greater than 2
    %
    % for example (X means value that is not of interest)
    % 
    % A = [
    %     X  X  X  X  X   X
    %     X  1  2  3  11  X
    %     X  4  5  6  12  X 
    %     X  7  8  9  13  X 
    %     X  X  X  X  X   X
    %     ]
    %
    % B = BoundMirrorEnsure(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """
    
    [m,n] = A.shape;
    
    if (m<3 | n<3):
        raise('either the number of rows or columns is smaller than 3');
    
    yi = np.arange(0, m-1);
    xi = np.arange(0, n-1);
    B = A;
    
    B[np.ix_([1-1, m-1,],[1-1, n-1,])] = \
        B[np.ix_([3-1, m-2-1,],[3-1, n-2-1,])]; # % mirror corners
    B[np.ix_([1-1, m-1,],xi)] = \
        B[np.ix_([3-1, m-2-1,],xi)]; #% mirror left and right boundary
    B[np.ix_(yi,[1-1, n-1,])] = \
        B[np.ix_(yi,[3-1, n-2-1,])]; #% mirror top and bottom boundary
    
    return B

def BoundMirrorExpand(A):
    """
    % Expand the matrix using mirror boundary condition
    % 
    % for example 
    %
    % A = [
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    %     ]
    %
    % B = BoundMirrorExpand(A) will yield
    %
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    # shift for matlab style

    [m,n] = A.shape;
    yi = np.arange(0, m+1-1);
    xi = np.arange(0, n+1-1);
    
    B = np.zeros((m+2, n+2));
    B[np.ix_(yi,xi)] = A;
    B[np.ix_([1-1, m+2-1,],[1-1, n+2-1,])] = \
      B[np.ix_([3-1, m-1,],[3-1, n-1,])];  #% mirror corners
    B[np.ix_([1-1, m+2-1,],xi)] = \
      B[np.ix_([3-1, m-1,],xi)]; #% mirror left and right boundary
    B[np.ix_(yi,[1-1, n+2-1,])] = \
      B[np.ix_(yi,[3-1, n-1,])]; #% mirror top and bottom boundary
    
    return B

def BoundMirrorShrink(A):
    """
    % Shrink the matrix to remove the padded mirror boundaries
    %
    % for example 
    %
    % A = [
    %     5  4  5  6  12  6
    %     2  1  2  3  11  3
    %     5  4  5  6  12  6 
    %     8  7  8  9  13  9 
    %     5  4  5  6  12  6
    %     ]
    % 
    % B = BoundMirrorShrink(A) will yield
    %
    %     1  2  3  11
    %     4  5  6  12
    %     7  8  9  13
    
    % Chenyang Xu and Jerry L. Prince, 9/9/1999
    % http://iacl.ece.jhu.edu/projects/gvf
    """

    [m,n] = A.shape;
    yi = np.arange(0, m-1);
    xi = np.arange(0, n-1);
    B = A[np.ix_(yi,xi)];
    
    return B


# %%
if __name__ == "__main__":
    from nurbs_active_contour.utils.image import ImageSequence
    file = "../../../examples/testfiles/aortic_cross_section.gif"
    # file = "../../../examples/testfiles/MRI_SUB1_subset/SUB1_subset/"
    # file = "../../examples/testfiles/CARDIAC_CT_ANGIO_(_Retro)_11/"
    imseq = ImageSequence(file)
    imseq.change_resolution((100, 100, -1))
    # imseq = imseq.gradient()
    
    img = np.array(imseq.return_image(1))
    
    # shape = img.shape
    
    # imgpad = np.zeros((np.array(img.shape)+3))
    # imgpad[:shape[0], :shape[1]] = img
    # img = imgpad
    
    from skimage.filters import gaussian
    img = gaussian(img, 1)
    
    out = np.array(GVF(img, 0.001, 1000, pad_image=10))
    
    # out = out[:, :shape[0], :shape[1]]
    
    x = np.linspace(0, img.shape[0]-1, out.shape[1])
    y = np.linspace(0, img.shape[1]-1, out.shape[2])
    xi, yi = np.meshgrid(x, y, indexing='ij')
    
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.imshow(img)
    plt.quiver(xi, yi, -out[0], -out[1], angles='xy') 
    plt.colorbar()
    
    plt.figure()
    plt.imshow(-np.linalg.norm(out, axis=0))
    
    plt.figure()
    plt.imshow(-np.array(imseq.gradient().return_image(0)))
    