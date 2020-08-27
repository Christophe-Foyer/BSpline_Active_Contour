"""
Author: Christophe Foyer
    
Description:
    Image import, export, storage, and manipulation tools.
"""

from PIL import Image as Image
import numpy as np
from scipy.interpolate import RegularGridInterpolator

import pydicom

from os import listdir
from os.path import isfile, join

from nurbs_active_contour.utils.plotting import animate, animate_3d

from scipy import ndimage

import pyvista as pv

# %% Gifs

def import_gif(filename, imageaxis="last", **kwargs):
    """
    Takes a gif file and converts it to a numpy array

    Input: path of gif file

    Returns: a numpy array based on a grayscale gif

    >>> import_gif("../../examples/testfiles/aortic_cross_section.gif").shape
    (7, 100, 100)
    >>>
    """

    array = None
    pic = Image.open(filename)
    for i in range(pic.n_frames):
        pic.seek(i)
        pix = np.array(pic.getdata()).reshape(pic.size[0], pic.size[1])
        if type(array) == np.ndarray:
            array = np.dstack((array, pix))
        else:
            array = pix

    if imageaxis == 'last':
        array = np.swapaxes(array, 0, 2)
        array = np.swapaxes(array, 1, 2)

    return array


# %% Standard image formats (png/jpg...)

def import_image(filename, **kwargs):
    """
    Takes a gif file and converts it to a numpy array

    Input: path of gif file

    Returns: a numpy array based on a grayscale gif
    """

    pic = Image.open(filename).convert('L')
    array = np.array(pic)

    return array


def import_image_dir(directory, imageaxis='last', **kwargs):
    """
    Takes a directory of files and converts it to a numpy array

    Input:
        Path of files

    Returns:
        A numpy array based on a grayscale file list
    """

    assert imageaxis in ['first', 'last'], 'imageaxis must be first or last'

    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    array = None
    for file in files:
        pix = import_image(directory + file)
        if type(array) == np.ndarray:
            array = np.dstack((array, pix))
        else:
            array = pix

    if imageaxis == 'last':
        array = np.swapaxes(array, 0, 2)
        array = np.swapaxes(array, 1, 2)

    return array


# %% DCM

def import_dcm(filename, **kwargs):
    """
    Imports dicom files as numpy arrays

    Input:
        File path

    Returns:
        2d numpy array

    """
    ds = pydicom.dcmread(filename)
    return np.array(ds.pixel_array)


def import_dcm_dir(directory, imageaxis='last', **kwargs):
    """
    Takes a directory of dcm files and converts it to a numpy array

    Input:
        Path of files

    Returns:
        A numpy array based on a grayscale file list
    """

    assert imageaxis in ['first', 'last'], 'imageaxis must be first or last'

    files = [f for f in listdir(directory) if isfile(join(directory, f))]

    array = None
    for file in files:
        pix = import_dcm(directory + file)
        if type(array) == np.ndarray:
            array = np.dstack((array, pix))
        else:
            array = pix

    if imageaxis == 'last':
        array = np.swapaxes(array, 0, 2)
        array = np.swapaxes(array, 1, 2)

    return array


# %% Class interface

class ImageSequence:
    """
    A class for reading image sequences/images

    Supports directories or single files

    TODO:
        - Support rotation
        - Support coordinates

    EXAMPLE:
    >>> file = "../../examples/testfiles/aortic_cross_section.gif"
    >>> imseq = ImageSequence(file)
    >>> imseq.array.shape
    (100, 100, 7)

    """

    _array = None     # Image array
    coords = None    # Spatial coordinates for each image
    path = None      # Filepath or directory
    filetype = None  # Filetype
    
    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        self._array = value  # Set value
        # xyz based on indices (might be improved on later)
        self.coords = np.indices(self.array.shape)

    def __init__(self, path=None, array=None, filetype=None):
        self.path = path

        assert path is not None or array is not None, "No data provided"

        if path is not None:
            # Remove the dot if needed
            if type(filetype) is str and filetype[0] == '.':
                filetype = filetype[1:]
    
            if isfile(path):
                if filetype is not None:
                    print("Path points to file, ignoring filetype")
    
                # Process file depending on the extension
                self._process_file(path)
            else:
                # Process files based on the extension
                self._process_dir(path, filetype)
        elif array is not None:
            self.array = array

    def _process_dir(self, dirname, filetype):
        files = [f for f in listdir(dirname) if isfile(join(dirname, f))]

        if filetype is not None:
            files = [f for f in files if f.split('.')[-1] == filetype]
        else:
            filetype = files[0].split('.')[-1]
            assert all([f.split('.')[-1] == filetype for f in files]),\
                "filetypes are not all the same, please specify filetype"

        self.filetype = filetype

        typemap = {
            'dcm': {'f': import_dcm_dir},
            'png': {'f': import_image_dir},
            'jpg': {'f': import_image_dir},
            }

        assert filetype in typemap.keys(), "Unsupported filetype. " + \
            "Supported filetypes are: " + str(list(typemap.keys()))

        self.array = typemap[filetype]['f'](dirname, imageaxis='first')

        # xyz based on indices (might be improved on later)
        self.coords = np.indices(self.array.shape)

    def _process_file(self, filename):
        filetype = filename.split('.')[-1]

        typemap = {
            'dcm': {"f": import_dcm, "dim": 2},
            'png': {"f": import_image, "dim": 2},
            'jpg': {"f": import_image, "dim": 2},
            'gif': {"f": import_gif, "dim": 3},
            }

        array = typemap[filetype]['f'](filename, imageaxis='first')

        # make 3d with third dimension = 1
        if typemap[filetype]['dim'] == 2:
            array == array.reshape((*array.shape, 1))

        self.array = array

        # xyz based on indices (might be improved on later)
        self.coords = np.indices(self.array.shape)
        
    def gradient(self, invert=True):
        """
        xyz image gradient generator
        """
        input_image = self.array
        
        # Get x-gradient in "sx"
        # sx = ndimage.sobel(input_image, axis=0, mode='constant')
        sx = np.gradient(input_image, axis=0)
        # Get y-gradient in "sy"
        # sy = ndimage.sobel(input_image, axis=1, mode='constant')
        sy = np.gradient(input_image, axis=1)
        # Get y-gradient in "sz"
        # sz = ndimage.sobel(input_image, axis=2, mode='constant')
        sz = np.gradient(input_image, axis=2)
        
        if invert:
            sgn = -1
        else:
            sgn = 1
        # Get square root of sum of squares
        image_gradients = sgn * np.linalg.norm(np.stack([sx,sy,sz]), axis=0)
        
        return ImageSequence(array=image_gradients)
        
    def multiresolution(self, maxiter=3, scaling=2.0):
        """
        Function to provide multiple resolutions of the image stack.
        The resolution is only scaled along the x/y plane.

        Parameters
        ----------
        maxiter : int
            Maximum number of downscaled images. 
            The function will end if resolution < 1 pixel for either x or y.
            The default is 3.
        scaling : float
            Scaling factor between images. The default is 2.

        Returns
        -------
        ImageSequence, list
            Returns an ImageSequence of stacked images with the original and a 
            list of ImageSequences of downscaled images.

        """
        
        # Multi-resolution along slices
        
        multires_imseq = np.zeros_like(self.array)
        
        multires_images = \
            [np.zeros((*((np.array(self.array.shape[:2])/scaling**k).astype(int)),
                       self.array.shape[2]))
             for k in range(1, maxiter+1)]
        
        for i in range(self.array.shape[2]):
            image = Image.fromarray(self.array[:, :, i])
            
            im_out = np.array(image)
            im_lr_list = [None]*maxiter
            
            k = 1
            while any(np.array(image.size)/scaling**k > 1) and k <= maxiter:
                # Divide resolution by roughly 2 and stack
                im_low_res = image.resize((np.array(image.size)/scaling**k).astype(int))
                im_low_res_rescaled = im_low_res.resize(image.size)
                
                im_out = im_out + im_low_res_rescaled
                
                im_lr_list[k-1] = im_low_res
                
                # Increase power
                k += 1
                
            multires_imseq[:, :, i] = im_out
            
            # Append
            for k in range(maxiter):
                if im_lr_list[k] is not None:
                    multires_images[k][:, :, i] = im_lr_list[k]
                
        # Filter out empty images and turn into imagesequences
        multires_images_filtered = []
        for image in multires_images:
            if (image != 0).any():
                multires_images_filtered.append(ImageSequence(array=image))
                    
        self.multires_images = multires_images_filtered
        
        multires_imseq = ImageSequence(array=multires_imseq)
            
        return multires_imseq, self.multires_images
        
    def change_resolution(self, shape, inplace=True):
        values = self.array
        
        shape = list(shape)
        
        assert len(shape) == 3
        for i in range(len(shape)):
            if shape[i] == -1:
                shape[i] = values.shape[i]
        
        points = [np.linspace(0, ind-1, ind) for ind in values.shape]
        
        # Assume no gradient outside bounds (fill value)
        interpolator = RegularGridInterpolator(points, values,
                                               bounds_error=True)
        x = np.linspace(0, values.shape[0]-1, shape[0])
        y = np.linspace(0, values.shape[1]-1, shape[1])
        z = np.linspace(0, values.shape[2]-1, shape[2])
        coords = np.stack(np.meshgrid(x, y, z), axis=-1)
        
        vals = interpolator(coords)
        
        if inplace:
            self.array = vals
            return self
        return ImageSequence(array=vals)

    def return_image(self, num):
        return Image.fromarray(self.array[:, :, num])

    def plot(self, plane_widget=True, **kwargs):
        
        # Display the arrows
        plotter = pv.BackgroundPlotter()
        
        cmap = kwargs.pop('cmap', 'bone')
        opacity = kwargs.pop('opacity', 0.5)
        
        grid = pv.UniformGrid()
        grid.dimensions = self.array.shape
        grid.point_arrays["values"] = self.array.flatten(order="F")
        plotter.add_mesh(grid, cmap=cmap, opacity=opacity, **kwargs)
        
        if plane_widget:
            plotter.add_mesh_clip_plane(grid)
        
        plotter.show_grid()
        plotter.show()
        
        return plotter

    def animate(self, renderer='plotly', plot=True, **kwargs):
        if renderer == 'plotly':
            return animate_3d(self.array, plot=plot, **kwargs)
        elif renderer == 'matplotlib':
            array = self.array
            array = np.swapaxes(array, 0, 2)
            array = np.swapaxes(array, 1, 2)
            return animate(array, **kwargs)
        
    def save_as_gif(self, filename):
        filetype = filename.split('.')[-1]
        assert filetype.lower() == 'gif'
        
        im = self.return_image(0)
        ims = [self.return_image(i) for i in range(1, self.array.shape[2])]
        im.save(filename, save_all=True, append_images=ims)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
