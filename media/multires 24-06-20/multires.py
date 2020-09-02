from nurbs_fit.utils.image import *

file = "../../examples/testfiles/aortic_cross_section.gif"
IS = ImageSequence(file)
IS.multiresolution(scaling=2)
IS.animate()
IS.multires_images[2].animate()