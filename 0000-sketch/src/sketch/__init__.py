import sketch.random_resnet

import numpy as np
import cairo

# im gonna use something like these, but i need to think about how to
# handle the difference in values, e.g. numbers in the range [0, 1],
# unsigned integers, and other things.

# def ndarray_from_cairo_image_surface(
#     surface: cairo.ImageSurface,
# ) -> np.ndarray:
#   shape = (
#     surface.get_height(), surface.get_width())
#   buffer = surface.get_data()
#   tensor = np.ndarray(
#     shape=shape, dtype=np.uint32, buffer=buffer)
#   return tensor

# def cairo_image_surface_from_ndarray(
#     tensor: np.ndarray,
# ) -> cairo.ImageSurface:
#   (height, width) = tensor.shape
#   surface = cairo.ImageSurface.create_for_data(
#     tensor, cairo.FORMAT_ARGB32, width, height)
#   return surface
