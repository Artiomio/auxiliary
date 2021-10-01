"""
  draw_polygon - is a wrapper for skimage.draw.polygon
"""

import numpy as np
import skimage.draw

def draw_polygon(img_arr, x_arr, y_arr, color):
    """draw_poly(img_arr, x_arr, y_arr, color)
       Draw filled polygon on an numpy array (img_arr) with given arrays
       of x and y (x_arr and y_arr) coordinates of the polygon
       Depending on the dimension of img_arr, the color parameter can be a number
       or a list/array/tuple:
           255 or [255] or [255, 255, 255] (white), [255, 0, 0, 100] (alpha channel)
    """
    y_poly, x_poly = skimage.draw.polygon(x_arr, y_arr)

    height, width = img_arr.shape[0], img_arr.shape[1]

    x_mask = x_poly < width
    y_mask = y_poly < height

    # print("x_mask", x_mask)
    # print("y_mask", x_mask)
    #     x_mask [ True  True  True ...,  True  True  True]
    #     y_mask [ True  True  False ...,  True  True  True]
    #
    x_poly = x_poly[x_mask & y_mask]
    y_poly = y_poly[x_mask & y_mask]
    img_arr[x_poly, y_poly] = color


if __name__ == '__main__':
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    draw_polygon(image, [0, 100, 350], [0, 100, 50], [255, 255, 255])

    from skimage.viewer import ImageViewer
    viewer = ImageViewer(image)
    viewer.show()
    