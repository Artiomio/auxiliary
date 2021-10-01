import cv2
import numpy as np


def fit_img_center(img, width, height, **kwargs):
    img_shape = list(img.shape)
    img_shape[0] = height
    img_shape[1] = width
    
    new_img = np.zeros(img_shape, dtype=img.dtype)
    w2h = width / height
    
    img_height, img_width = img.shape[0], img.shape[1]
    img_w2h = img.shape[1] / img.shape[1]
    
    alpha_width = width / img_width
    new_img_height = int(alpha_width * img_height)
    if new_img_height <= height:
        # ok, влезает
        scaled_img = cv2.resize(img, (width, new_img_height), **kwargs)
        y_start = height // 2 - new_img_height // 2
        new_img[y_start: y_start + new_img_height] = scaled_img
        return new_img
    else:
        alpha_height = height / img_height
        new_img_width = int(alpha_height * img_width)
        scaled_img = cv2.resize(img, (new_img_width, height), **kwargs)
        x_start = width // 2 - new_img_width // 2
        new_img[:, x_start: x_start + new_img_width] = scaled_img
        return new_img
    
    
        