imports = """
import sys, os, glob
import random
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from tqdm.notebook import tqdm
import requests
"""


for import_line in imports.split('\n'):
    try:
        exec(import_line)    
    except ImportError as e:
        print("ImportException:", e)


def crop_nonzero(img):
    Y, X = np.nonzero(img[:, :, 0])
    y1, y2, x1, x2 = Y.min(), Y.max(), X.min(), X.max()
    return img[y1:y2, x1:x2]



def smart_file_finder(file_name, start_path="."):
    if os.path.exists(file_name):
        return file_name
    file_name = glob.glob(
        os.path.join(start_path, "**", "*" + file_name), recursive=True)[:1]
    if file_name:
        print()#f"That's what I found: {file_name}")
        return file_name[0]
    else:
        print(f"Sorry, haven't found anything like this ({file_name})")


def show(
    a,
    BGR=None,
    *args,
    state={"BGR": False},
    **kwargs,
):
    if BGR is not None:
        state["BGR"] = BGR

    if state["BGR"]:
        a = RGB(a)

    plt.figure(figsize=(24, 12))
    try:

        plt.imshow(a, *args, **kwargs)
    except:
        plt.imshow(a.astype("uint8"), *args, **kwargs)

    plt.show()


s = show


def RGB_to_BGR(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def BGR_to_RGB(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def resize(img, height=None, width=None, **kwargs):
    if height is not None and width is not None:
        return cv2.resize(img, (width, height), **kwargs)

    w_over_h = img.shape[1] / img.shape[0]

    if height is None and width is not None:
        height = round(width / w_over_h)

    elif height is not None and width is None:
        width = round(height * w_over_h)

    if height is None and width is None:
        raise Exception

    return cv2.resize(img, (width, height), **kwargs)


RGB = RGB_to_BGR
BGR = BGR_to_RGB


def half_the_size(img, *args, **kwargs):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), *args, **kwargs)


def downsize(img, r=10, interpolation=cv2.INTER_NEAREST_EXACT, *args, **kwargs):
    if r < 1:
        r = 1 / r
    return cv2.resize(img, (img.shape[1] // r, img.shape[0] // r), *args, **kwargs)


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
        new_img[y_start : y_start + new_img_height] = scaled_img
        return new_img
    else:
        alpha_height = height / img_height
        new_img_width = int(alpha_height * img_width)
        scaled_img = cv2.resize(img, (new_img_width, height), **kwargs)
        x_start = width // 2 - new_img_width // 2
        new_img[:, x_start : x_start + new_img_width] = scaled_img
        return new_img




def send_to_artmonitor(img, secret="импредикабельность", jpg_quality=None, jpeg_quality=None, monitor_url="https://artmonitor.pythonanywhere.com"):
    jpg_quality = 20 or (jpeg_quality or jpg_quality)    
    url=f'{monitor_url}/{secret}/postimage/'
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    result, encimg_jpg = cv2.imencode('.jpg', img, encode_param)
    jpg_bytes = encimg_jpg.tobytes()
    start = time.time()
    res = requests.post(url,
                        data=jpg_bytes,
                        headers={'Content-Type': 'application/octet-stream'})
    # print(f"Response: {res}\nReady: {len(jpg_bytes)} bytes sent in {time.time() - start} sec")
