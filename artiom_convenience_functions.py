import sys, os, glob
import random
import time
imports = """
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from tqdm import tqdm
import requests
"""
for import_line in imports.split('\n'):
    try:
        exec(import_line)    
    except ImportError as e:
        print("ImportException:", e)



def it_is_jupyter_notebook():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' not in get_ipython().config:
            return False
    except ImportError:
        return False

    except:
        return False
    return True

def uint8(t):
    t = np.round(255*(t / t.max())).astype("uint8")
    return t



global vr, frame_counter, frame, key, frame_width, frame_height
global n_frames 


def run_video_loop(video_filename, inside_video_loop_func, 
every_n_th_frame=None):
    global vr, frame_counter, frame, key, frame_width, frame_height
    global n_frames 
    vr = cv2.VideoCapture(smart_file_finder(video_filename, start_path="."))
    n_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height, frame_width = vr.get(4), vr.get(3)
    key = -1
    for frame_counter in tqdm(range(n_frames)):
        ret, frame = vr.read()
        if every_n_th_frame is not None:
            for _ in range(0, every_n_th_frame - 1):
                vr.read()

        
        
        res = inside_video_loop_func(frame, frame_counter, key, frame_width, frame_height, n_frames)
        if res is None: frame_to_show = frame
        else: frame_to_show = res
        img = fit_img_center(frame_to_show, width=1900, height=1000)
        cv2.imshow("Main", img)
        key = cv2.waitKey(1)
        if key in (27, ord('q')):
            break

        if key in [ord('l'), 65361]:
            current_frame_number = vr.get(1) - 100*every_n_th_frame
            vr.set(1, current_frame_number)


        if key in [ord('r'), 65363]:
            current_frame_number = vr.get(1) + 100*every_n_th_frame
            vr.set(1, current_frame_number)

        if key == 32:
            print("Paused. Press SPACE to continue")
            while cv2.waitKey(1) & 0xFF != 32:
                key = cv2.waitKey(1)
                time.sleep(0.02)

    cv2.destroyAllWindows()
    vr.release()



def print_columns(*args):
    output  = [str(a).split('\n') for a in args]
    for x in zip(*output):
        for a in x:
            print(a, "    ", end="")
        print()    


def simple_tuple(a):
    s = []
    for x in a:
        if isinstance(x, Iterable):
            s += simple_tuple(x)
        else:
            s.append(x)
    return s

def rect_center(a):
    x1, y1, x2, y2 = open_list(a)
    return ((x1 + x2) / 2, (y1 + y2) / 2)



""" Finds the minimal horizontal rectangle containing given points
    points is a list of points e.g. [(3,5),(-1,4),(5,6)]
"""
def find_rect_range(points):
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]

    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    return((min_x, min_y), (max_x, max_y))



from math import pi

def rotate_image(image, center, angle):
    row,col = image.shape[: 2]
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image

def date_time_filename():
    return f'{time.ctime().replace(":", "-").replace(" ", "_")}.txt'

def write_to_file(*args, end="\n", sep=" ", file_name=None, self={}):
    if file_name is not None:
        self["file_name"] = file_name

    file_name = self.get('file_name', date_time_filename())

    with open(file_name, "a") as f:
        f.write(sep.join(map(str, args)) + end)

print_to_file = write_to_file

def pause(t, waitKey):
    dt = 0.3
    for i in range(0, int(t / dt)):
        key = waitKey(1)
        time.sleep(dt)
        key = waitKey(1)

    
    return key
    

def waitKey(max_pause=10):
    start = time.time()
    while time.time() - start < max_pause:
        key = cv2.waitKey(1)
        if key > 0: return key

    
def denormalize_coordinates(pt, width, height):
    try:
        x, y = pt
        res = round(x * width), round(y * height)
        return res
    except:
        res = [(round(x * width), round(y * height)) for (x, y) in pt]
        return res






def round_tuple(*args):
    if len(args) == 1:
        l = args[0]
    elif len(args) > 1:

        l = args

    return tuple([int(round(x))  for x in list(l)])





def rectagle_from_img(img, x1, y1, x2, y2):
    return img[y1: y2, x1: x2]

get_rectangle_from_img = rectagle_from_img


def crop_nonzero(img):
    Y, X = np.nonzero(img.sum(axis=2))
    y1, y2, x1, x2 = Y.min(), Y.max(), X.min(), X.max()
    return img[y1: y2, x1: x2]



def nonzero_subrectangle_coordinates(img):
    Y, X = np.nonzero(img.sum(axis=2))
    y1, y2, x1, x2 = Y.min(), Y.max(), X.min(), X.max()
    return x1, y1, x2, y2

nonzero_rectangle = nonzero_subrectangle_coordinates


def rectangle(img, *args, **kwargs):

    if isinstance(args[1], tuple) and len(args[0]) == 2:
        return cv2.rectangle(*args, **kwargs)
    elif isinstance(args[1], tuple) and len(args[0]) == 4:
        return cv2.rectangle(*(args[1:]), **kwargs)
    elif all(isinstance(t, numbers.Number) for t in args[: 4]):
        x1, y1, x2, y2 = simple_tuple(args)[: 4]
        return cv2.rectangle(img, (x1, y1), (x2, y2), *args, **kwargs)
    raise TypeError("What has it to do with a rectangle?")



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


    if height is None and width is None:
        width = round(1920 * 0.8)
        height = round(1080 * 0.8)
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


def fit_img_center(img, width=None, height=None, **kwargs):
    if width is None and height is None:
        width = round(1920 * 0.5)
        height = round(1080 * 0.5)
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
