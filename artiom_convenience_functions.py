import sys, os, glob
import random
import math
import time
from collections.abc import Iterable
import io
from PIL import Image

imports = """
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from tqdm import tqdm
import requests
import tkinter
from tkinter.filedialog import askopenfilename
open_file_dialog = askopenfilename

"""
for import_line in imports.split("\n"):
    try:
        exec(import_line)
    except ImportError as e:
        print("ImportException:", e)

# from videorecorder import save_to_video
_default_video_reader = None
_default_video_reader_return_code = 0
_last_frame = None



def use_artmonitor_as_imshow():
    cv2.imshow = lambda title, img : send_to_artmonitor(img)
    cv2.waitKey = lambda x: x


def limit_GPU_mem_usage(GPU_MEM_GB=4):
    import tensorflow as tf

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configurati11on(
                gpus[0],
                [
                    tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=GPU_MEM_GB * 1024
                    )
                ],
            )
        except RuntimeError as e:
            print(e)


"""def open_file_dialog(*args, **kwargs):
    filename = askopenfilename(*args, **kwargs)
    return filename
"""

def open_video(video_fname):
    global _default_video_reader
    _default_video_reader = cv2.VideoCapture(video_fname)
    return _default_video_reader


def read_frame(skip_frames=None):
    global _default_video_reader, _default_video_reader_return_code, _last_frame

    if skip_frames is not None:
        for _ in range(skip_frames):
            _default_video_reader.read()

    _default_video_reader_return_code, _last_frame = _default_video_reader.read()
    return _last_frame


def get_nvidia_temperature():
    process = subprocess.Popen(
        ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
        stdout=subprocess.PIPE,
    )
    out, err = process.communicate()
    temperature = int(out.decode().replace("\n", ""))
    return temperature



def put_text(
    img,
    message,
    coords=(20, 50),
    font=cv2.FONT_HERSHEY_COMPLEX_SMALL,
    size=1,
    *args,
    **kwargs,
):
    cv2.putText(img, message, coords, font, size, (0, 0, 0), 
    round(3 + size // 2))
    
    
    cv2.putText(
        img,
        message,
        coords,
        font,
        size,
        (255, 255, 255),
        round(1 + size // 5),
    )

def get_last_frame_obtained():
    return _last_frame


def random_RGB():
    return [random.randint(0, 255) for _ in range(3)]


def skip_frames(vr, delta_frames):
    current_frame_number = vr.get(cv2.CAP_PROP_POS_FRAMES) + delta_frames
    vr.set(1, current_frame_number)


def skip_frames_by_reading_one_by_one(vr, delta_frames):
    for i in range(delta_frames):
        vr.read()


def image_with_saturation(img, alpha=0.8):
    img_bw_ = (np.round(img.mean(axis=2))).astype("uint8")  # .reshape(img.shape[0], -1)

    img_bw = img.copy()
    img_bw[:, :, 0] = img_bw_
    img_bw[:, :, 1] = img_bw_
    img_bw[:, :, 2] = img_bw_

    res = np.round(img * alpha + img_bw * (1 - alpha)).astype("uint8")
    return res


def gradual_imshow(
    title, img, previous_img_l=[None], n_interframes=100, imshow=None, pause=1 / 50
):
    key = None
    last_key_pressed = None
    if previous_img_l[0] is not None or (
        previous_img_l[0] is not None and img.shape != previous_img_l[0].shape
    ):
        previous_img = previous_img_l[0]
        for t in np.linspace(0, 1, n_interframes):
            result_img = ((1 - t) * previous_img + t * img).astype("uint8")
            if imshow:
                imshow(title, result_img)
            else:
                cv2.imshow(title, result_img)
            key = cv2.waitKey(1)
            if key > -1: last_key_pressed = key
            time.sleep(pause)

    cv2.imshow(title, img)
    previous_img_l[0] = img.copy()
    return last_key_pressed

def corrected_file_name(file_name):
    if not os.path.isfile(file_name):
        return file_name
    file_name_, file_ext = os.path.splitext(file_name)

    j = 1
    new_name = file_name
    while os.path.isfile(new_name):
        new_name = file_name_ + "." + str(j) + file_ext
        j += 1

    return new_name


def img_center_xy(img):
    return (
        round(img.shape[1] / 2),
        round(img.shape[0] / 2),
    )


def width(img):
    return img.shape[1]


def height(img):
    return img.shape[0]


def distance_between(vec_1, vec_2):
    return math.sqrt(sum([(x - y) ** 2 for x, y in zip(vec_1, vec_2)]))


def add_img2_to_img_at_xy(img, img2, x, y):
    h1, w1 = img.shape[:2]
    h2, w2 = img2.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w1, x + w2)
    y2 = min(h1, y + h2)

    xA = -min(0, x)
    yA = -min(0, y)

    delta_x = x2 - x1
    delta_y = y2 - y1

    if x1 >= x2 or y1 >= y2:
        return
    img[y1:y2, x1:x2] = img[y1:y2, x1:x2] + img2[yA : yA + delta_y, xA : xA + delta_x]


def draw_shade(img, x1, y1, x2, y2, shade_dx=50, shade_dy=50):
    shade_img = img[y1 + shade_dy : y2 + shade_dy, x1 + shade_dx : x2 + shade_dx]
    shade_img -= shade_img // 4
    put_img2_to_img_at_xy(img, shade_img, x=x1, y=y1)


def put_img2_to_img_at_xy(img, img2, x, y=None):
    if y is None:
        x, y = x
    h1, w1 = img.shape[:2]
    h2, w2 = img2.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w1, x + w2)
    y2 = min(h1, y + h2)

    xA = -min(0, x)
    yA = -min(0, y)

    delta_x = x2 - x1
    delta_y = y2 - y1

    if x1 >= x2 or y1 >= y2:
        return
    img[y1:y2, x1:x2] = img2[yA : yA + delta_y, xA : xA + delta_x]


def put_img2_taken_by_center_to_img_at_xy(img, img2, x, y=None):
    delta_x = img.shape[1] // 2
    delta_y = img.shape[0] // 2
    put_img2_to_img_at_xy(img, img2, x=x - delta_x, y=y - delta_y)


# With shadow
def put_img2_to_img_at_xy_with_shadow(
    img, img2, x, y=None, shade_dx=50, shade_dy=50, coeff=4
):
    frame = img
    if y is None:
        x, y = x

    shade_img = frame[
        y + shade_dy : y + img2.shape[0] + shade_dy,
        x + shade_dx : x + img2.shape[1] + shade_dx,
    ]
    shade_img -= shade_img // coeff
    put_img2_to_img_at_xy(frame, img2, x=x, y=y)


def it_is_jupyter_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False

    except:
        return False
    return True


def uint8_normalized_std(array):
    array = np.round(255 * (array / 2 / array.max()))
    result = (array >= 0) * (array <= 255) * array

    return result.astype("uint8")


def uint8_normalized_mean(t):
    t = np.round(255 * (t / 2 / t.mean()))

    return t.astype("uint8")


def uint8_normalized(t):
    t = np.round(255 * (t / t.max()))

    return t.astype("uint8")


def uint8(t):
    t = np.round(t).astype("uint8")
    return t


global vr, frame_counter, frame, key, frame_width, frame_height
global n_frames


def get_one_first_n_frames_generator(video_filename, n=1):
    cap = cv2.VideoCapture(video_filename)
    for i in range(n):
        ret, frame = cap.read()
        yield frame


def run_video_loop(
    video_filename,
    inside_video_loop_func,
    every_n_th_frame=1,
    start_frame=0,
    framerate=60,
    full_screen=True,
    autoresize=True,
    save_video=0,
):
    global vr, frame_counter, frame, key, frame_width, frame_height
    global n_frames

    window_title = "Main"
    cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)
    if full_screen:

        cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)

        cv2.setWindowProperty(
            window_title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
        )
    cv2.moveWindow(window_title, 0, 0)
    if video_filename is not None:
        vr = cv2.VideoCapture(video_filename)
        if start_frame:
            vr.set(1, start_frame)
        n_frames = int(vr.get(cv2.CAP_PROP_FRAME_COUNT))
        if not n_frames > 1:
            n_frames = 10 ** 10

        frame_height, frame_width = vr.get(4), vr.get(3)
    else:
        n_frames = 1000000
        frame_height, frame_width = None, None
    key = -1
    frame_counter = -1 + start_frame
    for frame_counter_i in tqdm(range(n_frames)):
        frame_counter += 1
        if video_filename is not None:
            ret, frame = vr.read()
            if every_n_th_frame is not None and every_n_th_frame > 1:
                for _ in range(0, every_n_th_frame - 1):
                    vr.read()
        else:
            frame = None

        try:
            res = inside_video_loop_func(
                frame, frame_counter, key, frame_width, frame_height, n_frames
            )
        except StopIteration:
            break
        if res is None:
            frame_to_show = frame
        elif isinstance(res, int) and res == -1:
            continue
        else:
            frame_to_show = res

        if autoresize:
            img = fit_img_center(frame_to_show, width=1900, height=1000)
        else:
            img = frame_to_show
        cv2.imshow("Main", img)
        if save_video:
            save_to_video(img)

        delta_frames = 1000
        time.sleep(1 / framerate)
        key = cv2.waitKey(1)
        if key in (27, ord("q")):
            break

        if key in [ord("l"), 65361]:
            current_frame_number = vr.get(1) - delta_frames
            frame_counter -= delta_frames
            vr.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)

        if key in [ord("r"), 65363]:
            current_frame_number = vr.get(1) + delta_frames
            frame_counter += delta_frames
            vr.set(cv2.CAP_PROP_POS_FRAMES, current_frame_number)

        if key == 32:
            print("Paused. Press SPACE to continue")
            while cv2.waitKey(1) & 0xFF != 32:
                key = cv2.waitKey(1)
                time.sleep(0.02)

    cv2.destroyAllWindows()
    if video_filename is not None:
        vr.release()


def columns(*args):
    output = [str(a).split("\n") for a in args]
    for x in zip(*output):
        for a in x:
            print(a, "    ", end="")
        print("")


def print_columns(*args):
    output = [str(a).split("\n") for a in args]
    for x in zip(*output):
        for a in x:
            print(a, "    ", end="")
        print()


def get_columns_str(*args):
    output = [str(a).split("\n") for a in args]
    s = []
    for x in zip(*output):
        for a in x:
            s.append(a + "    ")
        s.append("\n")
    return "".join(s)


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
    return ((min_x, min_y), (max_x, max_y))


from math import pi
from math import sin, cos


def rotate(coords, origin, angle):
    """Rotates given point around given origin"""
    x, y = coords
    xc, yc = origin

    cos_angle = cos(angle)
    sin_angle = sin(angle)

    x_vector = x - xc
    y_vector = y - yc

    x_new = x_vector * cos(angle) - y_vector * sin(angle) + xc
    y_new = x_vector * sin(angle) + y_vector * cos(angle) + yc
    return (x_new, y_new)


def rotate_image(image, center, angle):
    row, col = image.shape[:2]
    rot_mat = cv2.getRotationMatrix2D(tuple(center), angle, 1.0)
    new_image = cv2.warpAffine(image, rot_mat, (col, row))
    return new_image


def date_time_filename():
    return f'{time.ctime().replace(":", "-").replace(" ", "_")}.txt'


def write_to_file(*args, end="\n", sep=" ", file_name=None, self={}):
    if file_name is not None:
        self["file_name"] = file_name

    file_name = self.get("file_name", date_time_filename())

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
        if key > 0:
            return key


def denormalize_coordinates(pt, width, height):
    pt = np.array(pt, dtype="float")
    original_shape = pt.shape
    pt = pt.reshape(-1, 2)
    pt[:, 0] *= width
    pt[:, 1] *= height
    return np.round(pt.reshape(original_shape)).astype("int")

    """
    try:
        x, y = pt
        res = round(x * width), round(y * height)
        return res
    except:
        res = [(round(x * width), round(y * height)) for (x, y) in pt]
        return res
     """


def normalize_coordinates(pt):
    original_shape = pt.shape
    pt = np.array(pt, dtype="float").reshape(-1, 2)
    x0 = pt[:, 0].min()
    y0 = pt[:, 1].min()

    pt[:, 0] -= x0
    pt[:, 1] -= y0

    pt[:, 0] /= pt[:, 0].max()
    pt[:, 1] /= pt[:, 1].max()
    return pt.reshape(original_shape)


def round_tuple(*args):
    if len(args) == 1:
        l = args[0]
    elif len(args) > 1:

        l = args

    return tuple([int(round(x)) for x in list(l)])


def rectangle_from_img(img, pt_1, pt_2, x2=None, y2=None):
    if x2 is None and y2 is None:
        x1, y1 = pt_1
        x2, y2 = pt_2
    elif x2 is not None and y2 is not None:
        x1 = pt_1
        y1 = pt_2

    # print(f"x1={x1}, y1={y1}, x2={x2}, y2={y2} ")
    if 0 <= x1 <= x2 <= 1 and 0 <= y1 <= y2 <= 1:
        (x1, y1), (x2, y2) = denormalize_coordinates(
            [(x1, y1), (x2, y2)], width=img.shape[1], height=img.shape[0]
        )
        # print(f"После денормализации: x1={x1}, y1={y1}, x2={x2}, y2={y2} ")

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    return img[y1:y2, x1:x2]


get_rectangle_from_img = rectangle_from_img


def crop_nonzero(img):
    Y, X = np.nonzero(img.sum(axis=2))
    y1, y2, x1, x2 = Y.min(), Y.max(), X.min(), X.max()
    return img[y1:y2, x1:x2]


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
    elif all(isinstance(t, numbers.Number) for t in args[:4]):
        x1, y1, x2, y2 = simple_tuple(args)[:4]
        return cv2.rectangle(img, (x1, y1), (x2, y2), *args, **kwargs)
    raise TypeError("What has it to do with a rectangle?")


def read_frame_and_show(cap, title="Main"):
    cap.set(0, ind)
    ret, frame = cap.read()
    cv2.imshow(title, frame)
    key = cv2.waitKey(1)
    return frame, key


def smart_file_finder(file_name, start_path=".", verbose=True):
    if file_name.startswith("http://") or file_name.startswith("https://"):
        input("хаха Press Enter...")
        return file_name
    if os.path.exists(file_name):
        return file_name

    fname_found = None
    for current_file_name in glob.iglob(
        os.path.join(start_path, "**", file_name), recursive=True
    ):

        fname_found = current_file_name
        break

    if fname_found:
        if verbose:
            print(f"That's what I found: {fname_found}")
        return fname_found
    else:
        if verbose:
            print(f"Sorry, haven't found anything like this ({fname_found})")


"""
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



  """


def random_file(mask, *args, **kwargs):
    file_list = glob.glob(os.path.join(mask), *args, **kwargs)
    file_name = random.choice(file_list)
    return file_name


def show(
    a,
    BGR=None,
    figsize=(24, 12),
    *args,
    state={"BGR": False},
    **kwargs,
):
    if BGR is not None:
        if isinstance(BGR, str):
            BGR = True if BGR.lower() == "bgr" else False
        state["BGR"] = BGR

    if state["BGR"]:
        a = RGB(a)

    plt.figure(figsize=figsize)
    try:

        plt.imshow(a, *args, **kwargs)
    except:
        plt.imshow(a.astype("uint8"), *args, **kwargs)

    plt.show()


s = show


def show_image_and_wait(img):
    cv2.imshow("Hello!", img)
    key = cv2.waitKey(0)
    if key in (27, ord("q")):
        sys.exit()
    return key


def imshow(img, title="Hello!"):
    if type(img) is str:
        img, title = title, img
    cv2.imshow(title, img)
    key = cv2.waitKey(1)
    if key in (27, ord("q")):
        sys.exit()

    return key


def RGB_to_BGR(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def BGR_to_RGB(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def stretch_horizontally(img, alpha):
    h, w = img.shape[:2]
    return resize(img, width=round(w * alpha))


def stretch_vertically(img, alpha):
    h, w = img.shape[:2]
    return cv2.resize(img, (w, round(h * alpha)))


def resize(img, height=None, width=None, **kwargs):
    if img.shape[0] * img.shape[1] == 0:
        return [0]

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


def downsize(img, r=10, interpolation=cv2.INTER_AREA, *args, **kwargs):
    if r < 1:
        r = 1 / r
    return cv2.resize(img, (img.shape[1] // r, img.shape[0] // r), *args, **kwargs)


def add_vertical_border(img, border_width):
    v_border = np.zeros((img.shape[0], border_width, 3), dtype="uint8")
    return np.concatenate([v_border.copy(), img, v_border.copy()], axis=1)


def add_horizontal_border(img, border_height):
    h_border = np.zeros((border_height, img.shape[1], 3), dtype="uint8")
    return np.concatenate([h_border.copy(), img, h_border.copy()], axis=0)


"""
def background_border(img, border_height, border_width, background):
    img = add_vertical_border(add_horizontal_border(img, border_height), 
    border_width)
    background = cv2.resize(background, (img.shape[1], img.shape[0]))

    background[]
    img[: border_height, :, :] = background[: border_height, :, :].copy()
    img[img.shape[0] - border_height:, :, :] = background[img.shape[0] - border_height:, :, :].copy()
    return img
   """





def fit_img_center(img, width=None, height=None, background=None, **kwargs):
    if width is None and height is None:
        width = round(1920 * 0.5)
        height = round(1080 * 0.5)
    img_shape = list(img.shape)
    img_shape[0] = height
    img_shape[1] = width

    
    if background is None:
        new_img = np.zeros(img_shape, dtype=img.dtype)
    else:
        
        new_img = background[: img_shape[0], : img_shape[1]].copy()


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
    
    

def send_to_artmonitor(
    img,
    secret="импредикабельность",
    jpg_quality=90,
    jpeg_quality=None,
    monitor_url="http://127.0.0.1:7893",  # https://artmonitor.pythonanywhere.com",
):
    jpg_quality = 20 or (jpeg_quality or jpg_quality)
    url = f"{monitor_url}/{secret}/postimage/"
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality]
    result, encimg_jpg = cv2.imencode(".jpg", img, encode_param)
    jpg_bytes = encimg_jpg.tobytes()
    start = time.time()
    res = requests.post(
        url, data=jpg_bytes, headers={"Content-Type": "application/octet-stream"}
    )
    # print(f"Response: {res}\nReady: {len(jpg_bytes)} bytes sent in {time.time() - start} sec")


def plot_as_array(*args, **kwargs):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_plot = Image.open(buf)
    img = np.asarray(img_plot)
    return img


def video_player(video_fname, callbacks=[], start_frame=0, number_of_frames_to_skip=0):
    try:
        cap = cv2.VideoCapture(video_fname)
    except:
        cap = video_fname

    if start_frame:
        cap.set(1, start_frame)

    current_frame_number = start_frame
    while True:

        for _ in range(number_of_frames_to_skip):
            ret, frame = cap.read()

        current_frame_number += number_of_frames_to_skip

        for cb in callbacks:
            cb()

        ret, frame = cap.read()
        cv2.imshow("Video", frame)

        key = cv2.waitKeyEx(1)
        print(key)

        if key & 0xFF == 27:
            print("Нажали Esc, выхожу из цикла")
            break

        if key == ord("l"):
            print("Left")
            current_frame_number -= 300
            cap.set(1, current_frame_number)

        if key == ord("r"):
            print("Right")
            current_frame_number += 300
            cap.set(1, current_frame_number)
