import random
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
import pylab as pl
from tqdm import tqdm


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


def half_the_size(img, *args, **kwargs):
    return cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), *args, **kwargs)


def downsize(img, r=7, *args, **kwargs):
    if r < 1:
        r = 1 / r

    return cv2.resize(img, (img.shape[1] // r, img.shape[0] // r), *args, **kwargs)


def show(a, *args, **kwargs):
    plt.figure(figsize=(24, 12))
    plt.imshow(a, *args, **kwargs)
    plt.show()


s = show


def downsize(img, r=7, *args, **kwargs):
    if r < 1:
        r = 1 / r

    return cv2.resize(img, (img.shape[1] // r, img.shape[0] // r), *args, **kwargs)


def RGB_to_BGR(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def BGR_to_RGB(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


RGB = RGB_to_BGR
BGR = BGR_to_RGB


def put_text(img, message, x, y):
    cv2.putText(img, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 15)
    cv2.putText(img, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 5)


class BackgroundChanger:
    def __init__(
        self,
        cap,
        n_frames,
        convert_BGR2RGB=False,
        display_progress=False,
        output="matplotlib",
    ):

        if isinstance(cap, str):
            self.file_name = cap
            self.cap = cv2.VideoCapture(cap)
        else:
            self.cap = cap

        self.n_frames = n_frames
        self.convert_BGR2RGB = convert_BGR2RGB
        self.display_progress = display_progress
        self.output = output

        self.__calculate_background()
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def __calculate_background(self, max_frames=10 ** 5):
        cap = self.cap
        n_frames = self.n_frames
        display_progress = self.display_progress

        total_frames_in_file = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        sum_of_frames = np.zeros_like(frame, dtype="int64")

        frames_to_skip = total_frames_in_file // n_frames
        i = 0
        ret = True

        if self.display_progress and self.output == "matplotlib":
            plt.figure(figsize=(16, 8))

        while cap.isOpened():

            cv2.waitKey(1)

            if frames_to_skip:
                cap.set(cv2.CAP_PROP_POS_FRAMES, i * (frames_to_skip + 1) + 1)

            ret, frame = cap.read()
            if not ret:
                break

            sum_of_frames += frame
            i += 1

            if display_progress:
                print(f"\rProcessed: {i}  out of {n_frames}", end="")

            if display_progress and i % 5 == 0:
                background_BGR = (np.round(sum_of_frames / i)).astype("uint8")

                put_text(
                    background_BGR,
                    f"Processed: {i} out of {n_frames}",
                    background_BGR.shape[1] // 2,
                    100,
                )

                if self.output == "matplotlib":
                    background = cv2.cvtColor(background_BGR, cv2.COLOR_BGR2RGB)

                    clear_output(wait=True)
                    plt.imshow(background)
                    display(pl.gcf())

                elif self.output == "opencv":
                    cv2.imshow(
                        "background",
                        fit_img_center(background_BGR, width=1000, height=800),
                    )

            if i > n_frames or i > max_frames:
                break

        background = (np.round(sum_of_frames / i)).astype("uint8")

        if self.convert_BGR2RGB:
            background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

        if self.output == "opencv":
            cv2.destroyAllWindows()

        self.background = background
        self.background_typed = background.astype("int16")
        clear_output()

    def get_background(self):
        return self.background

    def get_next_transformed_frame(self, mode=0, num_param=0.5, threshold=30):
        """
        Получить следующий кадр с преобразованным (напр. удалённым) фоном
        """
        ret, frame = self.cap.read()

        if not ret:
            raise Exception  # TODO

        self.frame = frame
        diff = np.abs(frame - self.background_typed).astype("uint8")
        diff_gray = diff.sum(axis=2)

        Y, X = np.where(diff_gray < threshold)

        if mode == 0:
            frame[Y, X, :3] = 0

        elif mode == 1:
            frame[Y, X] = self.background[Y, X]

        elif mode == 2:
            frame[Y, X] = (frame[Y, X] * num_param).astype("uint8")

        elif mode == 3:
            frame[Y, X] = (self.background[Y, X] * num_param).astype("uint8")

        return frame

    def show_background(self, convert_to_RGB=True):
        s(RGB(self.background))

    def show_next_frame(self, *args, **kwargs):
        s(RGB(self.get_next_transformed_frame(*args, **kwargs)))

    def set_frame_number(n):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, n)
