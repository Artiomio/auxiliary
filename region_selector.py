import os
import sys
import random
import time

import cv2
import numpy as np

import matplotlib
import matplotlib.pylab as plt
def RGB_to_BGR(a):
    return cv2.cvtColor(a, cv2.COLOR_RGB2BGR)


def BGR_to_RGB(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


RGB = RGB_to_BGR
BGR = BGR_to_RGB



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




class RegionSelector:
    def __init__(self, width=200, height=200, zoom=1, downsize=1, xc=1850, yc=50):
        self.width = width
        self.height = height
        self.zoom = zoom
        self.downsize = downsize
        self.xc = xc
        self.yc = yc
        self.prev_time_stamp = time.time()

    @staticmethod
    def put_text(img, message, x, y):
        cv2.putText(img, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        cv2.putText(
            img, message, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

    def select_region(self, frame_1, key=None):
        time_spent_outside_video_loop = time.time() - self.prev_time_stamp
        rect_1 = frame_1[
            self.yc : round(self.yc + self.height / self.zoom),
            self.xc : round(self.xc + self.width / self.zoom),
        ]
        frame_thumbnail = cv2.rectangle(
            frame_1.copy(),
            (
                self.xc,
                self.yc,
                round(self.width / self.zoom),
                round(self.height / self.zoom),
            ),
            100,
            30,
        )
        frame_thumbnail = fit_img_center(frame_thumbnail, width=700, height=300)
        img_height, img_width = rect_1.shape[:2]
        rect_1 = cv2.resize(
            rect_1,
            (round(img_width * self.downsize), round(img_height * self.downsize)),
        )
        RegionSelector.put_text(
            frame_thumbnail,
            f"Output: {rect_1.shape[0]}x{rect_1.shape[1]} x={self.xc}  y={self.yc}  scale={round(self.downsize, 2)}   Time {round(time_spent_outside_video_loop, 3)} sec",
            10,
            50,
        )

        cv2.imshow("full", frame_thumbnail)

        if key is None:
            key = cv2.waitKeyEx(1)

        self.key = key

        if key in (ord("="), ord("+")):
            self.zoom *= 1.1
            print(f"zoom={self.zoom}")

        if key == ord("-"):
            self.zoom *= 0.9
            print(f"zoom={self.zoom}")

        if key in (ord("0"),):
            self.downsize *= 1.1
            print(f"downsize={self.downsize}")

        if key == ord("9"):
            self.downsize *= 0.9
            print(f"downsize={self.downsize}")

        if key in (65361, ord("a")):
            self.xc = self.xc - 10
            print("Left")

        if key in (65363, ord("d")):
            self.xc = self.xc + 10
            print("Right")

        if key in (65362, ord("w")):
            self.yc = self.yc - 30

        if key in (65364, ord("s")):
            self.yc = self.yc + 30


        if key in (ord("1"),):
            self.width = round(0.95 * self.width)
            print(f"width={self.width}")

        if key in (ord("2"),):
            self.width = round(1.05 * self.width)
            print(f"width={self.width}")

        if key in (ord("3"),):
            self.height = round(0.95 * self.height)
            print(f"height={self.height}")

        if key in (ord("4"),):
            self.height = round(1.05 * self.height)
            print(f"height={self.height}")

        self.prev_time_stamp = time.time()
        return rect_1.copy()

