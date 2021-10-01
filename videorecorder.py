import atexit
import os
import time
from artiom_convenience_functions import fit_img_center, BGR

import cv2

def execute_at_exit():
    print("Saving video! See you soon!")
    if default_video_writer is not None:
        default_video_writer.release()

atexit.register(execute_at_exit)
default_video_writer = None
current_video_path = None


    
video_output_configs = dict(default_framerate=60)


def save_to_video(img, file_name_arg=None, frame_rate=None, width_arg=None, height_arg=None):
    global default_video_writer, current_video_path, video_width, video_height

    if not default_video_writer:
        if width_arg is None:
            width = img.shape[1]
            
        if height_arg is None:
            height = img.shape[0]

        video_height, video_width = img.shape[: 2]
        if not frame_rate:
            frame_rate = video_output_configs['default_framerate'] 

        if not file_name_arg:
            current_video_path = time.ctime().replace(":", "-") + ".mp4"
        else:
            current_video_path = file_name_arg





        default_video_writer = VideoRecorder(short_filename=current_video_path, 
                                                path='.',
                                                frame_rate=frame_rate,
                                                size=(video_width, video_height))

    if default_video_writer is not None:
        if img.shape != (video_height, video_width):
            img = fit_img_center(img, height=video_height, width=video_width)
        default_video_writer.write(img)


def new_imshow(title, img):
    global original_cv2_imshow
    original_cv2_imshow(title, img)
    save_to_video(img)


def get_new_cv2_imshow(func=None):
    """ from videorecorder import get_new_cv2_imshow
        cv2.imshow = get_new_cv2_imshow(cv2.imshow)
    """
    global original_cv2_imshow
    if not func:
        func = cv2.imshow
    original_cv2_imshow = func
    return new_imshow

class VideoRecorder:
    def __init__(
        self,
        short_filename=None,
        path=".",
        frame_rate=None,
        size=(1920, 1080),
        fourcc_str="XVID",
        RGB2BGR_transform=False,
    ):

        self.RGB2BGR_transform = RGB2BGR_transform
        self.base_filename = (
            short_filename
            if short_filename is not None
            else f'{time.ctime().replace(":", "-").replace(" ", "_")}.mp4'
        )

        self.size = size
        self.file_name = os.path.join(path, self.base_filename)
        fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.frame_rate = frame_rate
        self.out = cv2.VideoWriter(self.file_name, fourcc, self.frame_rate, self.size)

    def write_frame(self, img):
        if self.RGB2BGR_transform:
            img = BGR(img)
        self.frame = fit_img_center(img, width=self.size[0], height=self.size[1])
        self.out.write(self.frame)

    def write(self, img):
        self.write_frame(img)

    def finish(self):
        self.out.release()

    def release(self):
        self.finish()

