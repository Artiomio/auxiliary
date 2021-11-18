"""
   Для того, чтобы opencv-шная функция imshow записывала параллельно с показом в видеофайл
   достаточно вставить

   from videorecorder import get_new_cv2_imshow
   cv2.imshow = get_new_cv2_imshow(cv2.imshow)
   
   
   или

   from videorecorder import get_new_cv2_imshow
   cv2.imshow = get_new_cv2_imshow(cv2.imshow, video_frame_rate=10,
                                    redraw_only_every_nth_frame=10,
                                    last_frame_meditation_time_sec=3,
                                    fading_out_time_sec=5)
   
             
"""




import atexit
import os
import time
from artiom_convenience_functions import fit_img_center, BGR
import numpy as np
import cv2

def execute_at_exit():
    print("Saving video for you! See you soon!")
    if default_video_writer is not None:
        default_video_writer.release()

atexit.register(execute_at_exit)
default_video_writer = None
current_video_path = None
new_imshow_counter = {}


    
video_output_configs = dict(video_frame_rate=60, 
                            redraw_only_every_nth_frame=1,
                            last_frame_meditation_time_sec=5,
                            fading_out_time_sec=10
                            )


def save_to_video(img, file_name_arg=None, frame_rate=None, width_arg=None, height_arg=None, 
                    **more_video_params):
    global default_video_writer, current_video_path, video_width, video_height

    if not default_video_writer:
        if width_arg is None:
            width = img.shape[1]
            
        if height_arg is None:
            height = img.shape[0]

        video_height, video_width = img.shape[: 2]

        
        if not frame_rate:
            frame_rate = video_output_configs['video_frame_rate'] 

        if not file_name_arg:
            current_video_path = time.ctime().replace(":", "-") + ".mp4"
        else:
            current_video_path = file_name_arg


        if more_video_params:
            video_output_configs.update(more_video_params)


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

    new_imshow_counter[title] = new_imshow_counter.get(title, -1) + 1
    
    
    # Мы перерисовываем окно каждый энный раз
    if new_imshow_counter[title] % video_output_configs["redraw_only_every_nth_frame"] == 0:
        original_cv2_imshow(title, img)
    save_to_video(img)


def get_new_cv2_imshow(func=None, video_frame_rate=None, redraw_only_every_nth_frame=None):
    """ from videorecorder import get_new_cv2_imshow
        cv2.imshow = get_new_cv2_imshow(cv2.imshow)
    """
    global original_cv2_imshow

    if video_frame_rate:
        video_output_configs["video_frame_rate"] = video_frame_rate
    if not func:
        func = cv2.imshow


    if redraw_only_every_nth_frame is not None and redraw_only_every_nth_frame > 0:
        video_output_configs["redraw_only_every_nth_frame"] = redraw_only_every_nth_frame
        

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
        # Redirect-compatibility function
        self.write_frame(img)


    def write_frame_again(self):
        self.out.write(self.frame)
    
    
    
    def finish(self):
        # Real finish boilerplate 

        if video_output_configs.get("last_frame_meditation_time_sec", 0) > 0:
            n = int(video_output_configs["last_frame_meditation_time_sec"] *       
                 video_output_configs["video_frame_rate"])
            for i in range(n):
                self.write_frame_again()

        if video_output_configs.get("fading_out_time_sec", 0) > 0:
            n = int(video_output_configs["fading_out_time_sec"] *       
                 video_output_configs["video_frame_rate"])
            frame_0 = self.frame.copy()
            for i in range(n):
                self.frame = (np.round(np.cos(i*np.pi / 2 / n) * frame_0)).astype("uint8")
                #self.frame = ((n - i) * self.frame)).astype("uint8")
                self.write_frame_again()


                
        self.out.release()

    def release(self):
        # Compatibility redirect-function
        self.finish()

