import cv2
import numpy as np
from tqdm.notebook import tqdm
from artiom_convenience_functions import resize

def fancy_background_extractor(cap, n, show_progress_image=False, 
                               n_skip_frames=10):

    if type(cap) == str and os.path.isfile(cap):
        cap = cv2.VideoCapture(cap)

    for start_frame in [1000,]:
        cap.set(1, start_frame)
        mean_frame = cap.read()[1]
        threshold = 10
        indices = np.zeros(shape=mean_frame.shape[: 2], dtype="float")
        frames_sum = np.zeros_like(mean_frame, dtype="float")
        mean_img = frames_sum.copy()


        frames_sum_2 = np.zeros_like(mean_frame, dtype="float")
        frames_sum_2_N = 0

        frame_0 = cap.read()[1].astype("int")
        for _ in range(19):
            ret, frame = cap.read()
        frames_sum_2_N += 1
        frames_sum_2 += frame


        for i in tqdm(range(n)):
            frame_0 = cap.read()[1].astype("float")
            frames_sum_2_N += 1
            frames_sum_2 += frame_0

            for _ in range(n_skip_frames):
                ret, frame = cap.read()
            frames_sum_2_N += 1
            frames_sum_2 += frame

            diff = (np.abs(frame_0.astype("int") - frame)).mean(axis=2)
            if show_progress_image:
                cv2.imshow("Magic", resize(mean_img.astype("uint8"), width=1700))
                key = cv2.waitKeyEx(1)
                if key in [27, ]: break
                cv2.imshow("diff", resize(diff.astype("uint8"), width=1700))

            Y, X = np.where(diff < threshold)
            for channel in [0, 1, 2]:
                frames_sum[Y, X, channel] += frame[Y, X, channel]



            indices[Y, X] += 1

            Y, X = np.where(indices > 0)

            for channel in [0, 1, 2]:
                mean_img[Y, X, channel] = frames_sum[Y, X, channel] / indices[Y, X]


            #mean_frame = mean_img.copy()

    return mean_img.astype("uint8")
        
        
 