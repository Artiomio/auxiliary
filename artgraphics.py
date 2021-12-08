import random
import time
from itertools import product

import numpy as np
import cv2

def draw_text(

    *messages,
    x=None,
    y=None,
    self=dict(left_margin=30, cursor_x=30, cursor_y=30, lines_margin=15, words_margin=4),
    end="\n",
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1.5,
    thickness=2, 
    round_floats = 3,
    image=None,
    sep=" ",
    background_alpha=None

):
   
    message = sep.join([(str(round(message, round_floats)) if isinstance(message, float)  else str(message)) for message in messages] )
    if x is None:
        x = self["cursor_x"]
    else:
        self["cursor_x"] = x
        self["left_margin"] = x


    if y is None:
        y = self["cursor_y"]
    else:
        self["cursor_y"] = y

    if image is None and "image" in self:
        image = self["image"] 



    for message in (message + end).split("\n"):
        for dx, dy in product([ -2, -1, 0, 1, 2],[-2, -1, 0, 1, 2] ):
            if background_alpha is None:
                cv2.putText(image, message, (x + dx, y + dy), font, font_scale, (0, 0, 0), thickness)

        text_width, text_height = cv2.getTextSize(message, font, font_scale, thickness)[
      0
  ]
            
        cv2.putText(image, message, (x, y), font, font_scale, (255, 255, 255), thickness)
        y += text_height + self["lines_margin"]
        x = self["left_margin"]

    self["cursor_y"] = y + self["lines_margin"]
    self["cursor_x"] = self["left_margin"]
        # todo возможность end=""



def image_grid(img_arr, max_col, margin=1, background_color=0):
    if len(img_arr) < 1: return
    height, width = img_arr[0].shape[: 2]
    max_row = int(round(len(img_arr) / max_col + 0.5))
    canvas = np.zeros([(margin + height) * max_row, (margin + width) * max_col, 3], dtype='uint8') + background_color
    for i, img in enumerate(img_arr):
        y_i, x_i = (i // max_col), (i % max_col)
        y_coord =  y_i * (height + margin)
        x_coord = x_i * (width + margin)                                        
        canvas[y_coord: y_coord + height, x_coord: x_coord + width] = img       
                                                                                
    canvas = canvas[: -margin, : -margin]                                       
    return canvas                                                            


def random_RGB():
    return [random.randint(0, 255) for _ in range(3)]


def gradual_imshow(title, img, previous_img_l=[None], n_interframes=100):
    if previous_img_l[0] is not None:
        previous_img = previous_img_l[0]
        for t in np.linspace(0, 1, n_interframes):
            result_img = ((1 - t) * previous_img + t * img).astype("uint8")
            cv2.imshow(title, result_img)
            key = cv2.waitKey(1)
            time.sleep(1 / 50)

    cv2.imshow(title, img)
    previous_img_l[0] = img.copy()




def add_img2_to_img_at_xy(img, img2, x, y):
    h1, w1 = img.shape[: 2]
    h2, w2 = img2.shape[: 2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w1, x + w2)
    y2 = min(h1, y + h2)
    
    xA = -min(0, x)
    yA = -min(0, y)
    
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    if x1 >= x2 or y1 >= y2: return
    img[y1: y2, x1: x2] = (img[y1: y2, x1: x2] + 
                       img2[yA: yA + delta_y, xA: xA + delta_x]
                      )
  

def put_img2_to_img_at_xy(img, img2, x, y=None):
    if y is None:
        x, y = x
    h1, w1 = img.shape[: 2]
    h2, w2 = img2.shape[: 2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w1, x + w2)
    y2 = min(h1, y + h2)
    
    xA = -min(0, x)
    yA = -min(0, y)
    
    
    delta_x = x2 - x1
    delta_y = y2 - y1
    
    if x1 >= x2 or y1 >= y2: return
    img[y1: y2, x1: x2] =  img2[yA: yA + delta_y, xA: xA + delta_x]
                      


def put_with_alpha(background, img_with_alpha, x, y, copy=False):
    # Корректируем  размер накладываемого изображения чтобы исключить выходы за границы

    left = 0
    top = 0
    height, width = img_with_alpha.shape[0], img_with_alpha.shape[1]

    if x < 0:
        left = -x
        x = 0

    if y < 0:
        top = -y
        y = 0

    if y + img_with_alpha.shape[0] >= background.shape[0]:
        # Вылазит снизу
        height = max(
            0,
            img_with_alpha.shape[0]
            - (y + img_with_alpha.shape[0] - background.shape[0]),
        )


    if x + img_with_alpha.shape[1] >= background.shape[1]:
        # Вылазит справа
        width = max(
            0,
            img_with_alpha.shape[1]
            - (x + img_with_alpha.shape[1] - background.shape[1]),
        )

    img_with_alpha = img_with_alpha[top : top + height, left : left + width]
    # if x > background.shape[1]:
    #    return
    #
    # if y > background.shape[0]:


    if copy:
        background = background.copy()

    height, width = img_with_alpha.shape[0], img_with_alpha.shape[1]
    b_height, b_width = background.shape[0], background.shape[1]

    alpha = img_with_alpha[:, :, 3].astype(np.float64) / 255
    img_new = (img_with_alpha.copy()[:, :, 0:3]).astype(np.float64)

    c = background[
        max(y, 0) : min(y + height, b_height), max(x, 0) : min(x + width, b_width)
    ].astype(np.float64)

    img_new[:, :, 0] *= alpha
    img_new[:, :, 1] *= alpha
    img_new[:, :, 2] *= alpha

    c[:, :, 0] *= 1 - alpha
    c[:, :, 1] *= 1 - alpha
    c[:, :, 2] *= 1 - alpha

    c += img_new
    background[
        max(y, 0) : min(y + height, b_height), max(x, 0) : min(x + width, b_width)
    ] = c.astype(np.uint8)
    if copy:
        return background


def rectangle_xs_ys(pt1, pt2, x2=None, y2=None):
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
    w, h = x2 - x1, y2 - y1
   
    xs = list(range(x1, x2 + 1)) + [x2] * h + list(range(x1, x2 + 1))[:: -1] + [x1] * h
    ys =  [y1] * w + list(range(y1, y2 + 1)) +  [y2] * w  + list(range(y1, y2 + 1))[:: -1]
    
  
        
    return xs, ys

    
    


def draw_alpha_xs_ys(img, xs, ys, color_RGB=None, alpha=.5):
    color_RGB = np.array(color_RGB)
    
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = np.round(alpha * color_RGB + (1 - alpha) * img[ys, xs])
    
def draw_alpha_plus_invertion_xs_ys(img, xs, ys, color_RGB=None, alpha=.5):
    color_RGB = np.array(color_RGB)
    
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    tmp = np.round(alpha * color_RGB + (1 - alpha) * img[ys, xs])
    tmp = - tmp + (255, 255, 255)
    img[ys, xs, :3] = tmp


def draw_cross(img, x, y, color_RGB=(0, 0, 0), a=20):
    img[y - a: y + a, x] ^= np.array([255, 255, 255], dtype="uint8")
    img[y, x - a: x + a] ^= np.array([255, 255, 255], dtype="uint8")

def draw_thick_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=30, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_alpha_rectangle(img, (x1 - i, y1 - i ), (x2 + i, y2 + i),
                             color_RGB=color_RGB,
                            alpha=alpha)
                            
def draw_thick_alpha_plus_inversion_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=30, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_alpha_plus_inversion_rectangle(img, (x1 - i, y1 - i ), (x2 + i, y2 + i),
                             color_RGB=color_RGB,
                            alpha=alpha)
        
    

def draw_filled_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2

    xs, ys = zip(*list(product(list(range(x1, x2)), list(range(y1, y2)))))
    draw_alpha_xs_ys(img, xs, ys,
                            color_RGB=color_RGB,
                        alpha=alpha)
                        
        


def draw_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=None):
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = color_RGB
    

def draw_thick_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=1):
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_rectangle(img, (x1 + i, y1 + i ), (x2 - i, y2 - i), color_RGB=color_RGB)
        
    


def draw_alpha_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=None, alpha=1):
    color_RGB = np.array(color_RGB)
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = np.round(alpha * color_RGB + (1 - alpha) * img[ys, xs])
    
def draw_alpha_plus_inversion_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=None, alpha=1):
    color_RGB = np.array(color_RGB)
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    tmp  = np.round(alpha * color_RGB + (1 - alpha) * img[ys, xs])
    tmp = -tmp + (255, ) * 3
    img[ys, xs, :3] = tmp

def draw_thick_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=30, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_alpha_rectangle(img, (x1 - i, y1 - i ), (x2 + i, y2 + i),
                             color_RGB=color_RGB,
                            alpha=alpha)
                            
        
    



