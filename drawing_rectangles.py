import numpy as np

def rectangle_xs_ys(pt1, pt2, x2=None, y2=None):
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
    w, h = x2 - x1, y2 - y1
   
    xs = list(range(x1, x2 + 1)) + [x2] * h + list(range(x1, x2 + 1))[:: -1] + [x1] * h
    ys =  [y1] * w + list(range(y1, y2 + 1)) +  [y2] * w  + list(range(y1, y2 + 1))[:: -1]
    
  
        
    return xs, ys


def draw_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=None):
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = color_RGB
    #print(color_RGB)
    

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
    
    


def draw_alpha_xs_ys(img, xs, ys, color_RGB=None, alpha=.5):
    color_RGB = np.array(color_RGB)
    
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = np.round(alpha * color_RGB + (1 - alpha) * img[ys, xs])
    
    

def draw_thick_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=30, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_alpha_rectangle(img, (x1 - i, y1 - i ), (x2 + i, y2 + i),
                             color_RGB=color_RGB,
                            alpha=alpha)
                            
        
    

from itertools import product

def draw_filled_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2


    #xs = list(range(x1, x2)) * abs(y2 - y1) 
    #ys = list(range(y1, y2)) * abs(x2 - 11) 

    xs, ys = zip(*list(product(list(range(x1, x2)), list(range(y1, y2)))))
    draw_alpha_xs_ys(img, xs, ys,
                            color_RGB=color_RGB,
                        alpha=alpha)
                        
        
