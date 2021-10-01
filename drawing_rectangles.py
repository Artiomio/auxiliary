#!/usr/bin/env python
# coding: utf-8

# In[7]:




# In[8]:


from rectangles_tools import *
import matplotlib.pyplot as plt
import numpy as np
import cv2


# In[9]:


def draw_rect(x1, *args):
    print((x1, ) + tuple(args))
    if isinstance(x1, (list, tuple, np.ndarray)) and len(x1) == 4:
        x1, y1, x2, y2 = x1
        
    else:
        x1, y1, x2, y2 = (x1, ) + args
    cv2.rectangle(z, (x1, y1), (x2, y2), (100, 100, 100), 1)
    


# In[49]:


# Errorous
from itertools import product
def rectangles_intersect(r1, r2):
    rects_concat = [*r1, *r2]
    xs = rects_concat[:: 2]
    ys = rects_concat[1:: 2]
    all_interesting_points = (product(xs, ys))
    s1 = set([p for p in all_interesting_points if  point_is_inside_rect(p, r1)])
    s2 = set([p for p in all_interesting_points  if point_is_inside_rect(p, r2)])
    return len(s1.intersection(s2)) > 0


# In[43]:


from itertools import product
def rectangles_intersect(r1, r2):
    rects_concat = [*r1, *r2]
    xs = rects_concat[:: 2]
    ys = rects_concat[1:: 2]
    all_interesting_points = list((product(xs, ys)))
    s1 = set(p for p in all_interesting_points if point_is_inside_rect(p, r1))
    s2 = set(p for p in all_interesting_points if point_is_inside_rect(p, r2))
    return len(s1.intersection(s2)) > 0


# In[51]:


from itertools import product

def rectangles_intersect(r1, r2):
    rects_concat = [*r1, *r2]
    xs, ys = rects_concat[:: 2], rects_concat[1:: 2]
    all_interesting_points = product(xs, ys)
    s = [p for p in all_interesting_points 
            if point_is_inside_rect(p, r1)
                     and
            point_is_inside_rect(p, r2)]
    
    return len(s) > 0


# In[52]:


r1 = 4, 4, 10, 10
r2 = 5, 5, 20, 20

z = np.zeros((50, 50))
draw_rect(r1)
draw_rect(r2)
plt.imshow(z)
rectangles_intersect(r1, r2)


# In[53]:


rectangles_intersect(r2, r1)


# In[54]:



r1 = 4, 4, 10, 10
r2 = 15, 25, 80, 200

z = np.zeros((350, 350))
draw_rect(r1)
draw_rect(r2)
plt.imshow(z)

rectangles_intersect(r1, r2), rectangles_intersect(r2, r1)


# In[55]:



r1 = 40, 40, 100, 100
r2 = 15, 25, 80, 200

z = np.zeros((350, 350))
draw_rect(r1)
draw_rect(r2)
plt.imshow(z)

rectangles_intersect(r1, r2)


# In[56]:



r1 = 40, 40, 100, 100
r2 = 5, 25, 80, 200

z = np.zeros((350, 350))
draw_rect(r1)
draw_rect(r2)
plt.imshow(z)

rectangles_intersect(r1, r2), rectangles_intersect(r2, r1)


# In[57]:



r1 = 10, 40, 100, 100
r2 = 15, 25, 80, 200

z = np.zeros((350, 350))
draw_rect(r1)
draw_rect(r2)
plt.imshow(z)

rectangles_intersect(r1, r2), rectangles_intersect(r2, r1)


# In[58]:


get_horizontal_gap(r1, r2)


# In[59]:


get_vertical_gap(r1, r2)


# In[60]:


get_vertical_gap(r1, (20, 20, 23, 23))


# In[75]:


def rectangle_xs_ys(pt1, pt2, x2=None, y2=None):
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
    w, h = x2 - x1, y2 - y1
   
    xs = list(range(x1, x2 + 1)) + [x2] * h + list(range(x1, x2 + 1))[:: -1] + [x1] * h
    ys =  [y1] * w + list(range(y1, y2 + 1)) +  [y2] * w  + list(range(y1, y2 + 1))[:: -1]
    
  
        
    return xs, ys


# In[76]:


def draw_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=None):
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img[ys, xs, :3] = color_RGB
    print(color_RGB)
    


# In[77]:




z = np.zeros((51, 52, 3), dtype='uint8')
draw_rectangle(z, -10, 10, 20, 20, color_RGB=(100, 20, 2))
plt.imshow(z)


# In[85]:


def draw_thick_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=1):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_rectangle(img, (x1 + i, y1 + i ), (x2 - i, y2 - i), color_RGB=color_RGB)
        
    


# In[ ]:





# In[86]:




z = np.zeros((51, 52, 3), dtype='uint8')
draw_thick_rectangle(z, 10, 10, 20, 20, color_RGB=(100, 20, 2), thickness=4)
plt.imshow(z)


# In[104]:


def draw_alpha_rectangle(img, pt1, pt2, x2=None, y2=None, color_RGB=(10, 20, 30), alpha=1):
    xs, ys = rectangle_xs_ys(pt1, pt2, x2, y2)
    xs, ys = np.array(xs), np.array(ys)
    color_RGB = np.array(color_RGB)
    mask = (xs < img.shape[1]) & (xs >= 0) & (ys >= 0) & (ys < img.shape[0])
    xs, ys = xs[mask], ys[mask]
    img_points = img[ys, xs, :3].astype('float64')
    
    img_points = (img_points * (1.0 - alpha)).astype("float64")+  color_RGB
    
    img_points = np.round(img_points).astype("uint8")
    img[ys, xs] = img_points
    


# In[120]:




z = np.zeros((51, 52, 3), dtype='uint8')
draw_alpha_rectangle(z, -10, 10, 20, 20, color_RGB=(100, 120, 2), alpha=0.5)
draw_alpha_rectangle(z, 10, 20, 50, 50, color_RGB=(100, 120, 200), alpha=0.3)

draw_thick_alpha_rectangle(z, 15, 25, 55, 53, color_RGB=(100, 120, 200), alpha=0.3)
plt.imshow(z)


# In[113]:


def draw_thick_alpha_rectangle(img,  pt1, pt2, x2=None, y2=None, color_RGB=None, thickness=3, alpha=0.8):
    
    if x2 is not None and y2 is not None:
        x1, y1, x2, y2 = pt1, pt2, x2, y2
    else:
        (x1, y1), (x2, y2) = pt1, pt2
        
    for i in range(thickness):
        draw_alpha_rectangle(img, (x1 + i, y1 + i ), (x2 - i, y2 - i),
                             color_RGB=color_RGB,
                            alpha=alpha)
                            
        
    


# In[117]:




z = np.zeros((51, 52, 3), dtype='uint8')
draw_alpha_rectangle(z, -10, 10, 20, 20, color_RGB=(100, 120, 2), alpha=0.5)
draw_thick_alpha_rectangle(z, 10, 20, 50, 50, color_RGB=(100, 120, 200), alpha=0.1)
plt.imshow(z)


# In[ ]:





# In[ ]:





# In[ ]:





# In[83]:


['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
       'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
       'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
       'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
       'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
       'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
       'hair drier', 'toothbrush'].index('sports ball')


# In[84]:


alpha2_rectangle(z, (-10, -10), (20, 20))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




