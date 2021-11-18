def point_is_inside_rect(pt, r):
    x, y = pt
    x1, y1, x2, y2 = r
    return  (x1 <= x <= x2) and (y1 <= y <= y2)

def outter_rectangle(r1, r2):
    xs = (r1 + r2)[::2]
    ys = (r1 + r2)[1:: 2]
    new_rect = min(xs), min(ys), max(xs), max(ys)
    return new_rect

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


def get_horizontal_gap(r1, r2):
    if rectangles_intersect(r1, r2):
        print("They even intersect!")
        return None
    
    
    r1, r2 = sorted([r1, r2])
    print(r1, r2)
    x1_1, y1_1, x1_2, y1_2 = r1 
    x2_1, y2_1, x2_2, y2_2 = r2
    if x2_1 >= x1_1 and x2_1 >= x1_2:
        return (x2_1 - x1_2)
    else:
        return 0
    
    
    
    

def get_vertical_gap(r1, r2):
    if rectangles_intersect(r1, r2):
        print("They even intersect!")
        return 0
    
    r1, r2 = sorted([r1, r2], key=lambda x: x[1])
    v_gap = min(r2[1::2]) - max(r1[1::2])
    if v_gap < 0:
        return 0
    return v_gap
    





