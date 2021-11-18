from itertools import product

def point_is_inside_rect(pt, r):
    x, y = pt
    x1, y1, x2, y2 = r
    return  (x1 <= x <= x2) and (y1 <= y <= y2)
                


def get_horizontal_gap(r1, r2):
    if rectangles_intersect(r1, r2):
        return None
    
    r1, r2 = sorted([r1, r2])
    x1_1, y1_1, x1_2, y1_2 = r1 
    x2_1, y2_1, x2_2, y2_2 = r2
    if x2_1 >= x1_1 and x2_1 >= x1_2:
        return (x2_1 - x1_2)
    else:
        return 0
    

def get_vertical_gap(r1, r2):
    if rectangles_intersect(r1, r2):
        return 0
    
    r1, r2 = sorted([r1, r2], key=lambda x: x[1])
    v_gap = min(r2[1::2]) - max(r1[1::2])
    if v_gap < 0:
        return 0
    return v_gap
    






def outter_rectangle(r1, r2):
    xs = (r1 + r2)[::2]
    ys = (r1 + r2)[1:: 2]
    new_rect = min(xs), min(ys), max(xs), max(ys)
    return new_rect

def rectangles_intersect(r1, r2):
    rects_concat = [*r1, *r2]
    xs, ys = rects_concat[:: 2], rects_concat[1:: 2]
    all_interesting_points = list(product(xs, ys))
    s = [p for p in all_interesting_points 
            if point_is_inside_rect(p, r1)
                     and
            point_is_inside_rect(p, r2)]
    
    return len(s) > 0





    
def unite_intersecting_and_close_rects(boxes, horizontal_gap=20, vertical_gap=10):
    boxes = list(boxes)
    for m, n in list ( product(range(len(boxes)), range(len(boxes))))*2:
        if m == n: continue
        if boxes[m] is None or boxes[n] is None:
            continue

        horizontal_gap = get_horizontal_gap(boxes[m], boxes[n])
        vertical_gap   = get_vertical_gap(boxes[m], boxes[n])


        if  rectangles_intersect(boxes[m], boxes[n]) or (horizontal_gap <= 20 and vertical_gap < 10):
            boxes[m] = outter_rectangle(boxes[m], boxes[n])
            boxes[n] = None
            

    boxes = [x for x in boxes if x is not None]
    return boxes

