def find_min_square_range(points):
    ((min_x, min_y), (max_x, max_y)) = find_rect_range(points)
    width = max_x - min_x
    height = max_y - min_y

    delta = int(round(abs(width - height) / 2))
    
    if width > height:
        min_y -= delta 
        max_y += delta
    elif height > width:
        min_x -= delta 
        max_x += delta

    return ((min_x, min_y), (max_x, max_y))
