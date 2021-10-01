""" Finds the minimal horizontal rectangle containing given points
    points is a list of points e.g. [(3,5),(-1,4),(5,6)]
"""
def find_rect_range(points):
    min_x = min(points, key=lambda x: x[0])[0]
    max_x = max(points, key=lambda x: x[0])[0]


    min_y = min(points, key=lambda x: x[1])[1]
    max_y = max(points, key=lambda x: x[1])[1]
    return((min_x, min_y), (max_x, max_y))