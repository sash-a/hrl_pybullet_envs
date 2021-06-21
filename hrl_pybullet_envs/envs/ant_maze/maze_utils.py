from typing import NamedTuple

import numpy as np
from numpy import exp, abs, angle


# line segment code from: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

class Point(NamedTuple):
    x: float
    y: float


def on_segment(p, q, r):
    """Given three colinear points p, q, r, the function checks if point q lies on line segment 'pr'"""
    if ((q.x <= max(p.x, r.x)) and (q.x >= min(p.x, r.x)) and
            (q.y <= max(p.y, r.y)) and (q.y >= min(p.y, r.y))):
        return True
    return False


def orientation(p, q, r):
    """
    to find the orientation of an ordered triplet (p,q,r)
    function returns the following values:
    0 : Colinear points
    1 : Clockwise points
    2 : Counterclockwise
    """
    val = (float(q.y - p.y) * (r.x - q.x)) - (float(q.x - p.x) * (r.y - q.y))
    if val > 0:  # Clockwise orientation
        return 1
    elif val < 0:  # Counterclockwise orientation
        return 2
    else:  # Colinear orientation
        return 0


def intersection(p1, p2, p3, p4):
    inter = find_intersection(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)
    if inter is None:
        return None

    return Point(*inter)


# TODO numba would be great here
def find_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    d = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if d == 0:  # no intersection
        return None
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / d
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / d
    return [px, py]


def quadrant(p: Point):
    """returns the quadrant 1/2/3/4 of the point"""
    if p.x >= 0 and p.y >= 0:
        return 1
    elif p.x >= 0 and p.y <= 0:
        return 4
    elif p.x <= 0 and p.y >= 0:
        return 2
    elif p.x <= 0 and p.y <= 0:
        return 3
    else:
        raise Exception(f'This should never happen, attempting to get quadrant for point:{p}')


def cart2pol(x, y):
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
