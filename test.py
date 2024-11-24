import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

from math import atan2, cos, sin, sqrt

def change_radius(center, pt, change):
    """
    Returns a new point at the same angle but different distance from center.
    
    Args:
        center: Tuple of (x, y) representing the center point
        pt: Tuple of (x, y) representing the point to modify
        change: Float/int representing the new distance from center
    
    Returns:
        Tuple (x, y) representing the new point
    """
    # Get current x and y differences
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    
    # Calculate current angle using atan2
    angle = atan2(dy, dx)
    
    # Calculate new point using polar coordinates
    new_x = center[0] + change * cos(angle)
    new_y = center[1] + change * sin(angle)
    
    return (new_x, new_y)

c1 = (0, 0)
p1 = (1, 1)

c2 = (10, 10)
p2 = (9, 9)

print(change_radius(c1, p1, 2))
print(change_radius(c2, p2, 2))