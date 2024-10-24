import matplotlib.pyplot as plt
import numpy as np
import re
import uuid
import math
from kicad_funcs import create_antenna_spiral

class OutlineShape:
    def __init__(self, points):
        self.points = points
        # Calculate center as average of all points
        self.center = (
            (max(p[0] for p in points) - min(p[0] for p in points))/2,
            (max(p[1] for p in points) - min(p[1] for p in points))/2
        )

    def get_inset_shape(self, trace_spacing, rotation_offset=0):
        new_points = []
        
        for point in self.points:
            # Calculate angle from center to point
            dx = point[0] - self.center[0]
            dy = point[1] - self.center[1]
            angle = math.atan2(dy, dx)
            
            # Normalize angle to [0, 2π]
            angle = (angle + 2 * math.pi) % (2 * math.pi)
            
            # Add rotation offset (in radians)
            total_angle = angle + rotation_offset
            
            # Calculate inset amount: one full trace_spacing per 2π radians
            # The % (2 * math.pi) part makes it reset each full rotation
            inset_amount = trace_spacing * (total_angle % (2 * math.pi)) / (2 * math.pi)
            
            # Calculate distance from center
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Calculate new distance
            new_distance = distance - inset_amount
            
            # Calculate new point
            new_x = self.center[0] + (new_distance * math.cos(angle))
            new_y = self.center[1] + (new_distance * math.sin(angle))
            new_points.append((new_x, new_y))
            
        return OutlineShape(new_points)

    def generate_spiral(self, trace_spacing, num_rotations):
        paths = []
        
        # Generate points for each small angular step
        steps_per_rotation = 36  # adjust for smoothness
        total_steps = num_rotations * steps_per_rotation
        
        for i in range(total_steps):
            rotation = (2 * math.pi * i) / steps_per_rotation
            inset_shape = self.get_inset_shape(trace_spacing, rotation)
            paths.extend(inset_shape.points)
        
        return paths

if __name__ == "__main__":
    width = 50
    height = 25
    trace_spacing = 10
    trace_width = 1
    turns = 1
    via_points = [(0, 0), (25, 12.5), (50, 25)]  # Example via coordinates

    outline = OutlineShape([
        (0, 0),
        (width, 0),
        (width, height),
        (0, height)
    ])

    # Generate spiral points
    pts = [[0, 0],
           [0, 25],
           [25, 25],
           [25, 0],
           [50, 0],
           [50, 25],
           [25, 25],
           [25, 12.5],
           [0, 12.5],
           [0, 25],
           [25, 25],
           [25, 0],
           [0, 0]
           ]
    
    
    create_antenna_spiral("mycoil/mycoil.kicad_pcb", pts, mode="trace")
