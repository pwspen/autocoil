import matplotlib.pyplot as plt
import numpy as np
import re
import uuid
import math
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Point:
    x: float
    y: float
    
    def __iter__(self):
        return iter((self.x, self.y))
        
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
        
    def distance_to(self, other) -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

@dataclass
class Line:
    start: Point
    end: Point
    
    @property
    def direction(self) -> Point:
        """Returns normalized direction vector"""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = math.sqrt(dx*dx + dy*dy)
        return Point(dx/length, dy/length)
    
    def perpendicular(self, point: Point) -> Point:
        """Returns perpendicular vector at given point"""
        dir = self.direction
        return Point(-dir.y, dir.x)

def create_arc(line1: Line, line2: Line, radius: float) -> List[Point]:
    """
    Creates an arc tangent to both lines with specified radius.
    Returns list of points approximating the arc.
    """
    # Get perpendicular vectors
    perp1 = line1.perpendicular(line1.end)
    perp2 = line2.perpendicular(line2.start)
    
    # Center is intersection of offset lines
    center1 = Point(
        line1.end.x + perp1.x * radius,
        line1.end.y + perp1.y * radius
    )
    center2 = Point(
        line2.start.x + perp2.x * radius,
        line2.start.y + perp2.y * radius
    )
    
    # Calculate arc center
    dx = center2.x - center1.x
    dy = center2.y - center1.y
    dist = math.sqrt(dx*dx + dy*dy)
    center = Point(
        center1.x + dx * 0.5,
        center1.y + dy * 0.5
    )
    
    # Calculate start and end angles
    start_angle = math.atan2(line1.end.y - center.y, line1.end.x - center.x)
    end_angle = math.atan2(line2.start.y - center.y, line2.start.x - center.x)
    
    # Generate arc points
    points = []
    steps = 32  # Number of segments to approximate arc
    if end_angle < start_angle:
        end_angle += 2 * math.pi
    
    for i in range(steps + 1):
        t = i / steps
        angle = start_angle * (1-t) + end_angle * t
        points.append(Point(
            center.x + radius * math.cos(angle),
            center.y + radius * math.sin(angle)
        ))
    
    return points
from kicad_funcs import create_antenna_spiral

class OutlineShape:
    def __init__(self, points: List[Point]):
        self.points = points
        # Calculate center as average of all points
        self.center = Point(
            (max(p.x for p in points) - min(p.x for p in points))/2,
            (max(p.y for p in points) - min(p.y for p in points))/2
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

    # Create a rounded rectangle outline
    corner_points = [
        Point(0, 0),
        Point(width, 0),
        Point(width, height),
        Point(0, height)
    ]
    
    # Generate outline with rounded corners
    outline_points = []
    radius = 5.0  # Corner radius
    
    for i in range(len(corner_points)):
        p1 = corner_points[i]
        p2 = corner_points[(i + 1) % len(corner_points)]
        p3 = corner_points[(i + 2) % len(corner_points)]
        
        # Create lines for this corner
        line1 = Line(p1, p2)
        line2 = Line(p2, p3)
        
        # Add arc points
        arc_points = create_arc(line1, line2, radius)
        outline_points.extend(arc_points)
    
    # Convert to format expected by create_antenna_spiral
    pts = [[p.x, p.y] for p in outline_points]
    create_antenna_spiral("mycoil/mycoil.kicad_pcb", pts, mode="trace")
