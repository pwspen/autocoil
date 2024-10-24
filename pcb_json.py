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
        
    def offset(self, distance: float) -> 'Line':
        """Returns a new line parallel to this one, offset by distance"""
        perp = self.perpendicular(self.start)
        offset_start = Point(
            self.start.x + perp.x * distance,
            self.start.y + perp.y * distance
        )
        offset_end = Point(
            self.end.x + perp.x * distance,
            self.end.y + perp.y * distance
        )
        return Line(offset_start, offset_end)
        
    def plot(self, ax, color='blue', alpha=1.0, linestyle='-', label=None):
        """Plot the line on a matplotlib axis"""
        ax.plot([self.start.x, self.end.x], 
                [self.start.y, self.end.y], 
                color=color, alpha=alpha, linestyle=linestyle, label=label)
    
    def intersect(self, other: 'Line') -> Point:
        """Returns intersection point of two lines"""
        # Line 1 represented as a1x + b1y = c1
        a1 = self.end.y - self.start.y
        b1 = self.start.x - self.end.x
        c1 = a1 * self.start.x + b1 * self.start.y
        
        # Line 2 represented as a2x + b2y = c2
        a2 = other.end.y - other.start.y
        b2 = other.start.x - other.end.x
        c2 = a2 * other.start.x + b2 * other.start.y
        
        determinant = a1 * b2 - a2 * b1
        if abs(determinant) < 1e-10:  # Lines are parallel
            return None
            
        x = (b2 * c1 - b1 * c2) / determinant
        y = (a1 * c2 - a2 * c1) / determinant
        return Point(x, y)

def create_arc(line1: Line, line2: Line, radius: float, ax=None) -> List[Point]:
    """
    Creates an arc tangent to both lines with specified radius.
    Returns list of points approximating the arc.
    If ax is provided, visualizes construction process.
    """
    if ax:
        # Plot original lines
        line1.plot(ax, color='black', label='Original lines')
        line2.plot(ax, color='black')
        
    # Create offset lines
    offset_line1 = line1.offset(radius)
    offset_line2 = line2.offset(radius)
    
    if ax:
        # Plot offset lines
        offset_line1.plot(ax, color='red', alpha=0.5, linestyle='--', label='Offset lines')
        offset_line2.plot(ax, color='red', alpha=0.5, linestyle='--')
    
    # Find center point at intersection of offset lines
    center = offset_line1.intersect(offset_line2)
    if center is None:
        return []  # Lines are parallel, can't create arc
        
    if ax:
        # Plot center point
        ax.plot(center.x, center.y, 'go', label='Arc center')
    
    # Calculate start and end angles
    start_angle = math.atan2(line1.end.y - center.y, line1.end.x - center.x)
    end_angle = math.atan2(line2.start.y - center.y, line2.start.x - center.x)
    
    # Generate arc points
    points = []
    steps = 16  # Increased for smoother visualization
    if end_angle < start_angle:
        end_angle += 2 * math.pi
    
    for i in range(steps + 1):
        t = i / steps
        angle = start_angle * (1-t) + end_angle * t
        point = Point(
            center.x + radius * math.cos(angle),
            center.y + radius * math.sin(angle)
        )
        points.append(point)
        
        if ax and i > 0:
            # Plot arc segments
            prev = points[i-1]
            ax.plot([prev.x, point.x], [prev.y, point.y], 'b-', alpha=0.8)
            
    if ax:
        # Draw radius lines to start and end points
        ax.plot([center.x, points[0].x], [center.y, points[0].y], 'g--', alpha=0.5, label='Radius')
        ax.plot([center.x, points[-1].x], [center.y, points[-1].y], 'g--', alpha=0.5)
        ax.legend()
        ax.axis('equal')
    
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
            dx = point.x - self.center.x
            dy = point.y - self.center.y
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
            new_points.append(Point(new_x, new_y))
            
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

    # Create a rounded rectangle outline
    corner_points = [
        Point(0, 0),
        Point(width, 0),
        Point(width, height),
        Point(0, height)
    ]
    
    outline = OutlineShape(corner_points)

    # Set up the plotting
    fig = plt.figure(figsize=(15, 10))
    
    # Generate outline with rounded corners
    outline_points = []
    radius = 5.0  # Corner radius
    
    for i in range(len(corner_points)):
        p1 = corner_points[i]
        p2 = corner_points[(i + 1) % len(corner_points)]
        p3 = corner_points[(i + 2) % len(corner_points)]
        
        # Create subplot for this corner
        ax = fig.add_subplot(2, 2, i+1)
        ax.set_title(f'Corner {i+1}')
        
        # Create lines for this corner
        line1 = Line(p1, p2)
        line2 = Line(p2, p3)
        
        # Add arc points with visualization
        arc_points = create_arc(line1, line2, radius, ax=ax)
        outline_points.extend(arc_points)
    
    plt.tight_layout()
    plt.show()
    
    # Convert to format expected by create_antenna_spiral
    pts = [[p.x, p.y] for p in outline_points]
    create_antenna_spiral("mycoil/mycoil.kicad_pcb", pts, mode="trace")
