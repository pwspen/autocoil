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

def create_arc(line1: Line, line2: Line, radius: float, ax=None, debug=False) -> List[Point]:
    """
    Creates an arc tangent to both lines with specified radius.
    Returns list of points approximating the arc.
    If ax is provided, visualizes construction process.
    """
    # Find intersection of original lines
    corner = line1.intersect(line2)
    if corner is None:
        return []  # Lines are parallel
        
    # Create offset lines
    offset_line1 = line1.offset(radius)
    offset_line2 = line2.offset(radius)
    
    # Find center point at intersection of offset lines
    center = offset_line1.intersect(offset_line2)
    if center is None:
        return []  # Lines are parallel
        
    # Calculate tangent points
    dir1 = line1.direction
    dir2 = line2.direction
    
    # Tangent points are radius distance from corner along original lines
    tangent1 = Point(
        corner.x - dir1.x * radius,
        corner.y - dir1.y * radius
    )
    tangent2 = Point(
        corner.x + dir2.x * radius,
        corner.y + dir2.y * radius
    )
    
    # Calculate angles from center to tangent points
    start_angle = math.atan2(tangent1.y - center.y, tangent1.x - center.x)
    end_angle = math.atan2(tangent2.y - center.y, tangent2.x - center.x)
    
    if debug and ax:
        # Plot construction elements
        line1.plot(ax, color='black', label='Original lines')
        line2.plot(ax, color='black')
        offset_line1.plot(ax, color='red', alpha=0.5, linestyle='--', label='Offset lines')
        offset_line2.plot(ax, color='red', alpha=0.5, linestyle='--')
        ax.plot(center.x, center.y, 'go', label='Arc center')
        ax.plot(corner.x, corner.y, 'ro', label='Corner point')
        ax.plot(tangent1.x, tangent1.y, 'mo', label='Tangent points')
        ax.plot(tangent2.x, tangent2.y, 'mo')
    
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
    
    if debug and ax:
        # Draw radius lines to tangent points
        ax.plot([center.x, tangent1.x], [center.y, tangent1.y], 'g--', alpha=0.5, label='Radius')
        ax.plot([center.x, tangent2.x], [center.y, tangent2.y], 'g--', alpha=0.5)
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

    def generate_rectangular_spiral(width: float, height: float, spacing: float, turns: int) -> List[Point]:
        """Generate points for a rectangular spiral"""
        points = []
        for i in range(turns * 4):  # 4 sides per turn
            current_inset = (i // 4) * spacing
            w = width - 2 * current_inset
            h = height - 2 * current_inset
            
            if w <= 0 or h <= 0:
                break
                
            # Starting point for this rectangle
            x = current_inset
            y = current_inset
            
            # Add points for this rectangle level
            points.extend([
                Point(x, y),           # bottom-left
                Point(x + w, y),       # bottom-right
                Point(x + w, y + h),   # top-right
                Point(x, y + h),       # top-left
            ])
            
        return points

def round_corners(points: List[Point], radius: float, debug: bool = False) -> List[Point]:
    """Takes a list of points and returns a new list with rounded corners"""
    if len(points) < 3:
        return points
        
    rounded_points = []
    fig = None
    axes = None
    
    if debug:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
    
    for i in range(len(points)):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]
        p3 = points[(i + 2) % len(points)]
        
        # Create lines for this corner
        line1 = Line(p1, p2)
        line2 = Line(p2, p3)
        
        # Get the arc points
        ax = axes[i] if debug else None
        if ax:
            ax.set_title(f'Corner {i+1}')
            
        arc_points = create_arc(line1, line2, radius, ax=ax, debug=debug)
        rounded_points.extend(arc_points)
    
    if debug:
        plt.tight_layout()
        plt.show()
        
    return rounded_points

if __name__ == "__main__":
    # Parameters for the spiral
    width = 50
    height = 25
    spacing = 5
    turns = 2
    corner_radius = 2.0
    
    # Generate rectangular spiral points
    spiral_points = OutlineShape.generate_rectangular_spiral(width, height, spacing, turns)
    
    # Round all corners
    rounded_points = round_corners(spiral_points, corner_radius, debug=True)
    
    # Convert to format expected by create_antenna_spiral
    pts = [[p.x, p.y] for p in rounded_points]
    create_antenna_spiral("mycoil/mycoil.kicad_pcb", pts, mode="trace")
