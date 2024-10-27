import matplotlib.pyplot as plt
import numpy as np
import re
import uuid
import math
from dataclasses import dataclass
from typing import Tuple, List
from kicad_funcs import (
    create_antenna_spiral, 
    write_coils_to_file, 
    create_stack_group,
    create_radial_array,
    generate_coil_stack
)


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

def create_arc(line1: Tuple[Tuple[float, float], Tuple[float, float]], 
               line2: Tuple[Tuple[float, float], Tuple[float, float]], 
               radius: float,
               points: int = 3,
               debug: bool = True) -> List[Tuple[float, float]]:
    """
    Creates an arc tangent to both lines with specified radius.
    Returns list of points approximating the arc.
    
    Args:
        line1: tuple of (start, end) points for first line
        line2: tuple of (start, end) points for second line
        radius: desired radius of the arc
        points: number of points to generate for the arc (minimum 3)
        debug: whether to show construction elements and plot
    
    Returns:
        List of points approximating the arc
    """
    # Ensure minimum of 3 points
    points = max(3, points)
    
    # Extract line points
    (start1, end1) = line1
    (start2, end2) = line2
    assert end1 == start2, "Lines must share an endpoint"
    corner = end1  # The shared point
    
    # Convert points to numpy arrays for vector operations
    start1 = np.array(start1)
    end1 = np.array(end1)
    start2 = np.array(start2)
    end2 = np.array(end2)
    
    # Calculate unit vectors along each line
    v1 = end1 - start1
    v2 = end2 - start2
    
    len1 = np.linalg.norm(v1)
    len2 = np.linalg.norm(v2)
    
    # Validate vector lengths
    if len1 < 1e-10 or len2 < 1e-10:
        raise ValueError("Line segments are too short")
        
    dir1 = -v1 / len1  # Negative because we want vector pointing towards corner
    dir2 = v2 / len2
    
    # Calculate angle between lines
    dot_product = np.dot(dir1, dir2)
    if debug:
        print(f"Vector 1: {v1}, length: {len1}")
        print(f"Vector 2: {v2}, length: {len2}")
        print(f"Direction 1: {dir1}")
        print(f"Direction 2: {dir2}")
        print(f"Dot product: {dot_product}")
    
    angle_between = 2*np.pi - np.arccos(np.clip(dot_product, -1, 1))
    if debug:
        print(f"Angle between lines: {math.degrees(angle_between)} degrees")
    
    # Distance from corner to arc center
    center_distance = radius / math.sin(angle_between / 2)
    
    # Calculate bisector direction
    bisector = dir1 + dir2
    norm = np.linalg.norm(bisector)
    bisector = bisector / norm if norm > 1e-10 else dir1  # Use dir1 if bisector is too short
    
    # Calculate center point
    corner = np.array(corner)
    center = corner + bisector * center_distance
    
    # Calculate tangent points by moving back from corner along each line
    tangent_distance = radius / np.tan(angle_between / 2)
    
    tangent1 = corner - dir1 * tangent_distance
    tangent2 = corner - dir2 * tangent_distance
    
    # Calculate angles from center to tangent points
    start_angle = math.atan2(tangent1[1] - center[1], tangent1[0] - center[0])
    end_angle = math.atan2(tangent2[1] - center[1], tangent2[0] - center[0])
    
    # Ensure we draw the shorter arc
    if abs(end_angle - start_angle) > math.pi:
        if end_angle > start_angle:
            end_angle -= 2 * math.pi
        else:
            end_angle += 2 * math.pi
    
    # Generate arc points
    t = np.linspace(0, 1, points)
    angles = start_angle * (1-t) + end_angle * t
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    result_points = list(zip(x, y))
    
    if debug:
        fig, ax = plt.subplots()
        # Plot construction elements
        ax.plot([start1[0], end1[0]], [start1[1], end1[1]], 'k-', label='Original lines')
        ax.plot([start2[0], end2[0]], [start2[1], end2[1]], 'k-')
        ax.plot(dir1[0], dir1[1], 'b-', label='Direction vectors')
        ax.plot(dir2[0], dir2[1], 'b-')
        ax.plot(center[0], center[1], 'go', label='Arc center')
        ax.plot(corner[0], corner[1], 'ro', label='Corner point')
        ax.plot(tangent1[0], tangent1[1], 'mo', label='Tangent points')
        ax.plot(tangent2[0], tangent2[1], 'mo')
        
        # Plot arc
        for i in range(1, len(result_points)):
            prev = result_points[i-1]
            curr = result_points[i]
            ax.plot([prev[0], curr[0]], [prev[1], curr[1]], 'b-', alpha=0.8)
        
        # Draw radius lines
        ax.plot([center[0], tangent1[0]], [center[1], tangent1[1]], 'g--', alpha=0.5, label='Radius')
        ax.plot([center[0], tangent2[0]], [center[1], tangent2[1]], 'g--', alpha=0.5)
        ax.legend()
        ax.axis('equal')
        plt.show()
    
    return result_points

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
        """Generate points for a continuous rectangular spiral inwards with only horizontal and vertical lines"""
        points = []
        x_start = 0
        y_start = 0
        w = width
        h = height
        
        for turn in range(turns):
            if w <= 0 or h <= 0:
                break
                
            # Bottom edge (left to right)
            points.append(Point(x_start, y_start))
            points.append(Point(x_start + w, y_start))
            
            # Right edge (bottom to top)
            points.append(Point(x_start + w, y_start + h))
            
            # Top edge (right to left)
            points.append(Point(x_start + spacing, y_start + h))
            
            # Left edge (top to bottom, but stop at new start point)
            x_start += spacing
            y_start += spacing
            w -= 2 * spacing
            h -= 2 * spacing
            #points.append(Point(x_start, y_start))
            
        return points

def round_corners(points: List[tuple], radius: float, debug: bool = False, points_per_corner=6) -> List[tuple]:
    """Takes a list of points and returns a new list with rounded corners"""
    
    def dist(p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    if len(points) < 3:
        return points
        
    rounded_points = [points[0]]  # Start with first point
    fig = None
    axes = None
    
    if debug:
        # Calculate number of rows/columns needed for subplots
        n_corners = len(points) - 2  # Exclude first and last corners
        n_rows = int(np.ceil(np.sqrt(n_corners)))
        n_cols = int(np.ceil(n_corners / n_rows))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Process all points except first and last
    for i in range(len(points) - 2):
        p1 = points[i]
        p2 = points[i + 1]
        p3 = points[i + 2]
        
        # Calculate line lengths
        line1_length = dist(p1, p2)
        line2_length = dist(p2, p3)
        
        # TODO make this check angle before deciding to raise error
        if line1_length < 0.5*radius or line2_length < 0.5*radius:
            raise ValueError(f"Line segment at point {i+1} is too short for the specified radius. "
                           f"Line lengths: {line1_length:.2f}, {line2_length:.2f}, Required: {2*radius}")
        
        # Create line tuples for the arc function
        line1 = (p1, p2)
        line2 = (p2, p3)
        
        # Get the arc points
        ax = axes[i] if debug else None
        if ax:
            ax.set_title(f'Corner {i+1}')
            
        arc_points = create_arc(line1, line2, radius, debug=debug, points=points_per_corner)
        if arc_points:
            rounded_points.extend(arc_points)
            
    # Add the final point
    rounded_points.append(points[-1])
    
    if debug:
        plt.tight_layout()
        plt.show()
        
    return rounded_points

def plot(pts):
        plt.figure(figsize=(10, 10))
        x_coords = [p[0] for p in pts]
        y_coords = [p[1] for p in pts]
        plt.plot(x_coords, y_coords, 'bo', alpha=0.5)  # Removed the '-' to not show lines
        plt.axis('equal')
        plt.grid(True)
        plt.title('Rounded Rectangular Spiral Points')
        plt.show()

if __name__ == "__main__":
    # Parameters for the spiral
    width = 45
    height = 15.5
    spacing = 0.25
    turns = 20
    corner_radius = 0.3
    trace_width = 0.1
    num_layers = 4
    
    # Parameters for radial array
    num_copies = 24
    center_x = -15
    center_y = height/2
    start_angle = 0
    spacing_angle = 360/num_copies
    # Generate coil stack template
    coil_stack = generate_coil_stack(
        width=width,
        height=height,
        spacing=spacing,
        turns=turns,
        num_layers=num_layers,
        trace_width=trace_width
    )
    
    # Create radial array
    coil_stacks = create_radial_array(
        coil_stack=coil_stack,
        num_copies=num_copies,
        center_x=center_x,
        center_y=center_y,
        start_angle_deg=start_angle,
        spacing_deg=spacing_angle
    )
    
    # Generate KiCAD sections for each stack
    all_coil_sections = []
    stack_uuids = []
    for stack, stack_uuid in coil_stacks:
        stack_uuids.append(stack_uuid)
        for section in stack.sections:
            # Convert numpy points to list format
            pts = section.points
            
            pts = round_corners(pts, corner_radius)

            # Calculate total path length
            total_length = 0
            for i in range(len(pts)-1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                total_length += np.sqrt(dx*dx + dy*dy)
            print(f"Total path length: {total_length:.2f} mm")

            via_pts = section.via_points if section.via_points is not None else None
            
            # Create KiCAD sections
            main_section, via_section, group_section, group_uuid, element_uuids = create_antenna_spiral(
                pts, 
                mode=section.mode.value,
                trace_width=section.trace_width,
                via_points=via_pts,
                layer=section.layer
            )
            all_coil_sections.append((main_section, via_section, group_section, group_uuid, element_uuids))
    # Write all coils and stack groups to file
    write_coils_to_file("mycoil/mycoil.kicad_pcb", all_coil_sections, stack_uuids,
                       num_sections_per_stack=len(coil_stack.sections),
                       stack_name="Multi-Layer Coil Array")
    print('Saved to file')
