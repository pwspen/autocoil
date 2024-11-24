import numpy as np
import uuid
import math
from math import atan2, cos, sin
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum
import matplotlib.pyplot as plt

class CoilMode(Enum):
    POLYGON = "polygon"
    TRACE = "trace"

@dataclass
class CoilSection:
    points: np.ndarray  # Nx2 array of points
    via_points: Optional[np.ndarray]  # Mx2 array of via points
    mode: CoilMode
    trace_width: float
    layer: str
    
@dataclass
class CoilStack:
    sections: List[CoilSection]
    width: float
    height: float
    connection_points: List[np.ndarray] = None
    cut_points_each: List[np.ndarray] = None
    elec_holes: List[np.ndarray] = None
    mount_holes: List[np.ndarray] = None
    cut_points_once: List[np.ndarray] = None

def create_via_section(via_points):
    """
    Create KiCad PCB via sections from a list of coordinate points.
    
    Args:
        via_points (list): List of (x, y) coordinates for via placement
    
    Returns:
        tuple: (via_sections, via_uuids) - The formatted via text and list of UUIDs
    """
    via_sections = []
    via_uuids = []
    
    for x, y in via_points:
        via_uuid = str(uuid.uuid4())
        via_uuids.append(via_uuid)
        
        via_section = f'''(via
		(at {x} {y})
		(size 0.6)
		(drill 0.4)
		(layers "F.Cu" "B.Cu")
		(net 0)
		(uuid "{via_uuid}")
	)'''
        via_sections.append(via_section)
    
    return '\n'.join(via_sections), via_uuids

def create_trace(start_point, end_point, width=0.2, layer="F.Cu"):
    """
    Create a KiCad PCB trace segment between two points.
    
    Args:
        start_point (tuple): (x, y) coordinates of trace start
        end_point (tuple): (x, y) coordinates of trace end
        width (float): Width of the trace in mm
    
    Returns:
        tuple: (trace_section, uuid) - The formatted trace text and its UUID
    """
    trace_uuid = str(uuid.uuid4())
    
    trace_section = f'''(segment
		(start {start_point[0]} {start_point[1]})
		(end {end_point[0]} {end_point[1]})
		(width {width})
		(layer "{layer}")
		(net 0)
		(uuid "{trace_uuid}")
	)'''
    
    return trace_section, trace_uuid

def line_intersection(line1: tuple, line2: tuple) -> tuple:
    """
    Calculate the intersection point of two lines.
    
    Args:
        line1: tuple of two points ((x1, y1), (x2, y2)) defining the first line
        line2: tuple of two points ((x3, y3), (x4, y4)) defining the second line
    
    Returns:
        tuple: (x, y) intersection point coordinates
        
    Raises:
        ValueError: If lines are parallel or coincident
    """
    # Extract points
    (x1, y1), (x2, y2) = line1
    (x3, y3), (x4, y4) = line2
    
    # Convert to numpy arrays for easier calculation
    p1 = np.array([x1, y1])
    p2 = np.array([x2, y2])
    p3 = np.array([x3, y3])
    p4 = np.array([x4, y4])
    
    # Calculate direction vectors
    d1 = p2 - p1
    d2 = p4 - p3
    
    # Calculate cross product to check for parallel lines
    cross_product = np.cross(d1, d2)
    
    if abs(cross_product) < 1e-10:  # Using small epsilon for floating-point comparison
        raise ValueError("Lines are parallel or coincident")
    
    # Calculate intersection parameter for first line
    t = np.cross(p3 - p1, d2) / cross_product
    
    # Calculate intersection point
    intersection = p1 + t * d1
    
    return tuple(intersection)

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

def round_corners(points: List[tuple], radius: float, debug: bool = False, points_per_corner=6, closed_loop=False) -> List[tuple]:
    """Takes a list of points and returns a new list with rounded corners"""
    
    def dist(p1, p2):
        return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)
    
    if len(points) < 3:
        return points
        
    rounded_points = [points[0]] if not closed_loop else [] # Start with first point
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
    sub = 0 if closed_loop else 2
    for i in range(len(points) - sub):
        if closed_loop:
            p1 = points[i % len(points)]
            p2 = points[(i + 1) % len(points)]
            p3 = points[(i + 2) % len(points)]
        else:
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
    if not closed_loop:
        rounded_points.append(points[-1])
    
    if debug:
        plt.tight_layout()
        plt.show()
        
    return rounded_points

def rotate(center, pt, angle):
    # Convert angle to radians
    angle_rad = math.radians(angle)
    # Translate point to origin
    x = pt[0] - center[0]
    y = pt[1] - center[1]
    
    # Rotate point
    x_rot = x * math.cos(angle_rad) - y * math.sin(angle_rad)
    y_rot = x * math.sin(angle_rad) + y * math.cos(angle_rad)
    
    # Translate back
    x_final = x_rot + center[0]
    y_final = y_rot + center[1]
    
    return (x_final, y_final)

def calculate_angle(center, pt1, pt2, tolerance=0.001):    
    # Calculate vectors from center to points
    v1 = (pt1[0] - center[0], pt1[1] - center[1])
    v2 = (pt2[0] - center[0], pt2[1] - center[1])
    
    # Calculate radii
    r1 = math.sqrt(v1[0]**2 + v1[1]**2)
    r2 = math.sqrt(v2[0]**2 + v2[1]**2)
    
    # Check if radii are approximately equal
    if abs(r1 - r2) > tolerance:
        raise ValueError(f"Points are not equidistant from center. Radii: {r1:.6f}, {r2:.6f}")
    
    # Calculate the angle using atan2
    angle1 = math.atan2(v1[1], v1[0])
    angle2 = math.atan2(v2[1], v2[0])
    
    # Calculate the difference, ensuring it's positive and counterclockwise
    angle_diff = math.degrees(angle2 - angle1)
    
    # Normalize to [0, 360)
    angle_diff = angle_diff % 360
    
    return angle_diff

def scale(center, pt, factor):
    """
    Scale a point from a center point by a given factor.
    
    Args:
        center: Tuple (x, y) representing the center point
        pt: Tuple (x, y) representing the point to scale
        factor: Float representing the scale factor (>1 grows, <1 shrinks)
    
    Returns:
        Tuple (x, y) representing the scaled point
    """
    # Calculate vector from center to point
    x = pt[0] - center[0]
    y = pt[1] - center[1]
    
    # Scale the vector
    x_scaled = x * factor
    y_scaled = y * factor
    
    # Translate back relative to center
    x_final = x_scaled + center[0]
    y_final = y_scaled + center[1]
    
    return (x_final, y_final)

def change_radius(center, pt, change):
    """
    Returns a new point at the same angle but different distance from center.
    
    Args:
        center: Tuple of (x, y) representing the center point
        pt: Tuple of (x, y) representing the point to modify
        change: Float/int representing the change in distance from the center (pos or neg)
    
    Returns:
        Tuple (x, y) representing the new point
    """
    # Get current x and y differences
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    
    current_rad = (dx**2 + dy**2)**0.5
    new_rad = current_rad + change

    # Calculate current angle using atan2
    angle = atan2(dy, dx)
    
    # Calculate new point using polar coordinates
    new_x = center[0] + new_rad * cos(angle)
    new_y = center[1] + new_rad * sin(angle)
    
    return (new_x, new_y)

def generate_coil_stack(width: float, height: float, spacing: float, turns: int, 
                       num_layers: int, center: tuple, spacing_angle: float, trace_width: float = 0.2) -> CoilStack:
    """
    Generate a complete coil stack template without UUIDs.
    
    Args:
        width: Width of the coil
        height: Height of the coil
        spacing: Spacing between turns
        turns: Number of turns
        num_layers: Number of layers in the stack
        trace_width: Width of traces
        
    Returns:
        CoilStack object containing all sections
    """
    # Generate base spiral points using numpy
    spiral_points = []
    x_start = 0
    y_start = 0
    x_start_center = 0

    triangle_mode = False
    for turn in range(turns):

        # CORE OF PROGRAM 

        # Top left,
        # Top right,
        # Bottom right,
        # Bottom left

        # Trapezoid shape
        x_taper = 1
        y_taper = 1
        h_factor = 0.8
        right_y_factor = 1.2
        center_left_x_adjust = 1

        top_left = [x_start*x_taper, y_start*y_taper + height*0.5*h_factor]
        top_right = [width - x_start, y_start*right_y_factor]
        bottom_right = [width - x_start, -y_start*right_y_factor + height]
        bottom_left = [x_start*x_taper, -y_start*y_taper + height - height*0.5*h_factor]
        center_left = line_intersection((top_left, top_right), (bottom_left, bottom_right))
        center_left = (center_left[0] + spacing*center_left_x_adjust, center_left[1])

        if bottom_left[1] - top_left[1] < 2*spacing: # Transition to triangle after running out of room
            spiral_points.extend([
                    bottom_right,
                    center_left,
                    top_right
                ])
        else:
            spiral_points.extend([
                bottom_right,
                bottom_left,
                top_left,
                top_right
            ])
            x_start_center += spacing


        # For triangle shape
        # spiral_points.extend([
        #     [x_start*4, (height/2)],
        #     [x_start + w, y_start],
        #     [x_start + w, y_start + h],
        #     #[x_start + spacing, y_start + h]
        # ])

        # For rectangle shape:
        # spiral_points.extend([
        #     [x_start, y_start],
        #     [x_start + w, y_start],
        #     [x_start + w, y_start + h],
        #     [x_start + spacing, y_start + h]
        # ])
        
        x_start += spacing
        y_start += spacing
    
    spiral_points = np.array(spiral_points)
    
    # Inner via calculations
    via_count = num_layers // 2
    via_spacing = 3.5 * spacing
    inner_via_x = -turns * spacing + width - spacing*0.75
    inner_via_yspace = height - (spacing*turns*2)*right_y_factor

    # Below is kinda complicated - 
    # first evenly spaces out, then moves vias away from centerpoint
    # to prevent overlap with traces.
    # may not work well with 4+ layers
    squish_fac = 2
    inner_via_y = ((height/2 - (inner_via_yspace)/2 + i * inner_via_yspace/(via_count+1)) for i in range(1, via_count + 1))
    inner_via_y = ((height/2 - y) * squish_fac + height/2 for y in inner_via_y)

    vias = np.array([np.array([inner_via_x, y]) for y in inner_via_y])
    
    # Generate sections for each layer
    sections = []
    
    stack_endpoints = []
    for i in range(num_layers):
        add_vias = np.array([])
        # Create layer points
        layer_pts = spiral_points.copy()
        via = vias[i // 2]
        if i == 0 or (i == num_layers - 1):
            # Create hookup tail and terminating outer via
            orig_init_pt = spiral_points[0]
            new_orig_pt = np.array([[orig_init_pt[0] + spacing*1.5, orig_init_pt[1] - spacing*2]])
            add_vias = new_orig_pt
            stack_endpoints.append(new_orig_pt)
            layer_pts = np.vstack([new_orig_pt, layer_pts])
            layer = "F.Cu" if i == 0 else "B.Cu"
        else:
            layer = f"In{i}.Cu"
        
        # Add connection to inner via (every layer has this)
        #corner_pt = np.array([[via[0], height/2 - 0.7]])
        final_pt = np.array([via])
        layer_pts = np.vstack([layer_pts, final_pt])
        
        if i % 2 == 0:
            add_vias = np.array([via]) if add_vias.size == 0 else np.vstack([add_vias, via])

        if "In" in layer:
            # Non-terminating outer via
            via_connect_pt = np.array([layer_pts[0][0] + (spacing*1 + via_spacing*((i-1) // 2)), height*0.5])
            layer_pts = np.vstack([via_connect_pt, layer_pts])
            add_vias = np.array([via_connect_pt]) if add_vias.size == 0 else np.vstack([add_vias, via_connect_pt])

        
        # Flip points if needed
        if (i + 1) % 2 == 0:  # flip_y
            layer_pts[:, 1] = -layer_pts[:, 1] + height
            if add_vias.size > 0:
                add_vias[:, 1] = -add_vias[:, 1] + height
                
        sections.append(CoilSection(
            points=layer_pts,
            via_points=add_vias if add_vias.size > 0 else None,
            mode=CoilMode.TRACE,
            trace_width=trace_width,
            layer=layer
        ))
        add_vias = np.array([])
    
    # Use spacing angle(s) and centerpoint to make trace to next coil

    ipt = stack_endpoints[0][0]
    pt = stack_endpoints[-1][0]
    ang = calculate_angle(center, pt, ipt)


    interconnect_spacing_angle = 2*spacing_angle + (spacing_angle - ang)
    connection_points = []
    connection_points.append(np.array(pt))
    growfac = 1.013
    pt = scale(center, pt, growfac)
    num_pts = 10
    for i in range(0, num_pts):
        connection_points.append(np.array(rotate(center, pt, -interconnect_spacing_angle*(i/(num_pts-1)))))
    connection_points.append(scale(center, connection_points[-1], 1/growfac))
    connection_points = np.array(connection_points)
    
    # Triangle cutout
    cut_points = (
        (center_left[0] + spacing*12, height/2), 
        (inner_via_x - spacing*2.5, height/2 - inner_via_yspace*0.47), 
        (inner_via_x - spacing*2.5, height/2 + inner_via_yspace*0.47))
    cut_points = round_corners(cut_points, 0.5, points_per_corner=5, closed_loop=True)

    def gen_radial_pt(ang, radiuus):
        print(f'called with {ang=}, {radiuus=}')
        return (center[0] + radiuus*np.cos(np.deg2rad(ang)), center[1] + radiuus*np.sin(np.deg2rad(ang)))
    
    elec_holes = []
    mount_holes = []
    cut_points_once = []

    holerow1_ang = 0
    holerow2_ang = spacing_angle
    line_offset_ang = 6.5 # 1 deg

    cutrad = 62.5
    holerad = 64
    elecholespacing = 4
    mholespacing = 7
    cutdist = 12
    cutcurve_pts = 10
    hole_rot=4
    print(center)
    for i, a in enumerate((holerow1_ang, holerow2_ang)):
        h1 = gen_radial_pt(a, holerad)
        h2 = rotate(center=center, pt=h1, angle=-hole_rot)
        h3 = rotate(center=center, pt=h1, angle=hole_rot)
        elec_holes.append(h1)
        elec_holes.append(h2)
        elec_holes.append(h3)
        m1 = change_radius(center, h1, mholespacing)
        mount_holes.append(m1)

        if i == 0:
            line_ang = holerow1_ang - line_offset_ang
            c1 = gen_radial_pt(line_ang, cutrad)
            c2 = change_radius(center, c1, cutdist)
            cut_points_once.append(c1)
            cut_points_once.append(c2)

            for i in range(cutcurve_pts):
                c = rotate(center=center, pt=c2, angle=(holerow2_ang - holerow1_ang + 2*line_offset_ang)*(i/(cutcurve_pts-1)))
                cut_points_once.append(c)

            mount_holes.append(rotate(center=center, pt=m1, angle=spacing_angle/2))
        else:
            line_ang = holerow2_ang + line_offset_ang
            c1 = gen_radial_pt(line_ang, cutrad)
            c2 = change_radius(center, c1, cutdist)
            cut_points_once.append(c2)
            cut_points_once.append(c1)  
        

    return CoilStack(sections=sections, width=width, height=height, connection_points=connection_points, cut_points_each=cut_points, elec_holes=elec_holes, mount_holes=mount_holes, cut_points_once=cut_points_once)

def create_antenna_spiral(all_points, trace_width=0.2, via_points=None, flip_x=False, flip_y=False, name="Coil Layer", layer="F.Cu", mode="trace"):
    """
    Creates KiCad PCB sections for an antenna spiral pattern.
    
    Args:
        all_points (list): List of (x,y) coordinates defining the antenna shape
        mode (str): Either "polygon" or "trace" to determine how the antenna is created
        trace_width (float): Width of traces when in trace mode
        via_points (list): Optional list of (x, y) coordinates for via placement
        flip_x (bool): If True, flip points across y axis (negate x coordinates)
        flip_y (bool): If True, flip points across x axis (negate y coordinates)
        layer (str): KiCad layer name (e.g. "F.Cu", "B.Cu", etc)
        
    Returns:
        tuple: (main_section, via_sections, group_section, member_uuids)
    """
    
    # Calculate bounds before flipping
    min_x = min(x for x, y in all_points)
    max_x = max(x for x, y in all_points)
    min_y = min(y for x, y in all_points)
    max_y = max(y for x, y in all_points)
    width = max_x - min_x
    height = max_y - min_y

    # Apply flipping transformations
    if flip_x or flip_y:
        # First flip the points
        all_points = [((-x if flip_x else x), (-y if flip_y else y)) for x, y in all_points]
        if via_points:
            via_points = [((-x if flip_x else x), (-y if flip_y else y)) for x, y in via_points]
        
        # Then adjust position to maintain stacking
        if flip_x:
            # Shift points right by width
            all_points = [(x + width, y) for x, y in all_points]
            if via_points:
                via_points = [(x + width, y) for x, y in via_points]
        if flip_y:
            # Shift points up by height
            all_points = [(x, y + height) for x, y in all_points]
            if via_points:
                via_points = [(x, y + height) for x, y in via_points]

    # Generate UUIDs for elements
    trace_uuids = []
    via_uuids = []
    
    if mode == "trace":
        # Create traces between consecutive points
        trace_sections = []
        for i in range(len(all_points)-1):
            start = all_points[i]
            end = all_points[i+1]
            trace_section, trace_uuid = create_trace(start, end, width=trace_width, layer=layer)
            trace_sections.append(trace_section)
            trace_uuids.append(trace_uuid)
        main_section = "\n".join(trace_sections)

        # Generate via sections if via points were provided
        via_sections = ""
        if via_points:
            via_text, new_via_uuids = create_via_section(via_points)
            via_sections = via_text
            via_uuids.extend(new_via_uuids)

        # Create group section for this coil (including its traces/polygon and vias)
        all_element_uuids = trace_uuids + via_uuids
        group_uuid = str(uuid.uuid4())
        group_section, id = create_stack_group(all_element_uuids, name=f"{name} {layer}")
    elif mode == "polygon":
        puuid = uuid.uuid4()
        poly = f'''(gr_poly
            (pts
                {(f" ".join("(xy {} {})".format(x, y) for x, y in all_points))}
            )
            (stroke
                (width {trace_width})
                (type solid)
            )
            (fill none)
            (layer "{layer}")
            (uuid "{puuid}")
        )'''
        main_section = poly
        group_uuid = [puuid]
        all_element_uuids = [puuid]
        via_sections, group_section, via_uuids = "", "", [] # None supported for polygon
    else:
        raise NotImplementedError(f"Unsupported mode: {mode}")

    return main_section, via_sections, group_section, [group_uuid], all_element_uuids

def generate_holes(pts, size, drill):
    section = []
    all_uuids = []
    for pt in pts:
        id = str(uuid.uuid4())
        all_uuids.append(id)
        section.append(
        f'''(footprint ""
            (layer "F.Cu")
            (uuid "{id}")
            (at {pt[0]} {pt[1]})
            (pad "1" thru_hole circle
                (at 0 0)
                (size {size} {size})
                (drill {drill})
                (layers "*.Cu" "*.Mask")
                (remove_unused_layers no)
                (uuid "{str(uuid.uuid4())}")
            )
        )
        ''')

    return '\n'.join(section), "", "", [], [all_uuids]

def generate_lines(pts, width, layer):
    section = []
    all_uuids = []
    for i in range(len(pts) - 1):
        id = str(uuid.uuid4())
        all_uuids.append(id)
        section.append(
        f'''(gr_line
                (start {pts[i][0]} {pts[i][1]})
                (end {pts[i+1][0]} {pts[i+1][1]})
                (stroke
                    (width {width})
                    (type default)
                )
                (layer "{layer}")
                (uuid "{id}")
            )''')

    return '\n'.join(section), "", "", [], [all_uuids]

def create_stack_group(member_uuids, name="Coil Stack"):
    """
    Create a KiCad PCB group section for a stack of coils.
    
    Args:
        member_uuids (list): List of all UUIDs to include in the group
        name (str): Name for the group
        
    Returns:
        tuple: (group_section, group_uuid) - The formatted group text and its UUID
    """
    # Format member UUIDs with at most 2 per line
    members_text = []
    for i in range(0, len(member_uuids), 2):
        chunk = member_uuids[i:i + 2]
        members_text.append(f'\t\t{" ".join(f""""{uuid}" """ for uuid in chunk)}')
    
    group_uuid = str(uuid.uuid4())
    
    group_section = f'''(group "{name}"
		(uuid "{group_uuid}")
		(members
{"".join(members_text)}
		)
	)'''
    return group_section, group_uuid

def transform_point(point, center_x, center_y, angle_rad):
    """
    Transform a point around a center point by given angle in radians.
    
    Args:
        point (tuple): (x, y) coordinates to transform
        center_x (float): X coordinate of rotation center
        center_y (float): Y coordinate of rotation center
        angle_rad (float): Rotation angle in radians
        
    Returns:
        tuple: Transformed (x, y) coordinates
    """
    # Translate point to origin
    x = point[0] - center_x
    y = point[1] - center_y
    
    # Rotate
    cos_theta = math.cos(angle_rad)
    sin_theta = math.sin(angle_rad)
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta
    
    # Translate back
    return (new_x + center_x, new_y + center_y)

def transform_points(points, center_x, center_y, angle_rad):
    """Transform a list of points around a center by given angle"""
    return [transform_point(p, center_x, center_y, angle_rad) for p in points]

def create_radial_array(coil_stack: CoilStack, num_copies: int, 
                       center_x: float, center_y: float, 
                       start_angle_deg: float = 0, spacing_deg: float = 45) -> List[Tuple[CoilStack, str]]:
    """
    Create a radial array of coil stacks.
    
    Args:
        coil_sections (list): List of (main_section, via_sections, group_section, member_uuids) tuples
        num_copies (int): Number of copies to create including original
        center_x (float): X coordinate of rotation center
        center_y (float): Y coordinate of rotation center
        start_angle_deg (float): Starting angle in degrees
        spacing_deg (float): Angular spacing between copies in degrees
        
    Returns:
        list: List of tuples (transformed_stack, stack_uuid)
    """
    transformed_stacks = []
    
    for copy_num in range(num_copies):
        angle_rad = math.radians(start_angle_deg + copy_num * spacing_deg)
        stack_uuid = str(uuid.uuid4())
        
        phase = copy_num % 3

        if phase == 0:
            conn_layer = "F.Cu"
        elif phase == 1:
            conn_layer = "In1.Cu"
        elif phase == 2:
            conn_layer = "B.Cu"

        # Create a new stack with transformed sections
        transformed_sections = []
        for section in coil_stack.sections:
            # Transform points
            transformed_points = transform_points(section.points, center_x, center_y, angle_rad)
            transformed_via_points = None
            if section.via_points is not None:
                transformed_via_points = transform_points(section.via_points, center_x, center_y, angle_rad)
                
            transformed_sections.append(CoilSection(
                points=transformed_points,
                via_points=transformed_via_points,
                mode=section.mode,
                trace_width=section.trace_width,
                layer=section.layer
            ))
            
        transformed_stacks.append((
            CoilStack(
                sections=transformed_sections,
                width=coil_stack.width,
                height=coil_stack.height
            ),
            {
            "uuid": stack_uuid,
            "conn_layer": conn_layer,
            "conn_pts": transform_points(coil_stack.connection_points, center_x, center_y, angle_rad),
            "cut_pts_each": transform_points(coil_stack.cut_points_each, center_x, center_y, angle_rad),
            "elec_holes": coil_stack.elec_holes,
            "mount_holes": coil_stack.mount_holes,
            "cut_pts_once": coil_stack.cut_points_once
            }
        ))
    
    return transformed_stacks

def write_coils_to_file(filename, coil_sections, stack_uuids, num_sections_per_stack, stack_name="Coil Stack"):
    """
    Writes multiple coil sections to a KiCad PCB file.
    
    Args:
        filename (str): KiCad PCB filename
        coil_sections (list): List of (main_section, via_sections, group_section, member_uuids, element_uuids) tuples
        stack_uuids (list): List of UUIDs for each stack group
        num_sections_per_stack (int): Number of sections in each coil stack
        stack_name (str): Name for the final array group
    """
    # Read the KiCad PCB file
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Truncate at line 205 and add closing parenthesis
    base_content = ''.join(lines[:200]) + ')\n'
    
    # Combine all sections from all coils
    new_content = []
    all_stack_uuids = []
    
    # Group coils by stack
    current_stack_members = []
    for i, (main_section, via_sections, group_section, group_uuid, element_uuids) in enumerate(coil_sections):
        new_content.extend([main_section, via_sections, group_section])
        current_stack_members.extend(element_uuids + group_uuid)
        
        # When we've processed all sections for a stack, create its group
        if (i + 1) % num_sections_per_stack == 0:
            stack_idx = i // num_sections_per_stack
            stack_group, stack_uuid = create_stack_group(current_stack_members, 
                                           name=f"Coil Stack {stack_idx + 1}")
            new_content.append(stack_group)
            all_stack_uuids.append(stack_uuid)
            current_stack_members = []
    # Create a group for the entire array
    array_group, array_group_uuid = create_stack_group(all_stack_uuids, name=stack_name)
    new_content.append(array_group)
    
    # Filter out None values and ensure all items are strings
    filtered_content = [str(item) for item in new_content if item is not None]
    
    # Join all sections and add final parenthesis
    new_content = "\n\t".join(filtered_content) + "\n)"
    
    # Write the base content plus new sections
    with open(filename, 'w') as f:
        f.write(base_content[:-2])  # Remove the last \n)
        f.write(new_content)
