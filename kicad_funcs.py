import numpy as np
import re
import uuid
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple
from enum import Enum

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
		(drill 0.3)
		(layers "F.Cu" "B.Cu")
		(net 0)
		(uuid "{via_uuid}")
	)'''
        via_sections.append(via_section)
    
    return '\n'.join(via_sections), via_uuids

def create_group_section(member_uuids, name=""):
    
    """
    Create a KiCad PCB group section for the given member UUIDs.
    
    Args:
        member_uuids (list): List of UUID strings to include in the group
        name (str): Name for the group
        
    Returns:
        str: Formatted group section text
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
    
    return group_section

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

def generate_coil_stack(width: float, height: float, spacing: float, turns: int, 
                       num_layers: int, trace_width: float = 0.2) -> CoilStack:
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
    w = width
    h = height
    
    for turn in range(turns):
        if w <= 0 or h <= 0:
            break
            
        spiral_points.extend([
            [x_start, y_start],
            [x_start + w, y_start],
            [x_start + w, y_start + h],
            [x_start + spacing, y_start + h]
        ])
        
        x_start += spacing
        y_start += spacing
        w -= 2 * spacing
        h -= 2 * spacing
    
    spiral_points = np.array(spiral_points)
    
    # Calculate via positions
    via_count = num_layers // 2
    via_spacing = 3.5 * spacing
    via_start = turns * spacing
    vias = np.array([(via_start + i*via_spacing, height*0.5) for i in range(1, via_count + 1)])
    
    # Generate sections for each layer
    sections = []
    add_vias = vias.copy()
    
    for i in range(num_layers):
        if i == 0:
            layer = "F.Cu"
        elif i == num_layers - 1:
            layer = "B.Cu"
        else:
            layer = f"In{i}.Cu"
            
        # Create layer points
        layer_pts = spiral_points.copy()
        via = vias[i // 2]
        
        # Add connection to via
        corner_pt = np.array([[via[0], layer_pts[-1][1]]])
        final_pt = np.array([[via[0], via[1]]])
        layer_pts = np.vstack([layer_pts[:-1], corner_pt, final_pt])
        
        if "In" in layer:
            # Make room for via
            via_connect_pt = np.array([[layer_pts[0][0] - (spacing*1.25 + via_spacing*((i-1) // 2)), height*0.5]])
            layer_pts = np.vstack([via_connect_pt, layer_pts])
            
            if i % 2 == 0:
                add_vias = via_connect_pt if add_vias.size == 0 else np.vstack([add_vias, via_connect_pt])
        
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
        
    return CoilStack(sections=sections, width=width, height=height)

def create_antenna_spiral(all_points, mode="polygon", trace_width=0.2, via_points=None, flip_x=False, flip_y=False, layer="F.Cu"):
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
    coil_group_uuid = str(uuid.uuid4())
    
    if mode == "polygon":
        # Generate the points section string for polygon
        points_str = '\n\t\t\t'.join(f'(xy {x:.6f} {y:.6f})' for x, y in all_points)
        
        # Create the full polygon section
        poly_uuid = str(uuid.uuid4())
        trace_uuids.append(poly_uuid)
        main_section = f'''(gr_poly
            (pts
                {points_str}
            )
            (stroke
                (width 0)
                (type solid)
            )
            (fill solid)
            (layer "{layer}")
            (uuid "{poly_uuid}")
        )'''
    else:  # trace mode
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
    group_section = create_group_section(all_element_uuids, name=f"Coil Layer {layer}")
    
    return main_section, via_sections, group_section, [coil_group_uuid]

def create_stack_group(member_uuids, name="Coil Stack"):
    """
    Create a KiCad PCB group section for a stack of coils.
    
    Args:
        member_uuids (list): List of all UUIDs to include in the group
        name (str): Name for the group
        
    Returns:
        str: Formatted group section text
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
    
    return group_section

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
            stack_uuid
        ))
    
    return transformed_stacks

def write_coils_to_file(filename, coil_sections, stack_uuids, num_sections_per_stack, stack_name="Coil Stack"):
    """
    Writes multiple coil sections to a KiCad PCB file.
    
    Args:
        filename (str): KiCad PCB filename
        coil_sections (list): List of (main_section, via_sections, group_section, member_uuids) tuples
        stack_uuids (list): List of UUIDs for each stack group
        num_sections_per_stack (int): Number of sections in each coil stack
        stack_name (str): Name for the final array group
    """
    # Read the KiCad PCB file
    with open(filename, 'r') as f:
        pcb_content = f.read()

    # Find the last ) in the file to insert new content before it
    last_paren_index = pcb_content.rindex(')')
    
    # Combine all sections from all coils
    new_content = []
    all_stack_uuids = []
    
    # Group coils by stack
    current_stack_members = []
    for i, (main_section, via_sections, group_section, member_uuids) in enumerate(coil_sections):
        new_content.extend([main_section, via_sections, group_section])
        current_stack_members.extend(member_uuids)
        
        # When we've processed all sections for a stack, create its group
        if (i + 1) % num_sections_per_stack == 0:
            stack_idx = i // num_sections_per_stack
            stack_group = create_stack_group(current_stack_members, 
                                          name=f"Coil Stack {stack_idx + 1}")
            new_content.append(stack_group)
            all_stack_uuids.append(stack_uuids[stack_idx])
            current_stack_members = []
    
    # Create a group for the entire array
    array_group = create_stack_group(all_stack_uuids, name=stack_name)
    new_content.append(array_group)
    
    # Join all sections and add final parenthesis
    new_content = "\n\t".join(filter(None, new_content)) + "\n)"
    
    # Insert the new content
    updated_content = pcb_content[:last_paren_index] + new_content

    # Write the updated content back to the file
    with open(filename, 'w') as f:
        f.write(updated_content)
