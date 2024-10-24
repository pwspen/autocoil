import numpy as np
import re
import uuid
import math

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

def create_group_section(member_uuids):
    """
    Create a KiCad PCB group section for the given member UUIDs.
    
    Args:
        member_uuids (list): List of UUID strings to include in the group
    
    Returns:
        str: Formatted group section text
    """
    # Format member UUIDs with at most 2 per line
    members_text = []
    for i in range(0, len(member_uuids), 2):
        chunk = member_uuids[i:i + 2]
        members_text.append(f'\t\t{" ".join(f""""{uuid}" """ for uuid in chunk)}')
    
    group_uuid = str(uuid.uuid4())
    
    group_section = f'''(group ""
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
    
    # Apply flipping transformations
    if flip_x or flip_y:
        all_points = [((-x if flip_x else x), (-y if flip_y else y)) for x, y in all_points]
        if via_points:
            via_points = [((-x if flip_x else x), (-y if flip_y else y)) for x, y in via_points]

    # Generate UUID for the main element
    main_uuid = str(uuid.uuid4())
    
    # Initialize member_uuids list
    member_uuids = []
    
    if mode == "polygon":
        # Generate the points section string for polygon
        points_str = '\n\t\t\t'.join(f'(xy {x:.6f} {y:.6f})' for x, y in all_points)
        
        # Create the full polygon section
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
            (uuid "{main_uuid}")
        )'''
        member_uuids.append(main_uuid)
    else:  # trace mode
        # Create traces between consecutive points
        trace_sections = []
        trace_uuids = []
        for i in range(len(all_points)-1):
            start = all_points[i]
            end = all_points[i+1]
            trace_section, trace_uuid = create_trace(start, end, width=trace_width, layer=layer)
            trace_sections.append(trace_section)
            trace_uuids.append(trace_uuid)
        main_section = "\n".join(trace_sections)
        member_uuids.extend(trace_uuids)

    # Generate via sections if via points were provided
    via_sections = ""
    if via_points:
        via_text, via_uuids = create_via_section(via_points)
        via_sections = via_text
        member_uuids.extend(via_uuids)

    # Create the group section
    group_section = create_group_section(member_uuids)
    
    return main_section, via_sections, group_section, member_uuids

def write_coils_to_file(filename, coil_sections):
    """
    Writes multiple coil sections to a KiCad PCB file.
    
    Args:
        filename (str): KiCad PCB filename
        coil_sections (list): List of (main_section, via_sections, group_section) tuples
    """
    # Read the KiCad PCB file
    with open(filename, 'r') as f:
        pcb_content = f.read()

    # Find the last ) in the file to insert new content before it
    last_paren_index = pcb_content.rindex(')')
    
    # Combine all sections from all coils
    new_content = []
    for main_section, via_sections, group_section, _ in coil_sections:
        new_content.extend([main_section, via_sections, group_section])
    
    # Join all sections and add final parenthesis
    new_content = "\n\t".join(filter(None, new_content)) + "\n)"
    
    # Insert the new content
    updated_content = pcb_content[:last_paren_index] + new_content

    # Write the updated content back to the file
    with open(filename, 'w') as f:
        f.write(updated_content)
