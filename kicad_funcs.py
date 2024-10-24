import numpy as np
import re
import uuid
import math
import matplotlib.pyplot as plt

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

def create_antenna_spiral(filename, all_points, mode="polygon", via_points=None, show_plot=True):
    """
    Updates a KiCad PCB file with either a polygon or trace-based antenna pattern and vias.
    
    Args:
        filename (str): KiCad PCB filename
        all_points (list): List of (x,y) coordinates defining the antenna shape
        mode (str): Either "polygon" or "trace" to determine how the antenna is created
        via_points (list): Optional list of (x, y) coordinates for via placement
        show_plot (bool): Whether to display the matplotlib preview
    """
    
    # Generate UUID for the main element
    main_uuid = str(uuid.uuid4())
    
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
            (layer "F.Cu")
            (uuid "{main_uuid}")
        )'''
    else:  # trace mode
        # Create traces between consecutive points
        trace_sections = []
        for i in range(len(all_points)-1):
            start = all_points[i]
            end = all_points[i+1]
            trace_sections.append(create_trace(start, end))
        main_section = "\n".join(trace_sections)

    # Generate via sections if via points were provided
    via_sections = ""
    member_uuids = [main_uuid] if mode == "polygon" else []
    if via_points:
        via_text, via_uuids = create_via_section(via_points)
        via_sections = via_text
        member_uuids.extend(via_uuids)

    # Create the group section
    group_section = create_group_section(member_uuids)

    # Read the KiCad PCB file
    with open(filename, 'r') as f:
        pcb_content = f.read()

    # Find the last ) in the file to insert new content before it
    last_paren_index = pcb_content.rindex(')')
    
    # Combine all new sections
    new_content = f"{poly_section}\n\t{via_sections}\n\t{group_section}\n)"
    
    # Insert the new content
    updated_content = pcb_content[:last_paren_index] + new_content

    # Write the updated content back to the file
    with open(filename, 'w') as f:
        f.write(updated_content)
    
    if show_plot:
        plt.figure(figsize=(10, 10))
        
        # Plot antenna trace
        points_array = np.array(all_points)
        plt.plot(points_array[:, 0], points_array[:, 1], 'b-', linewidth=2)
        
        # Plot vias if they exist
        if via_points:
            via_points_array = np.array(via_points)
            plt.plot(via_points_array[:, 0], via_points_array[:, 1], 'rx', markersize=8, 
                    label='Vias')
        
        plt.axis('equal')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        if via_points:
            plt.legend()
        plt.show()
