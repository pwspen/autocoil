import matplotlib.pyplot as plt
import numpy as np
import math
from dataclasses import dataclass
from typing import Tuple, List
from kicad_funcs import (
    create_antenna_spiral, 
    write_coils_to_file, 
    create_radial_array,
    generate_coil_stack,
    round_corners,
    generate_holes,
    generate_lines
)

def generate_coil_array(width, height, spacing, turns, corner_radius, trace_width, num_layers,
                       num_copies, center_x, center_y, start_angle):
    """Generate a radial array of coil stacks"""
    spacing_angle = 360/num_copies
    
    # Generate coil stack template
    coil_stack = generate_coil_stack(
        width=width,
        height=height,
        spacing=spacing,
        turns=turns,
        num_layers=num_layers,
        spacing_angle=spacing_angle,
        center=(center_x, center_y),
        trace_width=trace_width
    )
    
    # Create radial array
    return create_radial_array(
        coil_stack=coil_stack,
        num_copies=num_copies,
        center_x=center_x,
        center_y=center_y,
        start_angle_deg=start_angle,
        spacing_deg=spacing_angle
    )

def process_coil_sections(coil_stacks, corner_radius):
    """Process coil sections and return KiCAD sections"""
    all_coil_sections = []
    stack_uuids = []
    coil=0
    for stack, info_dict in coil_stacks:
        stack_uuid = info_dict['uuid']
        conn_layer = info_dict['conn_layer']
        conn_pts = info_dict['conn_pts']
        cut_pts = info_dict['cut_pts_each']
        stack_uuids.append(stack_uuid)
        for section in stack.sections:
            pts = section.points
            pts = round_corners(pts, corner_radius)

            # Calculate total path length
            total_length = 0
            for i in range(len(pts)-1):
                dx = pts[i+1][0] - pts[i][0]
                dy = pts[i+1][1] - pts[i][1]
                total_length += np.sqrt(dx*dx + dy*dy)
            print(f"Total path length: {total_length:.2f} mm for coil {coil}")
            coil += 1

            via_pts = section.via_points if section.via_points is not None else None
            
            # Create KiCAD sections
            sections = create_antenna_spiral(
                pts, 
                trace_width=section.trace_width,
                via_points=via_pts,
                layer=section.layer
            )
            all_coil_sections.append(sections)
        interconnect = create_antenna_spiral(
            trace_width=0.5,
            all_points=conn_pts,
            name=f"Phase interconnect",
            layer=conn_layer
        )
        interconnect = list(interconnect)
        interconnect[3] = [] # Remove interconnects from grouping
        cutout = create_antenna_spiral(
            trace_width = 0.1,
            all_points = cut_pts,
            name=f"Cutout",
            layer="Edge.Cuts",
            mode="polygon"
        )
        all_coil_sections.append(interconnect)
        all_coil_sections.append(cutout)
    
    mhole_locs = info_dict["mount_holes"]
    ehole_locs = info_dict["elec_holes"]
    cut_pts_once = info_dict["cut_pts_once"]

    mholes = generate_holes(pts=mhole_locs, size=3.8, drill=2.2)
    elecholes = generate_holes(pts=ehole_locs, size=3, drill=2)
    edges = generate_lines(pts=cut_pts_once, width=0.1, layer="Edge.Cuts")

    all_coil_sections.append(mholes)
    all_coil_sections.append(elecholes)
    all_coil_sections.append(edges)

    return all_coil_sections, stack_uuids

if __name__ == "__main__":
    # Parameters
    params = {
        'width': 45,
        'height': 15,
        'spacing': 0.4,
        'turns': 10,
        'corner_radius': 0.5,
        'trace_width': 0.2,
        'num_layers': 4,
        'num_copies': 24,
        'center_x': -15,
        'center_y': 15/2,  # height/2
        'start_angle': 0
    }
    # Generate coil array
    coil_stacks = generate_coil_array(**params)
    
    # Process sections
    all_coil_sections, stack_uuids = process_coil_sections(coil_stacks, params['corner_radius'])
    
    # Write to file
    write_coils_to_file(
        "mycoil/mycoil.kicad_pcb", 
        all_coil_sections, 
        stack_uuids,
        num_sections_per_stack=params['num_layers'],
        stack_name="Multi-Layer Coil Array"
    )
    print('Saved to file')
