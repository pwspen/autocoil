import math
from typing import List, Tuple
import matplotlib.pyplot as plt

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
    
    # Calculate unit vectors along each line
    v1 = (end1[0] - start1[0], end1[1] - start1[1])
    v2 = (end2[0] - start2[0], end2[1] - start2[1])
    
    len1 = math.sqrt(v1[0]*v1[0] + v1[1]*v1[1])
    len2 = math.sqrt(v2[0]*v2[0] + v2[1]*v2[1])
    
    # Validate vector lengths
    if len1 < 1e-10 or len2 < 1e-10:
        raise ValueError("Line segments are too short")
        
    dir1 = (-v1[0]/len1, -v1[1]/len1)
    dir2 = (v2[0]/len2, v2[1]/len2)
    
    # Calculate angle between lines
    dot_product = dir1[0]*dir2[0] + dir1[1]*dir2[1]
    if debug:
        print(f"Vector 1: {v1}, length: {len1}")
        print(f"Vector 2: {v2}, length: {len2}")
        print(f"Direction 1: {dir1}")
        print(f"Direction 2: {dir2}")
        print(f"Dot product: {dot_product}")
    
    angle_between =  2*math.pi - math.acos(max(min(dot_product, 1), -1))
    if debug:
        print(f"Angle between lines: {math.degrees(angle_between)} degrees")
    
    # Distance from corner to arc center
    center_distance = radius / math.sin(angle_between / 2)
    
    # Calculate bisector direction
    bisector_x = dir1[0] + dir2[0]
    bisector_y = dir1[1] + dir2[1]
    bisector_length = math.sqrt(bisector_x*bisector_x + bisector_y*bisector_y)
    
    # Normalize bisector
    bisector = (
        bisector_x / bisector_length,
        bisector_y / bisector_length
    )
    
    # Calculate center point
    center = (
        corner[0] + bisector[0] * center_distance,
        corner[1] + bisector[1] * center_distance
    )
    
    # Calculate tangent points by moving back from corner along each line
    tangent_distance = radius / math.tan(angle_between / 2)
    
    tangent1 = (
        corner[0] - dir1[0] * tangent_distance,
        corner[1] - dir1[1] * tangent_distance
    )
    
    tangent2 = (
        corner[0] - dir2[0] * tangent_distance,
        corner[1] - dir2[1] * tangent_distance
    )
    
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
    result_points = []
    for i in range(points):
        t = i / (points - 1)
        angle = start_angle * (1-t) + end_angle * t
        point = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle)
        )
        result_points.append(point)
    
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

def test_create_arc(points_per_corner: int = 3):
    """
    Test cases for the create_arc function
    
    Args:
        points_per_corner: number of points to generate for each corner
    """
    # Test case 1: 90-degree angle
    line1 = ((0, 0), (1, 0))
    line2 = ((1, 0), (1, 1))
    radius = 0.2
    points = create_arc(line1, line2, radius, points=points_per_corner, debug=True)
    assert len(points) == points_per_corner, f"Should generate exactly {points_per_corner} points"
    
    # Test case 2: 45-degree angle
    line1 = ((0, 0), (1, 0))
    line2 = ((1, 0), (2, 1))
    radius = 0.3
    points = create_arc(line1, line2, radius, points=points_per_corner, debug=True)
    assert len(points) == points_per_corner, f"Should generate exactly {points_per_corner} points"
    
    # Test case 3: 135-degree angle
    line1 = ((0, 0), (1, 0))
    line2 = ((1, 0), (0, 1))
    radius = 0.25
    points = create_arc(line1, line2, radius, points=points_per_corner, debug=True)
    assert len(points) == points_per_corner, f"Should generate exactly {points_per_corner} points"
    
    # Test error cases
    try:
        # Test non-connected lines
        line1 = ((0, 0), (1, 0))
        line2 = ((2, 0), (3, 0))
        points = create_arc(line1, line2, radius, points=points_per_corner, debug=False)
        assert False, "Should raise assertion error for non-connected lines"
    except AssertionError:
        pass
    
    try:
        # Test zero-length line
        line1 = ((0, 0), (0, 0))
        line2 = ((0, 0), (1, 0))
        points = create_arc(line1, line2, radius, points=points_per_corner, debug=False)
        assert False, "Should raise ValueError for zero-length line"
    except ValueError:
        pass
    
    print("All tests passed!")

if __name__ == "__main__":
    # Test with different numbers of points
    test_create_arc(points_per_corner=6)  # minimum points
