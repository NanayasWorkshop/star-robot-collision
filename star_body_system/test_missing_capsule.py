#!/usr/bin/env python3
"""
Diagnostic: Find missing capsule
Compare expected bones vs generated capsules
"""

import numpy as np
import plotly.graph_objects as go
from star_body_system_pkg.core.star_interface import STARInterface
from star_body_system_pkg.core.body_definitions import BodyDefinitions
from star_body_system_pkg.layers.sphere_layer import SphereLayer
from star_body_system_pkg.layers.capsule_layer import CapsuleLayer

def main():
    # Load data
    star = STARInterface(gender='neutral')
    vertices, joints = star.get_neutral_pose()
    body_defs = BodyDefinitions()
    
    # Generate layers
    sphere_layer = SphereLayer()
    spheres = sphere_layer.generate_from_joints(joints, vertices)
    
    capsule_layer = CapsuleLayer()
    capsules = capsule_layer.generate_from_joints(joints, sphere_layer)
    
    # Expected vs actual
    expected_bones = [bone_name for _, _, bone_name in body_defs.DETAILED_BONES]
    generated_capsules = [name for _, _, _, name in capsules]
    
    print(f"Expected bones: {len(expected_bones)}")
    print(f"Generated capsules: {len(generated_capsules)}")
    
    # Find missing
    missing = set(expected_bones) - set(generated_capsules)
    extra = set(generated_capsules) - set(expected_bones)
    
    print(f"\nMissing capsules: {missing}")
    print(f"Extra capsules: {extra}")
    
    # Check bone definitions
    print(f"\nBone definition check:")
    for bone_name in expected_bones:
        has_def = bone_name in body_defs.BONE_DEFINITIONS
        print(f"  {bone_name}: {'✓' if has_def else '✗'}")
        if not has_def:
            print(f"    ^^^ MISSING BONE DEFINITION")
    
    # Transform coordinates
    def transform_coords(points):
        transformed = points.copy()
        transformed[:, [1, 2]] = transformed[:, [2, 1]]
        transformed[:, 1] *= -1
        return transformed
    
    joints_transformed = transform_coords(joints)
    
    # Visualize expected vs generated
    fig = go.Figure()
    
    # Expected bones (blue lines)
    for start_idx, end_idx, bone_name in body_defs.DETAILED_BONES:
        start_pos = joints_transformed[start_idx]
        end_pos = joints_transformed[end_idx]
        
        color = 'red' if bone_name in missing else 'blue'
        
        fig.add_trace(go.Scatter3d(
            x=[start_pos[0], end_pos[0]],
            y=[start_pos[1], end_pos[1]], 
            z=[start_pos[2], end_pos[2]],
            mode='lines',
            line=dict(color=color, width=6),
            name=f"Expected: {bone_name}",
            showlegend=True if bone_name in missing else False
        ))
    
    # Generated capsules (green lines)
    for start_pos, end_pos, radius, name in capsules:
        start_transformed = transform_coords(start_pos.reshape(1, -1))[0]
        end_transformed = transform_coords(end_pos.reshape(1, -1))[0]
        
        fig.add_trace(go.Scatter3d(
            x=[start_transformed[0], end_transformed[0]],
            y=[start_transformed[1], end_transformed[1]],
            z=[start_transformed[2], end_transformed[2]], 
            mode='lines',
            line=dict(color='green', width=4),
            name=f"Generated: {name}",
            showlegend=False
        ))
    
    # Joints
    fig.add_trace(go.Scatter3d(
        x=joints_transformed[:, 0],
        y=joints_transformed[:, 1],
        z=joints_transformed[:, 2],
        mode='markers+text',
        marker=dict(size=8, color='orange'),
        text=[f'{i}' for i in range(len(joints))],
        name='Joints'
    ))
    
    fig.update_layout(
        title=f"Expected: {len(expected_bones)} bones, Generated: {len(generated_capsules)} capsules",
        scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='data'),
        width=1200, height=900
    )
    
    fig.show()
    print("Visualization opened in browser")

if __name__ == "__main__":
    main()