#!/usr/bin/env python3
"""
Diagnostic: Visualize uncovered vertices
Shows all vertices (green) and uncovered vertices (red)
"""

import numpy as np
import plotly.graph_objects as go
from star_body_system_pkg.core.star_interface import STARInterface
from star_body_system_pkg.core.collision_mapping import VertexSphereMapper
from star_body_system_pkg.layers.sphere_layer import SphereLayer

def main():
    # Load STAR data
    star = STARInterface(gender='neutral')
    vertices, joints = star.get_neutral_pose()
    
    # Generate spheres
    sphere_layer = SphereLayer()
    spheres = sphere_layer.generate_from_joints(joints, vertices)
    
    # Get vertex assignments
    mapper = VertexSphereMapper()
    vertex_mapping = mapper.assign_vertices_to_spheres(vertices, spheres)
    
    # Find uncovered vertices
    uncovered_indices = []
    for vertex_idx in range(len(vertices)):
        assignments = vertex_mapping.get(vertex_idx, [])
        if len(assignments) == 0:
            uncovered_indices.append(vertex_idx)
    
    print(f"Found {len(uncovered_indices)} uncovered vertices")
    
    # Transform coordinates (Y-up to Z-up)
    def transform_coords(points):
        transformed = points.copy()
        transformed[:, [1, 2]] = transformed[:, [2, 1]]  # Swap Y and Z
        transformed[:, 1] *= -1  # Flip Y
        return transformed
    
    vertices_transformed = transform_coords(vertices)
    
    # Create plot
    fig = go.Figure()
    
    # Add spheres first (Layer 1)
    for center, radius, name in spheres:
        center_transformed = transform_coords(center.reshape(1, -1))[0]
        
        # Create sphere surface (low resolution for performance)
        u = np.linspace(0, 2 * np.pi, 8)
        v = np.linspace(0, np.pi, 8)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center_transformed[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center_transformed[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center_transformed[2]
        
        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'lightblue'], [1, 'lightblue']],
            showscale=False,
            name=f'Sphere_{name}',
            opacity=0.3,
            showlegend=False
        ))
    
    # All vertices in green
    fig.add_trace(go.Scatter3d(
        x=vertices_transformed[:, 0],
        y=vertices_transformed[:, 1], 
        z=vertices_transformed[:, 2],
        mode='markers',
        marker=dict(size=2, color='green', opacity=0.6),
        name=f'Covered vertices ({len(vertices) - len(uncovered_indices)})'
    ))
    
    # Uncovered vertices in red
    if uncovered_indices:
        uncovered_verts = vertices_transformed[uncovered_indices]
        fig.add_trace(go.Scatter3d(
            x=uncovered_verts[:, 0],
            y=uncovered_verts[:, 1],
            z=uncovered_verts[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', opacity=1.0),
            name=f'Uncovered vertices ({len(uncovered_indices)})'
        ))
    
    fig.update_layout(
        title=f"Vertex Coverage: {len(uncovered_indices)} uncovered vertices",
        scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='data'),
        width=1200, height=900
    )
    
    fig.show()
    print("Visualization opened in browser")

if __name__ == "__main__":
    main()