"""
Plotly Visualization for Body Layers
Pure visualization functions - separate from core logic
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px


class LayerVisualizer:
    """Plotly visualization for all body layers"""
    
    def __init__(self):
        # Color schemes for different layers
        self.layer_colors = {
            'sphere': {
                'torso': 'blue', 'head': 'red', 'leg_upper': 'green',
                'leg_lower': 'lightgreen', 'foot': 'purple', 'shoulder': 'orange',
                'arm_upper': 'yellow', 'arm_lower': 'lightyellow', 'hand': 'pink'
            },
            'capsule': {
                'torso': 'darkblue', 'head': 'darkred', 'leg_upper': 'darkgreen',
                'leg_lower': 'mediumseagreen', 'foot': 'indigo', 'shoulder': 'darkorange',
                'arm_upper': 'gold', 'arm_lower': 'khaki', 'hand': 'hotpink'
            },
            'simple': {
                'legs': 'forestgreen', 'core': 'navy', 'arms': 'crimson'
            }
        }
    
    def _transform_coordinates(self, points):
        """
        Transform STAR coordinates to standing upright position
        STAR: person lying on back (Y-up)
        Target: person standing upright (Z-up)
        """
        if isinstance(points, (list, tuple)):
            points = np.array(points)
        
        # Rotation matrix: Y-up to Z-up (90 degree rotation around X-axis)
        # Also flip Y to make person face forward
        if points.ndim == 1:
            # Single point
            x, y, z = points
            return np.array([x, -z, y])
        else:
            # Multiple points
            transformed = points.copy()
            transformed[:, [1, 2]] = transformed[:, [2, 1]]  # Swap Y and Z
            transformed[:, 1] *= -1  # Flip new Y (was Z) to face forward
            return transformed
    
    def visualize_layer0_mesh_joints(self, vertices, joints, title="Layer 0: STAR Mesh + Joints + Bones"):
        """
        Visualize Layer 0: Raw STAR mesh vertices, joints, and bone connections
        
        Args:
            vertices: Mesh vertices array (N, 3)
            joints: Joint positions array (24, 3) 
            title: Plot title
            
        Returns:
            plotly Figure
        """
        from ..core.body_definitions import BodyDefinitions
        body_defs = BodyDefinitions()
        
        fig = go.Figure()
        
        # Add mesh vertices (small, semi-transparent)
        if vertices is not None:
            vertices_transformed = self._transform_coordinates(vertices)
            fig.add_trace(
                go.Scatter3d(
                    x=vertices_transformed[:, 0], 
                    y=vertices_transformed[:, 1], 
                    z=vertices_transformed[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='lightgray', opacity=0.3),
                    name=f'Mesh Vertices ({len(vertices)})',
                    showlegend=True
                )
            )
        
        # Add joints (larger, colored spheres)
        if joints is not None:
            joints_transformed = self._transform_coordinates(joints)
            
            # Color joints by body region
            joint_colors = []
            for i, joint_name in enumerate(body_defs.JOINT_NAMES):
                if 'spine' in joint_name or 'pelvis' in joint_name or 'neck' in joint_name:
                    color = 'red'
                elif 'head' in joint_name:
                    color = 'darkred'
                elif 'left' in joint_name and ('hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name or 'foot' in joint_name):
                    color = 'blue'
                elif 'right' in joint_name and ('hip' in joint_name or 'knee' in joint_name or 'ankle' in joint_name or 'foot' in joint_name):
                    color = 'darkblue'
                elif 'left' in joint_name:
                    color = 'green'
                elif 'right' in joint_name:
                    color = 'darkgreen'
                else:
                    color = 'orange'
                joint_colors.append(color)
            
            fig.add_trace(
                go.Scatter3d(
                    x=joints_transformed[:, 0],
                    y=joints_transformed[:, 1], 
                    z=joints_transformed[:, 2],
                    mode='markers+text',
                    marker=dict(size=8, color=joint_colors, opacity=0.8),
                    text=[f'{i}:{name}' for i, name in enumerate(body_defs.JOINT_NAMES)],
                    textposition='top center',
                    textfont=dict(size=8),
                    name='Joints (24)',
                    showlegend=True
                )
            )
            
            # Add bone connections (lines between joints)
            for start_idx, end_idx, bone_name in body_defs.DETAILED_BONES:
                start_joint = joints_transformed[start_idx]
                end_joint = joints_transformed[end_idx]
                
                # Color bones by type
                if 'spine' in bone_name or 'pelvis' in bone_name or 'neck' in bone_name or 'head' in bone_name:
                    line_color = 'red'
                elif 'left' in bone_name and ('hip' in bone_name or 'knee' in bone_name or 'ankle' in bone_name or 'foot' in bone_name):
                    line_color = 'blue'
                elif 'right' in bone_name and ('hip' in bone_name or 'knee' in bone_name or 'ankle' in bone_name or 'foot' in bone_name):
                    line_color = 'darkblue'
                elif 'left' in bone_name:
                    line_color = 'green'
                elif 'right' in bone_name:
                    line_color = 'darkgreen'
                else:
                    line_color = 'orange'
                
                fig.add_trace(
                    go.Scatter3d(
                        x=[start_joint[0], end_joint[0]],
                        y=[start_joint[1], end_joint[1]],
                        z=[start_joint[2], end_joint[2]],
                        mode='lines',
                        line=dict(color=line_color, width=4),
                        name=bone_name,
                        showlegend=False,  # Too many bones for legend
                        hovertemplate=f'<b>{bone_name}</b><br>Start: {body_defs.JOINT_NAMES[start_idx]}<br>End: {body_defs.JOINT_NAMES[end_idx]}<extra></extra>'
                    )
                )
        
        # Set layout with proper scaling
        data_objects = []
        if vertices is not None:
            for vertex in vertices:
                data_objects.append((vertex, 0.01, 'vertex'))  # Fake radius for layout calculation
        
        self._set_3d_layout(fig, title, data_objects)
        return fig
    
    def visualize_spheres(self, spheres, title="Sphere Layer", show_mesh=None):
        """
        Visualize Layer 1 spheres
        
        Args:
            spheres: List of (center, radius, name) tuples
            title: Plot title
            show_mesh: Optional mesh vertices to show as background
            
        Returns:
            plotly Figure
        """
        fig = go.Figure()
        
        # Add mesh background if provided
        if show_mesh is not None:
            mesh_transformed = self._transform_coordinates(show_mesh)
            fig.add_trace(
                go.Scatter3d(
                    x=mesh_transformed[:, 0], y=mesh_transformed[:, 1], z=mesh_transformed[:, 2],
                    mode='markers',
                    marker=dict(size=0.5, color='lightgray', opacity=0.2),
                    name=f'Mesh ({len(show_mesh)} vertices)',
                    showlegend=True
                )
            )
        
        # Add spheres
        for center, radius, name in spheres:
            center_transformed = self._transform_coordinates(center)
            # Determine color by bone type
            bone_name = '_'.join(name.split('_')[:-1])
            # Simple heuristic for bone type from name
            if 'spine' in bone_name or 'pelvis' in bone_name:
                bone_type = 'torso'
            elif 'head' in bone_name or 'neck' in bone_name:
                bone_type = 'head'
            elif 'hip' in bone_name or 'knee' in bone_name:
                bone_type = 'leg_upper' if 'hip' in bone_name else 'leg_lower'
            elif 'ankle' in bone_name or 'foot' in bone_name:
                bone_type = 'foot'
            elif 'collar' in bone_name or 'shoulder' in bone_name:
                bone_type = 'shoulder'
            elif 'elbow' in bone_name:
                bone_type = 'arm_upper' if 'shoulder' in bone_name else 'arm_lower'
            elif 'wrist' in bone_name or 'hand' in bone_name:
                bone_type = 'hand'
            else:
                bone_type = 'torso'
            
            color = self.layer_colors['sphere'].get(bone_type, 'gray')
            
            # Create sphere surface
            sphere_surface = self._create_sphere_surface(center_transformed, radius, color, name)
            fig.add_trace(sphere_surface)
        
        # Set layout
        self._set_3d_layout(fig, title, spheres)
        return fig
    
    def visualize_capsules(self, capsules, title="Capsule Layer", layer_type='capsule'):
        """
        Visualize Layer 2 or Layer 3 capsules
        
        Args:
            capsules: List of (start_pos, end_pos, radius, name) tuples
            title: Plot title
            layer_type: 'capsule' or 'simple'
            
        Returns:
            plotly Figure
        """
        fig = go.Figure()
        
        for start_pos, end_pos, radius, name in capsules:
            start_transformed = self._transform_coordinates(start_pos)
            end_transformed = self._transform_coordinates(end_pos)
            # Determine color
            if layer_type == 'simple':
                if 'knee' in name or 'foot' in name:
                    color = self.layer_colors['simple']['legs']
                elif 'pelvis-head' in name:
                    color = self.layer_colors['simple']['core']
                else:
                    color = self.layer_colors['simple']['arms']
            else:
                # Use bone type logic similar to spheres
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['capsule'].get(bone_type, 'gray')
            
            # Create capsule surface
            capsule_surface = self._create_capsule_surface(start_transformed, end_transformed, radius, color, name)
            fig.add_trace(capsule_surface)
        
        # Set layout
        self._set_3d_layout(fig, title, capsules)
        return fig
    
    def visualize_sphere_capsule_comparison(self, sphere_layer, capsule_layer):
        """
        Visualize Layer 1 spheres and Layer 2 capsules together for comparison
        
        Args:
            sphere_layer: SphereLayer instance
            capsule_layer: CapsuleLayer instance
            
        Returns:
            plotly Figure
        """
        fig = go.Figure()
        
        # Add spheres (semi-transparent)
        if sphere_layer:
            spheres = sphere_layer.get_spheres()
            for center, radius, name in spheres:
                center_transformed = self._transform_coordinates(center)
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['sphere'].get(bone_type, 'gray')
                surface = self._create_sphere_surface(center_transformed, radius, color, f"Sphere_{name}", opacity=0.3)
                fig.add_trace(surface)
        
        # Add capsules (more opaque)
        if capsule_layer:
            capsules = capsule_layer.get_capsules()
            for start_pos, end_pos, radius, name in capsules:
                start_transformed = self._transform_coordinates(start_pos)
                end_transformed = self._transform_coordinates(end_pos)
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['capsule'].get(bone_type, 'gray')
                surface = self._create_capsule_surface(start_transformed, end_transformed, radius, color, f"Capsule_{name}", opacity=0.6)
                fig.add_trace(surface)
        
        fig.update_layout(
            title="Layer 1 Spheres + Layer 2 Capsules Comparison",
            scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='data'),
            width=1200, height=900
        )
        
        return fig
    
    def visualize_all_layers(self, sphere_layer=None, capsule_layer=None, simple_layer=None):
        """
        Visualize all layers together
        
        Args:
            sphere_layer: SphereLayer instance
            capsule_layer: CapsuleLayer instance  
            simple_layer: SimpleCapsuleLayer instance
            
        Returns:
            plotly Figure
        """
        fig = go.Figure()
        
        # Layer 3 (outermost, most transparent)
        if simple_layer:
            capsules = simple_layer.get_capsules()
            for start_pos, end_pos, radius, name in capsules:
                start_transformed = self._transform_coordinates(start_pos)
                end_transformed = self._transform_coordinates(end_pos)
                color = self.layer_colors['simple']['core'] if 'pelvis-head' in name else self.layer_colors['simple']['legs']
                surface = self._create_capsule_surface(start_transformed, end_transformed, radius, color, f"L3_{name}", opacity=0.2)
                fig.add_trace(surface)
        
        # Layer 2 (medium transparency)
        if capsule_layer:
            capsules = capsule_layer.get_capsules()
            for start_pos, end_pos, radius, name in capsules:
                start_transformed = self._transform_coordinates(start_pos)
                end_transformed = self._transform_coordinates(end_pos)
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['capsule'].get(bone_type, 'gray')
                surface = self._create_capsule_surface(start_transformed, end_transformed, radius, color, f"L2_{name}", opacity=0.4)
                fig.add_trace(surface)
        
        # Layer 1 (innermost, most opaque)
        if sphere_layer:
            spheres = sphere_layer.get_spheres()
            for center, radius, name in spheres[:20]:  # Limit for performance
                center_transformed = self._transform_coordinates(center)
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['sphere'].get(bone_type, 'gray')
                surface = self._create_sphere_surface(center_transformed, radius, color, f"L1_{name}", opacity=0.6)
                fig.add_trace(surface)
        
        fig.update_layout(
            title="Multi-Layer Body Representation",
            scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='data'),
            width=1200, height=900
        )
        
        return fig
    
    def _create_sphere_surface(self, center, radius, color, name, opacity=0.5):
        """Create a sphere surface for visualization"""
        u = np.linspace(0, 2 * np.pi, 15)  # Reduced resolution for performance
        v = np.linspace(0, np.pi, 15)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        return go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            opacity=opacity,
            hovertemplate=f'<b>{name}</b><br>Radius: {radius:.3f}<extra></extra>'
        )
    
    def _create_capsule_surface(self, start_pos, end_pos, radius, color, name, opacity=0.5):
        """Create a capsule surface for visualization"""
        # Simplified capsule as cylinder + spheres at ends
        direction = end_pos - start_pos
        length = np.linalg.norm(direction)
        
        if length == 0:
            # Degenerate case: just a sphere
            return self._create_sphere_surface(start_pos, radius, color, name, opacity)
        
        # Create cylinder
        theta = np.linspace(0, 2*np.pi, 12)
        z_cyl = np.linspace(0, length, 8)
        
        # Create coordinate system along capsule
        direction_norm = direction / length
        
        # Find perpendicular vectors
        if abs(direction_norm[2]) < 0.9:
            perp1 = np.cross(direction_norm, [0, 0, 1])
        else:
            perp1 = np.cross(direction_norm, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(direction_norm, perp1)
        
        # Generate cylinder points
        x_cyl = []
        y_cyl = []
        z_points = []
        
        for i, z in enumerate(z_cyl):
            current_center = start_pos + (z / length) * direction
            x_ring = []
            y_ring = []
            z_ring = []
            
            for t in theta:
                point = current_center + radius * (np.cos(t) * perp1 + np.sin(t) * perp2)
                x_ring.append(point[0])
                y_ring.append(point[1])
                z_ring.append(point[2])
            
            x_cyl.append(x_ring)
            y_cyl.append(y_ring)
            z_points.append(z_ring)
        
        return go.Surface(
            x=x_cyl, y=y_cyl, z=z_points,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            opacity=opacity,
            hovertemplate=f'<b>{name}</b><br>Radius: {radius:.3f}<br>Length: {length:.3f}<extra></extra>'
        )
    
    def _get_bone_type_from_name(self, name):
        """Get bone type from bone name"""
        if 'spine' in name or 'pelvis' in name:
            return 'torso'
        elif 'head' in name or 'neck' in name:
            return 'head'
        elif 'hip' in name:
            return 'leg_upper'
        elif 'knee' in name:
            return 'leg_lower'
        elif 'ankle' in name or 'foot' in name:
            return 'foot'
        elif 'collar' in name:
            return 'shoulder'
        elif 'shoulder' in name:
            return 'arm_upper'
        elif 'elbow' in name:
            return 'arm_lower'
        elif 'wrist' in name or 'hand' in name:
            return 'hand'
        else:
            return 'torso'
    
    def _set_3d_layout(self, fig, title, data_objects):
        """Set 3D layout for the figure with proper equal scaling"""
        fig.update_layout(
            title=title,
            scene=dict(
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='data',
                bgcolor='white'
            ),
            width=1200,
            height=900
        )


# Convenience functions for quick visualization
def visualize_layer0_mesh_joints(vertices, joints, title="Layer 0: STAR Mesh + Joints + Bones"):
    """Quick Layer 0 visualization"""
    viz = LayerVisualizer()
    return viz.visualize_layer0_mesh_joints(vertices, joints, title)


def visualize_spheres(spheres, title="Sphere Layer", mesh_vertices=None):
    """Quick sphere visualization"""
    viz = LayerVisualizer()
    return viz.visualize_spheres(spheres, title, mesh_vertices)


def visualize_capsules(capsules, title="Capsule Layer", layer_type='capsule'):
    """Quick capsule visualization"""
    viz = LayerVisualizer()
    return viz.visualize_capsules(capsules, title, layer_type)


def visualize_all_layers(sphere_layer=None, capsule_layer=None, simple_layer=None):
    """Quick multi-layer visualization"""
    viz = LayerVisualizer()
    return viz.visualize_all_layers(sphere_layer, capsule_layer, simple_layer)


def visualize_sphere_capsule_comparison(sphere_layer, capsule_layer):
    """Quick sphere + capsule comparison visualization"""
    viz = LayerVisualizer()
    return viz.visualize_sphere_capsule_comparison(sphere_layer, capsule_layer)