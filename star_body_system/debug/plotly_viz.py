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
            fig.add_trace(
                go.Scatter3d(
                    x=show_mesh[:, 0], y=show_mesh[:, 1], z=show_mesh[:, 2],
                    mode='markers',
                    marker=dict(size=0.5, color='lightgray', opacity=0.2),
                    name=f'Mesh ({len(show_mesh)} vertices)',
                    showlegend=True
                )
            )
        
        # Add spheres
        for center, radius, name in spheres:
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
            sphere_surface = self._create_sphere_surface(center, radius, color, name)
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
            capsule_surface = self._create_capsule_surface(start_pos, end_pos, radius, color, name)
            fig.add_trace(capsule_surface)
        
        # Set layout
        self._set_3d_layout(fig, title, capsules)
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
                color = self.layer_colors['simple']['core'] if 'pelvis-head' in name else self.layer_colors['simple']['legs']
                surface = self._create_capsule_surface(start_pos, end_pos, radius, color, f"L3_{name}", opacity=0.2)
                fig.add_trace(surface)
        
        # Layer 2 (medium transparency)
        if capsule_layer:
            capsules = capsule_layer.get_capsules()
            for start_pos, end_pos, radius, name in capsules:
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['capsule'].get(bone_type, 'gray')
                surface = self._create_capsule_surface(start_pos, end_pos, radius, color, f"L2_{name}", opacity=0.4)
                fig.add_trace(surface)
        
        # Layer 1 (innermost, most opaque)
        if sphere_layer:
            spheres = sphere_layer.get_spheres()
            for center, radius, name in spheres[:20]:  # Limit for performance
                bone_type = self._get_bone_type_from_name(name)
                color = self.layer_colors['sphere'].get(bone_type, 'gray')
                surface = self._create_sphere_surface(center, radius, color, f"L1_{name}", opacity=0.6)
                fig.add_trace(surface)
        
        fig.update_layout(
            title="Multi-Layer Body Representation",
            scene=dict(aspectratio=dict(x=1, y=1, z=1), aspectmode='cube'),
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
        """Set 3D layout for the figure"""
        # Calculate bounds from data
        all_points = []
        
        for obj in data_objects:
            if len(obj) == 3:  # Sphere: (center, radius, name)
                center, radius, _ = obj
                all_points.extend([
                    center - radius, center + radius
                ])
            elif len(obj) == 4:  # Capsule: (start, end, radius, name)
                start_pos, end_pos, radius, _ = obj
                all_points.extend([
                    start_pos - radius, start_pos + radius,
                    end_pos - radius, end_pos + radius
                ])
        
        if all_points:
            all_points = np.array(all_points)
            x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
            y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
            z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
            
            # Center and scale
            x_center = (x_range[0] + x_range[1]) / 2
            y_center = (y_range[0] + y_range[1]) / 2
            z_center = (z_range[0] + z_range[1]) / 2
            
            max_range = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]) * 1.2
            
            fig.update_layout(
                title=title,
                scene=dict(
                    xaxis=dict(range=[x_center - max_range/2, x_center + max_range/2]),
                    yaxis=dict(range=[y_center - max_range/2, y_center + max_range/2]),
                    zaxis=dict(range=[z_center - max_range/2, z_center + max_range/2]),
                    aspectratio=dict(x=1, y=1, z=1),
                    aspectmode='cube',
                    bgcolor='white'
                ),
                width=1200,
                height=900
            )


# Convenience functions for quick visualization
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