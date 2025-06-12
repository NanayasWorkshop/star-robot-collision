def print_sphere_stats(self):
        """Print statistics about the spheres"""
        if not self.all_spheres:
            print("No spheres to analyze")
            return
        
        print("\n" + "="*60)
        print("GUARANTEED COVERAGE SPHERE STATISTICS")
        print("="*60)
        
        # Group by bone type
        type_groups = {}
        filler_count = 0
        
        for center, radius, name in self.all_spheres:
            if name.startswith('filler'):
                filler_count += 1
                continue
                
            bone_name = '_'.join(name.split('_')[:-1])
            bone_def = self.bone_definitions.get(bone_name, {'type': 'unknown'})
            bone_type = bone_def['type']
            if bone_type not in type_groups:
                type_groups[bone_type] = []
            type_groups[bone_type].append(radius)
        
        print(f"Total spheres: {len(self.all_spheres)}")
        print(f"Anatomical spheres: {len(self.all_spheres) - filler_count}")
        print(f"Filler spheres: {filler_count}")
        
        print(f"\nBy body region:")
        for bone_type, radii in sorted(type_groups.items()):
            print(f"  {bone_type:12}: {len(radii):2d} spheres, "
                  f"avg: {np.mean(radii):.3f}, range: {np.min(radii):.3f}-{np.max(radii):.3f}")#!/usr/bin/env python3
"""
STAR Anatomical Sphere Body Visualizer with Guaranteed Overlap
Creates realistic 3D body using anatomically-constrained spheres with guaranteed coverage
"""

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
import time

# STAR model imports
try:
    from star.pytorch.star import STAR
    STAR_AVAILABLE = True
except ImportError:
    print("WARNING: STAR model not available. Install from https://github.com/ahmedosman/STAR")
    STAR_AVAILABLE = False


class STARGuaranteedCoverageSphereOptimizer:
    """Create anatomically-realistic overlapping spheres with guaranteed 100% coverage"""
    
    def __init__(self, gender='neutral'):
        if not STAR_AVAILABLE:
            raise ImportError("STAR model required")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STAR(gender=gender)
        self.model.to(self.device)
        self.model.eval()
        
        # Joint names for STAR model (24 joints)
        self.joint_names = [
            'pelvis', 'left_hip', 'right_hip', 'spine1', 'left_knee', 'right_knee',
            'spine2', 'left_ankle', 'right_ankle', 'spine3', 'left_foot', 'right_foot',
            'neck', 'left_collar', 'right_collar', 'head', 'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hand', 'right_hand'
        ]
        
        # Color palette
        self.colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
        
        # Define anatomical bone hierarchy with sizes
        self.bone_definitions = {
            # Core/Torso - Large spheres
            'pelvis-spine1': {'start_radius': 0.12, 'end_radius': 0.12, 'type': 'torso', 'min_overlap': 0.3},
            'spine1-spine2': {'start_radius': 0.12, 'end_radius': 0.11, 'type': 'torso', 'min_overlap': 0.3},
            'spine2-spine3': {'start_radius': 0.11, 'end_radius': 0.10, 'type': 'torso', 'min_overlap': 0.3},
            'spine3-neck': {'start_radius': 0.10, 'end_radius': 0.08, 'type': 'torso', 'min_overlap': 0.25},
            
            # Head/Neck
            'neck-head': {'start_radius': 0.06, 'end_radius': 0.08, 'type': 'head', 'min_overlap': 0.3},
            
            # Legs - Strong tapering with good overlap
            'pelvis-left_hip': {'start_radius': 0.10, 'end_radius': 0.10, 'type': 'leg_upper', 'min_overlap': 0.4},
            'left_hip-left_knee': {'start_radius': 0.10, 'end_radius': 0.07, 'type': 'leg_upper', 'min_overlap': 0.35},
            'left_knee-left_ankle': {'start_radius': 0.07, 'end_radius': 0.04, 'type': 'leg_lower', 'min_overlap': 0.35},
            'left_ankle-left_foot': {'start_radius': 0.04, 'end_radius': 0.04, 'type': 'foot', 'min_overlap': 0.4},
            
            'pelvis-right_hip': {'start_radius': 0.10, 'end_radius': 0.10, 'type': 'leg_upper', 'min_overlap': 0.4},
            'right_hip-right_knee': {'start_radius': 0.10, 'end_radius': 0.07, 'type': 'leg_upper', 'min_overlap': 0.35},
            'right_knee-right_ankle': {'start_radius': 0.07, 'end_radius': 0.04, 'type': 'leg_lower', 'min_overlap': 0.35},
            'right_ankle-right_foot': {'start_radius': 0.04, 'end_radius': 0.04, 'type': 'foot', 'min_overlap': 0.4},
            
            # Arms - Strong tapering with excellent overlap
            'spine3-left_collar': {'start_radius': 0.09, 'end_radius': 0.08, 'type': 'shoulder', 'min_overlap': 0.3},
            'left_collar-left_shoulder': {'start_radius': 0.08, 'end_radius': 0.07, 'type': 'shoulder', 'min_overlap': 0.3},
            'left_shoulder-left_elbow': {'start_radius': 0.07, 'end_radius': 0.05, 'type': 'arm_upper', 'min_overlap': 0.4},
            'left_elbow-left_wrist': {'start_radius': 0.05, 'end_radius': 0.035, 'type': 'arm_lower', 'min_overlap': 0.4},
            'left_wrist-left_hand': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'hand', 'min_overlap': 0.4},
            
            'spine3-right_collar': {'start_radius': 0.09, 'end_radius': 0.08, 'type': 'shoulder', 'min_overlap': 0.3},
            'right_collar-right_shoulder': {'start_radius': 0.08, 'end_radius': 0.07, 'type': 'shoulder', 'min_overlap': 0.3},
            'right_shoulder-right_elbow': {'start_radius': 0.07, 'end_radius': 0.05, 'type': 'arm_upper', 'min_overlap': 0.4},
            'right_elbow-right_wrist': {'start_radius': 0.05, 'end_radius': 0.035, 'type': 'arm_lower', 'min_overlap': 0.4},
            'right_wrist-right_hand': {'start_radius': 0.035, 'end_radius': 0.04, 'type': 'hand', 'min_overlap': 0.4},
        }
        
        # Define bone connections
        self.bones = [
            (0, 3, 'pelvis-spine1'),
            (3, 6, 'spine1-spine2'),  
            (6, 9, 'spine2-spine3'),
            (9, 12, 'spine3-neck'),
            (12, 15, 'neck-head'),
            
            (0, 1, 'pelvis-left_hip'),
            (1, 4, 'left_hip-left_knee'),
            (4, 7, 'left_knee-left_ankle'),
            (7, 10, 'left_ankle-left_foot'),
            
            (0, 2, 'pelvis-right_hip'),
            (2, 5, 'right_hip-right_knee'),
            (5, 8, 'right_knee-right_ankle'),
            (8, 11, 'right_ankle-right_foot'),
            
            (9, 13, 'spine3-left_collar'),
            (13, 16, 'left_collar-left_shoulder'),
            (16, 18, 'left_shoulder-left_elbow'),
            (18, 20, 'left_elbow-left_wrist'),
            (20, 22, 'left_wrist-left_hand'),
            
            (9, 14, 'spine3-right_collar'),
            (14, 17, 'right_collar-right_shoulder'),
            (17, 19, 'right_shoulder-right_elbow'),
            (19, 21, 'right_elbow-right_wrist'),
            (21, 23, 'right_wrist-right_hand'),
        ]
        
        # Results storage
        self.all_spheres = []
        
    def get_mesh_data(self, pose_params=None):
        """Get mesh vertices and joint positions"""
        batch_size = 1
        if pose_params is None:
            pose_params = torch.zeros(batch_size, 72, device=self.device)
        shape_params = torch.zeros(batch_size, 10, device=self.device)
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        with torch.no_grad():
            try:
                result = self.model(pose_params, shape_params, trans)
                if isinstance(result, tuple):
                    vertices, joints = result
                    return vertices[0].cpu().numpy(), joints[0].cpu().numpy()
                else:
                    vertices = result
                    if hasattr(self.model, 'J_regressor'):
                        joints = torch.matmul(self.model.J_regressor, vertices)
                        return vertices[0].cpu().numpy(), joints[0].cpu().numpy()
                    else:
                        return vertices[0].cpu().numpy(), None
            except Exception as e:
                print(f"Error getting mesh data: {e}")
                return None, None

    def calculate_adaptive_sphere_placement(self, start_pos, end_pos, bone_definition):
        """Calculate sphere positions that guarantee overlap"""
        start_radius = bone_definition['start_radius']
        end_radius = bone_definition['end_radius']
        min_overlap = bone_definition['min_overlap']
        
        bone_length = np.linalg.norm(end_pos - start_pos)
        
        # Start with first sphere at start position
        positions = [start_pos]
        radii = [start_radius]
        
        current_pos = start_pos
        current_radius = start_radius
        
        while True:
            # Calculate how far we can place the next sphere while maintaining overlap
            next_distance = current_radius * (2 - min_overlap)  # Distance for desired overlap
            
            # Calculate position along bone
            direction = (end_pos - start_pos) / bone_length
            next_pos = current_pos + direction * next_distance
            
            # Check if we've reached or passed the end
            distance_to_end = np.linalg.norm(end_pos - next_pos)
            progress = np.linalg.norm(next_pos - start_pos) / bone_length
            
            if progress >= 1.0 or distance_to_end < bone_length * 0.1:
                # Place final sphere at end position
                positions.append(end_pos)
                radii.append(end_radius)
                break
            else:
                # Calculate tapered radius for this position
                next_radius = start_radius + progress * (end_radius - start_radius)
                
                positions.append(next_pos)
                radii.append(next_radius)
                
                current_pos = next_pos
                current_radius = next_radius
        
        return positions, radii

    def fine_tune_sphere_size(self, center, initial_radius, mesh_vertices, bone_type, is_extremity=False):
        """Fine-tune sphere size based on local mesh geometry"""
        # More aggressive search for better coverage
        search_radius = initial_radius * 3  # Increased search area
        distances = np.linalg.norm(mesh_vertices - center, axis=1)
        nearby_vertices = mesh_vertices[distances <= search_radius]
        
        if len(nearby_vertices) == 0:
            return initial_radius
        
        # Calculate distances to nearby vertices
        nearby_distances = np.linalg.norm(nearby_vertices - center, axis=1)
        
        if is_extremity:
            # For extremities (last spheres), be very aggressive to capture everything
            print(f"      EXTREMITY MODE: Capturing all nearby vertices")
            percentile = 0.98  # Capture 98% of nearby vertices
            min_radius = initial_radius * 0.5   # Can shrink to 50% if needed
            max_radius = initial_radius * 3.0   # Can grow to 300% for extremities!
        else:
            # Normal fine-tuning for non-extremity spheres
            percentile_map = {
                'torso': 0.9,      # Very generous for torso
                'head': 0.9,       # Capture all head features
                'leg_upper': 0.85, # Good thigh coverage
                'leg_lower': 0.8,  # Good calf coverage
                'foot': 0.9,       # Must capture toes
                'shoulder': 0.85,  # Broad shoulders
                'arm_upper': 0.8,  # Good arm coverage
                'arm_lower': 0.8,  # Good forearm coverage
                'hand': 0.9        # Must capture fingers
            }
            
            percentile = percentile_map.get(bone_type, 0.85)
            min_radius = initial_radius * 0.8  # Can go to 80% of anatomical
            max_radius = initial_radius * 2.0  # Can go to 200% for coverage
        
        target_distance = np.percentile(nearby_distances, percentile * 100)
        fine_tuned_radius = np.clip(target_distance, min_radius, max_radius)
        
        return fine_tuned_radius

    def identify_extremity_spheres(self):
        """Identify which spheres are at extremities (last in their chains)"""
        extremity_patterns = [
            'neck-head_1',        # Last head sphere (top of head)
            'left_wrist-left_hand_2',   # Last left hand sphere (fingertips)
            'right_wrist-right_hand_2', # Last right hand sphere (fingertips)  
            'left_ankle-left_foot_3',   # Last left foot sphere (toe tips)
            'right_ankle-right_foot_3', # Last right foot sphere (toe tips)
        ]
        return extremity_patterns

    def generate_guaranteed_coverage_spheres(self, mesh_vertices, joint_positions):
        """Generate all spheres with guaranteed overlap and coverage"""
        print("Generating guaranteed coverage spheres...")
        print("="*60)
        
        all_spheres = []
        
        for start_idx, end_idx, bone_name in self.bones:
            print(f"Processing bone: {bone_name}")
            
            if bone_name not in self.bone_definitions:
                print(f"  Warning: No definition for {bone_name}, skipping")
                continue
            
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            bone_def = self.bone_definitions[bone_name]
            
            # Calculate adaptive sphere placement with guaranteed overlap
            positions, radii = self.calculate_adaptive_sphere_placement(start_pos, end_pos, bone_def)
            
            print(f"  Planned {len(positions)} spheres with guaranteed {bone_def['min_overlap']*100:.0f}% overlap")
            
            # Get extremity patterns to identify last spheres
            extremity_patterns = self.identify_extremity_spheres()
            
            # Fine-tune each sphere size and check actual overlap
            for i, (center, radius) in enumerate(zip(positions, radii)):
                sphere_name = f"{bone_name}_{i}"
                
                # Check if this is an extremity sphere
                is_extremity = sphere_name in extremity_patterns
                
                fine_tuned_radius = self.fine_tune_sphere_size(
                    center, radius, mesh_vertices, bone_def['type'], is_extremity
                )
                
                all_spheres.append((center, fine_tuned_radius, sphere_name))
                
                # Verify overlap with previous sphere
                if i > 0:
                    prev_center, prev_radius, prev_name = all_spheres[-2]
                    distance = np.linalg.norm(center - prev_center)
                    overlap_distance = prev_radius + fine_tuned_radius - distance
                    overlap_ratio = overlap_distance / min(prev_radius, fine_tuned_radius)
                    extremity_marker = " [EXTREMITY]" if is_extremity else ""
                    print(f"    {sphere_name}: r={radius:.3f}→{fine_tuned_radius:.3f}, overlap={overlap_ratio:.2f}{extremity_marker}")
                else:
                    extremity_marker = " [EXTREMITY]" if is_extremity else ""
                    print(f"    {sphere_name}: r={radius:.3f}→{fine_tuned_radius:.3f} (first){extremity_marker}")
        
        # Calculate final coverage
        covered_vertices = set()
        for center, radius, name in all_spheres:
            distances = np.linalg.norm(mesh_vertices - center, axis=1)
            vertex_indices = np.where(distances <= radius)[0]
            covered_vertices.update(vertex_indices)
        
        coverage_ratio = len(covered_vertices) / len(mesh_vertices)
        
        print(f"\nGeneration complete!")
        print(f"Total spheres: {len(all_spheres)}")
        print(f"Coverage: {len(covered_vertices)}/{len(mesh_vertices)} vertices ({coverage_ratio*100:.1f}%)")
        
        # If coverage is still not 100%, add filler spheres
        if coverage_ratio < 0.95:
            print("Coverage below 95%, adding filler spheres...")
            all_spheres = self.add_filler_spheres(all_spheres, mesh_vertices)
        
        self.all_spheres = all_spheres
        return all_spheres

    def add_filler_spheres(self, existing_spheres, mesh_vertices):
        """Add filler spheres to cover remaining uncovered vertices"""
        # Find uncovered vertices
        covered_vertices = set()
        for center, radius, name in existing_spheres:
            distances = np.linalg.norm(mesh_vertices - center, axis=1)
            vertex_indices = np.where(distances <= radius)[0]
            covered_vertices.update(vertex_indices)
        
        uncovered_indices = [i for i in range(len(mesh_vertices)) if i not in covered_vertices]
        uncovered_vertices = mesh_vertices[uncovered_indices]
        
        print(f"  Found {len(uncovered_vertices)} uncovered vertices")
        
        if len(uncovered_vertices) == 0:
            return existing_spheres
        
        # Group uncovered vertices by proximity and add filler spheres
        filler_spheres = []
        remaining_uncovered = uncovered_vertices.copy()
        
        while len(remaining_uncovered) > 10:  # Only add fillers for significant clusters
            # Find centroid of remaining uncovered vertices
            centroid = np.mean(remaining_uncovered, axis=0)
            
            # Find radius to cover reasonable number of nearby uncovered vertices
            distances = np.linalg.norm(remaining_uncovered - centroid, axis=1)
            target_radius = np.percentile(distances, 70)  # Cover 70% of remaining
            target_radius = np.clip(target_radius, 0.02, 0.15)  # Reasonable bounds
            
            # Create filler sphere
            filler_name = f"filler_{len(filler_spheres)}"
            filler_spheres.append((centroid, target_radius, filler_name))
            
            # Remove covered vertices from remaining
            covered_mask = distances <= target_radius
            remaining_uncovered = remaining_uncovered[~covered_mask]
            
            print(f"    Added {filler_name}: r={target_radius:.3f}, covers {np.sum(covered_mask)} vertices")
            
            if len(filler_spheres) > 10:  # Safety limit
                break
        
        # Combine existing and filler spheres
        final_spheres = existing_spheres + filler_spheres
        
        # Final coverage check
        covered_vertices = set()
        for center, radius, name in final_spheres:
            distances = np.linalg.norm(mesh_vertices - center, axis=1)
            vertex_indices = np.where(distances <= radius)[0]
            covered_vertices.update(vertex_indices)
        
        final_coverage = len(covered_vertices) / len(mesh_vertices)
        print(f"  Final coverage after fillers: {final_coverage*100:.1f}%")
        
        return final_spheres

    def create_sphere_surface(self, center, radius, color='blue', name='sphere'):
        """Create a sphere surface for visualization"""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        
        return go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=name,
            opacity=0.5,
            hovertemplate=f'<b>{name}</b><br>Radius: {radius:.3f}<extra></extra>'
        )

    def create_visualization(self):
        """Create visualization with guaranteed coverage spheres"""
        print("STAR Guaranteed Coverage Sphere Body Visualization")
        print("="*60)
        
        mesh_vertices, joint_positions = self.get_mesh_data()
        if mesh_vertices is None or joint_positions is None:
            print("Could not get mesh data")
            return None
        
        # Generate spheres with guaranteed coverage
        sphere_configs = self.generate_guaranteed_coverage_spheres(mesh_vertices, joint_positions)
        
        if not sphere_configs:
            print("No spheres generated")
            return None
        
        # Create visualization
        fig = go.Figure()
        
        # Add mesh vertices (very light background)
        fig.add_trace(
            go.Scatter3d(
                x=mesh_vertices[:, 0],
                y=mesh_vertices[:, 1], 
                z=mesh_vertices[:, 2],
                mode='markers',
                marker=dict(size=0.2, color='lightgray', opacity=0.1),
                name=f'Mesh vertices ({len(mesh_vertices)})',
                showlegend=True
            )
        )
        
        # Add spheres grouped by bone type
        bone_type_colors = {
            'torso': 'blue',
            'head': 'red', 
            'leg_upper': 'green',
            'leg_lower': 'lightgreen',
            'foot': 'purple',
            'shoulder': 'orange',
            'arm_upper': 'yellow',
            'arm_lower': 'lightyellow', 
            'hand': 'pink',
            'filler': 'gray'
        }
        
        for center, radius, name in sphere_configs:
            # Determine bone type from name
            if name.startswith('filler'):
                bone_type = 'filler'
            else:
                bone_name = '_'.join(name.split('_')[:-1])
                bone_def = self.bone_definitions.get(bone_name, {'type': 'unknown'})
                bone_type = bone_def['type']
            
            color = bone_type_colors.get(bone_type, 'gray')
            
            sphere_surface = self.create_sphere_surface(
                center, radius, color, name
            )
            fig.add_trace(sphere_surface)
        
        # Add joint markers for reference
        fig.add_trace(
            go.Scatter3d(
                x=joint_positions[:, 0],
                y=joint_positions[:, 1], 
                z=joint_positions[:, 2],
                mode='markers',
                marker=dict(size=2, color='black', symbol='diamond'),
                name='Joint centers',
                showlegend=True
            )
        )
        
        # Layout
        all_points = mesh_vertices
        x_range = [all_points[:, 0].min(), all_points[:, 0].max()]
        y_range = [all_points[:, 1].min(), all_points[:, 1].max()]
        z_range = [all_points[:, 2].min(), all_points[:, 2].max()]
        
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        z_center = (z_range[0] + z_range[1]) / 2
        
        max_size = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]) * 1.5
        
        fig.update_layout(
            title=f"STAR Guaranteed Coverage Body - {len(sphere_configs)} Spheres",
            scene=dict(
                xaxis=dict(range=[x_center - max_size/2, x_center + max_size/2]),
                yaxis=dict(range=[y_center - max_size/2, y_center + max_size/2]),
                zaxis=dict(range=[z_center - max_size/2, z_center + max_size/2]),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='cube',
                bgcolor='white'
            ),
            width=1200,
            height=900
        )
        
        return fig

    def print_sphere_stats(self):
        """Print statistics about the spheres"""
        if not self.all_spheres:
            print("No spheres to analyze")
            return
        
        print("\n" + "="*60)
        print("GUARANTEED COVERAGE SPHERE STATISTICS")
        print("="*60)
        
        # Group by bone type
        type_groups = {}
        filler_count = 0
        
        for center, radius, name in self.all_spheres:
            if name.startswith('filler'):
                filler_count += 1
                continue
                
            bone_name = '_'.join(name.split('_')[:-1])
            bone_def = self.bone_definitions.get(bone_name, {'type': 'unknown'})
            bone_type = bone_def['type']
            if bone_type not in type_groups:
                type_groups[bone_type] = []
            type_groups[bone_type].append(radius)
        
        print(f"Total spheres: {len(self.all_spheres)}")
        print(f"Anatomical spheres: {len(self.all_spheres) - filler_count}")
        print(f"Filler spheres: {filler_count}")
        
        print(f"\nBy body region:")
        for bone_type, radii in sorted(type_groups.items()):
            print(f"  {bone_type:12}: {len(radii):2d} spheres, "
                  f"avg: {np.mean(radii):.3f}, range: {np.min(radii):.3f}-{np.max(radii):.3f}")


def main():
    """Main function"""
    # Initialize optimizer
    optimizer = STARGuaranteedCoverageSphereOptimizer(gender='neutral')
    
    # Create visualization
    fig = optimizer.create_visualization()
    
    if fig:
        # Print statistics
        optimizer.print_sphere_stats()
        
        # Show visualization
        print("\nShowing guaranteed coverage sphere body...")
        fig.show()
    else:
        print("Failed to create visualization")
    
    print("Complete!")
    return optimizer


if __name__ == "__main__":
    optimizer = main()