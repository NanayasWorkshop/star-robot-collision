#!/usr/bin/env python3
"""
STAR Joint Influence Visualizer
Shows which joints influence which mesh triangles using color coding
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# STAR model imports
try:
    from star.pytorch.star import STAR
    STAR_AVAILABLE = True
except ImportError:
    print("WARNING: STAR model not available. Install from https://github.com/ahmedosman/STAR")
    STAR_AVAILABLE = False


class STARJointInfluenceAnalyzer:
    """Analyze and visualize joint influences on STAR mesh"""
    
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
        
        # Color palette for joints
        self.colors = px.colors.qualitative.Set3 + px.colors.qualitative.Pastel + px.colors.qualitative.Dark24
        
    def get_t_pose_mesh(self):
        """Get T-pose mesh (zero pose)"""
        batch_size = 1
        pose_params = torch.zeros(batch_size, 72, device=self.device)  # T-pose
        shape_params = torch.zeros(batch_size, 10, device=self.device)  # Neutral shape
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        with torch.no_grad():
            result = self.model(pose_params, shape_params, trans)
            if isinstance(result, tuple):
                vertices, joints = result
            else:
                vertices = result
        
        vertices_np = vertices[0].cpu().numpy()
        faces_np = self.model.faces.cpu().numpy()
        
        return vertices_np, faces_np
    
    def analyze_joint_influences(self, perturbation=0.1):
        """
        Analyze which vertices are influenced by each joint
        by applying small perturbations to each joint
        """
        print("Analyzing joint influences...")
        
        # Get T-pose reference
        ref_vertices, faces = self.get_t_pose_mesh()
        
        joint_influences = {}
        
        # Test each joint (24 joints * 3 axes = 72 parameters)
        for joint_idx in range(24):
            joint_name = self.joint_names[joint_idx]
            print(f"Testing joint {joint_idx}: {joint_name}")
            
            max_influence = np.zeros(len(ref_vertices))
            
            # Test each axis of the joint (x, y, z rotations)
            for axis in range(3):
                pose_param_idx = joint_idx * 3 + axis
                
                # Create perturbed pose
                pose_params = torch.zeros(1, 72, device=self.device)
                pose_params[0, pose_param_idx] = perturbation
                
                shape_params = torch.zeros(1, 10, device=self.device)
                trans = torch.zeros(1, 3, device=self.device)
                
                with torch.no_grad():
                    result = self.model(pose_params, shape_params, trans)
                    if isinstance(result, tuple):
                        perturbed_vertices, _ = result
                    else:
                        perturbed_vertices = result
                
                perturbed_vertices_np = perturbed_vertices[0].cpu().numpy()
                
                # Calculate vertex displacement
                displacement = np.linalg.norm(perturbed_vertices_np - ref_vertices, axis=1)
                
                # Keep maximum influence across all axes for this joint
                max_influence = np.maximum(max_influence, displacement)
            
            joint_influences[joint_name] = max_influence
        
        return ref_vertices, faces, joint_influences
    
    def assign_vertices_to_dominant_joints(self, joint_influences, threshold=1e-4):
        """
        Assign vertices exactly like Plotly does: "last one wins" approach
        Each joint draws its points in order, overwriting previous joints
        """
        print("Assigning vertices using 'last one wins' approach (mimicking Plotly)...")
        
        # Find maximum influence across all joints for normalization
        all_influences = np.array([influences for influences in joint_influences.values()])
        max_global_influence = np.max(all_influences)
        
        # Create array to track which joint "owns" each vertex (-1 = unassigned)
        n_vertices = len(next(iter(joint_influences.values())))
        vertex_owner = np.full(n_vertices, -1, dtype=int)
        
        # Process joints in order (like Plotly renders traces)
        # Each joint "paints" its vertices, overwriting any previous assignments
        for joint_idx, (joint_name, influences) in enumerate(joint_influences.items()):
            # Normalize influences to 0-100%
            normalized_influences = influences / max_global_influence * 100
            
            # This joint claims all vertices where it has significant influence
            # Use the same 10% threshold as the original combined plot
            significant_mask = normalized_influences >= 10
            
            # "Paint" these vertices with this joint's color (overwriting previous)
            vertex_indices_to_paint = np.where(significant_mask)[0]
            for vertex_idx in vertex_indices_to_paint:
                vertex_owner[vertex_idx] = joint_idx  # Last one wins!
        
        # Create final joint assignments based on who "owns" each vertex
        joint_assignments = {}
        for joint_idx, joint_name in enumerate(self.joint_names):
            owned_vertices = np.where(vertex_owner == joint_idx)[0]
            joint_assignments[joint_name] = owned_vertices
            
        return joint_assignments, vertex_owner, None
    
    def get_joint_positions(self):
        """Get joint positions from T-pose"""
        batch_size = 1
        pose_params = torch.zeros(batch_size, 72, device=self.device)  # T-pose
        shape_params = torch.zeros(batch_size, 10, device=self.device)  # Neutral shape
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        with torch.no_grad():
            try:
                # Try the standard forward pass first
                result = self.model(pose_params, shape_params, trans)
                if isinstance(result, tuple) and len(result) >= 2:
                    vertices, joints = result
                    print(f"Got joints from forward pass: {joints.shape}")
                    return joints[0].cpu().numpy()
                else:
                    print("Forward pass didn't return joints, trying alternative method...")
                    # Try accessing joints directly if available
                    if hasattr(self.model, 'J_regressor'):
                        vertices = result if not isinstance(result, tuple) else result[0]
                        joints = torch.matmul(self.model.J_regressor, vertices)
                        print(f"Got joints from J_regressor: {joints.shape}")
                        return joints[0].cpu().numpy()
                    else:
                        print("No J_regressor found, returning None")
                        return None
            except Exception as e:
                print(f"Error getting joint positions: {e}")
                return None

    def get_bone_connections(self):
        """Define bone connections for STAR skeleton (parent-child relationships)"""
        # STAR kinematic chain connections (parent -> child)
        connections = [
            (0, 3),   # pelvis -> spine1
            (3, 6),   # spine1 -> spine2  
            (6, 9),   # spine2 -> spine3
            (9, 12),  # spine3 -> neck
            (12, 15), # neck -> head
            
            # Left side
            (0, 1),   # pelvis -> left_hip
            (1, 4),   # left_hip -> left_knee
            (4, 7),   # left_knee -> left_ankle
            (7, 10),  # left_ankle -> left_foot
            
            # Right side  
            (0, 2),   # pelvis -> right_hip
            (2, 5),   # right_hip -> right_knee
            (5, 8),   # right_knee -> right_ankle
            (8, 11),  # right_ankle -> right_foot
            
            # Left arm
            (9, 13),  # spine3 -> left_collar
            (13, 16), # left_collar -> left_shoulder
            (16, 18), # left_shoulder -> left_elbow
            (18, 20), # left_elbow -> left_wrist
            (20, 22), # left_wrist -> left_hand
            
            # Right arm
            (9, 14),  # spine3 -> right_collar
            (14, 17), # right_collar -> right_shoulder
            (17, 19), # right_shoulder -> right_elbow
            (19, 21), # right_elbow -> right_wrist
            (21, 23), # right_wrist -> right_hand
        ]
        return connections

    def create_combined_and_individual_plots(self, vertices, faces, joint_influences, threshold=1e-4):
        """Create both combined plot and individual subplots showing exact vertex assignments"""
        
        # Get vertex assignments (which joint dominates each vertex)
        joint_assignments, dominant_joint, max_influence_per_vertex = self.assign_vertices_to_dominant_joints(joint_influences, threshold)
        
        # Get joint positions and bone connections
        try:
            joint_positions = self.get_joint_positions()
        except:
            print("Warning: Could not get joint positions, skeleton will not be shown")
            joint_positions = None
        
        # 1. Create COMBINED plot showing the final color assignment + skeleton
        fig_combined = go.Figure()
        
        # Add vertex assignments
        for joint_idx, (joint_name, assigned_vertex_indices) in enumerate(joint_assignments.items()):
            if len(assigned_vertex_indices) > 0:
                assigned_vertices = vertices[assigned_vertex_indices]
                
                fig_combined.add_trace(
                    go.Scatter3d(
                        x=assigned_vertices[:, 0],
                        y=assigned_vertices[:, 1],
                        z=assigned_vertices[:, 2],
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=self.colors[joint_idx % len(self.colors)],
                        ),
                        name=f'{joint_idx}: {joint_name}',
                        legendgroup=f'joint_{joint_idx}'
                    )
                )
        
        # Add skeleton if available
        if joint_positions is not None:
            print(f"Adding skeleton with {len(joint_positions)} joints")
            print(f"Joint positions range: X({joint_positions[:, 0].min():.3f}, {joint_positions[:, 0].max():.3f})")
            print(f"Joint positions range: Y({joint_positions[:, 1].min():.3f}, {joint_positions[:, 1].max():.3f})")
            print(f"Joint positions range: Z({joint_positions[:, 2].min():.3f}, {joint_positions[:, 2].max():.3f})")
            
            # Add joint positions as numbered spheres
            fig_combined.add_trace(
                go.Scatter3d(
                    x=joint_positions[:, 0],
                    y=joint_positions[:, 1], 
                    z=joint_positions[:, 2],
                    mode='markers+text',
                    marker=dict(
                        size=8,  # Smaller size
                        color=[self.colors[i % len(self.colors)] for i in range(len(joint_positions))],  # Match vertex colors
                        symbol='circle',
                        line=dict(color='black', width=2)  # Keep black outline
                    ),
                    text=[f'{i}' for i in range(len(joint_positions))],
                    textposition='middle center',
                    textfont=dict(color='white', size=10, family='Arial Black'),
                    name='Joint Centers',
                    showlegend=True,
                    legendgroup='skeleton'
                )
            )
            
            # Add bone connections
            bone_connections = self.get_bone_connections()
            print(f"Adding {len(bone_connections)} bone connections")
            for i, (parent_idx, child_idx) in enumerate(bone_connections):
                if parent_idx < len(joint_positions) and child_idx < len(joint_positions):
                    parent_pos = joint_positions[parent_idx]
                    child_pos = joint_positions[child_idx]
                    
                    fig_combined.add_trace(
                        go.Scatter3d(
                            x=[parent_pos[0], child_pos[0]],
                            y=[parent_pos[1], child_pos[1]],
                            z=[parent_pos[2], child_pos[2]],
                            mode='lines',
                            line=dict(color='black', width=6),  # Thicker lines
                            name=f'Bone {parent_idx}-{child_idx}',
                            showlegend=False
                        )
                    )
        else:
            print("No joint positions available - skeleton not shown")
        
        # Set up combined plot layout
        x_range = [vertices[:, 0].min(), vertices[:, 0].max()]
        y_range = [vertices[:, 1].min(), vertices[:, 1].max()]
        z_range = [vertices[:, 2].min(), vertices[:, 2].max()]
        
        x_center = (x_range[0] + x_range[1]) / 2
        y_center = (y_range[0] + y_range[1]) / 2
        z_center = (z_range[0] + z_range[1]) / 2
        
        max_size = max(x_range[1] - x_range[0], y_range[1] - y_range[0], z_range[1] - z_range[0]) * 1.1
        
        fig_combined.update_layout(
            title="Combined View - Vertices Assigned to Dominant Joints",
            scene=dict(
                xaxis=dict(range=[x_center - max_size/2, x_center + max_size/2]),
                yaxis=dict(range=[y_center - max_size/2, y_center + max_size/2]),
                zaxis=dict(range=[z_center - max_size/2, z_center + max_size/2]),
                aspectratio=dict(x=1, y=1, z=1),
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        # 2. Create INDIVIDUAL subplots showing each joint's assigned vertices
        cols = 6
        rows = (len(self.joint_names) + cols - 1) // cols
        
        fig_individual = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[f"{name} ({len(joint_assignments[name])} vertices)" 
                          for name in self.joint_names],
            specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
        )
        
        for joint_idx, joint_name in enumerate(self.joint_names):
            row = joint_idx // cols + 1
            col = joint_idx % cols + 1
            
            assigned_vertex_indices = joint_assignments[joint_name]
            
            if len(assigned_vertex_indices) > 0:
                assigned_vertices = vertices[assigned_vertex_indices]
                
                fig_individual.add_trace(
                    go.Scatter3d(
                        x=assigned_vertices[:, 0],
                        y=assigned_vertices[:, 1],
                        z=assigned_vertices[:, 2],
                        mode='markers',
                        marker=dict(
                            size=2,
                            color=self.colors[joint_idx % len(self.colors)],
                        ),
                        name=joint_name,
                        showlegend=False
                    ),
                    row=row, col=col
                )
                
                # Set individual subplot bounds
                if len(assigned_vertices) > 0:
                    x_min, x_max = assigned_vertices[:, 0].min(), assigned_vertices[:, 0].max()
                    y_min, y_max = assigned_vertices[:, 1].min(), assigned_vertices[:, 1].max()
                    z_min, z_max = assigned_vertices[:, 2].min(), assigned_vertices[:, 2].max()
                    
                    x_center_sub = (x_min + x_max) / 2
                    y_center_sub = (y_min + y_max) / 2
                    z_center_sub = (z_min + z_max) / 2
                    
                    max_size_sub = max(x_max - x_min, y_max - y_min, z_max - z_min) * 1.2
                    max_size_sub = max(max_size_sub, 0.1)  # Minimum size
                    
                    scene_key = f'scene{joint_idx + 1}' if joint_idx > 0 else 'scene'
                    fig_individual.layout[scene_key].update(
                        xaxis=dict(range=[x_center_sub - max_size_sub/2, x_center_sub + max_size_sub/2]),
                        yaxis=dict(range=[y_center_sub - max_size_sub/2, y_center_sub + max_size_sub/2]),
                        zaxis=dict(range=[z_center_sub - max_size_sub/2, z_center_sub + max_size_sub/2]),
                        aspectratio=dict(x=1, y=1, z=1),
                        aspectmode='cube'
                    )
        
        fig_individual.update_layout(
            title="Individual Joint Assignments - Exactly as in Combined Plot",
            height=200 * rows,
            showlegend=False
        )
        
        return fig_combined, fig_individual, joint_assignments
    
    def print_assignment_stats(self, joint_assignments, total_vertices):
        """Print statistics about vertex assignments"""
        print("\n=== Vertex Assignment Statistics ===")
        
        total_assigned = sum(len(vertices) for vertices in joint_assignments.values())
        
        for joint_name, assigned_vertices in joint_assignments.items():
            count = len(assigned_vertices)
            percentage = count / total_vertices * 100
            print(f"{joint_name:15}: {count:4d} vertices ({percentage:5.1f}%)")
        
        print(f"\nTotal assigned: {total_assigned} / {total_vertices} ({total_assigned/total_vertices*100:.1f}%)")
    
    def print_influence_stats(self, joint_influences, threshold=1e-4):
        """Print statistics about joint influences"""
        print("\n=== Joint Influence Statistics ===")
        total_vertices = len(next(iter(joint_influences.values())))
        
        for joint_name, influences in joint_influences.items():
            influenced_count = np.sum(influences > threshold)
            max_influence = np.max(influences)
            avg_influence = np.mean(influences[influences > threshold]) if influenced_count > 0 else 0
            
            print(f"{joint_name:15}: {influenced_count:4d} vertices ({influenced_count/total_vertices*100:5.1f}%) "
                  f"max={max_influence:.6f} avg={avg_influence:.6f}")


def main():
    """Main function to run joint influence analysis"""
    print("STAR Joint Influence Analysis")
    print("="*40)
    
    # Initialize analyzer
    analyzer = STARJointInfluenceAnalyzer(gender='neutral')
    
    # Analyze influences
    vertices, faces, joint_influences = analyzer.analyze_joint_influences(perturbation=0.1)
    
    # Print statistics
    analyzer.print_influence_stats(joint_influences)
    
    # Create visualizations
    print("\nCreating visualizations...")
    fig_combined, fig_individual, joint_assignments = analyzer.create_combined_and_individual_plots(
        vertices, faces, joint_influences
    )
    
    # Print assignment statistics
    analyzer.print_assignment_stats(joint_assignments, len(vertices))
    
    # Show plots
    print("Showing combined plot...")
    fig_combined.show()
    
    print("Showing individual joint assignments...")
    fig_individual.show()
    
    print("\nAnalysis complete!")
    return analyzer, vertices, faces, joint_influences, joint_assignments


if __name__ == "__main__":
    analyzer, vertices, faces, joint_influences, joint_assignments = main()