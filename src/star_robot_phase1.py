#!/usr/bin/env python3
"""
STAR-Robot Collision Avoidance System - Phase 1
Random STAR poses → FCL BVH with Plotly visualization

Dependencies:
    pip install torch numpy plotly python-fcl
    git clone https://github.com/ahmedosman/STAR
    cd STAR && pip install -e .
"""

import numpy as np
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import fcl
from typing import Tuple, Optional

# STAR model imports (assuming STAR is installed)
try:
    from star.pytorch.star import STAR
    STAR_AVAILABLE = True
except ImportError:
    print("WARNING: STAR model not available. Install from https://github.com/ahmedosman/STAR")
    STAR_AVAILABLE = False


class RandomJointGenerator:
    """Generate random joint angles for STAR model"""
    
    def __init__(self, seed: Optional[int] = None):
        """
        Initialize random joint generator
        
        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # STAR model has 72 pose parameters (24 joints * 3 axis-angle components)
        self.n_pose_params = 72
        
        # Reasonable joint angle limits (in radians)
        self.joint_limits = {
            'min': -np.pi/2,  # -90 degrees
            'max': np.pi/2,   # +90 degrees
        }
    
    def generate_random_pose(self) -> torch.Tensor:
        """
        Generate random joint angles for STAR model
        
        Returns:
            torch.Tensor: Random pose parameters [72]
        """
        # Generate random angles within limits
        angles = np.random.uniform(
            self.joint_limits['min'], 
            self.joint_limits['max'], 
            self.n_pose_params
        )
        
        # Convert to torch tensor
        pose_params = torch.tensor(angles, dtype=torch.float32)
        
        return pose_params
    
    def generate_sequence(self, n_poses: int = 5) -> torch.Tensor:
        """
        Generate sequence of random poses
        
        Args:
            n_poses: Number of poses to generate
            
        Returns:
            torch.Tensor: Batch of pose parameters [n_poses, 72]
        """
        poses = []
        for _ in range(n_poses):
            poses.append(self.generate_random_pose())
        
        return torch.stack(poses)


class STARMeshProcessor:
    """Process STAR model outputs for collision detection"""
    
    def __init__(self, gender: str = 'neutral'):
        """
        Initialize STAR mesh processor
        
        Args:
            gender: Gender for STAR model ('neutral', 'male', 'female')
        """
        if not STAR_AVAILABLE:
            raise ImportError("STAR model not available. Install from https://github.com/ahmedosman/STAR")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = STAR(gender=gender)
        self.model.to(self.device)
        self.model.eval()
        
        # Store current mesh data
        self.current_vertices = None
        self.current_faces = None
        
    def forward_pass(self, pose_params: torch.Tensor, 
                    shape_params: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run STAR forward pass to get mesh vertices
        
        Args:
            pose_params: Joint angles [72] or [batch, 72]
            shape_params: Body shape parameters [10] or [batch, 10]
            
        Returns:
            Tuple of (vertices, faces) as numpy arrays
        """
        # Ensure batch dimension
        if pose_params.dim() == 1:
            pose_params = pose_params.unsqueeze(0)
        
        batch_size = pose_params.shape[0]
        
        # Default shape parameters (neutral body)
        if shape_params is None:
            shape_params = torch.zeros(batch_size, 10, device=self.device)
        
        # Default translation (no translation)
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        # Move to device
        pose_params = pose_params.to(self.device)
        shape_params = shape_params.to(self.device)
        
        # Forward pass
        with torch.no_grad():
            result = self.model(pose_params, shape_params, trans)
            # STAR model returns only vertices, not joints
            if isinstance(result, tuple):
                vertices, joints = result
            else:
                vertices = result
        
        # Get vertices and faces
        vertices_np = vertices[0].cpu().numpy()  # Take first batch item
        faces_np = self.model.faces.cpu().numpy()
        
        # Store current mesh
        self.current_vertices = vertices_np
        self.current_faces = faces_np
        
        return vertices_np, faces_np


class FCLBVHManager:
    """Manage FCL BVH models for collision detection"""
    
    def __init__(self):
        """Initialize FCL BVH manager"""
        self.bvh_model = None
        self.collision_object = None
        
    def create_bvh_from_mesh(self, vertices: np.ndarray, faces: np.ndarray) -> fcl.BVHModel:
        """
        Create FCL BVH model from mesh data
        
        Args:
            vertices: Mesh vertices [n_vertices, 3]
            faces: Mesh faces [n_faces, 3]
            
        Returns:
            fcl.BVHModel: Created BVH model
        """
        # Convert to double precision for FCL and ensure contiguous arrays
        vertices = np.ascontiguousarray(vertices, dtype=np.float64)
        faces = np.ascontiguousarray(faces, dtype=np.int32)
        
        # Create BVH model using the simpler constructor
        bvh = fcl.BVHModel()
        
        # Use the direct mesh construction method
        bvh.beginModel()
        bvh.addSubModel(vertices, faces)
        bvh.endModel()
        
        # Store references and cache faces for updates
        self.bvh_model = bvh
        self.collision_object = fcl.CollisionObject(bvh, fcl.Transform())
        self._cached_faces = faces  # Cache faces for updates
        
        return bvh
    
    def update_bvh_vertices(self, vertices: np.ndarray) -> None:
        """
        Update BVH model vertices (refit)
        For python-fcl, we need to recreate the BVH model
        
        Args:
            vertices: New mesh vertices [n_vertices, 3]
        """
        if self.bvh_model is None:
            raise ValueError("BVH model not created yet")
        
        # Get the faces from the previous model
        # For python-fcl, we need to recreate the entire BVH
        # This is less efficient but works with the available API
        
        # Store the faces (we'll reuse them)
        if hasattr(self, '_cached_faces'):
            faces = self._cached_faces
        else:
            raise ValueError("No cached faces available for update")
        
        # Recreate the BVH model with new vertices
        self.create_bvh_from_mesh(vertices, faces)
    
    def get_bvh_info(self) -> dict:
        """Get BVH model information"""
        if self.bvh_model is None:
            return {"status": "No BVH model created"}
        
        return {
            "num_tris": self.bvh_model.num_tries_(),  # Call the method
            "node_type": self.bvh_model.getNodeType(),
            "status": "BVH model ready"
        }


class STARVisualization:
    """Plotly visualization for STAR meshes"""
    
    def __init__(self):
        """Initialize visualization"""
        self.fig = None
        
    def create_mesh_plot(self, vertices: np.ndarray, faces: np.ndarray, 
                        title: str = "STAR Mesh") -> go.Figure:
        """
        Create 3D mesh plot
        
        Args:
            vertices: Mesh vertices [n_vertices, 3]
            faces: Mesh faces [n_faces, 3]
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D mesh plot
        """
        # Create mesh3d trace
        mesh_trace = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1], 
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightblue',
            opacity=0.7,
            name='STAR Mesh'
        )
        
        # Create figure
        fig = go.Figure(data=[mesh_trace])
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        self.fig = fig
        return fig
    
    def create_vertex_plot(self, vertices: np.ndarray, 
                          title: str = "STAR Vertices") -> go.Figure:
        """
        Create scatter plot of vertices only
        
        Args:
            vertices: Mesh vertices [n_vertices, 3]
            title: Plot title
            
        Returns:
            plotly.graph_objects.Figure: 3D scatter plot
        """
        scatter_trace = go.Scatter3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=vertices[:, 1],  # Color by height
                colorscale='Viridis'
            ),
            name='Vertices'
        )
        
        fig = go.Figure(data=[scatter_trace])
        
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y', 
                zaxis_title='Z',
                aspectmode='cube'
            ),
            width=800,
            height=600
        )
        
        return fig
    
    def create_comparison_plot(self, vertices_list: list, titles: list) -> go.Figure:
        """
        Create comparison plot of multiple poses
        
        Args:
            vertices_list: List of vertex arrays
            titles: List of titles for each pose
            
        Returns:
            plotly.graph_objects.Figure: Comparison plot
        """
        # Create subplots
        n_plots = len(vertices_list)
        cols = min(3, n_plots)
        rows = (n_plots + cols - 1) // cols
        
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=titles,
            specs=[[{"type": "scatter3d"} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add traces
        for i, (vertices, title) in enumerate(zip(vertices_list, titles)):
            row = i // cols + 1
            col = i % cols + 1
            
            fig.add_trace(
                go.Scatter3d(
                    x=vertices[:, 0],
                    y=vertices[:, 1],
                    z=vertices[:, 2],
                    mode='markers',
                    marker=dict(size=1, color='blue'),
                    name=f'Pose {i+1}'
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title="STAR Pose Comparison",
            height=400 * rows,
            showlegend=False
        )
        
        return fig


class STARPhase1System:
    """Main system for Phase 1 implementation"""
    
    def __init__(self, gender: str = 'neutral', seed: int = 42):
        """
        Initialize Phase 1 system
        
        Args:
            gender: Gender for STAR model ('neutral', 'male', 'female')
            seed: Random seed for reproducibility
        """
        self.joint_generator = RandomJointGenerator(seed=seed)
        self.mesh_processor = STARMeshProcessor(gender=gender)
        self.fcl_manager = FCLBVHManager()
        self.visualizer = STARVisualization()
        
        print("STAR-Robot Phase 1 System Initialized")
        print(f"Device: {self.mesh_processor.device}")
    
    def run_single_pose_test(self) -> dict:
        """
        Run single pose generation and processing test
        
        Returns:
            dict: Test results and timing information
        """
        print("\n=== Single Pose Test ===")
        
        # Generate random pose
        start_time = time.time()
        pose_params = self.joint_generator.generate_random_pose()
        pose_gen_time = time.time() - start_time
        
        print(f"Generated pose parameters: shape={pose_params.shape}")
        print(f"Pose generation time: {pose_gen_time*1000:.2f}ms")
        
        # STAR forward pass
        start_time = time.time()
        vertices, faces = self.mesh_processor.forward_pass(pose_params)
        star_time = time.time() - start_time
        
        print(f"STAR forward pass: {star_time*1000:.2f}ms")
        print(f"Vertices: {vertices.shape}, Faces: {faces.shape}")
        print(f"Vertex bounds: X[{vertices[:,0].min():.3f}, {vertices[:,0].max():.3f}]")
        print(f"               Y[{vertices[:,1].min():.3f}, {vertices[:,1].max():.3f}]")
        print(f"               Z[{vertices[:,2].min():.3f}, {vertices[:,2].max():.3f}]")
        
        # Create FCL BVH
        start_time = time.time()
        bvh_model = self.fcl_manager.create_bvh_from_mesh(vertices, faces)
        fcl_time = time.time() - start_time
        
        print(f"FCL BVH creation: {fcl_time*1000:.2f}ms")
        print(f"BVH Info: {self.fcl_manager.get_bvh_info()}")
        
        return {
            'pose_params': pose_params,
            'vertices': vertices,
            'faces': faces,
            'timing': {
                'pose_generation': pose_gen_time * 1000,
                'star_forward': star_time * 1000,
                'fcl_creation': fcl_time * 1000,
                'total': (pose_gen_time + star_time + fcl_time) * 1000
            }
        }
    
    def run_update_test(self, n_updates: int = 5) -> dict:
        """
        Run BVH update/refit test with multiple poses
        
        Args:
            n_updates: Number of pose updates to test
            
        Returns:
            dict: Test results
        """
        print(f"\n=== BVH Update Test ({n_updates} updates) ===")
        
        # Initial pose
        initial_result = self.run_single_pose_test()
        
        update_times = []
        vertices_list = [initial_result['vertices']]
        
        # Update poses
        for i in range(n_updates):
            print(f"\nUpdate {i+1}/{n_updates}")
            
            # Generate new pose
            pose_params = self.joint_generator.generate_random_pose()
            vertices, faces = self.mesh_processor.forward_pass(pose_params)
            
            # Update BVH
            start_time = time.time()
            self.fcl_manager.update_bvh_vertices(vertices)
            update_time = time.time() - start_time
            
            update_times.append(update_time * 1000)
            vertices_list.append(vertices)
            
            print(f"BVH update time: {update_time*1000:.2f}ms")
        
        avg_update_time = np.mean(update_times)
        print(f"\nAverage BVH update time: {avg_update_time:.2f}ms")
        print(f"Update time range: [{min(update_times):.2f}, {max(update_times):.2f}]ms")
        
        return {
            'initial_result': initial_result,
            'update_times': update_times,
            'vertices_list': vertices_list,
            'avg_update_time': avg_update_time
        }
    
    def run_visualization_test(self, n_poses: int = 3) -> None:
        """
        Run visualization test with multiple poses
        
        Args:
            n_poses: Number of poses to visualize
        """
        print(f"\n=== Visualization Test ({n_poses} poses) ===")
        
        vertices_list = []
        titles = []
        
        # Generate poses
        for i in range(n_poses):
            pose_params = self.joint_generator.generate_random_pose()
            vertices, faces = self.mesh_processor.forward_pass(pose_params)
            
            vertices_list.append(vertices)
            titles.append(f"Random Pose {i+1}")
            
            # Show individual mesh plot for first pose
            if i == 0:
                fig = self.visualizer.create_mesh_plot(vertices, faces, f"STAR Mesh - Pose {i+1}")
                fig.show()
        
        # Show comparison plot
        if len(vertices_list) > 1:
            fig = self.visualizer.create_comparison_plot(vertices_list, titles)
            fig.show()
        
        print("Visualization complete - check browser for plots")
    
    def run_full_phase1_test(self) -> None:
        """Run complete Phase 1 test suite"""
        print("="*50)
        print("STAR-Robot Phase 1: Complete Test Suite")
        print("="*50)
        
        # Single pose test
        single_result = self.run_single_pose_test()
        
        # Update test
        update_result = self.run_update_test(n_updates=3)
        
        # Visualization test
        self.run_visualization_test(n_poses=3)
        
        # Performance summary
        print("\n" + "="*50)
        print("PHASE 1 PERFORMANCE SUMMARY")
        print("="*50)
        print(f"Single pose processing: {single_result['timing']['total']:.2f}ms")
        print(f"  - Pose generation: {single_result['timing']['pose_generation']:.2f}ms")
        print(f"  - STAR forward: {single_result['timing']['star_forward']:.2f}ms")
        print(f"  - FCL BVH creation: {single_result['timing']['fcl_creation']:.2f}ms")
        print(f"Average BVH update: {update_result['avg_update_time']:.2f}ms")
        
        # Target comparison
        print(f"\nTarget vs Actual:")
        print(f"  STAR Update Target: 1-5ms | Actual: {single_result['timing']['star_forward']:.2f}ms")
        print(f"  FCL Creation Target: 0.1-0.3ms | Actual: {single_result['timing']['fcl_creation']:.2f}ms")
        
        print("\nPhase 1 Complete! ✓")


# Example usage and testing
if __name__ == "__main__":
    # Initialize system
    system = STARPhase1System(seed=42)
    
    # Run full test suite
    system.run_full_phase1_test()
    
    # Additional individual tests
    print("\n" + "="*30)
    print("Additional Testing Options:")
    print("="*30)
    print("# Test single pose:")
    print("result = system.run_single_pose_test()")
    print()
    print("# Test BVH updates:")
    print("result = system.run_update_test(n_updates=10)")
    print()
    print("# Test visualization:")
    print("system.run_visualization_test(n_poses=5)")