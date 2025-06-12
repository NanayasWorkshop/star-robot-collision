#!/usr/bin/env python3
"""
Python wrapper for bone collision system with fixed bone diameters
Handles STAR integration and proper cylinder visualization
"""

import numpy as np
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import time
from typing import Dict, List, Tuple, Optional, Union
import bone_collision  # Our C++ module

# STAR model imports - REQUIRED, no fallbacks
try:
    from star.pytorch.star import STAR
except ImportError as e:
    print("ERROR: STAR model is required but not available!")
    print("Install STAR model from: https://github.com/ahmedosman/STAR")
    print("Commands:")
    print("  git clone https://github.com/ahmedosman/STAR")
    print("  cd STAR && pip install -e .")
    print(f"Import error: {e}")
    raise SystemExit("Cannot proceed without STAR model")

import plotly.express as px


class STARBoneExtractor:
    """Extract bone structure and joint positions from STAR model"""
    
    # STAR kinematic tree - 24 joints based on SMPL/STAR structure
    JOINT_NAMES = [
        'pelvis',           # 0
        'left_hip',         # 1
        'right_hip',        # 2
        'spine1',           # 3
        'left_knee',        # 4
        'right_knee',       # 5
        'spine2',           # 6
        'left_ankle',       # 7
        'right_ankle',      # 8
        'spine3',           # 9
        'left_foot',        # 10
        'right_foot',       # 11
        'neck',             # 12
        'left_collar',      # 13
        'right_collar',     # 14
        'head',             # 15
        'left_shoulder',    # 16
        'right_shoulder',   # 17
        'left_elbow',       # 18
        'right_elbow',      # 19
        'left_wrist',       # 20
        'right_wrist',      # 21
        'left_hand',        # 22
        'right_hand'        # 23
    ]
    
    # Bone connections - which joints form bones
    BONE_CONNECTIONS = [
        # Spine
        (0, 3),   # pelvis -> spine1
        (3, 6),   # spine1 -> spine2  
        (6, 9),   # spine2 -> spine3
        (9, 12),  # spine3 -> neck
        (12, 15), # neck -> head
        
        # Left leg
        (0, 1),   # pelvis -> left_hip
        (1, 4),   # left_hip -> left_knee
        (4, 7),   # left_knee -> left_ankle
        (7, 10),  # left_ankle -> left_foot
        
        # Right leg
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
    
    # Fixed bone diameters for thick-boned human collision
    FIXED_BONE_DIAMETERS = {
        # Spine (thick core)
        'pelvis_to_spine1': 0.25,
        'spine1_to_spine2': 0.22,
        'spine2_to_spine3': 0.20,
        'spine3_to_neck': 0.18,
        'neck_to_head': 0.15,
        
        # Legs (thick for stability)
        'pelvis_to_left_hip': 0.18,
        'left_hip_to_left_knee': 0.16,
        'left_knee_to_left_ankle': 0.12,
        'left_ankle_to_left_foot': 0.10,
        
        'pelvis_to_right_hip': 0.18,
        'right_hip_to_right_knee': 0.16,
        'right_knee_to_right_ankle': 0.12,
        'right_ankle_to_right_foot': 0.10,
        
        # Arms (medium thickness)
        'spine3_to_left_collar': 0.14,
        'left_collar_to_left_shoulder': 0.12,
        'left_shoulder_to_left_elbow': 0.10,
        'left_elbow_to_left_wrist': 0.08,
        'left_wrist_to_left_hand': 0.06,
        
        'spine3_to_right_collar': 0.14,
        'right_collar_to_right_shoulder': 0.12,
        'right_shoulder_to_right_elbow': 0.10,
        'right_elbow_to_right_wrist': 0.08,
        'right_wrist_to_right_hand': 0.06,
    }
    
    @classmethod
    def get_bone_names(cls) -> List[str]:
        """Get descriptive names for each bone"""
        bone_names = []
        for start_idx, end_idx in cls.BONE_CONNECTIONS:
            start_name = cls.JOINT_NAMES[start_idx]
            end_name = cls.JOINT_NAMES[end_idx]
            bone_names.append(f"{start_name}_to_{end_name}")
        return bone_names
    
    @classmethod
    def get_fixed_bone_radii(cls) -> np.ndarray:
        """Get fixed bone radii for thick-boned human"""
        bone_names = cls.get_bone_names()
        radii = []
        
        for bone_name in bone_names:
            radius = cls.FIXED_BONE_DIAMETERS.get(bone_name, 0.08)  # Default 8cm
            radii.append(radius)
        
        return np.array(radii, dtype=np.float64)
    
    def __init__(self, gender: str = 'neutral'):
        """Initialize STAR bone extractor"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # STAR model is required - no fallbacks
        self.model = STAR(gender=gender)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"STAR model loaded successfully on {self.device}")
        print(f"Gender: {gender}")
        print(f"Joints: {len(self.JOINT_NAMES)}")
        print(f"Bones: {len(self.BONE_CONNECTIONS)}")
    
    def get_t_pose_parameters(self) -> torch.Tensor:
        """Get T-pose parameters (all zeros for STAR)"""
        return torch.zeros(72, device=self.device, dtype=torch.float32)
    
    def extract_joint_positions(self, pose_params: torch.Tensor, 
                               shape_params: Optional[torch.Tensor] = None) -> np.ndarray:
        """
        Extract 3D joint positions from STAR model
        
        Args:
            pose_params: STAR pose parameters [72] or [batch, 72]
            shape_params: STAR shape parameters [10] or [batch, 10] (optional)
            
        Returns:
            Joint positions [24, 3] in world coordinates
        """
        # Ensure batch dimension
        if pose_params.dim() == 1:
            pose_params = pose_params.unsqueeze(0)
        
        batch_size = pose_params.shape[0]
        
        # Default shape (neutral body) and translation
        if shape_params is None:
            shape_params = torch.zeros(batch_size, 10, device=self.device)
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        # STAR forward pass
        with torch.no_grad():
            output = self.model(pose_params, shape_params, trans)
            
            # STAR returns vertices and joints
            if isinstance(output, tuple):
                vertices, joints = output
            else:
                # If only vertices returned, we need to compute joints
                vertices = output
                # Use STAR's joint regressor to get joint positions
                joints = torch.matmul(self.model.J_regressor, vertices)
        
        # Return first batch item as numpy array
        joint_positions = joints[0].cpu().numpy()  # [24, 3]
        
        if joint_positions.shape[0] != 24:
            raise ValueError(f"Expected 24 joints, got {joint_positions.shape[0]}")
        
        return joint_positions
    
    def get_mesh_data(self, pose_params: torch.Tensor,
                     shape_params: Optional[torch.Tensor] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mesh vertices and faces from STAR model
        
        Args:
            pose_params: STAR pose parameters [72] or [batch, 72] 
            shape_params: STAR shape parameters [10] or [batch, 10] (optional)
            
        Returns:
            Tuple of (vertices [6890, 3], faces [13776, 3])
        """
        # Ensure batch dimension
        if pose_params.dim() == 1:
            pose_params = pose_params.unsqueeze(0)
        
        batch_size = pose_params.shape[0]
        
        # Default shape and translation
        if shape_params is None:
            shape_params = torch.zeros(batch_size, 10, device=self.device)
        trans = torch.zeros(batch_size, 3, device=self.device)
        
        # STAR forward pass
        with torch.no_grad():
            output = self.model(pose_params, shape_params, trans)
            if isinstance(output, tuple):
                vertices, _ = output
            else:
                vertices = output
        
        # Get vertices and faces
        vertices_np = vertices[0].cpu().numpy()  # [6890, 3]
        faces_np = self.model.faces.cpu().numpy()  # [13776, 3]
        
        return vertices_np, faces_np
    
    def get_skinning_weights(self) -> np.ndarray:
        """
        Get STAR skinning weights matrix
        
        Returns:
            Skinning weights [6890, 24] - how much each vertex is influenced by each joint
        """
        weights = self.model.weights.cpu().numpy()  # [6890, 24]
        
        if weights.shape != (6890, 24):
            raise ValueError(f"Expected skinning weights shape (6890, 24), got {weights.shape}")
        
        return weights


class BoneCollisionManager:
    """High-level manager for bone-based collision detection with fixed diameters"""
    
    def __init__(self, gender: str = 'neutral', device: str = 'auto'):
        """
        Initialize bone collision manager
        
        Args:
            gender: STAR model gender ('neutral', 'male', 'female')
            device: Device for STAR model ('auto', 'cpu', 'cuda')
        """
        # Initialize STAR bone extractor (will fail if STAR not available)
        self.star_extractor = STARBoneExtractor(gender=gender)
        
        # Initialize C++ collision system
        self.collision_system = bone_collision.BoneCollisionSystem()
        
        # State tracking
        self.initialized = False
        self.current_pose = None
        self.using_fixed_diameters = True
        
        # Performance tracking
        self.timing_stats = {
            'star_forward': [],
            'joint_extraction': [],
            'collision_check': [],
            'total_update': []
        }
        
        print("BoneCollisionSystem initialized")
        print("BoneCollisionManager initialized successfully")
        print(f"STAR gender: {gender}")
        print(f"C++ collision system: Ready")
        print(f"Using fixed bone diameters: {self.using_fixed_diameters}")
    
    def initialize_with_fixed_diameters(self) -> Dict:
        """
        Initialize collision system with fixed bone diameters (no auto-tuning)
        
        Returns:
            Initialization info and statistics
        """
        print("Initializing bone collision system with fixed diameters...")
        start_time = time.time()
        
        # Get T-pose parameters
        t_pose_params = self.star_extractor.get_t_pose_parameters()
        
        # Extract joint positions in T-pose
        joint_start = time.time()
        joint_positions = self.star_extractor.extract_joint_positions(t_pose_params)
        joint_time = time.time() - joint_start
        
        # Get mesh data for triangle mapping
        mesh_start = time.time()
        vertices, faces = self.star_extractor.get_mesh_data(t_pose_params)
        skinning_weights = self.star_extractor.get_skinning_weights()  # [6890, 24]
        mesh_time = time.time() - mesh_start
        
        # Set up bone capsules with fixed radii
        setup_start = time.time()
        bone_connections = np.array(STARBoneExtractor.BONE_CONNECTIONS, dtype=np.int32)
        bone_names = STARBoneExtractor.get_bone_names()
        fixed_radii = STARBoneExtractor.get_fixed_bone_radii()
        
        print(f"Using fixed radii: {fixed_radii}")
        
        self.collision_system.setup_bone_capsules(
            joint_positions, bone_connections, fixed_radii, bone_names
        )
        setup_time = time.time() - setup_start
        
        # Convert joint-based skinning weights to bone-based (no auto-tuning)
        mapping_start = time.time()
        bone_skinning_weights = self._create_bone_skinning_weights(skinning_weights, bone_connections)
        
        # Build bone-triangle mapping using bone weights
        self.collision_system.build_bone_triangle_mapping(vertices, faces, bone_skinning_weights)
        mapping_time = time.time() - mapping_start
        
        # Store current state
        self.current_pose = t_pose_params
        self.initialized = True
        
        total_time = time.time() - start_time
        
        # Gather info
        info = {
            'status': 'initialized',
            'method': 'fixed_diameters',
            'vertices_shape': vertices.shape,
            'faces_shape': faces.shape,
            'joint_positions_shape': joint_positions.shape,
            'skinning_weights_shape': skinning_weights.shape,
            'bone_skinning_weights_shape': bone_skinning_weights.shape,
            'fixed_radii': fixed_radii.tolist(),
            'num_bones': self.collision_system.get_num_bones(),
            'bone_names': self.collision_system.get_bone_names(),
            'timing': {
                'joint_extraction': joint_time * 1000,
                'mesh_data': mesh_time * 1000,
                'cpp_setup': setup_time * 1000,
                'triangle_mapping': mapping_time * 1000,
                'total_init': total_time * 1000
            }
        }
        
        print(f"✓ Initialization complete!")
        print(f"  Vertices: {vertices.shape}")
        print(f"  Faces: {faces.shape}")
        print(f"  Joints: {joint_positions.shape[0]}")
        print(f"  Bones: {self.collision_system.get_num_bones()}")
        print(f"  Fixed radii range: {fixed_radii.min():.3f} - {fixed_radii.max():.3f}m")
        print(f"  Total time: {total_time*1000:.2f}ms")
        
        return info
    
    def _create_bone_skinning_weights(self, joint_weights: np.ndarray, 
                                    bone_connections: np.ndarray) -> np.ndarray:
        """
        Convert joint-based skinning weights to bone-based weights
        
        Args:
            joint_weights: [6890, 24] joint skinning weights
            bone_connections: [23, 2] bone connection indices
            
        Returns:
            bone_weights: [6890, 23] bone skinning weights
        """
        n_vertices = joint_weights.shape[0]
        n_bones = bone_connections.shape[0]
        
        bone_weights = np.zeros((n_vertices, n_bones), dtype=np.float64)
        
        for bone_id in range(n_bones):
            joint1_idx = bone_connections[bone_id, 0]
            joint2_idx = bone_connections[bone_id, 1]
            
            # Combine weights from both joints that form this bone
            # Use maximum weight (could also use sum or average)
            bone_weights[:, bone_id] = np.maximum(
                joint_weights[:, joint1_idx],
                joint_weights[:, joint2_idx]
            )
        
        print(f"Converted joint weights {joint_weights.shape} -> bone weights {bone_weights.shape}")
        return bone_weights
    
    def check_snake_segment_collision(self, segment_start: Union[List[float], np.ndarray],
                                    segment_end: Union[List[float], np.ndarray],
                                    segment_radius: float) -> Dict:
        """
        Check collision between snake segment and human body
        
        Args:
            segment_start: [x, y, z] start position of snake segment
            segment_end: [x, y, z] end position of snake segment
            segment_radius: Radius of snake segment
            
        Returns:
            Collision result with contact info and surface normal
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize_with_fixed_diameters() first.")
        
        # Convert to lists if numpy arrays
        if isinstance(segment_start, np.ndarray):
            segment_start = segment_start.tolist()
        if isinstance(segment_end, np.ndarray):
            segment_end = segment_end.tolist()
        
        # Check collision using C++ system
        result = self.collision_system.check_capsule_collision(
            segment_start[0], segment_start[1], segment_start[2],
            segment_end[0], segment_end[1], segment_end[2],
            segment_radius
        )
        
        # Convert C++ result to Python dict
        collision_info = {
            'collision': result.collision,
            'contact_point': [result.contact_point.x, result.contact_point.y, result.contact_point.z] if result.collision else None,
            'surface_normal': [result.surface_normal.x, result.surface_normal.y, result.surface_normal.z] if result.collision else None,
            'penetration_distance': result.penetration_distance if result.collision else 0.0,
            'triangle_id': result.triangle_id if result.collision else -1
        }
        
        return collision_info
    
    def check_robot_collision(self, robot_segments: List[Dict]) -> List[Dict]:
        """
        Check collision for multiple robot segments
        
        Args:
            robot_segments: List of dicts with 'start', 'end', 'radius' keys
            
        Returns:
            List of collision results for each segment
        """
        results = []
        
        for i, segment in enumerate(robot_segments):
            collision_info = self.check_snake_segment_collision(
                segment['start'], segment['end'], segment['radius']
            )
            
            results.append({
                'segment_id': i,
                'segment_data': segment,
                **collision_info
            })
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics from C++ system"""
        cpp_stats = self.collision_system.get_performance_stats()
        
        return {
            'avg_bone_filter_ms': cpp_stats[0] if len(cpp_stats) > 0 else 0.0,
            'avg_collision_check_ms': cpp_stats[1] if len(cpp_stats) > 1 else 0.0,
            'total_avg_ms': sum(cpp_stats) if cpp_stats else 0.0,
            'estimated_fps': 1000.0 / sum(cpp_stats) if cpp_stats and sum(cpp_stats) > 0 else 0.0
        }
    
    def visualize_thick_bone_capsules(self, pose_params: Optional[torch.Tensor] = None) -> go.Figure:
        """
        Create Plotly visualization of thick bone capsules as proper cylinders
        
        Args:
            pose_params: Pose parameters to visualize (uses current or T-pose if None)
            
        Returns:
            Plotly figure with mesh and proper bone cylinders
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize_with_fixed_diameters() first.")
        
        # Use provided pose or current pose or T-pose
        if pose_params is None:
            if self.current_pose is not None:
                pose_params = self.current_pose
            else:
                pose_params = self.star_extractor.get_t_pose_parameters()
        
        # Get mesh data
        vertices, faces = self.star_extractor.get_mesh_data(pose_params)
        joint_positions = self.star_extractor.extract_joint_positions(pose_params)
        
        # Create figure
        fig = go.Figure()
        
        # Add human mesh (more transparent)
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            color='lightgray',
            opacity=0.3,
            name='Human Mesh',
            showlegend=False
        ))
        
        # Add thick bone cylinders
        num_bones = self.collision_system.get_num_bones()
        bone_names = self.collision_system.get_bone_names()
        
        colors = px.colors.qualitative.Set3  # Color palette for bones
        
        for bone_id in range(num_bones):
            capsule_info = self.collision_system.get_bone_capsule_info(bone_id)
            if not capsule_info:
                continue
            
            start_x, start_y, start_z = capsule_info[0], capsule_info[1], capsule_info[2]
            end_x, end_y, end_z = capsule_info[3], capsule_info[4], capsule_info[5]
            radius = capsule_info[6]
            
            color = colors[bone_id % len(colors)]
            bone_name = bone_names[bone_id] if bone_id < len(bone_names) else f"Bone_{bone_id}"
            
            # Create proper cylinder mesh
            cylinder_mesh = self._create_cylinder_mesh(
                start_point=[start_x, start_y, start_z],
                end_point=[end_x, end_y, end_z],
                radius=radius,
                resolution=16
            )
            
            # Add cylinder as mesh
            fig.add_trace(go.Mesh3d(
                x=cylinder_mesh['x'],
                y=cylinder_mesh['y'],
                z=cylinder_mesh['z'],
                i=cylinder_mesh['i'],
                j=cylinder_mesh['j'],
                k=cylinder_mesh['k'],
                color=color,
                opacity=0.7,
                name=f"{bone_name} (r={radius:.3f})",
                showlegend=True
            ))
        
        # Add joint positions
        fig.add_trace(go.Scatter3d(
            x=joint_positions[:, 0],
            y=joint_positions[:, 1],
            z=joint_positions[:, 2],
            mode='markers',
            marker=dict(size=4, color='red'),
            name='Joints',
            text=[f"Joint {i}: {name}" for i, name in enumerate(STARBoneExtractor.JOINT_NAMES)],
            hovertemplate='%{text}<br>Position: (%{x:.3f}, %{y:.3f}, %{z:.3f})',
            showlegend=True
        ))
        
        # Update layout
        fig.update_layout(
            title="STAR Thick-Boned Human Collision System",
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)',
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=1),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=1000,
            height=800,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left", 
                x=0.01
            )
        )
        
        return fig
    
    def _create_cylinder_mesh(self, start_point: List[float], end_point: List[float], 
                            radius: float, resolution: int = 16) -> Dict:
        """
        Create cylinder mesh between two points
        
        Args:
            start_point: [x, y, z] cylinder start
            end_point: [x, y, z] cylinder end
            radius: Cylinder radius
            resolution: Number of sides for cylinder
            
        Returns:
            Dict with x, y, z coordinates and i, j, k triangle indices
        """
        start = np.array(start_point)
        end = np.array(end_point)
        
        # Calculate cylinder axis and length
        axis = end - start
        length = np.linalg.norm(axis)
        if length < 1e-8:
            # Degenerate case: start == end, create small sphere
            return self._create_sphere_mesh(start_point, radius, resolution)
        
        axis = axis / length
        
        # Create perpendicular vectors
        if abs(axis[2]) < 0.9:
            perp1 = np.cross(axis, [0, 0, 1])
        else:
            perp1 = np.cross(axis, [1, 0, 0])
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(axis, perp1)
        
        # Generate cylinder vertices
        vertices = []
        
        # Create circles at start and end
        for t in [0.0, 1.0]:  # start and end
            center = start + t * axis * length
            for i in range(resolution):
                angle = 2 * np.pi * i / resolution
                point = center + radius * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                vertices.append(point)
        
        vertices = np.array(vertices)
        
        # Create triangle indices
        triangles = []
        
        # Side faces
        for i in range(resolution):
            next_i = (i + 1) % resolution
            
            # Two triangles per side quad
            triangles.extend([
                [i, i + resolution, next_i],
                [next_i, i + resolution, next_i + resolution]
            ])
        
        # Cap faces (optional - makes cylinders closed)
        center_start_idx = len(vertices)
        center_end_idx = len(vertices) + 1
        
        # Add center points
        vertices = np.vstack([vertices, [start], [end]])
        
        # Start cap
        for i in range(resolution):
            next_i = (i + 1) % resolution
            triangles.append([center_start_idx, next_i, i])
        
        # End cap
        for i in range(resolution):
            next_i = (i + 1) % resolution
            triangles.append([center_end_idx, i + resolution, next_i + resolution])
        
        triangles = np.array(triangles)
        
        return {
            'x': vertices[:, 0],
            'y': vertices[:, 1],
            'z': vertices[:, 2],
            'i': triangles[:, 0],
            'j': triangles[:, 1],
            'k': triangles[:, 2]
        }
    
    def _create_sphere_mesh(self, center: List[float], radius: float, resolution: int) -> Dict:
        """Create sphere mesh for degenerate cylinder case"""
        # Simple sphere mesh
        phi = np.linspace(0, np.pi, resolution)
        theta = np.linspace(0, 2*np.pi, resolution)
        
        vertices = []
        for p in phi:
            for t in theta:
                x = center[0] + radius * np.sin(p) * np.cos(t)
                y = center[1] + radius * np.sin(p) * np.sin(t)
                z = center[2] + radius * np.cos(p)
                vertices.append([x, y, z])
        
        vertices = np.array(vertices)
        
        # Simple triangulation (not optimal but works)
        triangles = []
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                idx = i * resolution + j
                triangles.extend([
                    [idx, idx + 1, idx + resolution],
                    [idx + 1, idx + resolution + 1, idx + resolution]
                ])
        
        triangles = np.array(triangles)
        
        return {
            'x': vertices[:, 0],
            'y': vertices[:, 1], 
            'z': vertices[:, 2],
            'i': triangles[:, 0],
            'j': triangles[:, 1],
            'k': triangles[:, 2]
        }
    
    def save_bone_config(self, filename: str) -> None:
        """Save current bone configuration to JSON file"""
        if not self.initialized:
            print("WARNING: System not initialized yet!")
        
        config = {
            'method': 'fixed_diameters',
            'num_bones': self.collision_system.get_num_bones(),
            'bone_names': self.collision_system.get_bone_names(),
            'fixed_radii': STARBoneExtractor.get_fixed_bone_radii().tolist(),
            'bone_capsules': []
        }
        
        for bone_id in range(config['num_bones']):
            capsule_info = self.collision_system.get_bone_capsule_info(bone_id)
            if capsule_info:
                config['bone_capsules'].append({
                    'bone_id': bone_id,
                    'start': capsule_info[0:3],
                    'end': capsule_info[3:6],
                    'radius': capsule_info[6]
                })
        
        with open(filename, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Fixed diameter bone configuration saved to {filename}")


# Example usage and testing functions
def test_thick_bone_collision_system():
    """Test the bone collision system with fixed thick diameters"""
    print("="*60)
    print("THICK-BONED COLLISION SYSTEM TEST")
    print("="*60)
    
    try:
        # Initialize system
        manager = BoneCollisionManager(gender='neutral')
        
        # Initialize with fixed diameters (no auto-tuning)
        init_info = manager.initialize_with_fixed_diameters()
        
        print(f"\n=== Initialization Results ===")
        print(f"Status: {init_info['status']}")
        print(f"Method: {init_info['method']}")
        print(f"Bones: {init_info['num_bones']}")
        print(f"Fixed radii range: {min(init_info['fixed_radii']):.3f} - {max(init_info['fixed_radii']):.3f}m")
        print(f"Init time: {init_info['timing']['total_init']:.2f}ms")
        
        # Test collision detection with thick bones
        print(f"\n=== Collision Test ===")
        
        # Test 1: Segment through torso (should collide with thick spine)
        collision_result = manager.check_snake_segment_collision(
            segment_start=[0.0, 0.0, 0.6],   # Through torso
            segment_end=[0.0, 0.1, 0.6],
            segment_radius=0.02  # 2cm snake segment
        )
        
        print(f"Torso collision: {collision_result['collision']}")
        if collision_result['collision']:
            print(f"  Contact point: {collision_result['contact_point']}")
            print(f"  Surface normal: {collision_result['surface_normal']}")
            print(f"  Penetration: {collision_result['penetration_distance']:.4f}m")
        
        # Test 2: Segment through arm (should collide with thick arm)
        collision_result2 = manager.check_snake_segment_collision(
            segment_start=[0.3, 0.0, 1.0],   # Through left arm area
            segment_end=[0.4, 0.0, 1.0],
            segment_radius=0.02
        )
        
        print(f"Arm collision: {collision_result2['collision']}")
        if collision_result2['collision']:
            print(f"  Contact point: {collision_result2['contact_point']}")
            print(f"  Surface normal: {collision_result2['surface_normal']}")
            print(f"  Penetration: {collision_result2['penetration_distance']:.4f}m")
        
        # Test 3: Segment far away (should not collide)
        collision_result3 = manager.check_snake_segment_collision(
            segment_start=[2.0, 2.0, 2.0],   # Far away
            segment_end=[2.1, 2.0, 2.0],
            segment_radius=0.02
        )
        
        print(f"Far away collision: {collision_result3['collision']}")
        
        # Test multiple robot segments
        robot_segments = [
            {'start': [0.0, 0.0, 0.8], 'end': [0.1, 0.0, 0.8], 'radius': 0.02},  # Torso
            {'start': [0.3, 0.0, 1.0], 'end': [0.4, 0.0, 1.0], 'radius': 0.02},  # Arm
            {'start': [0.0, 0.3, 0.5], 'end': [0.0, 0.4, 0.5], 'radius': 0.02},  # Leg
        ]
        
        robot_collisions = manager.check_robot_collision(robot_segments)
        print(f"\n=== Multi-Segment Robot Test ===")
        for collision in robot_collisions:
            print(f"Segment {collision['segment_id']}: collision={collision['collision']}")
        
        # Performance stats
        stats = manager.get_performance_stats()
        print(f"\n=== Performance ===")
        print(f"Bone filter: {stats['avg_bone_filter_ms']:.3f}ms")
        print(f"Collision check: {stats['avg_collision_check_ms']:.3f}ms")
        print(f"Total: {stats['total_avg_ms']:.3f}ms")
        print(f"Estimated FPS: {stats['estimated_fps']:.1f}")
        
        # Create thick bone visualization
        print(f"\n=== Creating Thick-Bone Visualization ===")
        fig = manager.visualize_thick_bone_capsules()
        fig.show()
        
        # Save configuration
        manager.save_bone_config("thick_bone_config.json")
        
        return manager
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise


def demonstrate_thick_vs_thin():
    """Demonstrate difference between thick bone collision vs mesh collision"""
    print("\n" + "="*60)
    print("THICK BONE vs MESH COLLISION DEMONSTRATION")
    print("="*60)
    
    manager = BoneCollisionManager(gender='neutral')
    manager.initialize_with_fixed_diameters()
    
    # Test points around the body
    test_points = [
        # Around torso (should hit thick spine)
        {'name': 'Torso Center', 'pos': [0.0, 0.0, 0.8]},
        {'name': 'Torso Side', 'pos': [0.15, 0.0, 0.8]},
        {'name': 'Torso Wide', 'pos': [0.25, 0.0, 0.8]},
        
        # Around arm (should hit thick arm bones)
        {'name': 'Arm Close', 'pos': [0.3, 0.0, 1.0]},
        {'name': 'Arm Medium', 'pos': [0.4, 0.0, 1.0]},
        {'name': 'Arm Far', 'pos': [0.5, 0.0, 1.0]},
        
        # Around leg
        {'name': 'Leg Close', 'pos': [0.1, 0.0, 0.4]},
        {'name': 'Leg Medium', 'pos': [0.2, 0.0, 0.4]},
        {'name': 'Leg Far', 'pos': [0.3, 0.0, 0.4]},
    ]
    
    print(f"Testing collision at various distances from body:")
    print(f"(Shows how thick bones expand collision detection beyond mesh surface)")
    print()
    
    for test in test_points:
        collision = manager.check_snake_segment_collision(
            segment_start=test['pos'],
            segment_end=[test['pos'][0] + 0.01, test['pos'][1], test['pos'][2]],
            segment_radius=0.01
        )
        
        status = "🔴 COLLISION" if collision['collision'] else "🟢 SAFE"
        print(f"{test['name']:15} {test['pos']} → {status}")
    
    print(f"\nThis demonstrates the 'thick-boned human' concept:")
    print(f"- Points that would miss the thin mesh now hit thick bone cylinders")
    print(f"- Provides safety margin for robot collision avoidance")
    print(f"- Much faster than detailed mesh collision detection")


if __name__ == "__main__":
    # Run thick bone collision test
    manager = test_thick_bone_collision_system()
    
    # Demonstrate thick vs thin collision
    demonstrate_thick_vs_thin()
    
    print(f"\n=== Final Summary ===")
    stats = manager.get_performance_stats()
    if stats['total_avg_ms'] > 0:
        print(f"Thick-bone collision FPS: {stats['estimated_fps']:.1f}")
        print(f"Target 120Hz: {'✅ ACHIEVED' if stats['estimated_fps'] >= 120 else '❌ Not met'}")
    else:
        print(f"No performance data (no collisions detected yet)")
    
    print(f"\nThick-boned collision system ready! 🚀")
    print(f"Fixed diameters: 6cm (hands) to 25cm (spine)")
    print(f"Proper cylinder visualization with mesh overlay")