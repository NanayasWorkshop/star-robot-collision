#!/usr/bin/env python3
"""
Hybrid Collision System: Python STAR + C++ FCL
Combines Python STAR model with high-performance C++ collision detection
"""

import numpy as np
import torch
import time
from typing import List, Tuple, Dict, Optional
import fast_collision  # Our C++ module

# Import our existing STAR components
import sys
sys.path.append('src')
from star_robot_phase1 import (
    RandomJointGenerator, 
    STARMeshProcessor, 
    STARVisualization
)

class HybridCollisionSystem:
    """High-performance collision system combining Python STAR with C++ FCL"""
    
    def __init__(self, gender: str = 'neutral', seed: int = 42):
        """
        Initialize hybrid collision system
        
        Args:
            gender: STAR model gender ('neutral', 'male', 'female')
            seed: Random seed for reproducibility
        """
        # Python components
        self.joint_generator = RandomJointGenerator(seed=seed)
        self.star_processor = STARMeshProcessor(gender=gender)
        self.visualizer = STARVisualization()
        
        # C++ collision components
        self.human_collider = fast_collision.FastBVHCollider()
        self.collision_manager = fast_collision.CollisionManager()
        
        # State tracking
        self.initialized = False
        self.current_vertices = None
        self.current_faces = None
        
        # Performance metrics
        self.timing_stats = {
            'star_forward': [],
            'bvh_update': [],
            'collision_check': [],
            'total_update': []
        }
        
        print("Hybrid Collision System Initialized")
        print(f"STAR Device: {self.star_processor.device}")
        print(f"C++ FCL Extension: Available")
    
    def initialize_from_pose(self, pose_params: torch.Tensor) -> Dict:
        """
        Initialize collision system with initial pose
        
        Args:
            pose_params: STAR pose parameters [72]
            
        Returns:
            Dict with initialization info and timing
        """
        print("Initializing collision system...")
        start_time = time.time()
        
        # Generate initial mesh from STAR
        star_start = time.time()
        vertices, faces = self.star_processor.forward_pass(pose_params)
        star_time = time.time() - star_start
        
        # Create C++ BVH model
        bvh_start = time.time()
        self.human_collider.create_from_mesh(vertices, faces)
        bvh_time = time.time() - bvh_start
        
        # Add to collision manager
        self.collision_manager.add_object("human", self.human_collider)
        
        # Store current state
        self.current_vertices = vertices
        self.current_faces = faces
        self.initialized = True
        
        total_time = time.time() - start_time
        
        info = {
            'vertices_shape': vertices.shape,
            'faces_shape': faces.shape,
            'bvh_info': self.human_collider.get_info(),
            'bounding_box': self.human_collider.get_bounding_box(),
            'timing': {
                'star_forward': star_time * 1000,
                'bvh_creation': bvh_time * 1000,
                'total_init': total_time * 1000
            }
        }
        
        print(f"✓ Initialized - Vertices: {vertices.shape}, Faces: {faces.shape}")
        print(f"  STAR time: {star_time*1000:.2f}ms")
        print(f"  BVH creation: {bvh_time*1000:.2f}ms")
        print(f"  Total: {total_time*1000:.2f}ms")
        
        return info
    
    def update_pose(self, pose_params: torch.Tensor) -> Dict:
        """
        Update human pose with fast BVH refitting
        
        Args:
            pose_params: New STAR pose parameters [72]
            
        Returns:
            Dict with timing information
        """
        if not self.initialized:
            raise RuntimeError("System not initialized. Call initialize_from_pose first.")
        
        total_start = time.time()
        
        # Generate new mesh vertices with STAR
        star_start = time.time()
        vertices, _ = self.star_processor.forward_pass(pose_params)
        star_time = time.time() - star_start
        
        # Fast BVH update with C++ refitting
        bvh_update_time = self.human_collider.update_vertices(vertices)
        
        total_time = time.time() - total_start
        
        # Update stored state
        self.current_vertices = vertices
        
        # Record timing statistics
        timing = {
            'star_forward': star_time * 1000,
            'bvh_update': bvh_update_time,  # Already in ms
            'total_update': total_time * 1000
        }
        
        # Update rolling statistics
        self.timing_stats['star_forward'].append(timing['star_forward'])
        self.timing_stats['bvh_update'].append(timing['bvh_update'])
        self.timing_stats['total_update'].append(timing['total_update'])
        
        # Keep only last 100 measurements
        for key in self.timing_stats:
            if len(self.timing_stats[key]) > 100:
                self.timing_stats[key] = self.timing_stats[key][-100:]
        
        return timing
    
    def check_sphere_collision(self, x: float, y: float, z: float, radius: float) -> bool:
        """
        Check collision with sphere
        
        Args:
            x, y, z: Sphere center coordinates
            radius: Sphere radius
            
        Returns:
            True if collision detected
        """
        if not self.initialized:
            return False
        
        start_time = time.time()
        collision = self.human_collider.check_collision_with_sphere(x, y, z, radius)
        check_time = (time.time() - start_time) * 1000
        
        self.timing_stats['collision_check'].append(check_time)
        if len(self.timing_stats['collision_check']) > 100:
            self.timing_stats['collision_check'] = self.timing_stats['collision_check'][-100:]
        
        return collision
    
    def check_capsule_collision(self, start_point: List[float], end_point: List[float], 
                               radius: float) -> bool:
        """
        Check collision with capsule (robot link)
        
        Args:
            start_point: [x1, y1, z1] capsule start
            end_point: [x2, y2, z2] capsule end  
            radius: Capsule radius
            
        Returns:
            True if collision detected
        """
        if not self.initialized:
            return False
        
        start_time = time.time()
        collision = self.human_collider.check_collision_with_capsule(
            start_point[0], start_point[1], start_point[2],
            end_point[0], end_point[1], end_point[2],
            radius
        )
        check_time = (time.time() - start_time) * 1000
        
        self.timing_stats['collision_check'].append(check_time)
        if len(self.timing_stats['collision_check']) > 100:
            self.timing_stats['collision_check'] = self.timing_stats['collision_check'][-100:]
        
        return collision
    
    def check_robot_collision(self, robot_links: List[Dict]) -> List[Dict]:
        """
        Check collision with multiple robot links
        
        Args:
            robot_links: List of dicts with 'start', 'end', 'radius' keys
            
        Returns:
            List of collision results with link info
        """
        if not self.initialized:
            return []
        
        collisions = []
        for i, link in enumerate(robot_links):
            collision = self.check_capsule_collision(
                link['start'], link['end'], link['radius']
            )
            if collision:
                collisions.append({
                    'link_index': i,
                    'link_data': link,
                    'collision': True
                })
        
        return collisions
    
    def compute_distance_to_point(self, x: float, y: float, z: float, radius: float = 0.01) -> float:
        """
        Compute minimum distance to point (using small sphere)
        
        Args:
            x, y, z: Point coordinates
            radius: Small radius for distance computation
            
        Returns:
            Minimum distance to human mesh
        """
        if not self.initialized:
            return -1.0
        
        return self.human_collider.compute_distance_to_sphere(x, y, z, radius)
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.timing_stats['star_forward']:
            return {"status": "No timing data available"}
        
        stats = {}
        for key, times in self.timing_stats.items():
            if times:
                stats[key] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
        
        # Calculate effective frame rates
        if 'total_update' in stats:
            mean_total = stats['total_update']['mean']
            stats['effective_fps'] = 1000.0 / mean_total if mean_total > 0 else 0
            stats['target_120hz'] = mean_total <= 8.33  # 8.33ms = 120Hz
        
        return stats
    
    def run_performance_test(self, n_updates: int = 100) -> Dict:
        """
        Run performance test with random poses
        
        Args:
            n_updates: Number of pose updates to test
            
        Returns:
            Performance test results
        """
        print(f"\n=== Performance Test ({n_updates} updates) ===")
        
        if not self.initialized:
            # Initialize with random pose
            pose = self.joint_generator.generate_random_pose()
            self.initialize_from_pose(pose)
        
        # Clear previous stats
        for key in self.timing_stats:
            self.timing_stats[key] = []
        
        # Run updates
        print("Running pose updates...")
        for i in range(n_updates):
            if i % 20 == 0:
                print(f"  Update {i+1}/{n_updates}")
            
            pose = self.joint_generator.generate_random_pose()
            self.update_pose(pose)
        
        # Get final statistics
        stats = self.get_performance_stats()
        
        print(f"\n=== Performance Results ===")
        if 'total_update' in stats:
            print(f"Average total update time: {stats['total_update']['mean']:.2f}ms")
            print(f"  - STAR forward: {stats['star_forward']['mean']:.2f}ms")
            print(f"  - BVH refit: {stats['bvh_update']['mean']:.2f}ms")
            print(f"Effective FPS: {stats['effective_fps']:.1f}")
            print(f"120Hz capable: {'✓' if stats['target_120hz'] else '✗'}")
            print(f"Range: [{stats['total_update']['min']:.2f}, {stats['total_update']['max']:.2f}]ms")
        
        return stats
    
    def visualize_current_pose(self, title: str = "Current Human Pose"):
        """Visualize current human pose with Plotly"""
        if not self.initialized or self.current_vertices is None:
            print("No pose to visualize. Initialize system first.")
            return
        
        fig = self.visualizer.create_mesh_plot(
            self.current_vertices, self.current_faces, title
        )
        fig.show()
    
    def get_bounding_box(self) -> List[float]:
        """Get current mesh bounding box [min_x, min_y, min_z, max_x, max_y, max_z]"""
        if not self.initialized:
            return [0, 0, 0, 0, 0, 0]
        return self.human_collider.get_bounding_box()


class RobotCollisionChecker:
    """Robot collision checking using capsule approximations"""
    
    def __init__(self, hybrid_system: HybridCollisionSystem):
        """
        Initialize robot collision checker
        
        Args:
            hybrid_system: Initialized HybridCollisionSystem
        """
        self.hybrid_system = hybrid_system
        
    def create_robot_arm_links(self, joint_positions: List[List[float]], 
                              link_radii: List[float]) -> List[Dict]:
        """
        Create robot arm links as capsules
        
        Args:
            joint_positions: List of [x, y, z] joint positions
            link_radii: Radius for each link
            
        Returns:
            List of link dictionaries
        """
        links = []
        for i in range(len(joint_positions) - 1):
            links.append({
                'start': joint_positions[i],
                'end': joint_positions[i + 1],
                'radius': link_radii[i] if i < len(link_radii) else 0.05,
                'name': f'link_{i}'
            })
        return links
    
    def check_robot_path_collision(self, waypoints: List[Dict], 
                                  interpolation_steps: int = 10) -> Dict:
        """
        Check collision along robot path
        
        Args:
            waypoints: List of robot configurations (joint positions + radii)
            interpolation_steps: Steps between waypoints
            
        Returns:
            Collision analysis results
        """
        collisions = []
        
        for i in range(len(waypoints) - 1):
            # Interpolate between waypoints
            start_config = waypoints[i]
            end_config = waypoints[i + 1]
            
            for step in range(interpolation_steps):
                t = step / (interpolation_steps - 1)
                
                # Interpolate joint positions
                interpolated_joints = []
                for j in range(len(start_config['joints'])):
                    start_joint = start_config['joints'][j]
                    end_joint = end_config['joints'][j]
                    interp_joint = [
                        start_joint[k] + t * (end_joint[k] - start_joint[k])
                        for k in range(3)
                    ]
                    interpolated_joints.append(interp_joint)
                
                # Create links and check collision
                links = self.create_robot_arm_links(
                    interpolated_joints, 
                    start_config.get('radii', [0.05] * (len(interpolated_joints) - 1))
                )
                
                waypoint_collisions = self.hybrid_system.check_robot_collision(links)
                
                if waypoint_collisions:
                    collisions.append({
                        'waypoint_segment': i,
                        'interpolation_step': step,
                        't': t,
                        'colliding_links': waypoint_collisions
                    })
        
        return {
            'total_collisions': len(collisions),
            'collision_details': collisions,
            'path_safe': len(collisions) == 0
        }


# Example usage and testing functions
def test_hybrid_system():
    """Test the hybrid collision system"""
    print("="*60)
    print("HYBRID COLLISION SYSTEM TEST")
    print("="*60)
    
    # Initialize system
    system = HybridCollisionSystem(seed=42)
    
    # Generate initial pose and initialize
    pose = system.joint_generator.generate_random_pose()
    init_info = system.initialize_from_pose(pose)
    
    # Test single pose update
    print(f"\n=== Single Update Test ===")
    new_pose = system.joint_generator.generate_random_pose()
    timing = system.update_pose(new_pose)
    print(f"Update timing: {timing}")
    
    # Test collision checks
    print(f"\n=== Collision Tests ===")
    
    # Test sphere collision at human center
    bbox = system.get_bounding_box()
    center_x = (bbox[0] + bbox[3]) / 2
    center_y = (bbox[1] + bbox[4]) / 2  
    center_z = (bbox[2] + bbox[5]) / 2
    
    collision = system.check_sphere_collision(center_x, center_y, center_z, 0.1)
    print(f"Sphere collision at center: {collision}")
    
    collision_far = system.check_sphere_collision(10, 10, 10, 0.1)
    print(f"Sphere collision far away: {collision_far}")
    
    # Test capsule collision
    capsule_collision = system.check_capsule_collision(
        [center_x - 0.5, center_y, center_z],
        [center_x + 0.5, center_y, center_z], 
        0.1
    )
    print(f"Capsule collision through center: {capsule_collision}")
    
    # Run performance test
    perf_stats = system.run_performance_test(n_updates=50)
    
    return system

def test_robot_collision():
    """Test robot collision checking"""
    print(f"\n=== Robot Collision Test ===")
    
    # Initialize hybrid system
    system = HybridCollisionSystem(seed=42)
    pose = system.joint_generator.generate_random_pose()
    system.initialize_from_pose(pose)
    
    # Create robot collision checker
    robot_checker = RobotCollisionChecker(system)
    
    # Define simple robot arm
    robot_joints = [
        [0, 0, 0],      # Base
        [0.3, 0, 0.2],  # Shoulder  
        [0.6, 0, 0.4],  # Elbow
        [0.8, 0, 0.3],  # Wrist
        [1.0, 0, 0.2]   # End effector
    ]
    
    link_radii = [0.05, 0.04, 0.03, 0.02]
    
    # Create robot links
    links = robot_checker.create_robot_arm_links(robot_joints, link_radii)
    
    # Check collisions
    collisions = system.check_robot_collision(links)
    
    print(f"Robot links: {len(links)}")
    print(f"Collisions detected: {len(collisions)}")
    for collision in collisions:
        print(f"  Link {collision['link_index']} collision")
    
    return robot_checker

if __name__ == "__main__":
    # Run tests
    system = test_hybrid_system()
    robot_checker = test_robot_collision()
    
    print(f"\n=== Final Performance Summary ===")
    stats = system.get_performance_stats()
    if 'effective_fps' in stats:
        print(f"Achieved FPS: {stats['effective_fps']:.1f}")
        print(f"120Hz Target: {'✓ ACHIEVED' if stats['target_120hz'] else '✗ Not met'}")
    
    print(f"\nHybrid system ready for real-time collision detection! 🚀")