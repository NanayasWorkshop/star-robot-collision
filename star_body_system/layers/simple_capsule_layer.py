"""
Layer 3: Simple Capsule Layer
Generates 9 simplified capsules for fast collision detection
"""

import numpy as np
from ..core.body_definitions import BodyDefinitions


class SimpleCapsuleLayer:
    """Layer 3: Low-detail simplified representation (9 capsules)"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        self.capsules = []
        
    def generate_from_joints(self, joint_positions):
        """
        Generate simplified capsules from joint positions
        
        Args:
            joint_positions: Array of joint positions (24, 3)
            
        Returns:
            list: [(start_pos, end_pos, radius, name), ...] for each capsule
        """
        self.capsules = []
        
        for start_idx, end_idx, bone_name in self.body_defs.SIMPLE_BONES:
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            
            # Get radius from predefined simple bone radii
            radius = self.body_defs.DEFAULT_CAPSULE_RADII.get(bone_name, 0.08)
            
            self.capsules.append((start_pos, end_pos, radius, bone_name))
        
        return self.capsules
    
    def update_from_joints(self, joint_positions):
        """Update capsule positions from new joint positions"""
        return self.generate_from_joints(joint_positions)
    
    def get_capsules(self):
        """Get current capsule list"""
        return self.capsules
    
    def get_stats(self):
        """Get statistics about the simplified capsules"""
        if not self.capsules:
            return {"total": 0}
        
        # Group by body region
        region_groups = {
            'legs': [],
            'core': [],
            'arms': []
        }
        
        for start_pos, end_pos, radius, name in self.capsules:
            length = np.linalg.norm(end_pos - start_pos)
            capsule_info = {'radius': radius, 'length': length, 'name': name}
            
            if 'knee' in name or 'foot' in name:
                region_groups['legs'].append(capsule_info)
            elif 'pelvis-head' in name:
                region_groups['core'].append(capsule_info)
            elif 'shoulder' in name or 'elbow' in name or 'hand' in name:
                region_groups['arms'].append(capsule_info)
        
        stats = {
            "total": len(self.capsules),
            "by_region": {}
        }
        
        for region_name, capsules in region_groups.items():
            if capsules:
                radii = [c['radius'] for c in capsules]
                lengths = [c['length'] for c in capsules]
                
                stats["by_region"][region_name] = {
                    "count": len(capsules),
                    "avg_radius": np.mean(radii),
                    "avg_length": np.mean(lengths),
                    "radius_range": (np.min(radii), np.max(radii)),
                    "length_range": (np.min(lengths), np.max(lengths)),
                    "capsules": [c['name'] for c in capsules]
                }
        
        return stats
    
    def contains_point(self, point):
        """
        Check which capsules contain a given point
        
        Args:
            point: 3D point to test
            
        Returns:
            list: Names of capsules that contain the point
        """
        containing_capsules = []
        
        for start_pos, end_pos, radius, name in self.capsules:
            if self._point_in_capsule(point, start_pos, end_pos, radius):
                containing_capsules.append(name)
        
        return containing_capsules
    
    def _point_in_capsule(self, point, start_pos, end_pos, radius):
        """Check if a point is inside a capsule"""
        # Vector from start to end
        line_vec = end_pos - start_pos
        line_length = np.linalg.norm(line_vec)
        
        if line_length == 0:
            # Degenerate case: capsule is just a sphere
            return np.linalg.norm(point - start_pos) <= radius
        
        # Normalized line direction
        line_dir = line_vec / line_length
        
        # Vector from start to point
        point_vec = point - start_pos
        
        # Project point onto line
        projection_length = np.dot(point_vec, line_dir)
        
        # Clamp projection to line segment
        projection_length = np.clip(projection_length, 0, line_length)
        
        # Find closest point on line segment
        closest_point = start_pos + projection_length * line_dir
        
        # Check if point is within radius of closest point
        distance = np.linalg.norm(point - closest_point)
        return distance <= radius
    
    def get_collision_priority_order(self):
        """
        Get capsules in order of collision testing priority
        
        Returns:
            list: Capsule names in order of testing priority
        """
        # Test core first (most likely to hit), then extremities
        priority_order = [
            'pelvis-head',              # Core body
            'left_knee-pelvis',         # Upper legs
            'right_knee-pelvis',
            'left_shoulder-left_elbow', # Upper arms
            'right_shoulder-right_elbow',
            'left_foot-left_knee',      # Lower legs
            'right_foot-right_knee',
            'left_elbow-left_hand',     # Forearms/hands
            'right_elbow-right_hand'
        ]
        
        return priority_order