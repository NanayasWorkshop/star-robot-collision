"""
Layer 2: Capsule Layer  
Generates 24 anatomically accurate capsules
"""

import numpy as np
from ..core.body_definitions import BodyDefinitions


class CapsuleLayer:
    """Layer 2: Medium-detail capsule representation (24 capsules)"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        self.capsules = []
        
    def generate_from_joints(self, joint_positions):
        """
        Generate capsules from joint positions
        
        Args:
            joint_positions: Array of joint positions (24, 3)
            
        Returns:
            list: [(start_pos, end_pos, radius, name), ...] for each capsule
        """
        self.capsules = []
        
        for start_idx, end_idx, bone_name in self.body_defs.DETAILED_BONES:
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            
            # Get radius based on bone type
            bone_def = self.body_defs.BONE_DEFINITIONS.get(bone_name)
            if bone_def:
                bone_type = bone_def['type']
                radius = self.body_defs.DEFAULT_CAPSULE_RADII.get(bone_type, 0.05)
            else:
                radius = 0.05  # Default radius
            
            self.capsules.append((start_pos, end_pos, radius, bone_name))
        
        return self.capsules
    
    def update_from_joints(self, joint_positions):
        """Update capsule positions from new joint positions"""
        return self.generate_from_joints(joint_positions)
    
    def get_capsules(self):
        """Get current capsule list"""
        return self.capsules
    
    def get_stats(self):
        """Get statistics about the capsules"""
        if not self.capsules:
            return {"total": 0}
        
        # Group by bone type
        type_groups = {}
        for start_pos, end_pos, radius, name in self.capsules:
            bone_def = self.body_defs.BONE_DEFINITIONS.get(name, {'type': 'unknown'})
            bone_type = bone_def['type']
            
            if bone_type not in type_groups:
                type_groups[bone_type] = []
            type_groups[bone_type].append({
                'radius': radius,
                'length': np.linalg.norm(end_pos - start_pos)
            })
        
        stats = {
            "total": len(self.capsules),
            "by_type": {}
        }
        
        for bone_type, capsules in type_groups.items():
            radii = [c['radius'] for c in capsules]
            lengths = [c['length'] for c in capsules]
            
            stats["by_type"][bone_type] = {
                "count": len(capsules),
                "avg_radius": np.mean(radii),
                "avg_length": np.mean(lengths),
                "radius_range": (np.min(radii), np.max(radii)),
                "length_range": (np.min(lengths), np.max(lengths))
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