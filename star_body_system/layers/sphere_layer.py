"""
Layer 1: Sphere Layer
Generates 70-80 overlapping spheres with guaranteed coverage
"""

import numpy as np
from ..core.body_definitions import BodyDefinitions


class SphereLayer:
    """Layer 1: High-detail sphere representation"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        self.spheres = []
        
    def generate_from_joints(self, joint_positions, mesh_vertices=None):
        """
        Generate spheres from joint positions
        
        Args:
            joint_positions: Array of joint positions (24, 3)
            mesh_vertices: Optional mesh vertices for fine-tuning
            
        Returns:
            list: [(center, radius, name), ...] for each sphere
        """
        self.spheres = []
        
        for start_idx, end_idx, bone_name in self.body_defs.DETAILED_BONES:
            if bone_name not in self.body_defs.BONE_DEFINITIONS:
                continue
                
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            bone_def = self.body_defs.BONE_DEFINITIONS[bone_name]
            
            # Generate spheres along this bone
            bone_spheres = self._generate_bone_spheres(start_pos, end_pos, bone_def, bone_name)
            
            # Fine-tune with mesh if available
            if mesh_vertices is not None:
                bone_spheres = self._fine_tune_spheres(bone_spheres, mesh_vertices, bone_def['type'])
            
            self.spheres.extend(bone_spheres)
        
        return self.spheres
    
    def _generate_bone_spheres(self, start_pos, end_pos, bone_def, bone_name):
        """Generate spheres along a single bone with guaranteed overlap"""
        start_radius = bone_def['start_radius']
        end_radius = bone_def['end_radius']
        min_overlap = bone_def['min_overlap']
        
        bone_length = np.linalg.norm(end_pos - start_pos)
        
        # Start with first sphere
        positions = [start_pos]
        radii = [start_radius]
        
        current_pos = start_pos
        current_radius = start_radius
        
        while True:
            # Calculate distance for next sphere with guaranteed overlap
            next_distance = current_radius * (2 - min_overlap)
            
            # Move along bone direction
            direction = (end_pos - start_pos) / bone_length if bone_length > 0 else np.array([0, 0, 0])
            next_pos = current_pos + direction * next_distance
            
            # Check if we've reached the end
            distance_to_end = np.linalg.norm(end_pos - next_pos)
            progress = np.linalg.norm(next_pos - start_pos) / bone_length if bone_length > 0 else 1.0
            
            if progress >= 1.0 or distance_to_end < bone_length * 0.1:
                # Place final sphere at end
                positions.append(end_pos)
                radii.append(end_radius)
                break
            else:
                # Calculate tapered radius
                next_radius = start_radius + progress * (end_radius - start_radius)
                positions.append(next_pos)
                radii.append(next_radius)
                
                current_pos = next_pos
                current_radius = next_radius
        
        # Create sphere list with names
        bone_spheres = []
        for i, (pos, radius) in enumerate(zip(positions, radii)):
            sphere_name = f"{bone_name}_{i}"
            bone_spheres.append((pos, radius, sphere_name))
        
        return bone_spheres
    
    def _fine_tune_spheres(self, bone_spheres, mesh_vertices, bone_type):
        """Fine-tune sphere sizes based on mesh geometry"""
        fine_tuned = []
        
        for center, initial_radius, name in bone_spheres:
            # Find nearby mesh vertices
            distances = np.linalg.norm(mesh_vertices - center, axis=1)
            search_radius = initial_radius * 2.5
            nearby_mask = distances <= search_radius
            
            if np.any(nearby_mask):
                nearby_distances = distances[nearby_mask]
                
                # Determine coverage percentile based on bone type
                percentile_map = {
                    'torso': 0.9, 'head': 0.9, 'leg_upper': 0.85,
                    'leg_lower': 0.8, 'foot': 0.9, 'shoulder': 0.85,
                    'arm_upper': 0.8, 'arm_lower': 0.8, 'hand': 0.9
                }
                percentile = percentile_map.get(bone_type, 0.85)
                
                # Calculate fine-tuned radius
                target_distance = np.percentile(nearby_distances, percentile * 100)
                min_radius = initial_radius * 0.7
                max_radius = initial_radius * 2.5
                fine_tuned_radius = np.clip(target_distance, min_radius, max_radius)
            else:
                fine_tuned_radius = initial_radius
            
            fine_tuned.append((center, fine_tuned_radius, name))
        
        return fine_tuned
    
    def update_from_joints(self, joint_positions):
        """Update existing spheres from new joint positions (fast update)"""
        if not self.spheres:
            return self.generate_from_joints(joint_positions)
        
        # Quick update by repositioning existing spheres
        updated_spheres = []
        sphere_idx = 0
        
        for start_idx, end_idx, bone_name in self.body_defs.DETAILED_BONES:
            if bone_name not in self.body_defs.BONE_DEFINITIONS:
                continue
            
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            
            # Count how many spheres this bone had
            bone_sphere_count = sum(1 for _, _, name in self.spheres if name.startswith(bone_name))
            
            # Update positions proportionally
            for i in range(bone_sphere_count):
                if sphere_idx < len(self.spheres):
                    old_center, radius, name = self.spheres[sphere_idx]
                    
                    # Calculate new position
                    progress = i / max(1, bone_sphere_count - 1)
                    new_center = start_pos + progress * (end_pos - start_pos)
                    
                    updated_spheres.append((new_center, radius, name))
                    sphere_idx += 1
        
        self.spheres = updated_spheres
        return self.spheres
    
    def get_spheres(self):
        """Get current sphere list"""
        return self.spheres
    
    def get_stats(self):
        """Get statistics about the spheres"""
        if not self.spheres:
            return {"total": 0}
        
        # Group by bone type
        type_groups = {}
        for center, radius, name in self.spheres:
            bone_name = '_'.join(name.split('_')[:-1])
            bone_def = self.body_defs.BONE_DEFINITIONS.get(bone_name, {'type': 'unknown'})
            bone_type = bone_def['type']
            
            if bone_type not in type_groups:
                type_groups[bone_type] = []
            type_groups[bone_type].append(radius)
        
        stats = {
            "total": len(self.spheres),
            "by_type": {}
        }
        
        for bone_type, radii in type_groups.items():
            stats["by_type"][bone_type] = {
                "count": len(radii),
                "avg_radius": np.mean(radii),
                "radius_range": (np.min(radii), np.max(radii))
            }
        
        return stats