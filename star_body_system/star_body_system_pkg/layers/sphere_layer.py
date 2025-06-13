"""
Layer 1: Sphere Layer
Generates 70-80 overlapping spheres with guaranteed coverage
Two-phase approach: normal generation + endpoint expansion for uncovered vertices
"""

import numpy as np
from ..core.body_definitions import BodyDefinitions


class SphereLayer:
    """Layer 1: High-detail sphere representation"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        self.spheres = []
        
        # Define endpoint bones that should have expandable final spheres
        self.endpoint_bones = {
            'neck-head',           # Head endpoint
            'left_wrist-left_hand',   # Left hand endpoint
            'right_wrist-right_hand', # Right hand endpoint
            'left_ankle-left_foot',   # Left foot endpoint
            'right_ankle-right_foot'  # Right foot endpoint
        }
        
    def generate_from_joints(self, joint_positions, mesh_vertices=None):
        """
        Generate spheres from joint positions with two-phase endpoint expansion
        
        Args:
            joint_positions: Array of joint positions (24, 3)
            mesh_vertices: Optional mesh vertices for fine-tuning
            
        Returns:
            list: [(center, radius, name), ...] for each sphere
        """
        self.spheres = []
        
        # Phase 1: Generate all spheres with normal limits
        for start_idx, end_idx, bone_name in self.body_defs.DETAILED_BONES:
            if bone_name not in self.body_defs.BONE_DEFINITIONS:
                continue
                
            start_pos = joint_positions[start_idx]
            end_pos = joint_positions[end_idx]
            bone_def = self.body_defs.BONE_DEFINITIONS[bone_name]
            
            # Generate spheres along this bone
            bone_spheres = self._generate_bone_spheres(start_pos, end_pos, bone_def, bone_name)
            
            # Fine-tune with mesh if available (normal limits for all spheres)
            if mesh_vertices is not None:
                bone_spheres = self._fine_tune_spheres_phase1(bone_spheres, mesh_vertices, bone_def['type'])
            
            self.spheres.extend(bone_spheres)
        
        # Phase 2: Expand nearest spheres for uncovered vertices
        if mesh_vertices is not None:
            self._expand_nearest_spheres_for_uncovered_vertices(mesh_vertices)
        
        # Phase 3: Add 3% safety buffer to all spheres
        self._add_safety_buffer(buffer_percent=0.03)
        
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
    
    def _fine_tune_spheres_phase1(self, bone_spheres, mesh_vertices, bone_type):
        """Phase 1: Fine-tune sphere sizes with tighter limits for better mesh fit"""
        fine_tuned = []
        
        for center, initial_radius, name in bone_spheres:
            # Find nearby mesh vertices
            distances = np.linalg.norm(mesh_vertices - center, axis=1)
            search_radius = initial_radius * 2.0  # Reduced from 2.5
            nearby_mask = distances <= search_radius
            
            if np.any(nearby_mask):
                nearby_distances = distances[nearby_mask]
                
                # Lower percentiles for tighter fit, especially torso/legs
                percentile_map = {
                    'torso': 0.8,      # Reduced from 0.9
                    'head': 0.85,      # Reduced from 0.9
                    'leg_upper': 0.75, # Reduced from 0.85
                    'leg_lower': 0.75, # Reduced from 0.8
                    'foot': 0.85,      # Reduced from 0.9
                    'shoulder': 0.8,   # Reduced from 0.85
                    'arm_upper': 0.8,  # Same
                    'arm_lower': 0.8,  # Same
                    'hand': 0.85       # Reduced from 0.9
                }
                percentile = percentile_map.get(bone_type, 0.8)
                
                # Calculate target radius with tighter limits
                target_distance = np.percentile(nearby_distances, percentile * 100)
                min_radius = initial_radius * 0.7
                
                # Tighter max limits for torso/legs
                if bone_type in ['torso', 'leg_upper', 'leg_lower']:
                    max_radius = initial_radius * 1.8  # Much tighter than 2.5x
                else:
                    max_radius = initial_radius * 2.0  # Slightly tighter for others
                
                fine_tuned_radius = np.clip(target_distance, min_radius, max_radius)
            else:
                fine_tuned_radius = initial_radius
            
            fine_tuned.append((center, fine_tuned_radius, name))
        
        return fine_tuned
    
    def _expand_nearest_spheres_for_uncovered_vertices(self, mesh_vertices):
        """
        Phase 2: Expand nearest spheres (any sphere) to cover uncovered vertices
        """
        print("Phase 2: Expanding nearest spheres for uncovered vertices...")
        
        # Find uncovered vertices using current spheres
        uncovered_vertices = self._find_uncovered_vertices(mesh_vertices)
        
        if len(uncovered_vertices) == 0:
            print("   No uncovered vertices found - no expansion needed")
            return
        
        print(f"   Found {len(uncovered_vertices)} uncovered vertices")
        
        # For each uncovered vertex, find closest sphere (any sphere) and expand it
        expansions_made = {}
        
        for vertex_idx in uncovered_vertices:
            vertex_pos = mesh_vertices[vertex_idx]
            
            # Find closest sphere (any sphere, not just endpoints)
            closest_sphere_idx = None
            closest_distance = float('inf')
            
            for sphere_idx, (center, radius, name) in enumerate(self.spheres):
                distance = np.linalg.norm(vertex_pos - center)
                
                if distance < closest_distance:
                    closest_distance = distance
                    closest_sphere_idx = sphere_idx
            
            if closest_sphere_idx is not None:
                # Calculate required radius to reach this vertex
                center, current_radius, name = self.spheres[closest_sphere_idx]
                required_radius = np.linalg.norm(vertex_pos - center)
                
                # Expand sphere if needed
                if required_radius > current_radius:
                    if name not in expansions_made:
                        expansions_made[name] = {'old_radius': current_radius, 'new_radius': required_radius, 'count': 1, 'sphere_idx': closest_sphere_idx}
                    else:
                        # If this sphere needs to expand for multiple vertices, take the maximum
                        expansions_made[name]['new_radius'] = max(expansions_made[name]['new_radius'], required_radius)
                        expansions_made[name]['count'] += 1
        
        # Apply all expansions
        for name, expansion in expansions_made.items():
            sphere_idx = expansion['sphere_idx']
            center, old_radius, sphere_name = self.spheres[sphere_idx]
            new_radius = expansion['new_radius']
            
            # Update the sphere
            self.spheres[sphere_idx] = (center, new_radius, sphere_name)
            
            print(f"   Expanded {name}: {expansion['old_radius']:.3f} -> {expansion['new_radius']:.3f} "
                  f"(for {expansion['count']} vertices)")
        
        print(f"   Phase 2 complete - expanded {len(expansions_made)} spheres")
    
    def _add_safety_buffer(self, buffer_percent=0.03):
        """
        Phase 3: Add safety buffer to all spheres for clearance from mesh
        
        Args:
            buffer_percent: Percentage to increase all sphere radii (0.03 = 3%)
        """
        print(f"Phase 3: Adding {buffer_percent*100:.0f}% safety buffer to all spheres...")
        
        buffered_spheres = []
        total_expansion = 0
        
        for center, radius, name in self.spheres:
            new_radius = radius * (1 + buffer_percent)
            total_expansion += (new_radius - radius)
            buffered_spheres.append((center, new_radius, name))
        
        self.spheres = buffered_spheres
        avg_expansion = total_expansion / len(self.spheres) if self.spheres else 0
        
        print(f"   Applied buffer to {len(self.spheres)} spheres")
        print(f"   Average radius increase: {avg_expansion:.4f}")
        print(f"   Phase 3 complete")
    
    def _find_uncovered_vertices(self, mesh_vertices):
        """Find vertices not covered by any sphere"""
        uncovered = []
        
        for vertex_idx, vertex_pos in enumerate(mesh_vertices):
            is_covered = False
            
            for center, radius, name in self.spheres:
                distance = np.linalg.norm(vertex_pos - center)
                if distance <= radius:
                    is_covered = True
                    break
            
            if not is_covered:
                uncovered.append(vertex_idx)
        
        return uncovered
    
    def _get_endpoint_sphere_indices(self):
        """Get indices of endpoint spheres (last sphere of each endpoint bone)"""
        endpoint_indices = []
        
        for bone_name in self.endpoint_bones:
            # Find the highest numbered sphere for this bone (the endpoint)
            bone_spheres = [(i, name) for i, (_, _, name) in enumerate(self.spheres) 
                           if name.startswith(bone_name)]
            
            if bone_spheres:
                # Sort by sphere number (last part of name) and take the highest
                bone_spheres.sort(key=lambda x: int(x[1].split('_')[-1]))
                endpoint_indices.append(bone_spheres[-1][0])
        
        return endpoint_indices
    
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
        expanded_spheres = {}
        
        for center, radius, name in self.spheres:
            bone_name = '_'.join(name.split('_')[:-1])
            bone_def = self.body_defs.BONE_DEFINITIONS.get(bone_name, {'type': 'unknown'})
            bone_type = bone_def['type']
            
            if bone_type not in type_groups:
                type_groups[bone_type] = []
            type_groups[bone_type].append(radius)
            
            # Track expanded spheres (those significantly larger than initial)
            if bone_def and 'start_radius' in bone_def:
                initial_radius = bone_def['start_radius']
                if radius > initial_radius * 1.5:  # 50% larger than initial
                    expanded_spheres[name] = {'initial': initial_radius, 'final': radius, 'expansion': radius/initial_radius}
        
        stats = {
            "total": len(self.spheres),
            "by_type": {},
            "expanded_spheres": expanded_spheres
        }
        
        for bone_type, radii in type_groups.items():
            stats["by_type"][bone_type] = {
                "count": len(radii),
                "avg_radius": np.mean(radii),
                "radius_range": (np.min(radii), np.max(radii))
            }
        
        return stats