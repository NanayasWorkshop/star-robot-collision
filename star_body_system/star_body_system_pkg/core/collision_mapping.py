"""
Phase 1: Collision Mapping System
Static mapping generation for hierarchical collision detection
"""

import numpy as np
from scipy.spatial import cKDTree
import h5py
import time
from typing import Dict, List, Tuple, Optional
from ..core.body_definitions import BodyDefinitions


class VertexSphereMapper:
    """Fast vertex-to-sphere assignment using spatial indexing"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        
    def assign_vertices_to_spheres(self, vertices: np.ndarray, spheres: List[Tuple]) -> Dict[int, List[str]]:
        """
        Assign vertices to spheres using simple distance check with spatial optimization
        
        Args:
            vertices: STAR mesh vertices (N, 3)
            spheres: List of (center, radius, name) tuples from SphereLayer
            
        Returns:
            Dict mapping vertex_idx -> list of sphere names
        """
        print(f"Assigning {len(vertices)} vertices to {len(spheres)} spheres...")
        start_time = time.time()
        
        # Build spatial index for fast O(log n) queries
        sphere_centers = np.array([center for center, _, _ in spheres])
        sphere_tree = cKDTree(sphere_centers)
        
        vertex_to_spheres = {}
        max_radius = max(radius for _, radius, _ in spheres)
        
        # Process vertices in batches for progress reporting
        batch_size = 1000
        total_assignments = 0
        
        for batch_start in range(0, len(vertices), batch_size):
            batch_end = min(batch_start + batch_size, len(vertices))
            batch_vertices = vertices[batch_start:batch_end]
            
            for local_idx, vertex_pos in enumerate(batch_vertices):
                vertex_idx = batch_start + local_idx
                assigned_spheres = set()  # Automatic deduplication
                
                # Query potential spheres efficiently using spatial index
                potential_indices = sphere_tree.query_ball_point(vertex_pos, max_radius)
                
                # Test each potential sphere with simple distance check
                for sphere_idx in potential_indices:
                    center, radius, name = spheres[sphere_idx]
                    distance = np.linalg.norm(vertex_pos - center)
                    
                    if distance <= radius:  # Simple inside check
                        assigned_spheres.add(name)
                
                vertex_to_spheres[vertex_idx] = list(assigned_spheres)
                total_assignments += len(assigned_spheres)
            
            # Progress reporting
            print(f"  Processed {batch_end}/{len(vertices)} vertices ({batch_end/len(vertices)*100:.1f}%)")
        
        elapsed = time.time() - start_time
        avg_assignments = total_assignments / len(vertices)
        print(f"Assignment completed in {elapsed:.2f} seconds")
        print(f"Average assignments per vertex: {avg_assignments:.2f}")
        
        return vertex_to_spheres


class HierarchyBuilder:
    """Build containment hierarchies between collision layers"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        
    def build_hierarchy(self, spheres: List[Tuple], capsule_layer, simple_layer) -> Dict:
        """
        Build complete hierarchy mappings using existing containment definitions
        
        Args:
            spheres: List of (center, radius, name) from SphereLayer
            capsule_layer: CapsuleLayer instance 
            simple_layer: SimpleCapsuleLayer instance
            
        Returns:
            Complete hierarchy mapping structure
        """
        print("Building containment hierarchies...")
        
        capsules = capsule_layer.get_capsules()
        simple_capsules = simple_layer.get_capsules()
        
        # Create name-to-ID mappings for fast lookup
        sphere_name_to_id = {name: idx for idx, (_, _, name) in enumerate(spheres)}
        capsule_name_to_id = {name: idx for idx, (_, _, _, name) in enumerate(capsules)}
        simple_name_to_id = {name: idx for idx, (_, _, _, name) in enumerate(simple_capsules)}
        
        print(f"  Mapping {len(spheres)} spheres, {len(capsules)} capsules, {len(simple_capsules)} simple capsules")
        
        # Build Layer 1 → Layer 2 mapping (sphere → capsule)
        sphere_to_capsule = {}
        for sphere_idx, (_, _, sphere_name) in enumerate(spheres):
            # Extract bone name from sphere name (remove "_X" suffix)
            bone_name = '_'.join(sphere_name.split('_')[:-1])
            
            if bone_name in capsule_name_to_id:
                sphere_to_capsule[sphere_idx] = capsule_name_to_id[bone_name]
            else:
                print(f"    WARNING: No capsule found for sphere bone '{bone_name}'")
        
        # Build Layer 2 → Layer 3 mapping (capsule → simple)
        capsule_to_simple = {}
        for capsule_idx, (_, _, _, capsule_name) in enumerate(capsules):
            # Find which simple capsule contains this capsule
            found_parent = False
            for simple_name, contained_capsules in self.body_defs.LAYER3_TO_LAYER2_CONTAINMENT.items():
                if capsule_name in contained_capsules:
                    if simple_name in simple_name_to_id:
                        capsule_to_simple[capsule_idx] = simple_name_to_id[simple_name]
                        found_parent = True
                        break
            
            if not found_parent:
                print(f"    WARNING: No simple capsule found for capsule '{capsule_name}'")
        
        hierarchy = {
            'sphere_name_to_id': sphere_name_to_id,
            'capsule_name_to_id': capsule_name_to_id,
            'simple_name_to_id': simple_name_to_id,
            'sphere_to_capsule': sphere_to_capsule,
            'capsule_to_simple': capsule_to_simple
        }
        
        print(f"  Built {len(sphere_to_capsule)}/{len(spheres)} sphere→capsule mappings")
        print(f"  Built {len(capsule_to_simple)}/{len(capsules)} capsule→simple mappings")
        
        return hierarchy


class CollisionDataStore:
    """Convert mappings to optimized arrays for C++ consumption"""
    
    def __init__(self):
        self.vertex_sphere_assignments = None
        self.assignment_lengths = None
        self.sphere_to_capsule = None
        self.capsule_to_simple = None
        self.metadata = {}
        
    def build_optimized_structures(self, vertex_mapping: Dict[int, List[str]], 
                                 hierarchy: Dict, num_vertices: int):
        """
        Convert all mappings to padded integer arrays for maximum C++ performance
        
        Args:
            vertex_mapping: vertex_idx -> list of sphere names
            hierarchy: hierarchy mappings from HierarchyBuilder
            num_vertices: total number of STAR vertices
        """
        print("Building optimized data structures...")
        
        sphere_name_to_id = hierarchy['sphere_name_to_id']
        
        # Convert vertex assignments from names to IDs
        vertex_assignments_by_id = []
        max_assignments = 0
        
        for vertex_idx in range(num_vertices):
            sphere_names = vertex_mapping.get(vertex_idx, [])
            # Convert names to IDs, skip invalid names
            sphere_ids = []
            for name in sphere_names:
                if name in sphere_name_to_id:
                    sphere_ids.append(sphere_name_to_id[name])
                else:
                    print(f"    WARNING: Unknown sphere name '{name}' for vertex {vertex_idx}")
            
            vertex_assignments_by_id.append(sphere_ids)
            max_assignments = max(max_assignments, len(sphere_ids))
        
        print(f"  Maximum assignments per vertex: {max_assignments}")
        
        # Create padded arrays for C++ convenience
        self.vertex_sphere_assignments = np.full((num_vertices, max_assignments), -1, dtype=np.int32)
        self.assignment_lengths = np.zeros(num_vertices, dtype=np.int32)
        
        for vertex_idx, sphere_ids in enumerate(vertex_assignments_by_id):
            if sphere_ids:
                self.vertex_sphere_assignments[vertex_idx, :len(sphere_ids)] = sphere_ids
                self.assignment_lengths[vertex_idx] = len(sphere_ids)
        
        # Convert hierarchy mappings to direct arrays
        num_spheres = len(sphere_name_to_id)
        num_capsules = len(hierarchy['capsule_name_to_id'])
        
        self.sphere_to_capsule = np.full(num_spheres, -1, dtype=np.int32)
        self.capsule_to_simple = np.full(num_capsules, -1, dtype=np.int32)
        
        # Fill hierarchy arrays
        for sphere_id, capsule_id in hierarchy['sphere_to_capsule'].items():
            self.sphere_to_capsule[sphere_id] = capsule_id
            
        for capsule_id, simple_id in hierarchy['capsule_to_simple'].items():
            self.capsule_to_simple[capsule_id] = simple_id
        
        # Store metadata for C++ and debugging
        self.metadata = {
            'num_vertices': num_vertices,
            'num_spheres': num_spheres,
            'num_capsules': num_capsules,
            'num_simple': len(hierarchy['simple_name_to_id']),
            'max_assignments_per_vertex': max_assignments,
            'sphere_to_capsule_mappings': len(hierarchy['sphere_to_capsule']),
            'capsule_to_simple_mappings': len(hierarchy['capsule_to_simple'])
        }
        
        memory_mb = self._estimate_memory_mb()
        print(f"  Optimized structures built - Memory usage: ~{memory_mb:.1f} MB")
        
    def _estimate_memory_mb(self) -> float:
        """Estimate total memory usage in MB"""
        if self.vertex_sphere_assignments is None:
            return 0.0
            
        vertex_array_mb = self.vertex_sphere_assignments.nbytes / (1024 * 1024)
        length_array_mb = self.assignment_lengths.nbytes / (1024 * 1024)
        hierarchy_mb = (self.sphere_to_capsule.nbytes + self.capsule_to_simple.nbytes) / (1024 * 1024)
        
        return vertex_array_mb + length_array_mb + hierarchy_mb
    
    def save_to_hdf5(self, filepath: str):
        """Save collision data in HDF5 format for fast C++ loading"""
        print(f"Saving collision data to {filepath}...")
        
        with h5py.File(filepath, 'w') as f:
            # Main data arrays with compression for storage efficiency
            f.create_dataset('vertex_sphere_assignments', data=self.vertex_sphere_assignments, 
                           compression='gzip', compression_opts=9)
            f.create_dataset('assignment_lengths', data=self.assignment_lengths)
            f.create_dataset('sphere_to_capsule', data=self.sphere_to_capsule)
            f.create_dataset('capsule_to_simple', data=self.capsule_to_simple)
            
            # Metadata as attributes
            for key, value in self.metadata.items():
                f.attrs[key] = value
            
            # Format versioning for future compatibility
            f.attrs['format_version'] = '1.0'
            f.attrs['creation_time'] = time.time()
        
        file_size_mb = self._get_file_size_mb(filepath)
        print(f"  Saved {filepath} ({file_size_mb:.1f} MB on disk)")
    
    def _get_file_size_mb(self, filepath: str) -> float:
        """Get file size in MB"""
        try:
            import os
            return os.path.getsize(filepath) / (1024 * 1024)
        except:
            return 0.0


class MappingValidator:
    """Validation and quality checking for collision mappings"""
    
    def __init__(self):
        self.body_defs = BodyDefinitions()
        
    def validate_coverage(self, vertex_mapping: Dict[int, List[str]], num_vertices: int) -> Dict:
        """
        Validate vertex-to-sphere coverage completeness
        
        Args:
            vertex_mapping: vertex assignments
            num_vertices: total vertex count
            
        Returns:
            Coverage statistics dictionary
        """
        uncovered_vertices = []
        assignment_counts = []
        total_assignments = 0
        
        for vertex_idx in range(num_vertices):
            assignments = vertex_mapping.get(vertex_idx, [])
            assignment_count = len(assignments)
            assignment_counts.append(assignment_count)
            total_assignments += assignment_count
            
            if assignment_count == 0:
                uncovered_vertices.append(vertex_idx)
        
        coverage_stats = {
            'total_vertices': num_vertices,
            'covered_vertices': num_vertices - len(uncovered_vertices),
            'uncovered_vertices': len(uncovered_vertices),
            'coverage_percent': (1 - len(uncovered_vertices)/num_vertices) * 100,
            'total_assignments': total_assignments,
            'avg_assignments_per_vertex': total_assignments / num_vertices if num_vertices > 0 else 0,
            'max_assignments_per_vertex': max(assignment_counts) if assignment_counts else 0,
            'min_assignments_per_vertex': min(assignment_counts) if assignment_counts else 0,
            'vertices_with_multiple_assignments': sum(1 for count in assignment_counts if count > 1),
            'uncovered_sample': uncovered_vertices[:10]  # First 10 for debugging
        }
        
        return coverage_stats
    
    def validate_hierarchy(self, hierarchy: Dict) -> Dict:
        """
        Validate hierarchy mapping completeness
        
        Args:
            hierarchy: hierarchy mappings
            
        Returns:
            Hierarchy validation statistics
        """
        total_spheres = len(hierarchy['sphere_name_to_id'])
        total_capsules = len(hierarchy['capsule_name_to_id'])
        total_simple = len(hierarchy['simple_name_to_id'])
        
        mapped_spheres = len(hierarchy['sphere_to_capsule'])
        mapped_capsules = len(hierarchy['capsule_to_simple'])
        
        hierarchy_stats = {
            'total_spheres': total_spheres,
            'total_capsules': total_capsules,
            'total_simple': total_simple,
            'mapped_spheres': mapped_spheres,
            'mapped_capsules': mapped_capsules,
            'sphere_mapping_percent': (mapped_spheres / total_spheres * 100) if total_spheres > 0 else 0,
            'capsule_mapping_percent': (mapped_capsules / total_capsules * 100) if total_capsules > 0 else 0,
            'unmapped_spheres': total_spheres - mapped_spheres,
            'unmapped_capsules': total_capsules - mapped_capsules
        }
        
        return hierarchy_stats
    
    def print_validation_report(self, coverage_stats: Dict, hierarchy_stats: Dict):
        """Print comprehensive validation report"""
        print("\n" + "="*60)
        print("COLLISION MAPPING VALIDATION REPORT")
        print("="*60)
        
        # Coverage Report
        print(f"\nVertex Coverage:")
        print(f"  Total vertices: {coverage_stats['total_vertices']:,}")
        print(f"  Covered vertices: {coverage_stats['covered_vertices']:,}")
        print(f"  Coverage: {coverage_stats['coverage_percent']:.2f}%")
        
        if coverage_stats['uncovered_vertices'] > 0:
            print(f"  ⚠️  WARNING: {coverage_stats['uncovered_vertices']} uncovered vertices!")
            if coverage_stats['uncovered_sample']:
                print(f"    Sample uncovered vertices: {coverage_stats['uncovered_sample']}")
        else:
            print(f"  ✅ Perfect vertex coverage!")
        
        print(f"\nAssignment Statistics:")
        print(f"  Total assignments: {coverage_stats['total_assignments']:,}")
        print(f"  Average per vertex: {coverage_stats['avg_assignments_per_vertex']:.2f}")
        print(f"  Range: {coverage_stats['min_assignments_per_vertex']} - {coverage_stats['max_assignments_per_vertex']}")
        print(f"  Vertices with multiple assignments: {coverage_stats['vertices_with_multiple_assignments']:,}")
        
        # Hierarchy Report
        print(f"\nHierarchy Completeness:")
        print(f"  Sphere→Capsule: {hierarchy_stats['mapped_spheres']}/{hierarchy_stats['total_spheres']} " +
              f"({hierarchy_stats['sphere_mapping_percent']:.1f}%)")
        print(f"  Capsule→Simple: {hierarchy_stats['mapped_capsules']}/{hierarchy_stats['total_capsules']} " +
              f"({hierarchy_stats['capsule_mapping_percent']:.1f}%)")
        
        if hierarchy_stats['unmapped_spheres'] > 0:
            print(f"  ⚠️  WARNING: {hierarchy_stats['unmapped_spheres']} unmapped spheres!")
        if hierarchy_stats['unmapped_capsules'] > 0:
            print(f"  ⚠️  WARNING: {hierarchy_stats['unmapped_capsules']} unmapped capsules!")
        
        if (hierarchy_stats['unmapped_spheres'] == 0 and hierarchy_stats['unmapped_capsules'] == 0):
            print(f"  ✅ Complete hierarchy mappings!")
        
        print("="*60)


def build_complete_collision_data(star_interface, output_filepath: Optional[str] = None) -> CollisionDataStore:
    """
    Complete Phase 1: Generate all collision mappings and export optimized data
    
    Args:
        star_interface: STARInterface instance for getting mesh data
        output_filepath: Optional HDF5 file path to save data
        
    Returns:
        CollisionDataStore with all mappings ready for C++
    """
    print("="*60)
    print("PHASE 1: COLLISION MAPPING SYSTEM")
    print("="*60)
    total_start = time.time()
    
    # Step 1: Get T-pose data from STAR
    print("\n1. Getting T-pose data from STAR model...")
    vertices, joints = star_interface.get_neutral_pose()
    if vertices is None or joints is None:
        raise RuntimeError("Failed to get STAR mesh and joint data")
    
    print(f"   Retrieved {len(vertices)} vertices and {len(joints)} joints")
    
    # Step 2: Generate all collision layers using existing classes
    print("\n2. Generating hierarchical collision layers...")
    from ..layers.sphere_layer import SphereLayer
    from ..layers.capsule_layer import CapsuleLayer
    from ..layers.simple_capsule_layer import SimpleCapsuleLayer
    
    sphere_layer = SphereLayer()
    capsule_layer = CapsuleLayer()
    simple_layer = SimpleCapsuleLayer()
    
    # Generate layers with proper containment
    spheres = sphere_layer.generate_from_joints(joints, vertices)
    capsules = capsule_layer.generate_from_joints(joints, sphere_layer)
    simple_capsules = simple_layer.generate_from_joints(joints, capsule_layer)
    
    print(f"   Generated {len(spheres)} spheres, {len(capsules)} capsules, {len(simple_capsules)} simple capsules")
    
    # Step 3: Create vertex-to-sphere assignments
    print("\n3. Creating vertex-to-sphere assignments...")
    mapper = VertexSphereMapper()
    vertex_mapping = mapper.assign_vertices_to_spheres(vertices, spheres)
    
    # Step 4: Build containment hierarchy
    print("\n4. Building containment hierarchy...")
    hierarchy_builder = HierarchyBuilder()
    hierarchy = hierarchy_builder.build_hierarchy(spheres, capsule_layer, simple_layer)
    
    # Step 5: Validate all mappings
    print("\n5. Validating mappings...")
    validator = MappingValidator()
    coverage_stats = validator.validate_coverage(vertex_mapping, len(vertices))
    hierarchy_stats = validator.validate_hierarchy(hierarchy)
    
    # Print validation report
    validator.print_validation_report(coverage_stats, hierarchy_stats)
    
    # Step 6: Build optimized data structures
    print("\n6. Building optimized data structures...")
    collision_store = CollisionDataStore()
    collision_store.build_optimized_structures(vertex_mapping, hierarchy, len(vertices))
    
    # Step 7: Save to file if requested
    if output_filepath:
        print("\n7. Saving collision data...")
        collision_store.save_to_hdf5(output_filepath)
    
    # Final summary
    total_time = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"PHASE 1 COMPLETE in {total_time:.1f} seconds")
    print(f"Collision mapping system ready for C++ integration!")
    if output_filepath:
        print(f"Data saved to: {output_filepath}")
    print(f"{'='*60}")
    
    return collision_store


# Convenience function for quick testing
def quick_build_and_save(star_interface, filepath: str = "collision_data.h5"):
    """Quick build and save for testing"""
    return build_complete_collision_data(star_interface, filepath)


# Example usage
if __name__ == "__main__":
    # This would be used like:
    # from star_body_system.core.star_interface import STARInterface
    # star = STARInterface(gender='neutral')
    # collision_data = build_complete_collision_data(star, "collision_data.h5")
    pass