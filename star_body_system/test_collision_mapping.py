"""
Test script for collision mapping system
Run this to verify Phase 1 implementation works correctly
"""

import numpy as np
import sys
import os

try:
    from star_body_system_pkg.core.star_interface import STARInterface
    from star_body_system_pkg.core.collision_mapping import build_complete_collision_data
    from star_body_system_pkg.debug.plotly_viz import visualize_layer0_mesh_joints
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you've installed the package with: pip install -e .")
    sys.exit(1)


def test_star_interface():
    """Test basic STAR interface functionality"""
    print("\n" + "="*50)
    print("Testing STAR Interface...")
    
    try:
        star = STARInterface(gender='neutral')
        vertices, joints = star.get_neutral_pose()
        
        if vertices is None or joints is None:
            print("❌ Failed to get STAR data - check STAR installation")
            return None
        
        print(f"✅ STAR interface working:")
        print(f"   Vertices: {len(vertices)} (expected ~6890)")
        print(f"   Joints: {len(joints)} (expected 24)")
        print(f"   Vertex shape: {vertices.shape}")
        print(f"   Joint shape: {joints.shape}")
        
        # Basic sanity checks
        if len(vertices) < 5000:
            print(f"⚠️  WARNING: Vertex count seems low ({len(vertices)})")
        if len(joints) != 24:
            print(f"⚠️  WARNING: Joint count not 24 ({len(joints)})")
            
        return star
        
    except Exception as e:
        print(f"❌ STAR interface error: {e}")
        return None


def test_layer_generation(star):
    """Test individual layer generation"""
    print("\n" + "="*50)
    print("Testing Layer Generation...")
    
    try:
        from star_body_system_pkg.layers.sphere_layer import SphereLayer
        from star_body_system_pkg.layers.capsule_layer import CapsuleLayer
        from star_body_system_pkg.layers.simple_capsule_layer import SimpleCapsuleLayer
        
        vertices, joints = star.get_neutral_pose()
        
        # Test sphere layer
        sphere_layer = SphereLayer()
        spheres = sphere_layer.generate_from_joints(joints, vertices)
        print(f"✅ Sphere layer: {len(spheres)} spheres generated")
        
        # Test capsule layer
        capsule_layer = CapsuleLayer()
        capsules = capsule_layer.generate_from_joints(joints, sphere_layer)
        print(f"✅ Capsule layer: {len(capsules)} capsules generated")
        
        # Test simple layer
        simple_layer = SimpleCapsuleLayer()
        simple_capsules = simple_layer.generate_from_joints(joints, capsule_layer)
        print(f"✅ Simple layer: {len(simple_capsules)} simple capsules generated")
        
        # Sanity checks
        if len(spheres) < 50:
            print(f"⚠️  WARNING: Low sphere count ({len(spheres)})")
        if len(capsules) != 24:
            print(f"⚠️  WARNING: Capsule count not 24 ({len(capsules)})")
        if len(simple_capsules) != 9:
            print(f"⚠️  WARNING: Simple capsule count not 9 ({len(simple_capsules)})")
            
        return True
        
    except Exception as e:
        print(f"❌ Layer generation error: {e}")
        return False


def test_collision_mapping(star):
    """Test the main collision mapping system"""
    print("\n" + "="*50)
    print("Testing Collision Mapping System...")
    
    try:
        # Run the complete collision mapping
        collision_data = build_complete_collision_data(star)
        
        print(f"✅ Collision mapping completed successfully!")
        
        # Check the results
        metadata = collision_data.metadata
        print(f"\nMapping Results:")
        print(f"   Vertices: {metadata['num_vertices']}")
        print(f"   Spheres: {metadata['num_spheres']}")
        print(f"   Capsules: {metadata['num_capsules']}")
        print(f"   Simple capsules: {metadata['num_simple']}")
        print(f"   Max assignments per vertex: {metadata['max_assignments_per_vertex']}")
        print(f"   Memory usage: ~{collision_data._estimate_memory_mb():.1f} MB")
        
        # Sanity checks
        if metadata['num_vertices'] < 5000:
            print(f"⚠️  WARNING: Low vertex count")
        if metadata['max_assignments_per_vertex'] == 0:
            print(f"❌ ERROR: No vertex assignments found!")
            return False
        if metadata['max_assignments_per_vertex'] > 20:
            print(f"⚠️  WARNING: Very high assignments per vertex ({metadata['max_assignments_per_vertex']})")
            
        return collision_data
        
    except Exception as e:
        print(f"❌ Collision mapping error: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_hdf5_export(collision_data):
    """Test HDF5 export functionality"""
    print("\n" + "="*50)
    print("Testing HDF5 Export...")
    
    try:
        test_filepath = "test_collision_data.h5"
        collision_data.save_to_hdf5(test_filepath)
        
        # Try to read it back
        import h5py
        with h5py.File(test_filepath, 'r') as f:
            print(f"✅ HDF5 export successful:")
            print(f"   File size: {collision_data._get_file_size_mb(test_filepath):.1f} MB")
            print(f"   Datasets: {list(f.keys())}")
            print(f"   Format version: {f.attrs['format_version']}")
            
            # Check array shapes
            vertex_assignments = f['vertex_sphere_assignments']
            print(f"   Vertex assignments shape: {vertex_assignments.shape}")
            print(f"   Sphere to capsule mappings: {len(f['sphere_to_capsule'])}")
        
        # Clean up test file
        os.remove(test_filepath)
        print(f"✅ Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ HDF5 export error: {e}")
        return False


def test_visualization(star):
    """Test basic visualization to ensure everything renders"""
    print("\n" + "="*50)
    print("Testing Visualization...")
    
    try:
        vertices, joints = star.get_neutral_pose()
        
        # Create a simple visualization
        fig = visualize_layer0_mesh_joints(vertices, joints, "Test Visualization")
        
        print(f"✅ Visualization created successfully")
        print(f"   Figure has {len(fig.data)} traces")
        
        # Optionally save to HTML for viewing
        try:
            fig.write_html("test_visualization.html")
            print(f"✅ Visualization saved to test_visualization.html")
            print(f"   Open this file in a browser to view the 3D model")
        except:
            print(f"⚠️  Could not save HTML file (optional)")
            
        return True
        
    except Exception as e:
        print(f"❌ Visualization error: {e}")
        return False


def run_all_tests():
    """Run comprehensive test suite"""
    print("COLLISION MAPPING SYSTEM TEST SUITE")
    print("="*60)
    
    # Test 1: STAR interface
    star = test_star_interface()
    if star is None:
        print("\n❌ STAR interface failed - cannot continue")
        return False
    
    # Test 2: Layer generation
    if not test_layer_generation(star):
        print("\n❌ Layer generation failed - cannot continue")
        return False
    
    # Test 3: Collision mapping (main test)
    collision_data = test_collision_mapping(star)
    if collision_data is None:
        print("\n❌ Collision mapping failed - core functionality broken")
        return False
    
    # Test 4: HDF5 export
    if not test_hdf5_export(collision_data):
        print("\n❌ HDF5 export failed - data persistence broken")
        return False
    
    # Test 5: Visualization (optional)
    test_visualization(star)  # Don't fail on visualization issues
    
    # Final summary
    print("\n" + "="*60)
    print("✅ ALL CORE TESTS PASSED!")
    print("Phase 1 collision mapping system is working correctly.")
    print("Ready to proceed to Phase 2 (collision detection engine).")
    print("="*60)
    
    return True


def quick_performance_test(star):
    """Quick performance measurement"""
    print("\n" + "="*50)
    print("Performance Test...")
    
    import time
    start_time = time.time()
    
    collision_data = build_complete_collision_data(star)
    
    total_time = time.time() - start_time
    vertex_count = collision_data.metadata['num_vertices']
    sphere_count = collision_data.metadata['num_spheres']
    
    print(f"✅ Performance Results:")
    print(f"   Total time: {total_time:.1f} seconds")
    print(f"   Vertices per second: {vertex_count/total_time:.0f}")
    print(f"   Assignments per second: {collision_data.metadata.get('total_assignments', 0)/total_time:.0f}")
    
    if total_time > 60:
        print(f"⚠️  WARNING: Performance seems slow (>{total_time:.1f}s)")
    else:
        print(f"✅ Performance acceptable")


if __name__ == "__main__":
    print("Starting collision mapping tests...\n")
    
    # Check if we want to run performance test
    run_performance = "--performance" in sys.argv
    
    try:
        success = run_all_tests()
        
        if success and run_performance:
            star = STARInterface(gender='neutral')
            quick_performance_test(star)
        
        if success:
            print(f"\n🎉 All tests completed successfully!")
            print(f"Your collision mapping system is ready!")
        else:
            print(f"\n💥 Some tests failed - check the output above")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print(f"\n⏹️  Tests interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)