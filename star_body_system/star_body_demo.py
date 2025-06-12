#!/usr/bin/env python3
"""
STAR Body System Demo
Demonstrates multi-layer body representation generation and visualization
"""

import numpy as np
from star_body_system_pkg import STARInterface, SphereLayer, CapsuleLayer, SimpleCapsuleLayer
from star_body_system_pkg.debug.plotly_viz import LayerVisualizer

def main():
    print("=== STAR Body System Demo ===\n")
    
    # 1. Generate Multi-Layer Body Representation
    print("1. Initializing STAR model and generating layers...")
    
    try:
        # Initialize STAR model
        star = STARInterface(gender='neutral')
        
        # Get mesh and joint data for neutral pose
        vertices, joints = star.get_neutral_pose()
        
        if vertices is None or joints is None:
            print("ERROR: Could not load STAR model data")
            return
        
        print(f"   Loaded mesh with {len(vertices)} vertices")
        print(f"   Loaded {len(joints)} joint positions")
        
        # Create all three layers
        sphere_layer = SphereLayer()
        capsule_layer = CapsuleLayer() 
        simple_layer = SimpleCapsuleLayer()
        
        # Generate layers from joint positions WITH HIERARCHICAL CONTAINMENT
        spheres = sphere_layer.generate_from_joints(joints, vertices)
        capsules = capsule_layer.generate_from_joints(joints, sphere_layer)  # Pass sphere_layer for containment
        simple_capsules = simple_layer.generate_from_joints(joints, capsule_layer)  # Pass capsule_layer for containment
        
        print(f"   Generated {len(spheres)} spheres, {len(capsules)} capsules, {len(simple_capsules)} simple capsules\n")
        
    except Exception as e:
        print(f"ERROR initializing STAR: {e}")
        print("Make sure STAR model is installed and model files are available")
        return
    
    # 2. Print Layer Statistics
    print("2. Layer Statistics:")
    
    sphere_stats = sphere_layer.get_stats()
    capsule_stats = capsule_layer.get_stats()
    simple_stats = simple_layer.get_stats()
    
    print(f"   Sphere Layer: {sphere_stats['total']} spheres")
    for bone_type, stats in sphere_stats.get('by_type', {}).items():
        print(f"     {bone_type}: {stats['count']} spheres, avg radius: {stats['avg_radius']:.3f}")
    
    print(f"   Capsule Layer: {capsule_stats['total']} capsules")
    for bone_type, stats in capsule_stats.get('by_type', {}).items():
        print(f"     {bone_type}: {stats['count']} capsules, avg radius: {stats['avg_radius']:.3f}")
    
    print(f"   Simple Layer: {simple_stats['total']} capsules")
    for region, stats in simple_stats.get('by_region', {}).items():
        print(f"     {region}: {stats['count']} capsules, avg radius: {stats['avg_radius']:.3f}")
    
    print()
    
    # 3. Visualize the Layers
    print("3. Creating visualizations...")
    
    viz = LayerVisualizer()
    
    # Visualize individual layers
    print("   Creating Layer 0 (mesh + joints + bones) visualization...")
    layer0_fig = viz.visualize_layer0_mesh_joints(vertices, joints, "Layer 0: STAR Mesh + Joints + Bones")
    
    print("   Creating sphere visualization...")
    sphere_fig = viz.visualize_spheres(spheres, "Layer 1: Spheres", show_mesh=vertices)
    
    print("   Creating capsule visualization...")
    capsule_fig = viz.visualize_capsules(capsules, "Layer 2: Capsules")
    
    print("   Creating simple capsule visualization...")
    simple_fig = viz.visualize_capsules(simple_capsules, "Layer 3: Simple", layer_type='simple')
    
    print("   Creating combined visualization...")
    combined_fig = viz.visualize_all_layers(sphere_layer, capsule_layer, simple_layer)
    
    print("   Creating sphere + capsule comparison...")
    sphere_capsule_fig = viz.visualize_sphere_capsule_comparison(sphere_layer, capsule_layer)
    
    # Show visualizations (will open in browser)
    print("   Opening visualizations in browser...")
    layer0_fig.show()
    sphere_fig.show()
    capsule_fig.show() 
    simple_fig.show()
    combined_fig.show()
    sphere_capsule_fig.show()
    
    # 4. Test Different Poses
    print("\n4. Testing pose updates...")
    
    # Create custom pose (bend spine forward)
    pose_params = np.zeros(72)
    pose_params[6:9] = [0.5, 0, 0]  # Bend spine forward
    
    print("   Generating posed body (spine bent forward)...")
    vertices_posed, joints_posed = star.get_mesh_and_joints(pose_params)
    
    if vertices_posed is not None and joints_posed is not None:
        # Update all layers WITH HIERARCHICAL CONTAINMENT
        spheres_posed = sphere_layer.update_from_joints(joints_posed)
        capsules_posed = capsule_layer.update_from_joints(joints_posed, sphere_layer)  # Pass sphere_layer
        simple_posed = simple_layer.update_from_joints(joints_posed, capsule_layer)    # Pass capsule_layer
        
        print(f"   Updated layers for new pose")
        
        # Visualize the posed body
        posed_fig = viz.visualize_all_layers(sphere_layer, capsule_layer, simple_layer)
        posed_fig.update_layout(title="Posed Body (Spine Bent Forward)")
        posed_fig.show()
    
    # 5. Collision Detection Test
    print("\n5. Testing collision detection...")
    
    # Test points around the body
    test_points = [
        np.array([0.0, 0.0, 0.0]),   # Center (should be inside)
        np.array([0.1, 0.5, 0.0]),   # Torso area
        np.array([0.3, -0.5, 0.0]),  # Leg area
        np.array([0.5, 0.8, 0.0]),   # Arm area
        np.array([1.0, 1.0, 1.0]),   # Outside body
    ]
    
    for i, test_point in enumerate(test_points):
        print(f"   Test point {i+1}: {test_point}")
        
        # Check spheres
        containing_spheres = [name for center, radius, name in spheres 
                             if np.linalg.norm(test_point - center) <= radius]
        
        # Check capsules
        containing_capsules = capsule_layer.contains_point(test_point)
        containing_simple = simple_layer.contains_point(test_point)
        
        print(f"     Contained in {len(containing_spheres)} spheres")
        print(f"     Contained in capsules: {containing_capsules}")
        print(f"     Contained in simple capsules: {containing_simple}")
        print()
    
    print("=== Demo Complete ===")
    print("Check your browser for the 3D visualizations!")

if __name__ == "__main__":
    main()