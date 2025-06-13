/**
 * Complete example demonstrating the Phase 2 Collision Detection Engine
 * Shows integration with robot planning and real-time usage at 120+ FPS
 */

#include "collision_detection_engine.hpp"
#include "capsule_creation_block.hpp"
#include <iostream>
#include <vector>
#include <chrono>
#include <random>

using namespace delta;

/**
 * Example: Integration with STAR model and robot planning
 */
class RoboticsCollisionDemo {
private:
    std::unique_ptr<CollisionDetectionEngine> collision_engine_;
    std::unique_ptr<CapsuleCreationBlock> capsule_creator_;
    
    // Demo parameters
    int demo_frames_;
    bool show_detailed_stats_;
    
public:
    RoboticsCollisionDemo() : demo_frames_(1000), show_detailed_stats_(false) {
        capsule_creator_ = std::make_unique<CapsuleCreationBlock>();
    }
    
    /**
     * Initialize the collision system with Phase 1 data
     */
    bool initialize(const std::string& collision_data_path) {
        std::cout << "==============================================================" << std::endl;
        std::cout << "PHASE 2: REAL-TIME COLLISION DETECTION ENGINE DEMO" << std::endl;
        std::cout << "==============================================================" << std::endl;
        
        // Generate example T-pose mesh vertices (in real application, this comes from STAR)
        std::vector<Eigen::Vector3d> base_vertices = generate_example_star_mesh();
        
        // Initialize collision detection engine
        collision_engine_ = create_collision_engine(collision_data_path, base_vertices);
        
        if (!collision_engine_) {
            std::cerr << "❌ Failed to initialize collision detection engine" << std::endl;
            return false;
        }
        
        // Configure for high-performance robotics use
        collision_engine_->configure(
            2,      // 2 frame cooldown for fast response
            5,      // 5 max contacts per capsule (sufficient for path planning)
            1e-5    // 0.01mm penetration tolerance
        );
        
        std::cout << "✅ Collision system initialized for robotics application" << std::endl;
        std::cout << "   Target: 120+ FPS collision detection" << std::endl;
        std::cout << "   Use case: Robot path planning with force feedback" << std::endl;
        
        return true;
    }
    
    /**
     * Simulate real-time collision detection loop
     */
    void run_realtime_demo() {
        std::cout << "\n" << "="*60 << std::endl;
        std::cout << "REAL-TIME COLLISION DETECTION SIMULATION" << std::endl;
        std::cout << "="*60 << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Statistics tracking
        std::vector<double> frame_times;
        int collision_frames = 0;
        double total_collision_time = 0.0;
        
        frame_times.reserve(demo_frames_);
        
        // Simulate camera/tracking updates at 60-90 FPS
        // Collision detection should be much faster (120+ FPS)
        for (int frame = 0; frame < demo_frames_; ++frame) {
            auto frame_start = std::chrono::high_resolution_clock::now();
            
            // 1. Get current human pose (normally from STAR + camera tracking)
            auto bone_positions = simulate_human_movement(frame);
            
            // 2. Get current robot configuration (normally from robot planner)
            auto robot_s_points = simulate_robot_movement(frame);
            
            // 3. Convert robot S-points to capsule chain
            auto capsule_result = capsule_creator_->create_capsule_chain(robot_s_points, 0.05); // 5cm robot radius
            
            if (!capsule_result.success) {
                std::cerr << "⚠️  Frame " << frame << ": Failed to create robot capsules" << std::endl;
                continue;
            }
            
            // 4. MAIN COLLISION DETECTION CALL
            auto collision_result = collision_engine_->detect_collisions(bone_positions, capsule_result.capsules);
            
            // 5. Process results for robot control
            if (collision_result.has_collision) {
                collision_frames++;
                total_collision_time += collision_result.computation_time_ms;
                
                // In real application: send collision data to robot controller
                process_collision_for_robot_control(collision_result);
            }
            
            auto frame_end = std::chrono::high_resolution_clock::now();
            double frame_time_ms = std::chrono::duration<double, std::milli>(frame_end - frame_start).count();
            frame_times.push_back(frame_time_ms);
            
            // Progress reporting
            if (frame % 100 == 0) {
                double avg_frame_time = std::accumulate(frame_times.end() - std::min(100, static_cast<int>(frame_times.size())), 
                                                       frame_times.end(), 0.0) / std::min(100, static_cast<int>(frame_times.size()));
                double fps = 1000.0 / avg_frame_time;
                
                std::cout << "  Frame " << frame << "/" << demo_frames_ 
                          << " - Avg FPS: " << fps 
                          << " - Collisions: " << collision_frames << std::endl;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        double total_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
        
        // Performance analysis
        analyze_performance_results(frame_times, collision_frames, total_time_seconds, total_collision_time);
    }
    
    /**
     * Demonstrate hierarchical collision detection benefits
     */
    void demonstrate_hierarchical_benefits() {
        std::cout << "\n" << "="*60 << std::endl;
        std::cout << "HIERARCHICAL COLLISION DETECTION BENEFITS" << std::endl;
        std::cout << "="*60 << std::endl;
        
        // Test collision detection with different pipeline stages
        auto bone_positions = simulate_human_movement(500); // Mid-demo pose
        auto robot_s_points = simulate_collision_scenario(); // Intentional collision
        auto capsule_result = capsule_creator_->create_capsule_chain(robot_s_points, 0.05);
        
        if (!capsule_result.success) {
            std::cerr << "❌ Failed to create test capsules" << std::endl;
            return;
        }
        
        // Test 1: Only Layer 3 (fastest, coarse detection)
        collision_engine_->set_pipeline_stages(false, false, false);
        auto result_layer3 = collision_engine_->detect_collisions(bone_positions, capsule_result.capsules);
        
        // Test 2: Layer 3 + Layer 2 (medium detail)
        collision_engine_->set_pipeline_stages(true, false, false);
        auto result_layer2 = collision_engine_->detect_collisions(bone_positions, capsule_result.capsules);
        
        // Test 3: Layer 3 + Layer 2 + Layer 1 (high detail)
        collision_engine_->set_pipeline_stages(true, true, false);
        auto result_layer1 = collision_engine_->detect_collisions(bone_positions, capsule_result.capsules);
        
        // Test 4: Full pipeline (precise depth + normals)
        collision_engine_->set_pipeline_stages(true, true, true);
        auto result_full = collision_engine_->detect_collisions(bone_positions, capsule_result.capsules);
        
        // Compare results
        std::cout << "Pipeline Stage Comparison:" << std::endl;
        std::cout << "  Layer 3 only:  " << result_layer3.computation_time_ms << " ms - " 
                  << (result_layer3.has_collision ? "COLLISION" : "no collision") << std::endl;
        std::cout << "  + Layer 2:     " << result_layer2.computation_time_ms << " ms - " 
                  << (result_layer2.has_collision ? "COLLISION" : "no collision") << std::endl;
        std::cout << "  + Layer 1:     " << result_layer1.computation_time_ms << " ms - " 
                  << (result_layer1.has_collision ? "COLLISION" : "no collision") << std::endl;
        std::cout << "  Full pipeline: " << result_full.computation_time_ms << " ms - " 
                  << result_full.contacts.size() << " precise contacts" << std::endl;
        
        if (!result_full.contacts.empty()) {
            std::cout << "  Max penetration: " << result_full.max_penetration_depth << " units" << std::endl;
            std::cout << "  Sample normal: (" << result_full.contacts[0].surface_normal.transpose() << ")" << std::endl;
        }
        
        // Restore full pipeline
        collision_engine_->set_pipeline_stages(true, true, true);
    }
    
    /**
     * Show final performance statistics
     */
    void show_final_statistics() {
        std::cout << "\n" << "="*60 << std::endl;
        std::cout << "FINAL PERFORMANCE STATISTICS" << std::endl;
        std::cout << "="*60 << std::endl;
        
        auto breakdown = collision_engine_->get_performance_breakdown();
        auto layer_stats = collision_engine_->get_layer_statistics();
        
        std::cout << "Overall Performance:" << std::endl;
        std::cout << "  Total frames processed: " << breakdown.total_frames << std::endl;
        std::cout << "  Collision frames: " << breakdown.collision_frames << std::endl;
        std::cout << "  Collision rate: " << breakdown.collision_rate << "%" << std::endl;
        std::cout << "  Average computation time: " << breakdown.total_time_ms << " ms/frame" << std::endl;
        
        if (breakdown.total_time_ms > 0) {
            double target_fps = 120.0;
            double max_time_per_frame = 1000.0 / target_fps; // 8.33 ms for 120 FPS
            std::cout << "  Target 120 FPS budget: " << max_time_per_frame << " ms" << std::endl;
            std::cout << "  Performance margin: " << (max_time_per_frame - breakdown.total_time_ms) << " ms" << std::endl;
            
            if (breakdown.total_time_ms <= max_time_per_frame) {
                std::cout << "  ✅ TARGET ACHIEVED - Suitable for 120+ FPS robotics" << std::endl;
            } else {
                std::cout << "  ⚠️  Target missed - optimization needed" << std::endl;
            }
        }
        
        std::cout << "\nPipeline Stage Breakdown:" << std::endl;
        std::cout << "  Stage 1 (Layer 3): " << breakdown.stage1_time_ms << " ms" << std::endl;
        std::cout << "  Stage 2 (Layer 2): " << breakdown.stage2_time_ms << " ms" << std::endl;
        std::cout << "  Stage 3 (Layer 1): " << breakdown.stage3_time_ms << " ms" << std::endl;
        std::cout << "  Stage 4 (Layer 0): " << breakdown.stage4_time_ms << " ms" << std::endl;
        
        std::cout << "\nMemory Usage:" << std::endl;
        std::cout << "  Active Layer 2: " << breakdown.active_layer2_count << " primitives" << std::endl;
        std::cout << "  Active Layer 1: " << breakdown.active_layer1_count << " primitives" << std::endl;
        std::cout << "  Loaded Layer 0: " << breakdown.loaded_layer0_count << " vertex groups" << std::endl;
        std::cout << "  Total memory: " << breakdown.memory_usage_mb << " MB" << std::endl;
        
        std::cout << "\n" << collision_engine_->get_debug_info() << std::endl;
    }

private:
    
    /**
     * Generate example STAR mesh vertices (placeholder)
     * In real application, this comes from STAR model
     */
    std::vector<Eigen::Vector3d> generate_example_star_mesh() {
        std::vector<Eigen::Vector3d> vertices;
        vertices.reserve(6890); // STAR has ~6890 vertices
        
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> dist(0.0, 0.3);
        
        // Generate human-shaped point cloud (very simplified)
        for (int i = 0; i < 6890; ++i) {
            Eigen::Vector3d vertex;
            
            // Create rough human shape
            double height_factor = static_cast<double>(i) / 6890.0;
            vertex.y() = height_factor * 1.8; // 1.8m tall human
            vertex.x() = dist(gen) * (0.3 + 0.2 * sin(height_factor * M_PI)); // Torso width variation
            vertex.z() = dist(gen) * 0.2; // Body depth
            
            vertices.push_back(vertex);
        }
        
        return vertices;
    }
    
    /**
     * Simulate human movement (bone positions)
     */
    std::vector<Eigen::Vector3d> simulate_human_movement(int frame) {
        std::vector<Eigen::Vector3d> bone_positions(24);
        
        // Simulate simple walking motion
        double time = frame * 0.016; // 60 FPS time step
        double walk_cycle = sin(time * 2.0) * 0.1;
        
        // Simplified bone positions (normally from STAR forward kinematics)
        bone_positions[0] = Eigen::Vector3d(0, 0.9 + walk_cycle, 0);    // pelvis
        bone_positions[15] = Eigen::Vector3d(0, 1.7 + walk_cycle, 0);   // head
        bone_positions[16] = Eigen::Vector3d(-0.4, 1.4 + walk_cycle, 0); // left shoulder
        bone_positions[17] = Eigen::Vector3d(0.4, 1.4 + walk_cycle, 0);  // right shoulder
        bone_positions[22] = Eigen::Vector3d(-0.6, 1.0 + walk_cycle, 0); // left hand
        bone_positions[23] = Eigen::Vector3d(0.6, 1.0 + walk_cycle, 0);  // right hand
        
        // Fill remaining bones with interpolated positions
        for (int i = 1; i < 24; ++i) {
            if (bone_positions[i].norm() == 0) {
                bone_positions[i] = bone_positions[0] + Eigen::Vector3d(
                    (i % 3 - 1) * 0.2, 
                    (i / 12) * 0.3 + walk_cycle, 
                    0
                );
            }
        }
        
        return bone_positions;
    }
    
    /**
     * Simulate robot movement (S-points for capsule chain)
     */
    std::vector<Eigen::Vector3d> simulate_robot_movement(int frame) {
        std::vector<Eigen::Vector3d> s_points;
        
        // Simulate robot arm moving towards human
        double time = frame * 0.016;
        double approach_factor = 0.5 + 0.3 * sin(time * 0.5); // Slow approach motion
        
        // 6-DOF robot arm (7 S-points for 6 capsules)
        s_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));                    // Base
        s_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.3));                    // Joint 1
        s_points.push_back(Eigen::Vector3d(0.8, 0.0, 0.6));                    // Joint 2
        s_points.push_back(Eigen::Vector3d(0.5, 0.0, 0.9));                    // Joint 3
        s_points.push_back(Eigen::Vector3d(0.2 * approach_factor, 0.0, 1.1));  // Joint 4 (approaching)
        s_points.push_back(Eigen::Vector3d(0.0 * approach_factor, 0.0, 1.3));  // Joint 5 (approaching)
        s_points.push_back(Eigen::Vector3d(-0.1 * approach_factor, 0.0, 1.4)); // End effector (approaching)
        
        return s_points;
    }
    
    /**
     * Create intentional collision scenario for testing
     */
    std::vector<Eigen::Vector3d> simulate_collision_scenario() {
        std::vector<Eigen::Vector3d> s_points;
        
        // Robot arm penetrating human torso area
        s_points.push_back(Eigen::Vector3d(1.0, 0.0, 0.0));   // Base
        s_points.push_back(Eigen::Vector3d(0.8, 0.0, 0.3));   // Joint 1
        s_points.push_back(Eigen::Vector3d(0.6, 0.0, 0.6));   // Joint 2
        s_points.push_back(Eigen::Vector3d(0.3, 0.0, 0.9));   // Joint 3
        s_points.push_back(Eigen::Vector3d(0.0, 0.0, 1.2));   // Joint 4 (near torso)
        s_points.push_back(Eigen::Vector3d(-0.2, 0.0, 1.4));  // Joint 5 (penetrating)
        s_points.push_back(Eigen::Vector3d(-0.3, 0.0, 1.5));  // End effector (deep penetration)
        
        return s_points;
    }
    
    /**
     * Process collision results for robot control
     * In real application: send to robot controller for path modification
     */
    void process_collision_for_robot_control(const CollisionResult& collision_result) {
        // This is where the collision data would be sent to:
        // 1. Robot path planner (for collision avoidance)
        // 2. Force feedback controller (for haptic response)
        // 3. Safety system (for emergency stops)
        
        if (show_detailed_stats_) {
            std::cout << "    COLLISION DETECTED:" << std::endl;
            std::cout << "      Contacts: " << collision_result.contacts.size() << std::endl;
            std::cout << "      Max depth: " << collision_result.max_penetration_depth << std::endl;
            std::cout << "      Computation: " << collision_result.computation_time_ms << " ms" << std::endl;
            
            // Show first contact details
            if (!collision_result.contacts.empty()) {
                const auto& contact = collision_result.contacts[0];
                std::cout << "      Contact point: (" << contact.contact_point.transpose() << ")" << std::endl;
                std::cout << "      Surface normal: (" << contact.surface_normal.transpose() << ")" << std::endl;
                std::cout << "      Robot capsule: " << contact.robot_capsule_index << std::endl;
            }
        }
        
        // Example: Calculate force feedback vector for haptic device
        if (!collision_result.contacts.empty()) {
            for (const auto& contact : collision_result.contacts) {
                // Force magnitude proportional to penetration depth
                double force_magnitude = contact.penetration_depth * 1000.0; // N/m spring constant
                
                // Force direction along surface normal (pushing robot away)
                Eigen::Vector3d force_vector = contact.surface_normal * force_magnitude;
                
                // In real application: send force_vector to haptic device
                // haptic_device.apply_force(contact.robot_capsule_index, force_vector);
            }
        }
    }
    
    /**
     * Analyze performance results and determine if 120+ FPS target is met
     */
    void analyze_performance_results(const std::vector<double>& frame_times, 
                                    int collision_frames, 
                                    double total_time_seconds,
                                    double total_collision_time) {
        
        std::cout << "\n" << "="*60 << std::endl;
        std::cout << "PERFORMANCE ANALYSIS RESULTS" << std::endl;
        std::cout << "="*60 << std::endl;
        
        // Calculate statistics
        double avg_frame_time = std::accumulate(frame_times.begin(), frame_times.end(), 0.0) / frame_times.size();
        double max_frame_time = *std::max_element(frame_times.begin(), frame_times.end());
        double min_frame_time = *std::min_element(frame_times.begin(), frame_times.end());
        
        // Calculate percentiles
        std::vector<double> sorted_times = frame_times;
        std::sort(sorted_times.begin(), sorted_times.end());
        double p95_frame_time = sorted_times[static_cast<size_t>(sorted_times.size() * 0.95)];
        double p99_frame_time = sorted_times[static_cast<size_t>(sorted_times.size() * 0.99)];
        
        // FPS calculations
        double avg_fps = 1000.0 / avg_frame_time;
        double min_fps = 1000.0 / max_frame_time;
        double p95_fps = 1000.0 / p95_frame_time;
        
        std::cout << "Frame Time Statistics:" << std::endl;
        std::cout << "  Average: " << avg_frame_time << " ms (" << avg_fps << " FPS)" << std::endl;
        std::cout << "  Minimum: " << min_frame_time << " ms (" << 1000.0/min_frame_time << " FPS)" << std::endl;
        std::cout << "  Maximum: " << max_frame_time << " ms (" << min_fps << " FPS)" << std::endl;
        std::cout << "  95th percentile: " << p95_frame_time << " ms (" << p95_fps << " FPS)" << std::endl;
        std::cout << "  99th percentile: " << p99_frame_time << " ms (" << 1000.0/p99_frame_time << " FPS)" << std::endl;
        
        std::cout << "\nCollision Statistics:" << std::endl;
        std::cout << "  Total frames: " << frame_times.size() << std::endl;
        std::cout << "  Collision frames: " << collision_frames << std::endl;
        std::cout << "  Collision rate: " << (static_cast<double>(collision_frames) / frame_times.size() * 100.0) << "%" << std::endl;
        
        if (collision_frames > 0) {
            double avg_collision_time = total_collision_time / collision_frames;
            std::cout << "  Average collision detection time: " << avg_collision_time << " ms" << std::endl;
        }
        
        // 120 FPS target analysis
        double target_120fps_budget = 1000.0 / 120.0; // 8.33 ms
        double target_240fps_budget = 1000.0 / 240.0; // 4.17 ms
        
        std::cout << "\nPerformance Target Analysis:" << std::endl;
        std::cout << "  120 FPS target (8.33 ms budget):" << std::endl;
        
        if (p95_frame_time <= target_120fps_budget) {
            std::cout << "    ✅ ACHIEVED - 95% of frames meet 120 FPS target" << std::endl;
        } else {
            std::cout << "    ❌ MISSED - 95th percentile: " << p95_frame_time << " ms" << std::endl;
        }
        
        std::cout << "  240 FPS stretch target (4.17 ms budget):" << std::endl;
        if (p95_frame_time <= target_240fps_budget) {
            std::cout << "    ✅ ACHIEVED - System capable of 240+ FPS" << std::endl;
        } else {
            std::cout << "    ❌ MISSED - Average: " << avg_frame_time << " ms" << std::endl;
        }
        
        // Recommendations
        std::cout << "\nRecommendations:" << std::endl;
        if (avg_fps >= 120) {
            std::cout << "  ✅ System ready for production robotics use" << std::endl;
            std::cout << "  ✅ Sufficient performance margin for real-world conditions" << std::endl;
        } else if (avg_fps >= 60) {
            std::cout << "  ⚠️  Acceptable for development, optimization recommended for production" << std::endl;
        } else {
            std::cout << "  ❌ Performance insufficient - significant optimization needed" << std::endl;
        }
        
        // Efficiency analysis
        double total_actual_fps = frame_times.size() / total_time_seconds;
        std::cout << "  Overall simulation FPS: " << total_actual_fps << std::endl;
        
        if (collision_frames > 0) {
            double collision_overhead = (total_collision_time / collision_frames) / avg_frame_time * 100.0;
            std::cout << "  Collision detection overhead: " << collision_overhead << "% of frame time" << std::endl;
        }
    }
};

/**
 * Main function - demonstrates complete collision detection system
 */
int main(int argc, char* argv[]) {
    try {
        // Check command line arguments
        std::string collision_data_path = "collision_data.h5";
        if (argc > 1) {
            collision_data_path = argv[1];
        }
        
        std::cout << "Phase 2 Collision Detection Engine Demo" << std::endl;
        std::cout << "Using collision data: " << collision_data_path << std::endl;
        std::cout << std::endl;
        
        // Create and initialize demo
        RoboticsCollisionDemo demo;
        
        if (!demo.initialize(collision_data_path)) {
            std::cerr << "❌ Failed to initialize collision demo" << std::endl;
            return 1;
        }
        
        // Run real-time simulation
        demo.run_realtime_demo();
        
        // Demonstrate hierarchical benefits
        demo.demonstrate_hierarchical_benefits();
        
        // Show final statistics
        demo.show_final_statistics();
        
        std::cout << "\n" << "="*60 << std::endl;
        std::cout << "DEMO COMPLETED SUCCESSFULLY" << std::endl;
        std::cout << "="*60 << std::endl;
        std::cout << "Phase 2 collision detection engine is ready for integration" << std::endl;
        std::cout << "with robot control systems and path planners." << std::endl;
        std::cout << std::endl;
        std::cout << "Next steps:" << std::endl;
        std::cout << "1. Integrate with your robot control system" << std::endl;
        std::cout << "2. Connect to STAR model for real human pose data" << std::endl;
        std::cout << "3. Implement force feedback based on collision normals/depths" << std::endl;
        std::cout << "4. Add safety protocols based on collision detection results" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Demo failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "❌ Demo failed with unknown exception" << std::endl;
        return 1;
    }
}

/**
 * Alternative simple usage example
 */
void simple_usage_example() {
    // 1. Initialize collision engine
    std::vector<Eigen::Vector3d> star_vertices; // From STAR model
    auto engine = create_collision_engine("collision_data.h5", star_vertices);
    
    if (!engine) {
        std::cerr << "Failed to initialize collision engine" << std::endl;
        return;
    }
    
    // 2. Configure for your use case
    engine->configure(3, 10, 1e-6); // 3 frame cooldown, 10 max contacts, 1e-6 tolerance
    
    // 3. In your control loop:
    while (true) {
        // Get current human pose (from STAR + camera tracking)
        std::vector<Eigen::Vector3d> bone_positions(24); // Current human bone positions
        
        // Get current robot configuration
        std::vector<Eigen::Vector3d> robot_s_points;     // Current robot S-points
        
        // Convert robot to capsules
        CapsuleCreationBlock capsule_creator;
        auto capsule_result = capsule_creator.create_capsule_chain(robot_s_points, 0.05);
        
        if (capsule_result.success) {
            // MAIN COLLISION DETECTION CALL
            auto collision_result = engine->detect_collisions(bone_positions, capsule_result.capsules);
            
            // Process results
            if (collision_result.has_collision) {
                // Send collision data to robot controller
                for (const auto& contact : collision_result.contacts) {
                    // Apply force feedback: contact.penetration_depth, contact.surface_normal
                    // Modify robot path to avoid collision
                }
            }
        }
        
        // Continue control loop...
    }
}

/**
 * CMakeLists.txt content for building this system:
 * 
 * cmake_minimum_required(VERSION 3.12)
 * project(CollisionDetectionEngine)
 * 
 * set(CMAKE_CXX_STANDARD 17)
 * set(CMAKE_CXX_STANDARD_REQUIRED ON)
 * 
 * # Find required packages
 * find_package(Eigen3 REQUIRED)
 * find_package(HDF5 REQUIRED COMPONENTS CXX)
 * find_package(Threads REQUIRED)
 * 
 * # Source files
 * set(SOURCES
 *     layer_manager.cpp
 *     mesh_collision.cpp
 *     collision_detection_engine.cpp
 *     capsule_creation_block.cpp
 *     collision_example.cpp
 * )
 * 
 * # Create executable
 * add_executable(collision_demo ${SOURCES})
 * 
 * # Link libraries
 * target_link_libraries(collision_demo 
 *     Eigen3::Eigen
 *     ${HDF5_CXX_LIBRARIES}
 *     Threads::Threads
 * )
 * 
 * # Include directories
 * target_include_directories(collision_demo PRIVATE 
 *     ${HDF5_INCLUDE_DIRS}
 * )
 * 
 * # Compiler flags for performance
 * target_compile_options(collision_demo PRIVATE
 *     -O3 -march=native -DNDEBUG
 * )
 */