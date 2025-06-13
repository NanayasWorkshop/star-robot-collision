#include "collision_detection_engine.hpp"
#include "capsule_creation_block.hpp" // For CapsuleData
#include <iostream>
#include <algorithm>
#include <execution>
#include <numeric>

namespace delta {

CollisionDetectionEngine::CollisionDetectionEngine() 
    : initialized_(false), use_parallel_processing_(true), num_threads_(0),
      total_computation_time_ms_(0.0), stage1_time_ms_(0.0), stage2_time_ms_(0.0),
      stage3_time_ms_(0.0), stage4_time_ms_(0.0), total_frame_count_(0),
      collision_frame_count_(0), enable_stage2_(true), enable_stage3_(true), 
      enable_stage4_(true) {
    
    layer_manager_ = std::make_unique<LayerManager>();
    mesh_detector_ = std::make_unique<MeshCollisionDetector>();
}

CollisionDetectionEngine::~CollisionDetectionEngine() {
}

// =============================================================================
// INITIALIZATION AND CONFIGURATION
// =============================================================================

bool CollisionDetectionEngine::initialize(const std::string& hdf5_filepath,
                                          const std::vector<Eigen::Vector3d>& base_vertices,
                                          bool use_parallel,
                                          int num_threads) {
    
    std::cout << "Initializing Collision Detection Engine (Clean Selective Branch Loading)..." << std::endl;
    
    // Set threading configuration
    use_parallel_processing_ = use_parallel;
    num_threads_ = (num_threads <= 0) ? get_optimal_thread_count() : num_threads;
    
    std::cout << "  Parallel processing: " << (use_parallel_processing_ ? "enabled" : "disabled") << std::endl;
    std::cout << "  Thread count: " << num_threads_ << std::endl;
    
    // Initialize layer manager
    if (!layer_manager_->load_hdf5_data(hdf5_filepath)) {
        std::cerr << "❌ Failed to load HDF5 collision data" << std::endl;
        return false;
    }
    
    if (!layer_manager_->initialize_base_mesh(base_vertices)) {
        std::cerr << "❌ Failed to initialize base mesh" << std::endl;
        return false;
    }
    
    // Configure components with default settings
    layer_manager_->set_configuration(3); // 3 frame cooldown
    mesh_detector_->set_parameters(1e-6, 10, true); // Default mesh collision settings
    
    initialized_ = true;
    
    std::cout << "✅ Collision Detection Engine initialized successfully" << std::endl;
    std::cout << "  Architecture: Clean selective branch loading for maximum efficiency" << std::endl;
    std::cout << "  Target: 120+ FPS with adaptive LOD collision detection" << std::endl;
    
    return true;
}

void CollisionDetectionEngine::configure(int layer_cooldown_frames,
                                        int max_contacts_per_capsule,
                                        double penetration_tolerance) {
    
    if (!initialized_) {
        std::cerr << "⚠️  Engine not initialized - configuration ignored" << std::endl;
        return;
    }
    
    layer_manager_->set_configuration(layer_cooldown_frames);
    mesh_detector_->set_parameters(penetration_tolerance, max_contacts_per_capsule, true);
    
    std::cout << "✅ Engine configured for clean selective branch loading:" << std::endl;
    std::cout << "  Layer cooldown: " << layer_cooldown_frames << " frames" << std::endl;
    std::cout << "  Max contacts per capsule: " << max_contacts_per_capsule << std::endl;
    std::cout << "  Penetration tolerance: " << penetration_tolerance << std::endl;
}

// =============================================================================
// MAIN COLLISION DETECTION INTERFACE
// =============================================================================

CollisionResult CollisionDetectionEngine::detect_collisions(
    const std::vector<Eigen::Vector3d>& bone_positions,
    const std::vector<CapsuleData>& robot_capsules) {
    
    ScopedTimer total_timer(total_computation_time_ms_);
    
    CollisionResult result;
    total_frame_count_++;
    
    // Validate inputs
    if (!initialized_) {
        result.debug_info = "Engine not initialized";
        return result;
    }
    
    if (!validate_inputs(bone_positions, robot_capsules)) {
        result.debug_info = "Invalid input parameters";
        return result;
    }
    
    // Advance frame and update human pose
    layer_manager_->advance_frame();
    
    if (!layer_manager_->update_human_pose(bone_positions)) {
        result.debug_info = "Failed to update human pose";
        return result;
    }
    
    // Reset stage timers
    stage1_time_ms_ = 0.0;
    stage2_time_ms_ = 0.0;
    stage3_time_ms_ = 0.0;
    stage4_time_ms_ = 0.0;
    
    std::cout << "\n🔍 CLEAN SELECTIVE BRANCH COLLISION DETECTION (Frame " << total_frame_count_ << ")" << std::endl;
    std::cout << "Robot capsules: " << robot_capsules.size() << std::endl;
    
    // Execute 4-stage selective collision pipeline
    
    // Stage 1: Robot vs Layer 3 (always executed - coarse detection)
    std::vector<int> layer3_hits = execute_stage1_layer3_collision(robot_capsules, result);
    
    std::vector<int> layer2_hits;
    std::vector<int> layer1_hits;
    
    // Stage 2: Robot vs SELECTIVE Layer 2 (only hit branches)
    if (enable_stage2_ && !layer3_hits.empty()) {
        layer2_hits = execute_stage2_selective_layer2_collision(robot_capsules, layer3_hits, result);
        
        // Stage 3: Robot vs SELECTIVE Layer 1 (only hit branches)
        if (enable_stage3_ && !layer2_hits.empty()) {
            layer1_hits = execute_stage3_selective_layer1_collision(robot_capsules, layer2_hits, result);
            
            // Stage 4: Robot vs SELECTIVE Layer 0 (only hit vertices)
            if (enable_stage4_ && !layer1_hits.empty()) {
                execute_stage4_selective_layer0_collision(robot_capsules, layer1_hits, result);
            }
        }
    }
    
    // Cool down unused layers
    layer_manager_->cool_down_unused_layers();
    
    // Update result statistics
    update_collision_statistics(result);
    
    // Track collision frames
    if (result.has_collision) {
        collision_frame_count_++;
    }
    
    // Show selective loading efficiency
    auto branch_stats = layer_manager_->get_collision_branch_statistics();
    std::cout << "💡 Efficiency: " << std::fixed << std::setprecision(1) << branch_stats.selectivity_ratio 
              << "% system active, " << branch_stats.memory_efficiency << "% memory saved" << std::endl;
    
    return result;
}

// =============================================================================
// SELECTIVE COLLISION PIPELINE STAGES
// =============================================================================

std::vector<int> CollisionDetectionEngine::execute_stage1_layer3_collision(
    const std::vector<CapsuleData>& robot_capsules,
    CollisionResult& result) {
    
    ScopedTimer timer(stage1_time_ms_);
    
    std::cout << "🔍 Stage 1: Testing robot vs Layer 3 (9 simple capsules)..." << std::endl;
    
    const auto& layer3_primitives = layer_manager_->get_layer3_primitives();
    result.layer3_tests = static_cast<int>(layer3_primitives.size());
    
    std::vector<bool> collision_flags;
    
    // Execute collision tests (parallel or sequential)
    if (use_parallel_processing_ && robot_capsules.size() > 1) {
        collision_flags = parallel_stage1_execution(robot_capsules, layer3_primitives);
    } else {
        // Sequential execution with detailed logging
        collision_flags.resize(layer3_primitives.size(), false);
        
        for (size_t primitive_idx = 0; primitive_idx < layer3_primitives.size(); ++primitive_idx) {
            const auto& primitive = layer3_primitives[primitive_idx];
            
            bool primitive_hit = false;
            for (size_t capsule_idx = 0; capsule_idx < robot_capsules.size(); ++capsule_idx) {
                const auto& capsule = robot_capsules[capsule_idx];
                
                if (test_capsule_vs_layer3(capsule, primitive)) {
                    primitive_hit = true;
                    std::cout << "  ✓ HIT: Robot capsule[" << capsule_idx << "] vs Layer3[" 
                              << primitive_idx << "] (" << primitive.name << ")" << std::endl;
                    break; // Found collision, no need to test other capsules against this primitive
                }
            }
            collision_flags[primitive_idx] = primitive_hit;
        }
    }
    
    // Collect indices of primitives with collisions
    std::vector<int> hit_indices;
    for (size_t i = 0; i < collision_flags.size(); ++i) {
        if (collision_flags[i]) {
            hit_indices.push_back(static_cast<int>(i));
        }
    }
    
    // Set basic collision flag
    if (!hit_indices.empty()) {
        result.has_collision = true;
        std::cout << "  → " << hit_indices.size() << "/" << layer3_primitives.size() 
                  << " Layer 3 primitives hit" << std::endl;
    } else {
        std::cout << "  → No Layer 3 collisions detected" << std::endl;
    }
    
    return hit_indices;
}

std::vector<int> CollisionDetectionEngine::execute_stage2_selective_layer2_collision(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<int>& layer3_hits,
    CollisionResult& result) {
    
    ScopedTimer timer(stage2_time_ms_);
    
    std::cout << "🔍 Stage 2: Selective Layer 2 loading..." << std::endl;
    
    // SELECTIVE ACTIVATION: Only activate Layer 2 children of hit Layer 3 primitives
    layer_manager_->activate_layer2_primitives(layer3_hits);
    
    // Get ONLY the activated Layer 2 primitives (not all 23!)
    auto active_layer2_primitives = layer_manager_->get_active_layer2_primitives();
    result.layer2_activations = static_cast<int>(active_layer2_primitives.size());
    
    if (active_layer2_primitives.empty()) {
        std::cout << "  → No Layer 2 primitives activated" << std::endl;
        return {};
    }
    
    std::cout << "  Testing robot vs " << active_layer2_primitives.size() 
              << " activated Layer 2 primitives..." << std::endl;
    
    std::vector<bool> collision_flags;
    
    // Execute collision tests on ONLY the activated Layer 2 primitives
    if (use_parallel_processing_ && robot_capsules.size() > 1) {
        collision_flags = parallel_stage2_execution(robot_capsules, active_layer2_primitives);
    } else {
        // Sequential execution with detailed logging
        collision_flags.resize(active_layer2_primitives.size(), false);
        
        for (size_t primitive_idx = 0; primitive_idx < active_layer2_primitives.size(); ++primitive_idx) {
            const auto& primitive = active_layer2_primitives[primitive_idx];
            
            bool primitive_hit = false;
            for (size_t capsule_idx = 0; capsule_idx < robot_capsules.size(); ++capsule_idx) {
                const auto& capsule = robot_capsules[capsule_idx];
                
                if (test_capsule_vs_layer2(capsule, primitive)) {
                    primitive_hit = true;
                    std::cout << "    ✓ HIT: Robot capsule[" << capsule_idx << "] vs Layer2 " 
                              << primitive.name << std::endl;
                    break;
                }
            }
            collision_flags[primitive_idx] = primitive_hit;
        }
    }
    
    // Collect hit indices - need to map back to global Layer 2 indices
    std::vector<int> hit_indices;
    const auto& all_layer2 = layer_manager_->get_layer_states().layer2_primitives;
    
    for (size_t i = 0; i < collision_flags.size(); ++i) {
        if (collision_flags[i]) {
            const auto& hit_primitive = active_layer2_primitives[i];
            
            // Find global index of this Layer 2 primitive
            for (size_t global_idx = 0; global_idx < all_layer2.size(); ++global_idx) {
                if (all_layer2[global_idx].name == hit_primitive.name) {
                    hit_indices.push_back(static_cast<int>(global_idx));
                    break;
                }
            }
        }
    }
    
    if (!hit_indices.empty()) {
        std::cout << "  → " << hit_indices.size() << "/" << active_layer2_primitives.size() 
                  << " Layer 2 primitives hit (selective)" << std::endl;
    } else {
        std::cout << "  → No Layer 2 collisions detected" << std::endl;
    }
    
    return hit_indices;
}

std::vector<int> CollisionDetectionEngine::execute_stage3_selective_layer1_collision(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<int>& layer2_hits,
    CollisionResult& result) {
    
    ScopedTimer timer(stage3_time_ms_);
    
    std::cout << "🔍 Stage 3: Selective Layer 1 loading..." << std::endl;
    
    // SELECTIVE ACTIVATION: Only activate Layer 1 children of hit Layer 2 primitives
    layer_manager_->activate_layer1_primitives(layer2_hits);
    
    // Get ONLY the activated Layer 1 primitives (not all 76!)
    auto active_layer1_primitives = layer_manager_->get_active_layer1_primitives();
    result.layer1_activations = static_cast<int>(active_layer1_primitives.size());
    
    if (active_layer1_primitives.empty()) {
        std::cout << "  → No Layer 1 primitives activated" << std::endl;
        return {};
    }
    
    std::cout << "  Testing robot vs " << active_layer1_primitives.size() 
              << " activated Layer 1 spheres..." << std::endl;
    
    std::vector<bool> collision_flags;
    
    // Execute collision tests on ONLY the activated Layer 1 primitives
    if (use_parallel_processing_ && robot_capsules.size() > 1) {
        collision_flags = parallel_stage3_execution(robot_capsules, active_layer1_primitives);
    } else {
        // Sequential execution with detailed logging
        collision_flags.resize(active_layer1_primitives.size(), false);
        
        for (size_t primitive_idx = 0; primitive_idx < active_layer1_primitives.size(); ++primitive_idx) {
            const auto& primitive = active_layer1_primitives[primitive_idx];
            
            bool primitive_hit = false;
            for (size_t capsule_idx = 0; capsule_idx < robot_capsules.size(); ++capsule_idx) {
                const auto& capsule = robot_capsules[capsule_idx];
                
                if (test_capsule_vs_layer1(capsule, primitive)) {
                    primitive_hit = true;
                    std::cout << "    ✓ HIT: Robot capsule[" << capsule_idx << "] vs Layer1 " 
                              << primitive.name << std::endl;
                    break;
                }
            }
            collision_flags[primitive_idx] = primitive_hit;
        }
    }
    
    // Collect hit indices - map back to global Layer 1 indices
    std::vector<int> hit_indices;
    const auto& all_layer1 = layer_manager_->get_layer_states().layer1_primitives;
    
    for (size_t i = 0; i < collision_flags.size(); ++i) {
        if (collision_flags[i]) {
            const auto& hit_primitive = active_layer1_primitives[i];
            
            // Find global index of this Layer 1 primitive
            for (size_t global_idx = 0; global_idx < all_layer1.size(); ++global_idx) {
                if (all_layer1[global_idx].name == hit_primitive.name) {
                    hit_indices.push_back(static_cast<int>(global_idx));
                    break;
                }
            }
        }
    }
    
    if (!hit_indices.empty()) {
        std::cout << "  → " << hit_indices.size() << "/" << active_layer1_primitives.size() 
                  << " Layer 1 spheres hit (selective)" << std::endl;
    } else {
        std::cout << "  → No Layer 1 collisions detected" << std::endl;
    }
    
    return hit_indices;
}

void CollisionDetectionEngine::execute_stage4_selective_layer0_collision(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<int>& layer1_hits,
    CollisionResult& result) {
    
    ScopedTimer timer(stage4_time_ms_);
    
    std::cout << "🔍 Stage 4: Selective Layer 0 vertex loading..." << std::endl;
    
    // SELECTIVE LOADING: Only load vertices for hit Layer 1 spheres
    layer_manager_->load_layer0_vertices(layer1_hits);
    
    // Get ONLY the loaded vertex groups (not all vertices!)
    auto loaded_vertex_groups = layer_manager_->get_loaded_layer0_vertices();
    result.layer0_activations = static_cast<int>(loaded_vertex_groups.size());
    
    if (loaded_vertex_groups.empty()) {
        std::cout << "  → No Layer 0 vertices loaded" << std::endl;
        return;
    }
    
    int total_vertices = 0;
    for (const auto& vertex_group : loaded_vertex_groups) {
        total_vertices += static_cast<int>(vertex_group.vertices.size());
    }
    
    std::cout << "  Testing robot vs " << total_vertices 
              << " loaded vertices (from " << loaded_vertex_groups.size() << " groups)..." << std::endl;
    
    // Test each vertex group against all robot capsules for precise collision
    for (const auto& vertex_group : loaded_vertex_groups) {
        if (!vertex_group.is_loaded || vertex_group.vertices.empty()) {
            continue;
        }
        
        // Use mesh collision detector for precise depth and normal calculation
        auto contacts = mesh_detector_->detect_multi_capsule_mesh_collision(
            robot_capsules, vertex_group.vertices, vertex_group.triangles);
        
        // Log precise contacts found
        if (!contacts.empty()) {
            std::cout << "    ✓ PRECISE: Found " << contacts.size() 
                      << " contacts in vertex group (parent Layer1[" 
                      << vertex_group.parent_layer1_index << "])" << std::endl;
            
            for (const auto& contact : contacts) {
                std::cout << "      - Depth: " << std::fixed << std::setprecision(4) 
                          << contact.penetration_depth << ", Normal: (" 
                          << contact.surface_normal.transpose() << ")" << std::endl;
            }
        }
        
        // Add contacts to result
        result.contacts.insert(result.contacts.end(), contacts.begin(), contacts.end());
        
        // Update maximum penetration depth
        for (const auto& contact : contacts) {
            result.max_penetration_depth = std::max(result.max_penetration_depth, 
                                                   contact.penetration_depth);
        }
    }
    
    // Update collision flag if we found precise contacts
    if (!result.contacts.empty()) {
        result.has_collision = true;
        std::cout << "  → " << result.contacts.size() 
                  << " precise contacts found, max depth: " << std::fixed << std::setprecision(4)
                  << result.max_penetration_depth << std::endl;
    } else {
        std::cout << "  → No precise contacts detected" << std::endl;
    }
}

// =============================================================================
// PARALLEL PROCESSING HELPERS (unchanged)
// =============================================================================

bool CollisionDetectionEngine::test_capsule_vs_layer3(const CapsuleData& capsule,
                                                      const Layer3Primitive& primitive) const {
    
    return capsule_vs_capsule_collision(
        capsule.start_point, capsule.end_point, capsule.radius,
        primitive.start_point, primitive.end_point, primitive.radius);
}

bool CollisionDetectionEngine::test_capsule_vs_layer2(const CapsuleData& capsule,
                                                      const Layer2Primitive& primitive) const {
    
    return capsule_vs_capsule_collision(
        capsule.start_point, capsule.end_point, capsule.radius,
        primitive.start_point, primitive.end_point, primitive.radius);
}

bool CollisionDetectionEngine::test_capsule_vs_layer1(const CapsuleData& capsule,
                                                      const Layer1Primitive& primitive) const {
    
    return capsule_vs_sphere_collision(
        capsule.start_point, capsule.end_point, capsule.radius,
        primitive.center, primitive.radius);
}

std::vector<bool> CollisionDetectionEngine::parallel_stage1_execution(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<Layer3Primitive>& layer3_primitives) {
    
    std::vector<bool> results(layer3_primitives.size(), false);
    
    // Use std::for_each with execution policy for parallel processing
    std::for_each(std::execution::par_unseq,
                  layer3_primitives.begin(), layer3_primitives.end(),
                  [&](const Layer3Primitive& primitive) {
                      size_t primitive_idx = &primitive - &layer3_primitives[0];
                      
                      for (const auto& capsule : robot_capsules) {
                          if (test_capsule_vs_layer3(capsule, primitive)) {
                              results[primitive_idx] = true;
                              break;
                          }
                      }
                  });
    
    return results;
}

std::vector<bool> CollisionDetectionEngine::parallel_stage2_execution(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<Layer2Primitive>& layer2_primitives) {
    
    std::vector<bool> results(layer2_primitives.size(), false);
    
    std::for_each(std::execution::par_unseq,
                  layer2_primitives.begin(), layer2_primitives.end(),
                  [&](const Layer2Primitive& primitive) {
                      size_t primitive_idx = &primitive - &layer2_primitives[0];
                      
                      for (const auto& capsule : robot_capsules) {
                          if (test_capsule_vs_layer2(capsule, primitive)) {
                              results[primitive_idx] = true;
                              break;
                          }
                      }
                  });
    
    return results;
}

std::vector<bool> CollisionDetectionEngine::parallel_stage3_execution(
    const std::vector<CapsuleData>& robot_capsules,
    const std::vector<Layer1Primitive>& layer1_primitives) {
    
    std::vector<bool> results(layer1_primitives.size(), false);
    
    std::for_each(std::execution::par_unseq,
                  layer1_primitives.begin(), layer1_primitives.end(),
                  [&](const Layer1Primitive& primitive) {
                      size_t primitive_idx = &primitive - &layer1_primitives[0];
                      
                      for (const auto& capsule : robot_capsules) {
                          if (test_capsule_vs_layer1(capsule, primitive)) {
                              results[primitive_idx] = true;
                              break;
                          }
                      }
                  });
    
    return results;
}

// =============================================================================
// PERFORMANCE AND DIAGNOSTICS
// =============================================================================

CollisionDetectionEngine::PerformanceBreakdown CollisionDetectionEngine::get_performance_breakdown() const {
    PerformanceBreakdown breakdown;
    
    // Timing information
    breakdown.total_time_ms = total_computation_time_ms_;
    breakdown.stage1_time_ms = stage1_time_ms_;
    breakdown.stage2_time_ms = stage2_time_ms_;
    breakdown.stage3_time_ms = stage3_time_ms_;
    breakdown.stage4_time_ms = stage4_time_ms_;
    
    // Get layer manager timings
    layer_manager_->get_performance_timings(breakdown.layer_update_ms, breakdown.vertex_loading_ms);
    
    // Frame statistics
    breakdown.total_frames = total_frame_count_;
    breakdown.collision_frames = collision_frame_count_;
    breakdown.collision_rate = (total_frame_count_ > 0) ? 
        (static_cast<double>(collision_frame_count_) / total_frame_count_ * 100.0) : 0.0;
    
    // Clean branch statistics (no more global arrays)
    auto branch_stats = layer_manager_->get_collision_branch_statistics();
    breakdown.active_layer2_count = branch_stats.active_layer2_primitives;
    breakdown.active_layer1_count = branch_stats.active_layer1_primitives;
    breakdown.loaded_layer0_count = branch_stats.loaded_vertices;
    breakdown.memory_usage_mb = layer_manager_->get_layer_statistics().memory_usage_mb;
    
    return breakdown;
}

void CollisionDetectionEngine::reset_performance_statistics() {
    total_computation_time_ms_ = 0.0;
    stage1_time_ms_ = 0.0;
    stage2_time_ms_ = 0.0;
    stage3_time_ms_ = 0.0;
    stage4_time_ms_ = 0.0;
    total_frame_count_ = 0;
    collision_frame_count_ = 0;
    
    mesh_detector_->reset_performance_timings();
}

LayerManager::LayerStats CollisionDetectionEngine::get_layer_statistics() const {
    if (!initialized_) {
        return LayerManager::LayerStats{};
    }
    
    return layer_manager_->get_layer_statistics();
}

std::string CollisionDetectionEngine::get_debug_info() const {
    if (!initialized_) {
        return "CollisionDetectionEngine: Not initialized";
    }
    
    auto breakdown = get_performance_breakdown();
    auto branch_stats = layer_manager_->get_collision_branch_statistics();
    
    std::string debug_info = "CollisionDetectionEngine Debug Info (Clean Selective Branch Loading):\n";
    debug_info += "  Status: " + std::string(initialized_ ? "Initialized" : "Not initialized") + "\n";
    debug_info += "  Parallel processing: " + std::string(use_parallel_processing_ ? "enabled" : "disabled") + "\n";
    debug_info += "  Thread count: " + std::to_string(num_threads_) + "\n";
    debug_info += "\nClean Selective Loading Efficiency:\n";
    debug_info += "  Overall selectivity: " + std::to_string(branch_stats.selectivity_ratio) + "% of system active\n";
    debug_info += "  Memory efficiency: " + std::to_string(branch_stats.memory_efficiency) + "% memory saved\n";
    debug_info += "  Active branches L2: " + std::to_string(branch_stats.total_active_branches_layer2) + "\n";
    debug_info += "  Active branches L1: " + std::to_string(branch_stats.total_active_branches_layer1) + "\n";
    debug_info += "  Active branches L0: " + std::to_string(branch_stats.total_active_branches_layer0) + "\n";
    debug_info += "  Active Layer 2: " + std::to_string(branch_stats.active_layer2_primitives) + "/23\n";
    debug_info += "  Active Layer 1: " + std::to_string(branch_stats.active_layer1_primitives) + "/76\n";
    debug_info += "  Loaded vertices: " + std::to_string(branch_stats.loaded_vertices) + "/6890\n";
    debug_info += "\nTiming Breakdown (last frame):\n";
    debug_info += "  Total: " + std::to_string(breakdown.total_time_ms) + " ms\n";
    debug_info += "  Stage 1 (Layer 3): " + std::to_string(breakdown.stage1_time_ms) + " ms\n";
    debug_info += "  Stage 2 (Selective Layer 2): " + std::to_string(breakdown.stage2_time_ms) + " ms\n";
    debug_info += "  Stage 3 (Selective Layer 1): " + std::to_string(breakdown.stage3_time_ms) + " ms\n";
    debug_info += "  Stage 4 (Selective Layer 0): " + std::to_string(breakdown.stage4_time_ms) + " ms\n";
    debug_info += "  Layer update: " + std::to_string(breakdown.layer_update_ms) + " ms\n";
    debug_info += "  Vertex loading: " + std::to_string(breakdown.vertex_loading_ms) + " ms\n";
    debug_info += "\nFrame Statistics:\n";
    debug_info += "  Total frames: " + std::to_string(breakdown.total_frames) + "\n";
    debug_info += "  Collision frames: " + std::to_string(breakdown.collision_frames) + "\n";
    debug_info += "  Collision rate: " + std::to_string(breakdown.collision_rate) + "%\n";
    debug_info += "  Memory usage: " + std::to_string(breakdown.memory_usage_mb) + " MB\n";
    
    return debug_info;
}

// =============================================================================
// ADVANCED CONFIGURATION (Optional)
// =============================================================================

void CollisionDetectionEngine::set_pipeline_stages(bool enable_layer2,
                                                   bool enable_layer1,
                                                   bool enable_layer0) {
    enable_stage2_ = enable_layer2;
    enable_stage3_ = enable_layer1;
    enable_stage4_ = enable_layer0;
    
    std::cout << "Clean pipeline stages configured:" << std::endl;
    std::cout << "  Layer 2 (clean selective): " << (enable_stage2_ ? "enabled" : "disabled") << std::endl;
    std::cout << "  Layer 1 (clean selective): " << (enable_stage3_ ? "enabled" : "disabled") << std::endl;
    std::cout << "  Layer 0 (clean selective): " << (enable_stage4_ ? "enabled" : "disabled") << std::endl;
}

void CollisionDetectionEngine::force_activate_layers(const std::vector<int>& layer2_indices,
                                                     const std::vector<int>& layer1_indices) {
    if (!initialized_) {
        std::cerr << "⚠️  Engine not initialized - cannot force activate layers" << std::endl;
        return;
    }
    
    if (!layer2_indices.empty()) {
        layer_manager_->activate_layer2_primitives(layer2_indices);
        std::cout << "Force activated " << layer2_indices.size() << " Layer 2 primitives (clean selective)" << std::endl;
    }
    
    if (!layer1_indices.empty()) {
        layer_manager_->activate_layer1_primitives(layer1_indices);
        std::cout << "Force activated " << layer1_indices.size() << " Layer 1 primitives (clean selective)" << std::endl;
    }
}

// =============================================================================
// UTILITY AND HELPER METHODS
// =============================================================================

bool CollisionDetectionEngine::validate_inputs(const std::vector<Eigen::Vector3d>& bone_positions,
                                               const std::vector<CapsuleData>& robot_capsules) const {
    
    // Check bone positions
    if (bone_positions.size() != 24) {
        std::cerr << "❌ Invalid bone positions: expected 24, got " << bone_positions.size() << std::endl;
        return false;
    }
    
    // Check for finite values in bone positions
    for (size_t i = 0; i < bone_positions.size(); ++i) {
        if (!bone_positions[i].allFinite()) {
            std::cerr << "❌ Invalid bone position at index " << i << ": contains non-finite values" << std::endl;
            return false;
        }
    }
    
    // Check robot capsules
    if (robot_capsules.empty()) {
        std::cerr << "❌ No robot capsules provided" << std::endl;
        return false;
    }
    
    // Validate each robot capsule
    for (size_t i = 0; i < robot_capsules.size(); ++i) {
        const auto& capsule = robot_capsules[i];
        
        if (!capsule.start_point.allFinite() || !capsule.end_point.allFinite()) {
            std::cerr << "❌ Invalid robot capsule at index " << i << ": contains non-finite positions" << std::endl;
            return false;
        }
        
        if (capsule.radius <= 0.0 || !std::isfinite(capsule.radius)) {
            std::cerr << "❌ Invalid robot capsule at index " << i << ": invalid radius " << capsule.radius << std::endl;
            return false;
        }
        
        if (capsule.length < 0.0 || !std::isfinite(capsule.length)) {
            std::cerr << "❌ Invalid robot capsule at index " << i << ": invalid length " << capsule.length << std::endl;
            return false;
        }
    }
    
    return true;
}

void CollisionDetectionEngine::update_collision_statistics(CollisionResult& result) const {
    // Set computation time
    result.computation_time_ms = total_computation_time_ms_;
    
    // Update debug information with clean selective loading stats
    auto branch_stats = layer_manager_->get_collision_branch_statistics();
    
    if (result.contacts.empty()) {
        result.debug_info = "No collisions detected - " + 
                           std::to_string(branch_stats.selectivity_ratio) + "% system tested (clean)";
    } else {
        result.debug_info = "Found " + std::to_string(result.contacts.size()) + " collision contacts, " +
                           "max depth: " + std::to_string(result.max_penetration_depth) + 
                           " (clean selective: " + std::to_string(branch_stats.selectivity_ratio) + "% active)";
    }
    
    // Ensure has_collision flag is consistent with contacts
    result.has_collision = !result.contacts.empty();
}

int CollisionDetectionEngine::get_optimal_thread_count() const {
    // Get hardware concurrency
    int hardware_threads = static_cast<int>(std::thread::hardware_concurrency());
    
    if (hardware_threads <= 0) {
        // Fallback if hardware_concurrency() fails
        return 4;
    }
    
    // Use all available threads for collision detection
    // Selective loading makes threading even more effective
    return hardware_threads;
}

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

std::unique_ptr<CollisionDetectionEngine> create_collision_engine(
    const std::string& hdf5_filepath,
    const std::vector<Eigen::Vector3d>& base_vertices) {
    
    auto engine = std::make_unique<CollisionDetectionEngine>();
    
    if (!engine->initialize(hdf5_filepath, base_vertices)) {
        std::cerr << "❌ Failed to create clean selective collision detection engine" << std::endl;
        return nullptr;
    }
    
    return engine;
}

bool quick_collision_test(CollisionDetectionEngine& engine,
                         const std::vector<Eigen::Vector3d>& bone_positions,
                         const std::vector<CapsuleData>& robot_capsules) {
    
    if (!engine.is_initialized()) {
        std::cerr << "❌ Engine not initialized" << std::endl;
        return false;
    }
    
    auto result = engine.detect_collisions(bone_positions, robot_capsules);
    return result.has_collision;
}

} // namespace delta

// =============================================================================
// UTILITY FUNCTIONS IMPLEMENTATION (from collision_types.hpp)
// =============================================================================

namespace delta {

double point_to_line_distance(const Eigen::Vector3d& point, 
                              const Eigen::Vector3d& line_start,
                              const Eigen::Vector3d& line_end) {
    
    Eigen::Vector3d line_vec = line_end - line_start;
    double line_length = line_vec.norm();
    
    if (line_length < 1e-12) {
        // Degenerate line: distance to start point
        return (point - line_start).norm();
    }
    
    // Normalized line direction
    Eigen::Vector3d line_dir = line_vec / line_length;
    
    // Vector from line start to point
    Eigen::Vector3d point_vec = point - line_start;
    
    // Project point onto line
    double projection_length = point_vec.dot(line_dir);
    projection_length = std::max(0.0, std::min(projection_length, line_length));
    
    // Find closest point on line segment
    Eigen::Vector3d closest_point = line_start + projection_length * line_dir;
    
    // Return distance
    return (point - closest_point).norm();
}

bool point_in_capsule(const Eigen::Vector3d& point,
                     const Eigen::Vector3d& capsule_start,
                     const Eigen::Vector3d& capsule_end,
                     double capsule_radius) {
    
    double distance = point_to_line_distance(point, capsule_start, capsule_end);
    return distance <= capsule_radius;
}

bool capsule_vs_capsule_collision(const Eigen::Vector3d& cap1_start,
                                 const Eigen::Vector3d& cap1_end,
                                 double cap1_radius,
                                 const Eigen::Vector3d& cap2_start,
                                 const Eigen::Vector3d& cap2_end,
                                 double cap2_radius) {
    
    // Simplified capsule vs capsule collision test
    // Test endpoints of each capsule against the other capsule
    
    double threshold = cap1_radius + cap2_radius;
    
    // Test cap1 endpoints against cap2
    if (point_to_line_distance(cap1_start, cap2_start, cap2_end) <= threshold ||
        point_to_line_distance(cap1_end, cap2_start, cap2_end) <= threshold) {
        return true;
    }
    
    // Test cap2 endpoints against cap1
    if (point_to_line_distance(cap2_start, cap1_start, cap1_end) <= threshold ||
        point_to_line_distance(cap2_end, cap1_start, cap1_end) <= threshold) {
        return true;
    }
    
    // Test midpoint distances for more accuracy
    Eigen::Vector3d cap1_mid = (cap1_start + cap1_end) * 0.5;
    Eigen::Vector3d cap2_mid = (cap2_start + cap2_end) * 0.5;
    
    if (point_to_line_distance(cap1_mid, cap2_start, cap2_end) <= threshold ||
        point_to_line_distance(cap2_mid, cap1_start, cap1_end) <= threshold) {
        return true;
    }
    
    return false;
}

bool capsule_vs_sphere_collision(const Eigen::Vector3d& capsule_start,
                                const Eigen::Vector3d& capsule_end,
                                double capsule_radius,
                                const Eigen::Vector3d& sphere_center,
                                double sphere_radius) {
    
    double distance = point_to_line_distance(sphere_center, capsule_start, capsule_end);
    double threshold = capsule_radius + sphere_radius;
    
    return distance <= threshold;
}

} // namespace delta