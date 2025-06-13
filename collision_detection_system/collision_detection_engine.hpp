#pragma once

#include "collision_types.hpp"
#include "layer_manager.hpp"
#include "mesh_collision.hpp"
#include <memory>
#include <thread>
#include <vector>
#include <future>

namespace delta {

// Forward declaration
struct CapsuleData;

/**
 * Main collision detection engine - orchestrates the 4-stage hierarchical pipeline
 * Public interface for the entire collision detection system
 */
class CollisionDetectionEngine {
private:
    // Core components
    std::unique_ptr<LayerManager> layer_manager_;
    std::unique_ptr<MeshCollisionDetector> mesh_detector_;
    
    // Configuration
    bool initialized_;
    bool use_parallel_processing_;
    int num_threads_;
    
    // Performance tracking
    mutable double total_computation_time_ms_;
    mutable double stage1_time_ms_;  // Layer 3 collision time
    mutable double stage2_time_ms_;  // Layer 2 collision time
    mutable double stage3_time_ms_;  // Layer 1 collision time
    mutable double stage4_time_ms_;  // Layer 0 collision time
    
    // Statistics
    mutable int total_frame_count_;
    mutable int collision_frame_count_;
    
public:
    CollisionDetectionEngine();
    ~CollisionDetectionEngine();
    
    // =============================================================================
    // INITIALIZATION AND CONFIGURATION
    // =============================================================================
    
    /**
     * Initialize the collision detection engine
     * @param hdf5_filepath Path to collision data file from Phase 1
     * @param base_vertices T-pose mesh vertices from STAR
     * @param use_parallel Enable parallel processing (default: true)
     * @param num_threads Number of threads to use (default: std::thread::hardware_concurrency())
     * @return true if initialization successful
     */
    bool initialize(const std::string& hdf5_filepath,
                   const std::vector<Eigen::Vector3d>& base_vertices,
                   bool use_parallel = true,
                   int num_threads = 0);
    
    /**
     * Configure collision detection parameters
     * @param layer_cooldown_frames Frames to wait before cooling down layers (default: 3)
     * @param max_contacts_per_capsule Maximum contacts to calculate per robot capsule (default: 10)
     * @param penetration_tolerance Minimum penetration depth to report (default: 1e-6)
     */
    void configure(int layer_cooldown_frames = 3,
                   int max_contacts_per_capsule = 10,
                   double penetration_tolerance = 1e-6);
    
    /**
     * Check if engine is properly initialized
     * @return true if ready for collision detection
     */
    bool is_initialized() const { return initialized_; }
    
    // =============================================================================
    // MAIN COLLISION DETECTION INTERFACE
    // =============================================================================
    
    /**
     * Detect collisions between robot and human - MAIN FUNCTION
     * This is the primary interface for the collision detection system
     * 
     * @param bone_positions Current human bone positions (24 joints)
     * @param robot_capsules Robot represented as chain of capsules
     * @return Complete collision results with depth and normals
     */
    CollisionResult detect_collisions(const std::vector<Eigen::Vector3d>& bone_positions,
                                     const std::vector<CapsuleData>& robot_capsules);
    
    // =============================================================================
    // PERFORMANCE AND DIAGNOSTICS
    // =============================================================================
    
    /**
     * Get detailed performance breakdown
     */
    struct PerformanceBreakdown {
        double total_time_ms;
        double stage1_time_ms;      // Layer 3 collision tests
        double stage2_time_ms;      // Layer 2 collision tests  
        double stage3_time_ms;      // Layer 1 collision tests
        double stage4_time_ms;      // Layer 0 mesh collision tests
        double layer_update_ms;     // Layer transformation time
        double vertex_loading_ms;   // Vertex loading time
        
        int total_frames;
        int collision_frames;
        double collision_rate;      // Percentage of frames with collisions
        
        // Layer activation stats
        int active_layer2_count;
        int active_layer1_count;
        int loaded_layer0_count;
        double memory_usage_mb;
    };
    
    PerformanceBreakdown get_performance_breakdown() const;
    
    /**
     * Reset performance statistics
     */
    void reset_performance_statistics();
    
    /**
     * Get current layer statistics
     */
    LayerManager::LayerStats get_layer_statistics() const;
    
    /**
     * Get debug information string
     * @return Formatted debug information
     */
    std::string get_debug_info() const;
    
    // =============================================================================
    // ADVANCED CONFIGURATION (Optional)
    // =============================================================================
    
    /**
     * Enable/disable specific pipeline stages for testing
     * @param enable_layer2 Enable Layer 2 collision detection
     * @param enable_layer1 Enable Layer 1 collision detection  
     * @param enable_layer0 Enable Layer 0 precise collision detection
     */
    void set_pipeline_stages(bool enable_layer2 = true,
                            bool enable_layer1 = true,
                            bool enable_layer0 = true);
    
    /**
     * Force activation of specific layers (for debugging)
     * @param layer2_indices Layer 2 primitives to activate
     * @param layer1_indices Layer 1 primitives to activate
     */
    void force_activate_layers(const std::vector<int>& layer2_indices = {},
                              const std::vector<int>& layer1_indices = {});

private:
    // Pipeline stage control
    bool enable_stage2_;
    bool enable_stage3_;
    bool enable_stage4_;
    
    // =============================================================================
    // PIPELINE STAGES IMPLEMENTATION
    // =============================================================================
    
    /**
     * Stage 1: Robot capsules vs Layer 3 (9 simple capsules)
     * Always executed - provides coarse collision detection
     * @param robot_capsules Robot capsule chain
     * @param result Collision result to populate
     * @return Indices of Layer 3 primitives with collisions
     */
    std::vector<int> execute_stage1_layer3_collision(
        const std::vector<CapsuleData>& robot_capsules,
        CollisionResult& result);
    
    /**
     * Stage 2: Robot capsules vs active Layer 2 (detailed capsules)
     * Only executed if Stage 1 found collisions
     * @param robot_capsules Robot capsule chain
     * @param layer3_hits Layer 3 primitives that had collisions
     * @param result Collision result to populate
     * @return Indices of Layer 2 primitives with collisions
     */
    std::vector<int> execute_stage2_layer2_collision(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<int>& layer3_hits,
        CollisionResult& result);
    
    /**
     * Stage 3: Robot capsules vs active Layer 1 (spheres)
     * Only executed if Stage 2 found collisions
     * @param robot_capsules Robot capsule chain
     * @param layer2_hits Layer 2 primitives that had collisions
     * @param result Collision result to populate
     * @return Indices of Layer 1 primitives with collisions
     */
    std::vector<int> execute_stage3_layer1_collision(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<int>& layer2_hits,
        CollisionResult& result);
    
    /**
     * Stage 4: Robot capsules vs loaded Layer 0 (mesh vertices)
     * Only executed if Stage 3 found collisions - provides precise depth/normals
     * @param robot_capsules Robot capsule chain
     * @param layer1_hits Layer 1 primitives that had collisions
     * @param result Collision result to populate (adds precise contacts)
     */
    void execute_stage4_layer0_collision(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<int>& layer1_hits,
        CollisionResult& result);
    
    // =============================================================================
    // PARALLEL PROCESSING HELPERS
    // =============================================================================
    
    /**
     * Test robot capsule against Layer 3 primitive
     * @param capsule Robot capsule
     * @param primitive Layer 3 primitive
     * @return true if collision detected
     */
    bool test_capsule_vs_layer3(const CapsuleData& capsule,
                                const Layer3Primitive& primitive) const;
    
    /**
     * Test robot capsule against Layer 2 primitive
     * @param capsule Robot capsule
     * @param primitive Layer 2 primitive
     * @return true if collision detected
     */
    bool test_capsule_vs_layer2(const CapsuleData& capsule,
                                const Layer2Primitive& primitive) const;
    
    /**
     * Test robot capsule against Layer 1 primitive
     * @param capsule Robot capsule
     * @param primitive Layer 1 primitive
     * @return true if collision detected
     */
    bool test_capsule_vs_layer1(const CapsuleData& capsule,
                                const Layer1Primitive& primitive) const;
    
    /**
     * Parallel execution wrapper for Stage 1
     * @param robot_capsules Robot capsules to test
     * @param layer3_primitives Layer 3 primitives to test against
     * @return Hit results for each Layer 3 primitive
     */
    std::vector<bool> parallel_stage1_execution(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<Layer3Primitive>& layer3_primitives);
    
    /**
     * Parallel execution wrapper for Stage 2
     * @param robot_capsules Robot capsules to test
     * @param layer2_primitives Layer 2 primitives to test against
     * @return Hit results for each Layer 2 primitive
     */
    std::vector<bool> parallel_stage2_execution(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<Layer2Primitive>& layer2_primitives);
    
    /**
     * Parallel execution wrapper for Stage 3
     * @param robot_capsules Robot capsules to test
     * @param layer1_primitives Layer 1 primitives to test against
     * @return Hit results for each Layer 1 primitive
     */
    std::vector<bool> parallel_stage3_execution(
        const std::vector<CapsuleData>& robot_capsules,
        const std::vector<Layer1Primitive>& layer1_primitives);
    
    // =============================================================================
    // UTILITY AND HELPER METHODS
    // =============================================================================
    
    /**
     * Validate input parameters
     * @param bone_positions Bone positions to validate
     * @param robot_capsules Robot capsules to validate
     * @return true if inputs are valid
     */
    bool validate_inputs(const std::vector<Eigen::Vector3d>& bone_positions,
                        const std::vector<CapsuleData>& robot_capsules) const;
    
    /**
     * Update collision result statistics
     * @param result Collision result to update
     */
    void update_collision_statistics(CollisionResult& result) const;
    
    /**
     * Get optimal number of threads for current hardware
     * @return Recommended thread count
     */
    int get_optimal_thread_count() const;
    
    /**
     * Thread worker function for parallel collision testing
     * @param robot_capsules Robot capsules to test
     * @param primitives_start Start index of primitives to test
     * @param primitives_end End index of primitives to test
     * @param test_function Function to call for each test
     * @param results Output array for results
     */
    template<typename PrimitiveType, typename TestFunction>
    void collision_worker_thread(const std::vector<CapsuleData>& robot_capsules,
                                 const std::vector<PrimitiveType>& primitives,
                                 int start_idx, int end_idx,
                                 TestFunction test_function,
                                 std::vector<bool>& results);
};

// =============================================================================
// CONVENIENCE FUNCTIONS
// =============================================================================

/**
 * Create and initialize a collision detection engine
 * @param hdf5_filepath Path to Phase 1 collision data
 * @param base_vertices T-pose mesh vertices
 * @return Initialized collision detection engine (nullptr if failed)
 */
std::unique_ptr<CollisionDetectionEngine> create_collision_engine(
    const std::string& hdf5_filepath,
    const std::vector<Eigen::Vector3d>& base_vertices);

/**
 * Quick collision test - simplified interface for basic usage
 * @param engine Initialized collision detection engine
 * @param bone_positions Current bone positions
 * @param robot_capsules Robot capsule chain
 * @return true if any collision detected
 */
bool quick_collision_test(CollisionDetectionEngine& engine,
                         const std::vector<Eigen::Vector3d>& bone_positions,
                         const std::vector<CapsuleData>& robot_capsules);

} // namespace delta