#pragma once

#include "collision_types.hpp"
#include <string>
#include <unordered_map>
#include <memory>

namespace delta {

class LayerManager {
private:
    // HDF5 data loaded from Phase 1
    HierarchyMappings hierarchy_mappings_;
    
    // Layer state management
    LayerStates layer_states_;
    
    // Current human pose
    std::vector<Eigen::Vector3d> current_bone_positions_;
    std::vector<Eigen::Vector3d> base_mesh_vertices_;  // T-pose vertices from STAR
    
    // Configuration
    int cooldown_frames_;                              // Frames to wait before cooling down layers
    bool hdf5_loaded_;
    std::string hdf5_filepath_;
    
    // Performance tracking
    mutable double layer_update_time_ms_;
    mutable double vertex_loading_time_ms_;
    
public:
    LayerManager();
    ~LayerManager();
    
    // =============================================================================
    // INITIALIZATION
    // =============================================================================
    
    /**
     * Load collision data from HDF5 file (generated in Phase 1)
     * @param filepath Path to HDF5 collision data file
     * @return true if successful
     */
    bool load_hdf5_data(const std::string& filepath);
    
    /**
     * Initialize base mesh vertices (T-pose from STAR)
     * @param vertices Base mesh vertices in T-pose
     * @return true if successful
     */
    bool initialize_base_mesh(const std::vector<Eigen::Vector3d>& vertices);
    
    /**
     * Set configuration parameters
     * @param cooldown_frames Number of frames to wait before cooling down unused layers
     */
    void set_configuration(int cooldown_frames = 3);
    
    // =============================================================================
    // LAYER UPDATE AND MANAGEMENT
    // =============================================================================
    
    /**
     * Update human pose and transform all layers
     * @param bone_positions Current bone positions (24 joints)
     * @return true if update successful
     */
    bool update_human_pose(const std::vector<Eigen::Vector3d>& bone_positions);
    
    /**
     * Activate Layer 2 primitives for collision testing
     * @param layer3_indices Indices of Layer 3 primitives that had collisions
     */
    void activate_layer2_primitives(const std::vector<int>& layer3_indices);
    
    /**
     * Activate Layer 1 primitives for collision testing
     * @param layer2_indices Indices of Layer 2 primitives that had collisions
     */
    void activate_layer1_primitives(const std::vector<int>& layer2_indices);
    
    /**
     * Load Layer 0 mesh vertices for precise collision
     * @param layer1_indices Indices of Layer 1 primitives that had collisions
     */
    void load_layer0_vertices(const std::vector<int>& layer1_indices);
    
    /**
     * Cool down layers that haven't had collisions recently
     * Called once per frame to manage memory and performance
     */
    void cool_down_unused_layers();
    
    /**
     * Advance frame counter and reset active flags
     */
    void advance_frame();
    
    // =============================================================================
    // DATA ACCESS
    // =============================================================================
    
    /**
     * Get all Layer 3 primitives (always active)
     * @return const reference to Layer 3 primitives
     */
    const std::vector<Layer3Primitive>& get_layer3_primitives() const;
    
    /**
     * Get currently active Layer 2 primitives
     * @return vector of active Layer 2 primitives
     */
    std::vector<Layer2Primitive> get_active_layer2_primitives() const;
    
    /**
     * Get currently active Layer 1 primitives  
     * @return vector of active Layer 1 primitives
     */
    std::vector<Layer1Primitive> get_active_layer1_primitives() const;
    
    /**
     * Get loaded Layer 0 vertex groups
     * @return vector of loaded vertex groups
     */
    std::vector<Layer0Vertices> get_loaded_layer0_vertices() const;
    
    /**
     * Get layer states for direct access (for collision engine)
     * @return reference to layer states
     */
    const LayerStates& get_layer_states() const { return layer_states_; }
    LayerStates& get_layer_states_mutable() { return layer_states_; }
    
    /**
     * Get hierarchy mappings for direct access
     * @return const reference to hierarchy mappings
     */
    const HierarchyMappings& get_hierarchy_mappings() const { return hierarchy_mappings_; }
    
    // =============================================================================
    // PERFORMANCE AND DIAGNOSTICS
    // =============================================================================
    
    /**
     * Get performance timing information
     * @param layer_update_ms Time spent updating layers
     * @param vertex_loading_ms Time spent loading vertices
     */
    void get_performance_timings(double& layer_update_ms, double& vertex_loading_ms) const;
    
    /**
     * Get current layer activation statistics
     */
    struct LayerStats {
        int total_layer3;
        int active_layer2; 
        int active_layer1;
        int loaded_layer0;
        double memory_usage_mb;
    };
    
    /**
     * Get collision branch statistics - shows selective loading efficiency
     */
    struct CollisionBranchStats {
        int total_layer3_primitives;
        int hit_layer3_primitives;
        
        int total_layer2_primitives;  
        int active_layer2_primitives;
        
        int total_layer1_primitives;
        int active_layer1_primitives;
        
        int total_vertices;
        int loaded_vertices;
        
        double selectivity_ratio;     // Percentage of total system that's active
        double memory_efficiency;    // Memory saved by selective loading
    };
    
    CollisionBranchStats get_collision_branch_statistics() const;
    
    /**
     * Get debug information about current state
     * @return debug string with layer states and timings
     */
    std::string get_debug_info() const;
    
private:
    // =============================================================================
    // INTERNAL METHODS
    // =============================================================================
    
    /**
     * Load hierarchy mappings from HDF5 file
     */
    bool load_hierarchy_mappings_from_hdf5();
    
    /**
     * Build reverse lookup mappings for faster access
     */
    void build_reverse_mappings();
    
    /**
     * Initialize all layer primitives in T-pose
     */
    void initialize_layer_primitives();
    
    /**
     * Transform layer primitives based on current bone positions
     * Uses forward kinematics to move primitives with bones
     */
    void transform_layer_primitives();
    
    /**
     * Transform specific vertices based on bone deformation
     * @param vertex_indices Indices of vertices to transform
     * @return Transformed vertices in world space
     */
    std::vector<Eigen::Vector3d> transform_vertices(const std::vector<int>& vertex_indices);
    
    /**
     * Calculate transformed position based on bone movement
     * Simple bone-based transformation (can be replaced with STAR's forward kinematics)
     * @param original_position Position in T-pose
     * @param bone_index Primary bone influencing this position
     * @return Transformed position
     */
    Eigen::Vector3d calculate_bone_transformation(const Eigen::Vector3d& original_position, 
                                                 int bone_index);
    
    /**
     * Cool down specific layer type
     * @param current_frame Current frame number
     */
    void cool_down_layer2(int current_frame);
    void cool_down_layer1(int current_frame);  
    void cool_down_layer0(int current_frame);
    
    /**
     * Estimate memory usage of currently loaded data
     * @return Memory usage in MB
     */
    double estimate_memory_usage() const;
};

} // namespace delta