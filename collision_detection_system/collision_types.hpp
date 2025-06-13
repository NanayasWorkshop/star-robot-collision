#pragma once

#include <vector>
#include <Eigen/Dense>
#include <string>
#include <memory>

namespace delta {

// Forward declarations
struct CapsuleData;

// =============================================================================
// COLLISION RESULT STRUCTURES
// =============================================================================

struct CollisionContact {
    Eigen::Vector3d contact_point;      // World space contact point
    Eigen::Vector3d surface_normal;     // Surface normal at contact (pointing away from human)
    double penetration_depth;           // How deep robot penetrated into human
    int robot_capsule_index;           // Which robot capsule had this contact
    
    CollisionContact() : penetration_depth(0.0), robot_capsule_index(-1) {}
    
    CollisionContact(const Eigen::Vector3d& point, const Eigen::Vector3d& normal, 
                    double depth, int capsule_idx)
        : contact_point(point), surface_normal(normal), 
          penetration_depth(depth), robot_capsule_index(capsule_idx) {}
};

struct CollisionResult {
    std::vector<CollisionContact> contacts;  // All collision contacts found
    bool has_collision;                      // Quick boolean check
    double max_penetration_depth;           // Maximum penetration across all contacts
    double computation_time_ms;             // Performance timing
    std::string debug_info;                 // Optional debug information
    
    // Statistics for performance monitoring
    int layer3_tests;                       // Number of Layer 3 collision tests
    int layer2_activations;                 // Number of Layer 2 primitives activated
    int layer1_activations;                 // Number of Layer 1 primitives activated
    int layer0_activations;                 // Number of Layer 0 mesh tests
    
    CollisionResult() : has_collision(false), max_penetration_depth(0.0), 
                       computation_time_ms(0.0), layer3_tests(0), layer2_activations(0),
                       layer1_activations(0), layer0_activations(0) {}
};

// =============================================================================
// LAYER PRIMITIVE STRUCTURES
// =============================================================================

struct Layer3Primitive {
    Eigen::Vector3d start_point;
    Eigen::Vector3d end_point;
    double radius;
    std::string name;
    bool is_active;                         // Currently being tested for collision
    int last_collision_frame;              // Last frame this primitive had collision
    
    Layer3Primitive() : radius(0.0), is_active(false), last_collision_frame(-1) {}
    
    Layer3Primitive(const Eigen::Vector3d& start, const Eigen::Vector3d& end, 
                   double r, const std::string& n)
        : start_point(start), end_point(end), radius(r), name(n), 
          is_active(false), last_collision_frame(-1) {}
};

struct Layer2Primitive {
    Eigen::Vector3d start_point;
    Eigen::Vector3d end_point;
    double radius;
    std::string name;
    bool is_active;
    int last_collision_frame;
    int parent_layer3_index;                // Which Layer 3 primitive contains this
    
    Layer2Primitive() : radius(0.0), is_active(false), last_collision_frame(-1),
                       parent_layer3_index(-1) {}
    
    Layer2Primitive(const Eigen::Vector3d& start, const Eigen::Vector3d& end, 
                   double r, const std::string& n, int parent_idx)
        : start_point(start), end_point(end), radius(r), name(n), 
          is_active(false), last_collision_frame(-1), parent_layer3_index(parent_idx) {}
};

struct Layer1Primitive {
    Eigen::Vector3d center;
    double radius;
    std::string name;
    bool is_active;
    int last_collision_frame;
    int parent_layer2_index;                // Which Layer 2 primitive contains this
    std::vector<int> vertex_indices;        // Vertices assigned to this sphere (from HDF5)
    
    Layer1Primitive() : radius(0.0), is_active(false), last_collision_frame(-1),
                       parent_layer2_index(-1) {}
    
    Layer1Primitive(const Eigen::Vector3d& center_pos, double r, const std::string& n, 
                   int parent_idx)
        : center(center_pos), radius(r), name(n), is_active(false), 
          last_collision_frame(-1), parent_layer2_index(parent_idx) {}
};

struct Layer0Vertices {
    std::vector<Eigen::Vector3d> vertices;  // Actual mesh vertices in world space
    std::vector<Eigen::Vector3i> triangles; // Triangle indices (if using triangulated mesh)
    int parent_layer1_index;                // Which Layer 1 sphere these vertices belong to
    bool is_loaded;                         // Whether vertices are currently loaded
    int last_access_frame;                  // Last frame these vertices were accessed
    
    Layer0Vertices() : parent_layer1_index(-1), is_loaded(false), last_access_frame(-1) {}
};

// =============================================================================
// LAYER STATE MANAGEMENT (Clean Selective Branch Loading)
// =============================================================================

struct LayerStates {
    std::vector<Layer3Primitive> layer3_primitives;    // 9 simple capsules
    std::vector<Layer2Primitive> layer2_primitives;    // 23 detailed capsules  
    std::vector<Layer1Primitive> layer1_primitives;    // 76 spheres
    std::vector<Layer0Vertices> layer0_vertex_groups;  // Vertex groups per sphere
    
    // Enhanced tracking for selective branch loading
    struct BranchActivation {
        int parent_primitive_index;              // Index of parent that caused activation
        std::vector<int> activated_children;     // Indices of activated children
        int activation_frame;                    // Frame when activated
        bool is_collision_branch;               // True if this branch has active collisions
    };
    
    // Track active branches per layer
    std::vector<BranchActivation> active_layer2_branches;  // Layer 3 → Layer 2 activations
    std::vector<BranchActivation> active_layer1_branches;  // Layer 2 → Layer 1 activations  
    std::vector<BranchActivation> active_layer0_branches;  // Layer 1 → Layer 0 activations
    
    // Parent-child relationship tracking
    struct ParentChildMap {
        std::vector<std::vector<int>> layer3_to_layer2_active;  // [layer3_idx] → active layer2 children
        std::vector<std::vector<int>> layer2_to_layer1_active;  // [layer2_idx] → active layer1 children
        std::vector<std::vector<int>> layer1_to_layer0_active;  // [layer1_idx] → loaded vertex groups
    };
    
    ParentChildMap active_relationships;
    
    // Frame tracking for cooling down
    int current_frame;
    
    LayerStates() : current_frame(0) {}
    
    void advance_frame() { 
        current_frame++; 
        cleanup_old_branches();
    }
    
    void reset_for_new_frame() {
        // Clear active relationship tracking for new frame
        active_relationships.layer3_to_layer2_active.clear();
        active_relationships.layer2_to_layer1_active.clear();
        active_relationships.layer1_to_layer0_active.clear();
        
        // Resize relationship maps
        active_relationships.layer3_to_layer2_active.resize(layer3_primitives.size());
        active_relationships.layer2_to_layer1_active.resize(layer2_primitives.size());
        active_relationships.layer1_to_layer0_active.resize(layer1_primitives.size());
    }
    
    // Branch management methods
    void activate_branch_layer2(int layer3_parent, const std::vector<int>& layer2_children) {
        BranchActivation branch;
        branch.parent_primitive_index = layer3_parent;
        branch.activated_children = layer2_children;
        branch.activation_frame = current_frame;
        branch.is_collision_branch = true;
        
        active_layer2_branches.push_back(branch);
        
        // Update parent-child relationship tracking
        if (layer3_parent >= 0 && layer3_parent < static_cast<int>(active_relationships.layer3_to_layer2_active.size())) {
            active_relationships.layer3_to_layer2_active[layer3_parent] = layer2_children;
        }
        
        // Mark individual primitives as active
        for (int child_idx : layer2_children) {
            if (child_idx >= 0 && child_idx < static_cast<int>(layer2_primitives.size())) {
                layer2_primitives[child_idx].is_active = true;
                layer2_primitives[child_idx].last_collision_frame = current_frame;
            }
        }
    }
    
    void activate_branch_layer1(int layer2_parent, const std::vector<int>& layer1_children) {
        BranchActivation branch;
        branch.parent_primitive_index = layer2_parent;
        branch.activated_children = layer1_children;
        branch.activation_frame = current_frame;
        branch.is_collision_branch = true;
        
        active_layer1_branches.push_back(branch);
        
        // Update parent-child relationship tracking
        if (layer2_parent >= 0 && layer2_parent < static_cast<int>(active_relationships.layer2_to_layer1_active.size())) {
            active_relationships.layer2_to_layer1_active[layer2_parent] = layer1_children;
        }
        
        // Mark individual primitives as active
        for (int child_idx : layer1_children) {
            if (child_idx >= 0 && child_idx < static_cast<int>(layer1_primitives.size())) {
                layer1_primitives[child_idx].is_active = true;
                layer1_primitives[child_idx].last_collision_frame = current_frame;
            }
        }
    }
    
    void activate_branch_layer0(int layer1_parent, const std::vector<int>& vertex_groups) {
        BranchActivation branch;
        branch.parent_primitive_index = layer1_parent;
        branch.activated_children = vertex_groups;
        branch.activation_frame = current_frame;
        branch.is_collision_branch = true;
        
        active_layer0_branches.push_back(branch);
        
        // Update parent-child relationship tracking
        if (layer1_parent >= 0 && layer1_parent < static_cast<int>(active_relationships.layer1_to_layer0_active.size())) {
            active_relationships.layer1_to_layer0_active[layer1_parent] = vertex_groups;
        }
        
        // Mark individual vertex groups as loaded
        for (int child_idx : vertex_groups) {
            if (child_idx >= 0 && child_idx < static_cast<int>(layer0_vertex_groups.size())) {
                layer0_vertex_groups[child_idx].last_access_frame = current_frame;
            }
        }
    }
    
    // Get active children for a specific parent
    std::vector<int> get_active_layer2_children(int layer3_parent) const {
        if (layer3_parent >= 0 && layer3_parent < static_cast<int>(active_relationships.layer3_to_layer2_active.size())) {
            return active_relationships.layer3_to_layer2_active[layer3_parent];
        }
        return {};
    }
    
    std::vector<int> get_active_layer1_children(int layer2_parent) const {
        if (layer2_parent >= 0 && layer2_parent < static_cast<int>(active_relationships.layer2_to_layer1_active.size())) {
            return active_relationships.layer2_to_layer1_active[layer2_parent];
        }
        return {};
    }
    
    std::vector<int> get_active_layer0_children(int layer1_parent) const {
        if (layer1_parent >= 0 && layer1_parent < static_cast<int>(active_relationships.layer1_to_layer0_active.size())) {
            return active_relationships.layer1_to_layer0_active[layer1_parent];
        }
        return {};
    }
    
    // Get all currently active primitives by layer
    std::vector<int> get_all_active_layer2_indices() const {
        std::vector<int> active_indices;
        for (size_t i = 0; i < layer2_primitives.size(); ++i) {
            if (layer2_primitives[i].is_active) {
                active_indices.push_back(static_cast<int>(i));
            }
        }
        return active_indices;
    }
    
    std::vector<int> get_all_active_layer1_indices() const {
        std::vector<int> active_indices;
        for (size_t i = 0; i < layer1_primitives.size(); ++i) {
            if (layer1_primitives[i].is_active) {
                active_indices.push_back(static_cast<int>(i));
            }
        }
        return active_indices;
    }
    
    std::vector<int> get_all_loaded_layer0_indices() const {
        std::vector<int> loaded_indices;
        for (size_t i = 0; i < layer0_vertex_groups.size(); ++i) {
            if (layer0_vertex_groups[i].is_loaded) {
                loaded_indices.push_back(static_cast<int>(i));
            }
        }
        return loaded_indices;
    }
    
    // Get branch statistics
    struct BranchStats {
        int total_active_branches_layer2;
        int total_active_branches_layer1;
        int total_active_branches_layer0;
        int total_active_primitives_layer2;
        int total_active_primitives_layer1;
        int total_loaded_vertices;
        double branch_efficiency_percent;  // How selective the loading is
    };
    
    BranchStats get_branch_statistics() const {
        BranchStats stats{};
        stats.total_active_branches_layer2 = static_cast<int>(active_layer2_branches.size());
        stats.total_active_branches_layer1 = static_cast<int>(active_layer1_branches.size());
        stats.total_active_branches_layer0 = static_cast<int>(active_layer0_branches.size());
        
        // Count total active primitives by checking individual primitive states
        stats.total_active_primitives_layer2 = 0;
        for (const auto& primitive : layer2_primitives) {
            if (primitive.is_active) stats.total_active_primitives_layer2++;
        }
        
        stats.total_active_primitives_layer1 = 0;
        for (const auto& primitive : layer1_primitives) {
            if (primitive.is_active) stats.total_active_primitives_layer1++;
        }
        
        stats.total_loaded_vertices = 0;
        for (const auto& vertex_group : layer0_vertex_groups) {
            if (vertex_group.is_loaded) {
                stats.total_loaded_vertices += static_cast<int>(vertex_group.vertices.size());
            }
        }
        
        // Calculate branch efficiency (fewer active primitives = more efficient)
        int total_possible_primitives = static_cast<int>(layer2_primitives.size() + layer1_primitives.size() + layer0_vertex_groups.size());
        int total_active_primitives = stats.total_active_primitives_layer2 + stats.total_active_primitives_layer1;
        int total_loaded_groups = 0;
        for (const auto& vertex_group : layer0_vertex_groups) {
            if (vertex_group.is_loaded) total_loaded_groups++;
        }
        total_active_primitives += total_loaded_groups;
        
        if (total_possible_primitives > 0) {
            stats.branch_efficiency_percent = (1.0 - static_cast<double>(total_active_primitives) / total_possible_primitives) * 100.0;
        }
        
        return stats;
    }
    
    // Check if a specific primitive is part of an active collision branch
    bool is_in_active_collision_branch_layer2(int layer2_idx) const {
        for (const auto& branch : active_layer2_branches) {
            if (branch.is_collision_branch) {
                for (int child_idx : branch.activated_children) {
                    if (child_idx == layer2_idx) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
    bool is_in_active_collision_branch_layer1(int layer1_idx) const {
        for (const auto& branch : active_layer1_branches) {
            if (branch.is_collision_branch) {
                for (int child_idx : branch.activated_children) {
                    if (child_idx == layer1_idx) {
                        return true;
                    }
                }
            }
        }
        return false;
    }
    
private:
    void cleanup_old_branches() {
        // Remove branches that are too old
        const int max_branch_age = 10; // frames
        
        auto remove_old = [this, max_branch_age](std::vector<BranchActivation>& branches) {
            branches.erase(
                std::remove_if(branches.begin(), branches.end(),
                    [this, max_branch_age](const BranchActivation& branch) {
                        return (current_frame - branch.activation_frame) > max_branch_age;
                    }),
                branches.end()
            );
        };
        
        remove_old(active_layer2_branches);
        remove_old(active_layer1_branches);
        remove_old(active_layer0_branches);
    }
};

// =============================================================================
// HDF5 DATA STRUCTURES (from Phase 1)
// =============================================================================

struct HierarchyMappings {
    // Direct arrays from HDF5 file
    std::vector<std::vector<int>> vertex_sphere_assignments;  // [vertex_idx] -> list of sphere indices
    std::vector<int> sphere_to_capsule;                       // [sphere_idx] -> capsule_idx
    std::vector<int> capsule_to_simple;                       // [capsule_idx] -> simple_capsule_idx
    
    // Reverse mappings for faster lookup
    std::vector<std::vector<int>> layer3_to_layer2;           // [simple_idx] -> list of capsule indices
    std::vector<std::vector<int>> layer2_to_layer1;           // [capsule_idx] -> list of sphere indices
    std::vector<std::vector<int>> layer1_to_layer0;           // [sphere_idx] -> list of vertex indices
    
    // Metadata
    int num_vertices;
    int num_spheres; 
    int num_capsules;
    int num_simple;
    int max_assignments_per_vertex;
    
    HierarchyMappings() : num_vertices(0), num_spheres(0), num_capsules(0), 
                         num_simple(0), max_assignments_per_vertex(0) {}
};

// =============================================================================
// TIMING AND PERFORMANCE
// =============================================================================

class ScopedTimer {
private:
    double& time_accumulator_;
    std::chrono::high_resolution_clock::time_point start_time_;
    
public:
    explicit ScopedTimer(double& accumulator) 
        : time_accumulator_(accumulator), 
          start_time_(std::chrono::high_resolution_clock::now()) {}
    
    ~ScopedTimer() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time_);
        time_accumulator_ += duration.count() / 1000.0; // Convert to milliseconds
    }
};

// =============================================================================
// UTILITY FUNCTIONS
// =============================================================================

// Distance from point to line segment
double point_to_line_distance(const Eigen::Vector3d& point, 
                              const Eigen::Vector3d& line_start,
                              const Eigen::Vector3d& line_end);

// Check if point is inside capsule
bool point_in_capsule(const Eigen::Vector3d& point,
                     const Eigen::Vector3d& capsule_start,
                     const Eigen::Vector3d& capsule_end,
                     double capsule_radius);

// Capsule vs capsule collision test
bool capsule_vs_capsule_collision(const Eigen::Vector3d& cap1_start,
                                 const Eigen::Vector3d& cap1_end,
                                 double cap1_radius,
                                 const Eigen::Vector3d& cap2_start,
                                 const Eigen::Vector3d& cap2_end,
                                 double cap2_radius);

// Capsule vs sphere collision test
bool capsule_vs_sphere_collision(const Eigen::Vector3d& capsule_start,
                                const Eigen::Vector3d& capsule_end,
                                double capsule_radius,
                                const Eigen::Vector3d& sphere_center,
                                double sphere_radius);

} // namespace delta