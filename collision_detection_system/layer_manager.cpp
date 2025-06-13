#include "layer_manager.hpp"
#include <h5cpp/hdf5.hpp>
#include <iostream>
#include <algorithm>
#include <chrono>

namespace delta {

LayerManager::LayerManager() 
    : cooldown_frames_(3), hdf5_loaded_(false), layer_update_time_ms_(0.0), 
      vertex_loading_time_ms_(0.0) {
}

LayerManager::~LayerManager() {
}

// =============================================================================
// INITIALIZATION
// =============================================================================

bool LayerManager::load_hdf5_data(const std::string& filepath) {
    ScopedTimer timer(layer_update_time_ms_);
    
    try {
        hdf5_filepath_ = filepath;
        
        // Open HDF5 file
        auto file = hdf5::file::open(filepath, hdf5::file::AccessFlags::READONLY);
        
        // Load basic metadata
        hierarchy_mappings_.num_vertices = file.attributes["num_vertices"].read<int>();
        hierarchy_mappings_.num_spheres = file.attributes["num_spheres"].read<int>();
        hierarchy_mappings_.num_capsules = file.attributes["num_capsules"].read<int>();
        hierarchy_mappings_.num_simple = file.attributes["num_simple"].read<int>();
        hierarchy_mappings_.max_assignments_per_vertex = file.attributes["max_assignments_per_vertex"].read<int>();
        
        std::cout << "Loading HDF5 collision data:" << std::endl;
        std::cout << "  Vertices: " << hierarchy_mappings_.num_vertices << std::endl;
        std::cout << "  Spheres: " << hierarchy_mappings_.num_spheres << std::endl;
        std::cout << "  Capsules: " << hierarchy_mappings_.num_capsules << std::endl;
        std::cout << "  Simple capsules: " << hierarchy_mappings_.num_simple << std::endl;
        
        // Load vertex sphere assignments (2D array)
        auto vertex_assignments_dataset = file.root()["vertex_sphere_assignments"];
        auto vertex_assignments_data = vertex_assignments_dataset.read<std::vector<std::vector<int>>>();
        
        // Load assignment lengths
        auto assignment_lengths_dataset = file.root()["assignment_lengths"];
        auto assignment_lengths = assignment_lengths_dataset.read<std::vector<int>>();
        
        // Process vertex assignments (remove padding -1 values)
        hierarchy_mappings_.vertex_sphere_assignments.resize(hierarchy_mappings_.num_vertices);
        for (int vertex_idx = 0; vertex_idx < hierarchy_mappings_.num_vertices; ++vertex_idx) {
            int num_assignments = assignment_lengths[vertex_idx];
            hierarchy_mappings_.vertex_sphere_assignments[vertex_idx].clear();
            
            for (int i = 0; i < num_assignments; ++i) {
                int sphere_idx = vertex_assignments_data[vertex_idx][i];
                if (sphere_idx >= 0) {  // Skip padding -1 values
                    hierarchy_mappings_.vertex_sphere_assignments[vertex_idx].push_back(sphere_idx);
                }
            }
        }
        
        // Load sphere to capsule mappings
        auto sphere_to_capsule_dataset = file.root()["sphere_to_capsule"];
        hierarchy_mappings_.sphere_to_capsule = sphere_to_capsule_dataset.read<std::vector<int>>();
        
        // Load capsule to simple mappings
        auto capsule_to_simple_dataset = file.root()["capsule_to_simple"];
        hierarchy_mappings_.capsule_to_simple = capsule_to_simple_dataset.read<std::vector<int>>();
        
        // Build reverse mappings for faster lookup
        build_reverse_mappings();
        
        hdf5_loaded_ = true;
        std::cout << "✅ HDF5 collision data loaded successfully" << std::endl;
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "❌ Failed to load HDF5 data: " << e.what() << std::endl;
        return false;
    }
}

bool LayerManager::initialize_base_mesh(const std::vector<Eigen::Vector3d>& vertices) {
    if (vertices.size() != static_cast<size_t>(hierarchy_mappings_.num_vertices)) {
        std::cerr << "❌ Vertex count mismatch: expected " << hierarchy_mappings_.num_vertices 
                  << ", got " << vertices.size() << std::endl;
        return false;
    }
    
    base_mesh_vertices_ = vertices;
    
    // Initialize layer primitives in T-pose
    initialize_layer_primitives();
    
    std::cout << "✅ Base mesh initialized with " << vertices.size() << " vertices" << std::endl;
    return true;
}

void LayerManager::set_configuration(int cooldown_frames) {
    cooldown_frames_ = cooldown_frames;
}

// =============================================================================
// LAYER UPDATE AND MANAGEMENT  
// =============================================================================

bool LayerManager::update_human_pose(const std::vector<Eigen::Vector3d>& bone_positions) {
    ScopedTimer timer(layer_update_time_ms_);
    
    if (bone_positions.size() != 24) {
        std::cerr << "❌ Invalid bone positions: expected 24, got " << bone_positions.size() << std::endl;
        return false;
    }
    
    current_bone_positions_ = bone_positions;
    
    // Transform all layer primitives based on new bone positions
    transform_layer_primitives();
    
    return true;
}

void LayerManager::activate_layer2_primitives(const std::vector<int>& layer3_indices) {
    // SELECTIVE BRANCH LOADING: Only activate specific Layer 2 children of hit Layer 3 primitives
    
    for (int layer3_idx : layer3_indices) {
        if (layer3_idx < 0 || layer3_idx >= static_cast<int>(hierarchy_mappings_.layer3_to_layer2.size())) {
            std::cerr << "⚠️  Invalid Layer 3 index: " << layer3_idx << std::endl;
            continue;
        }
        
        // Get ONLY the specific Layer 2 children of this hit Layer 3 primitive
        const auto& layer2_children = hierarchy_mappings_.layer3_to_layer2[layer3_idx];
        
        std::cout << "  Activating Layer 2 branch for Layer 3[" << layer3_idx << "]: " 
                  << layer2_children.size() << " children" << std::endl;
        
        // Use the new branch activation system
        layer_states_.activate_branch_layer2(layer3_idx, layer2_children);
        
        std::cout << "    ✓ Activated " << layer2_children.size() << " Layer 2 primitives for branch" << std::endl;
    }
    
    // Log selective activation summary
    auto active_indices = layer_states_.get_all_active_layer2_indices();
    std::cout << "  → Total active Layer 2 primitives: " << active_indices.size() << "/23" << std::endl;
}

void LayerManager::activate_layer1_primitives(const std::vector<int>& layer2_indices) {
    // SELECTIVE BRANCH LOADING: Only activate specific Layer 1 children of hit Layer 2 primitives
    
    for (int layer2_idx : layer2_indices) {
        if (layer2_idx < 0 || layer2_idx >= static_cast<int>(hierarchy_mappings_.layer2_to_layer1.size())) {
            std::cerr << "⚠️  Invalid Layer 2 index: " << layer2_idx << std::endl;
            continue;
        }
        
        // Get ONLY the specific Layer 1 children of this hit Layer 2 primitive
        const auto& layer1_children = hierarchy_mappings_.layer2_to_layer1[layer2_idx];
        
        std::cout << "  Activating Layer 1 branch for Layer 2[" << layer2_idx << "]: " 
                  << layer1_children.size() << " children" << std::endl;
        
        // Use the new branch activation system
        layer_states_.activate_branch_layer1(layer2_idx, layer1_children);
        
        std::cout << "    ✓ Activated " << layer1_children.size() << " Layer 1 primitives for branch" << std::endl;
    }
    
    // Log selective activation summary
    auto active_indices = layer_states_.get_all_active_layer1_indices();
    std::cout << "  → Total active Layer 1 primitives: " << active_indices.size() << "/76" << std::endl;
}

void LayerManager::load_layer0_vertices(const std::vector<int>& layer1_indices) {
    ScopedTimer timer(vertex_loading_time_ms_);
    
    // SELECTIVE VERTEX LOADING: Only load vertices for specific hit Layer 1 spheres
    
    for (int layer1_idx : layer1_indices) {
        if (layer1_idx < 0 || layer1_idx >= static_cast<int>(layer_states_.layer0_vertex_groups.size())) {
            std::cerr << "⚠️  Invalid Layer 1 index: " << layer1_idx << std::endl;
            continue;
        }
        
        Layer0Vertices& vertex_group = layer_states_.layer0_vertex_groups[layer1_idx];
        
        if (!vertex_group.is_loaded) {
            // Get ONLY the specific vertices assigned to THIS sphere
            const auto& vertex_indices = layer_states_.layer1_primitives[layer1_idx].vertex_indices;
            
            std::cout << "  Loading vertices for Layer 1[" << layer1_idx << "]: " 
                      << vertex_indices.size() << " vertices" << std::endl;
            
            // Transform only these specific vertices to current pose
            vertex_group.vertices = transform_vertices(vertex_indices);
            vertex_group.is_loaded = true;
            
            std::cout << "    ✓ Loaded " << vertex_group.vertices.size() 
                      << " vertices for sphere " << layer_states_.layer1_primitives[layer1_idx].name << std::endl;
        } else {
            std::cout << "  Layer 1[" << layer1_idx << "] vertices already loaded (cached)" << std::endl;
        }
        
        vertex_group.last_access_frame = layer_states_.current_frame;
    }
    
    // Use the new branch activation system
    layer_states_.activate_branch_layer0(-1, layer1_indices); // -1 indicates multiple parents
    
    // Log selective loading summary
    auto loaded_indices = layer_states_.get_all_loaded_layer0_indices();
    
    // Calculate total loaded vertices
    int total_vertices = 0;
    for (int idx : loaded_indices) {
        if (idx >= 0 && idx < static_cast<int>(layer_states_.layer0_vertex_groups.size())) {
            total_vertices += static_cast<int>(layer_states_.layer0_vertex_groups[idx].vertices.size());
        }
    }
    
    std::cout << "  → Total loaded vertex groups: " << loaded_indices.size() << "/76" << std::endl;
    std::cout << "  → Total loaded vertices: " << total_vertices << " (selective loading)" << std::endl;
}

void LayerManager::cool_down_unused_layers() {
    cool_down_layer2(layer_states_.current_frame);
    cool_down_layer1(layer_states_.current_frame);
    cool_down_layer0(layer_states_.current_frame);
}

void LayerManager::advance_frame() {
    layer_states_.advance_frame();
    layer_states_.reset_for_new_frame();
    
    // Reset performance timers
    layer_update_time_ms_ = 0.0;
    vertex_loading_time_ms_ = 0.0;
}

// =============================================================================
// DATA ACCESS
// =============================================================================

const std::vector<Layer3Primitive>& LayerManager::get_layer3_primitives() const {
    return layer_states_.layer3_primitives;
}

std::vector<Layer2Primitive> LayerManager::get_active_layer2_primitives() const {
    std::vector<Layer2Primitive> active_primitives;
    
    // Use the new branch system to get active primitives
    auto active_indices = layer_states_.get_all_active_layer2_indices();
    
    for (int idx : active_indices) {
        if (idx >= 0 && idx < static_cast<int>(layer_states_.layer2_primitives.size())) {
            active_primitives.push_back(layer_states_.layer2_primitives[idx]);
        }
    }
    
    return active_primitives;
}

std::vector<Layer1Primitive> LayerManager::get_active_layer1_primitives() const {
    std::vector<Layer1Primitive> active_primitives;
    
    // Use the new branch system to get active primitives
    auto active_indices = layer_states_.get_all_active_layer1_indices();
    
    for (int idx : active_indices) {
        if (idx >= 0 && idx < static_cast<int>(layer_states_.layer1_primitives.size())) {
            active_primitives.push_back(layer_states_.layer1_primitives[idx]);
        }
    }
    
    return active_primitives;
}

std::vector<Layer0Vertices> LayerManager::get_loaded_layer0_vertices() const {
    std::vector<Layer0Vertices> loaded_vertices;
    
    // Use the new branch system to get loaded vertex groups
    auto loaded_indices = layer_states_.get_all_loaded_layer0_indices();
    
    for (int idx : loaded_indices) {
        if (idx >= 0 && idx < static_cast<int>(layer_states_.layer0_vertex_groups.size()) &&
            layer_states_.layer0_vertex_groups[idx].is_loaded) {
            loaded_vertices.push_back(layer_states_.layer0_vertex_groups[idx]);
        }
    }
    
    return loaded_vertices;
}

// =============================================================================
// PERFORMANCE AND DIAGNOSTICS
// =============================================================================

void LayerManager::get_performance_timings(double& layer_update_ms, double& vertex_loading_ms) const {
    layer_update_ms = layer_update_time_ms_;
    vertex_loading_ms = vertex_loading_time_ms_;
}

LayerManager::CollisionBranchStats LayerManager::get_collision_branch_statistics() const {
    CollisionBranchStats stats;
    
    // Layer 3 (always all active)
    stats.total_layer3_primitives = static_cast<int>(layer_states_.layer3_primitives.size());
    stats.hit_layer3_primitives = stats.total_layer3_primitives; // Always test all Layer 3
    
    // Layer 2 (selective activation)
    stats.total_layer2_primitives = static_cast<int>(layer_states_.layer2_primitives.size());
    stats.active_layer2_primitives = static_cast<int>(std::count(layer_states_.active_layer2.begin(), 
                                                                 layer_states_.active_layer2.end(), true));
    
    // Layer 1 (selective activation)
    stats.total_layer1_primitives = static_cast<int>(layer_states_.layer1_primitives.size());
    stats.active_layer1_primitives = static_cast<int>(std::count(layer_states_.active_layer1.begin(), 
                                                                 layer_states_.active_layer1.end(), true));
    
    // Layer 0 (selective vertex loading)
    stats.total_vertices = hierarchy_mappings_.num_vertices;
    stats.loaded_vertices = 0;
    
    for (size_t i = 0; i < layer_states_.layer0_vertex_groups.size(); ++i) {
        if (layer_states_.active_layer0[i] && layer_states_.layer0_vertex_groups[i].is_loaded) {
            stats.loaded_vertices += static_cast<int>(layer_states_.layer0_vertex_groups[i].vertices.size());
        }
    }
    
    // Calculate efficiency metrics
    int total_possible_primitives = stats.total_layer3_primitives + stats.total_layer2_primitives + 
                                   stats.total_layer1_primitives + stats.total_vertices;
    int active_primitives = stats.hit_layer3_primitives + stats.active_layer2_primitives + 
                           stats.active_layer1_primitives + stats.loaded_vertices;
    
    stats.selectivity_ratio = (total_possible_primitives > 0) ? 
        (static_cast<double>(active_primitives) / total_possible_primitives * 100.0) : 0.0;
    
    // Memory efficiency (how much memory we saved by not loading everything)
    double potential_memory_mb = static_cast<double>(stats.total_vertices) * sizeof(Eigen::Vector3d) / (1024.0 * 1024.0);
    double actual_memory_mb = static_cast<double>(stats.loaded_vertices) * sizeof(Eigen::Vector3d) / (1024.0 * 1024.0);
    stats.memory_efficiency = (potential_memory_mb > 0) ? 
        ((potential_memory_mb - actual_memory_mb) / potential_memory_mb * 100.0) : 0.0;
    
    return stats;
}

LayerManager::LayerStats LayerManager::get_layer_statistics() const {
    LayerStats stats;
    stats.total_layer3 = static_cast<int>(layer_states_.layer3_primitives.size());
    
    // Use the new branch system for statistics
    auto active_layer2 = layer_states_.get_all_active_layer2_indices();
    auto active_layer1 = layer_states_.get_all_active_layer1_indices();
    auto loaded_layer0 = layer_states_.get_all_loaded_layer0_indices();
    
    stats.active_layer2 = static_cast<int>(active_layer2.size());
    stats.active_layer1 = static_cast<int>(active_layer1.size());
    stats.loaded_layer0 = static_cast<int>(loaded_layer0.size());
    stats.memory_usage_mb = estimate_memory_usage();
    
    return stats;
}

std::string LayerManager::get_debug_info() const {
    auto stats = get_layer_statistics();
    auto branch_stats = get_collision_branch_statistics();
    
    std::string debug_info = "LayerManager Debug Info (Selective Branch Loading):\n";
    debug_info += "  Frame: " + std::to_string(layer_states_.current_frame) + "\n";
    
    // Show selective loading efficiency
    debug_info += "\nSelective Loading Statistics:\n";
    debug_info += "  Layer 3: " + std::to_string(branch_stats.total_layer3_primitives) + " (always active)\n";
    debug_info += "  Layer 2: " + std::to_string(branch_stats.active_layer2_primitives) + "/" + 
                  std::to_string(branch_stats.total_layer2_primitives) + " active (" + 
                  std::to_string(static_cast<int>(static_cast<double>(branch_stats.active_layer2_primitives) / 
                  branch_stats.total_layer2_primitives * 100.0)) + "%)\n";
    debug_info += "  Layer 1: " + std::to_string(branch_stats.active_layer1_primitives) + "/" + 
                  std::to_string(branch_stats.total_layer1_primitives) + " active (" + 
                  std::to_string(static_cast<int>(static_cast<double>(branch_stats.active_layer1_primitives) / 
                  branch_stats.total_layer1_primitives * 100.0)) + "%)\n";
    debug_info += "  Layer 0: " + std::to_string(branch_stats.loaded_vertices) + "/" + 
                  std::to_string(branch_stats.total_vertices) + " vertices loaded (" + 
                  std::to_string(static_cast<int>(static_cast<double>(branch_stats.loaded_vertices) / 
                  branch_stats.total_vertices * 100.0)) + "%)\n";
    
    debug_info += "\nEfficiency Metrics:\n";
    debug_info += "  Overall selectivity: " + std::to_string(branch_stats.selectivity_ratio) + "% of total system active\n";
    debug_info += "  Memory efficiency: " + std::to_string(branch_stats.memory_efficiency) + "% memory saved\n";
    debug_info += "  Total memory: " + std::to_string(stats.memory_usage_mb) + " MB\n";
    
    debug_info += "\nPerformance Timings:\n";
    debug_info += "  Layer update time: " + std::to_string(layer_update_time_ms_) + " ms\n";
    debug_info += "  Vertex loading time: " + std::to_string(vertex_loading_time_ms_) + " ms\n";
    
    return debug_info;
}

// =============================================================================
// INTERNAL METHODS
// =============================================================================

void LayerManager::build_reverse_mappings() {
    // Build Layer 3 → Layer 2 mapping
    hierarchy_mappings_.layer3_to_layer2.resize(hierarchy_mappings_.num_simple);
    for (int capsule_idx = 0; capsule_idx < hierarchy_mappings_.num_capsules; ++capsule_idx) {
        int simple_idx = hierarchy_mappings_.capsule_to_simple[capsule_idx];
        if (simple_idx >= 0 && simple_idx < hierarchy_mappings_.num_simple) {
            hierarchy_mappings_.layer3_to_layer2[simple_idx].push_back(capsule_idx);
        }
    }
    
    // Build Layer 2 → Layer 1 mapping
    hierarchy_mappings_.layer2_to_layer1.resize(hierarchy_mappings_.num_capsules);
    for (int sphere_idx = 0; sphere_idx < hierarchy_mappings_.num_spheres; ++sphere_idx) {
        int capsule_idx = hierarchy_mappings_.sphere_to_capsule[sphere_idx];
        if (capsule_idx >= 0 && capsule_idx < hierarchy_mappings_.num_capsules) {
            hierarchy_mappings_.layer2_to_layer1[capsule_idx].push_back(sphere_idx);
        }
    }
    
    // Build Layer 1 → Layer 0 mapping (already in vertex_sphere_assignments)
    hierarchy_mappings_.layer1_to_layer0.resize(hierarchy_mappings_.num_spheres);
    for (int vertex_idx = 0; vertex_idx < hierarchy_mappings_.num_vertices; ++vertex_idx) {
        const auto& sphere_assignments = hierarchy_mappings_.vertex_sphere_assignments[vertex_idx];
        for (int sphere_idx : sphere_assignments) {
            if (sphere_idx >= 0 && sphere_idx < hierarchy_mappings_.num_spheres) {
                hierarchy_mappings_.layer1_to_layer0[sphere_idx].push_back(vertex_idx);
            }
        }
    }
}

void LayerManager::initialize_layer_primitives() {
    if (!hdf5_loaded_) {
        std::cerr << "❌ Cannot initialize primitives: HDF5 data not loaded" << std::endl;
        return;
    }
    
    // Initialize Layer 3 primitives (9 simple capsules)
    // These are hardcoded from BodyDefinitions.SIMPLE_BONES
    layer_states_.layer3_primitives.clear();
    layer_states_.layer3_primitives.reserve(9);
    
    // Hardcoded simple bone definitions (from your Python BodyDefinitions)
    struct SimpleBoneDef {
        int start_joint, end_joint;
        std::string name;
        double radius;
    };
    
    std::vector<SimpleBoneDef> simple_bones = {
        {10, 4, "left_foot-left_knee", 0.08},      // left lower leg
        {11, 5, "right_foot-right_knee", 0.08},    // right lower leg
        {4, 0, "left_knee-pelvis", 0.12},          // left upper leg  
        {5, 0, "right_knee-pelvis", 0.12},         // right upper leg
        {0, 15, "pelvis-head", 0.15},              // entire torso
        {16, 18, "left_shoulder-left_elbow", 0.08}, // left upper arm
        {17, 19, "right_shoulder-right_elbow", 0.08}, // right upper arm
        {18, 22, "left_elbow-left_hand", 0.06},    // left forearm+hand
        {19, 23, "right_elbow-right_hand", 0.06}   // right forearm+hand
    };
    
    for (const auto& bone_def : simple_bones) {
        Layer3Primitive primitive;
        primitive.start_point = Eigen::Vector3d::Zero(); // Will be set in transform_layer_primitives()
        primitive.end_point = Eigen::Vector3d::Zero();
        primitive.radius = bone_def.radius;
        primitive.name = bone_def.name;
        primitive.is_active = true; // Layer 3 is always active
        layer_states_.layer3_primitives.push_back(primitive);
    }
    
    // Initialize Layer 2 primitives (23 detailed capsules)
    layer_states_.layer2_primitives.clear();
    layer_states_.layer2_primitives.reserve(hierarchy_mappings_.num_capsules);
    layer_states_.active_layer2.resize(hierarchy_mappings_.num_capsules, false);
    
    for (int i = 0; i < hierarchy_mappings_.num_capsules; ++i) {
        Layer2Primitive primitive;
        primitive.start_point = Eigen::Vector3d::Zero();
        primitive.end_point = Eigen::Vector3d::Zero();
        primitive.radius = 0.08; // Default radius, will be refined
        primitive.name = "capsule_" + std::to_string(i);
        primitive.is_active = false;
        primitive.parent_layer3_index = hierarchy_mappings_.capsule_to_simple[i];
        layer_states_.layer2_primitives.push_back(primitive);
    }
    
    // Initialize Layer 1 primitives (76 spheres)
    layer_states_.layer1_primitives.clear();
    layer_states_.layer1_primitives.reserve(hierarchy_mappings_.num_spheres);
    layer_states_.active_layer1.resize(hierarchy_mappings_.num_spheres, false);
    
    for (int i = 0; i < hierarchy_mappings_.num_spheres; ++i) {
        Layer1Primitive primitive;
        primitive.center = Eigen::Vector3d::Zero();
        primitive.radius = 0.05; // Default radius, will be refined
        primitive.name = "sphere_" + std::to_string(i);
        primitive.is_active = false;
        primitive.parent_layer2_index = hierarchy_mappings_.sphere_to_capsule[i];
        primitive.vertex_indices = hierarchy_mappings_.layer1_to_layer0[i];
        layer_states_.layer1_primitives.push_back(primitive);
    }
    
    // Initialize Layer 0 vertex groups (one per sphere)
    layer_states_.layer0_vertex_groups.clear();
    layer_states_.layer0_vertex_groups.reserve(hierarchy_mappings_.num_spheres);
    layer_states_.active_layer0.resize(hierarchy_mappings_.num_spheres, false);
    
    for (int i = 0; i < hierarchy_mappings_.num_spheres; ++i) {
        Layer0Vertices vertex_group;
        vertex_group.parent_layer1_index = i;
        vertex_group.is_loaded = false;
        layer_states_.layer0_vertex_groups.push_back(vertex_group);
    }
    
    std::cout << "✅ Layer primitives initialized:" << std::endl;
    std::cout << "  Layer 3: " << layer_states_.layer3_primitives.size() << " primitives" << std::endl;
    std::cout << "  Layer 2: " << layer_states_.layer2_primitives.size() << " primitives" << std::endl;
    std::cout << "  Layer 1: " << layer_states_.layer1_primitives.size() << " primitives" << std::endl;
    std::cout << "  Layer 0: " << layer_states_.layer0_vertex_groups.size() << " vertex groups" << std::endl;
}

void LayerManager::transform_layer_primitives() {
    if (current_bone_positions_.size() != 24) {
        return;
    }
    
    // Transform Layer 3 primitives based on bone positions
    struct SimpleBoneDef {
        int start_joint, end_joint;
        std::string name;
    };
    
    std::vector<SimpleBoneDef> simple_bones = {
        {10, 4, "left_foot-left_knee"},
        {11, 5, "right_foot-right_knee"},  
        {4, 0, "left_knee-pelvis"},
        {5, 0, "right_knee-pelvis"},
        {0, 15, "pelvis-head"},
        {16, 18, "left_shoulder-left_elbow"},
        {17, 19, "right_shoulder-right_elbow"},
        {18, 22, "left_elbow-left_hand"},
        {19, 23, "right_elbow-right_hand"}
    };
    
    for (size_t i = 0; i < layer_states_.layer3_primitives.size() && i < simple_bones.size(); ++i) {
        const auto& bone_def = simple_bones[i];
        layer_states_.layer3_primitives[i].start_point = current_bone_positions_[bone_def.start_joint];
        layer_states_.layer3_primitives[i].end_point = current_bone_positions_[bone_def.end_joint];
    }
    
    // Transform Layer 2 primitives (simplified - use closest bones)
    // In a full implementation, this would use the exact bone mappings from STAR
    for (auto& primitive : layer_states_.layer2_primitives) {
        // For now, use a simple approximation based on parent Layer 3
        if (primitive.parent_layer3_index >= 0 && 
            primitive.parent_layer3_index < static_cast<int>(layer_states_.layer3_primitives.size())) {
            
            const auto& parent = layer_states_.layer3_primitives[primitive.parent_layer3_index];
            
            // Position Layer 2 primitives along parent Layer 3 primitive
            // This is a simplification - real implementation would use bone weights
            Eigen::Vector3d direction = parent.end_point - parent.start_point;
            double length = direction.norm();
            
            if (length > 0) {
                direction.normalize();
                
                // Simple positioning along parent primitive
                double t = 0.5; // Middle of parent primitive (simplified)
                primitive.start_point = parent.start_point + direction * (t * length - primitive.radius);
                primitive.end_point = parent.start_point + direction * (t * length + primitive.radius);
            }
        }
    }
    
    // Transform Layer 1 primitives (simplified - use parent Layer 2)
    for (auto& primitive : layer_states_.layer1_primitives) {
        if (primitive.parent_layer2_index >= 0 && 
            primitive.parent_layer2_index < static_cast<int>(layer_states_.layer2_primitives.size())) {
            
            const auto& parent = layer_states_.layer2_primitives[primitive.parent_layer2_index];
            
            // Position sphere at midpoint of parent capsule (simplified)
            primitive.center = (parent.start_point + parent.end_point) * 0.5;
        }
    }
}

std::vector<Eigen::Vector3d> LayerManager::transform_vertices(const std::vector<int>& vertex_indices) {
    std::vector<Eigen::Vector3d> transformed_vertices;
    transformed_vertices.reserve(vertex_indices.size());
    
    for (int vertex_idx : vertex_indices) {
        if (vertex_idx >= 0 && vertex_idx < static_cast<int>(base_mesh_vertices_.size())) {
            // For now, use simple transformation
            // In full implementation, this would use STAR's forward kinematics
            Eigen::Vector3d original_vertex = base_mesh_vertices_[vertex_idx];
            
            // Simple bone-based transformation (placeholder)
            // This should be replaced with proper STAR mesh deformation
            Eigen::Vector3d transformed_vertex = calculate_bone_transformation(original_vertex, 0);
            
            transformed_vertices.push_back(transformed_vertex);
        }
    }
    
    return transformed_vertices;
}

Eigen::Vector3d LayerManager::calculate_bone_transformation(const Eigen::Vector3d& original_position, 
                                                           int bone_index) {
    // Simplified bone transformation
    // In full implementation, this would use proper bone weights and STAR's deformation model
    
    if (bone_index >= 0 && bone_index < static_cast<int>(current_bone_positions_.size())) {
        // For now, just apply a simple offset based on bone movement
        // This is a placeholder - real implementation needs proper skinning
        Eigen::Vector3d bone_offset = current_bone_positions_[bone_index];
        return original_position + bone_offset * 0.1; // Very simplified
    }
    
    return original_position;
}

void LayerManager::cool_down_layer2(int current_frame) {
    for (size_t i = 0; i < layer_states_.layer2_primitives.size(); ++i) {
        auto& primitive = layer_states_.layer2_primitives[i];
        
        if (primitive.is_active && 
            (current_frame - primitive.last_collision_frame) > cooldown_frames_) {
            primitive.is_active = false;
            // Note: No need to update global active flags as they don't exist anymore
        }
    }
}

void LayerManager::cool_down_layer1(int current_frame) {
    for (size_t i = 0; i < layer_states_.layer1_primitives.size(); ++i) {
        auto& primitive = layer_states_.layer1_primitives[i];
        
        if (primitive.is_active && 
            (current_frame - primitive.last_collision_frame) > cooldown_frames_) {
            primitive.is_active = false;
            // Note: No need to update global active flags as they don't exist anymore
        }
    }
}

void LayerManager::cool_down_layer0(int current_frame) {
    for (size_t i = 0; i < layer_states_.layer0_vertex_groups.size(); ++i) {
        auto& vertex_group = layer_states_.layer0_vertex_groups[i];
        
        if (vertex_group.is_loaded && 
            (current_frame - vertex_group.last_access_frame) > cooldown_frames_) {
            vertex_group.is_loaded = false;
            vertex_group.vertices.clear(); // Free memory
            // Note: No need to update global active flags as they don't exist anymore
        }
    }
}

double LayerManager::estimate_memory_usage() const {
    double memory_mb = 0.0;
    
    // Base mesh vertices
    memory_mb += base_mesh_vertices_.size() * sizeof(Eigen::Vector3d) / (1024.0 * 1024.0);
    
    // Layer primitives (small)
    memory_mb += layer_states_.layer3_primitives.size() * sizeof(Layer3Primitive) / (1024.0 * 1024.0);
    memory_mb += layer_states_.layer2_primitives.size() * sizeof(Layer2Primitive) / (1024.0 * 1024.0);
    memory_mb += layer_states_.layer1_primitives.size() * sizeof(Layer1Primitive) / (1024.0 * 1024.0);
    
    // Loaded Layer 0 vertices (this is the big one)
    for (const auto& vertex_group : layer_states_.layer0_vertex_groups) {
        if (vertex_group.is_loaded) {
            memory_mb += vertex_group.vertices.size() * sizeof(Eigen::Vector3d) / (1024.0 * 1024.0);
        }
    }
    
    // HDF5 mapping data
    memory_mb += hierarchy_mappings_.vertex_sphere_assignments.size() * sizeof(std::vector<int>) / (1024.0 * 1024.0);
    memory_mb += hierarchy_mappings_.sphere_to_capsule.size() * sizeof(int) / (1024.0 * 1024.0);
    memory_mb += hierarchy_mappings_.capsule_to_simple.size() * sizeof(int) / (1024.0 * 1024.0);
    
    return memory_mb;
}

} // namespace delta