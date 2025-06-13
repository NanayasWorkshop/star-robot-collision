#include "mesh_collision.hpp"
#include "capsule_creation_block.hpp" // For CapsuleData
#include <algorithm>
#include <cmath>
#include <limits>

namespace delta {

MeshCollisionDetector::MeshCollisionDetector() 
    : collision_computation_time_ms_(0.0), normal_computation_time_ms_(0.0),
      penetration_tolerance_(1e-6), max_contacts_per_capsule_(10), 
      use_spatial_acceleration_(true) {
    reset_performance_timings();
}

MeshCollisionDetector::~MeshCollisionDetector() {
}

// =============================================================================
// CONFIGURATION
// =============================================================================

void MeshCollisionDetector::set_parameters(double penetration_tolerance, 
                                          int max_contacts,
                                          bool use_spatial_accel) {
    penetration_tolerance_ = penetration_tolerance;
    max_contacts_per_capsule_ = max_contacts;
    use_spatial_acceleration_ = use_spatial_accel;
}

// =============================================================================
// MAIN COLLISION DETECTION INTERFACE
// =============================================================================

std::vector<CollisionContact> MeshCollisionDetector::detect_capsule_mesh_collision(
    const CapsuleData& capsule,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles) {
    
    ScopedTimer timer(collision_computation_time_ms_);
    
    current_stats_ = CollisionStats{}; // Reset stats
    
    if (vertices.empty()) {
        return {};
    }
    
    // Choose detection method based on whether triangles are provided
    if (!triangles.empty()) {
        return detect_capsule_triangulated_collision(capsule, vertices, triangles);
    } else {
        return detect_capsule_pointcloud_collision(capsule, vertices);
    }
}

std::vector<CollisionContact> MeshCollisionDetector::detect_multi_capsule_mesh_collision(
    const std::vector<CapsuleData>& capsules,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles) {
    
    ScopedTimer timer(collision_computation_time_ms_);
    
    std::vector<CollisionContact> all_contacts;
    current_stats_ = CollisionStats{};
    
    if (vertices.empty() || capsules.empty()) {
        return all_contacts;
    }
    
    // Build spatial acceleration if enabled and mesh is large
    if (use_spatial_acceleration_ && vertices.size() > 1000) {
        build_spatial_grid(vertices, triangles);
    }
    
    // Test each capsule against the mesh
    for (size_t capsule_idx = 0; capsule_idx < capsules.size(); ++capsule_idx) {
        const auto& capsule = capsules[capsule_idx];
        
        std::vector<CollisionContact> capsule_contacts;
        
        if (!triangles.empty()) {
            capsule_contacts = detect_capsule_triangulated_collision(capsule, vertices, triangles);
        } else {
            capsule_contacts = detect_capsule_pointcloud_collision(capsule, vertices);
        }
        
        // Set capsule index for all contacts
        for (auto& contact : capsule_contacts) {
            contact.robot_capsule_index = static_cast<int>(capsule_idx);
        }
        
        // Add to total results
        all_contacts.insert(all_contacts.end(), capsule_contacts.begin(), capsule_contacts.end());
    }
    
    return all_contacts;
}

// =============================================================================
// SPECIALIZED COLLISION TESTS
// =============================================================================

std::vector<CollisionContact> MeshCollisionDetector::detect_capsule_pointcloud_collision(
    const CapsuleData& capsule,
    const std::vector<Eigen::Vector3d>& vertices) {
    
    std::vector<CollisionContact> contacts;
    contacts.reserve(max_contacts_per_capsule_);
    
    current_stats_.total_vertex_tests += static_cast<int>(vertices.size());
    
    // Test each vertex against the capsule
    for (const auto& vertex : vertices) {
        Eigen::Vector3d closest_point;
        double penetration_depth;
        
        if (point_in_capsule_detailed(vertex, capsule.start_point, capsule.end_point, 
                                     capsule.radius, closest_point, penetration_depth)) {
            
            if (penetration_depth > penetration_tolerance_) {
                // Calculate surface normal (from contact point toward vertex)
                Eigen::Vector3d surface_normal = vertex - closest_point;
                double normal_length = surface_normal.norm();
                
                if (normal_length > 1e-12) {
                    surface_normal /= normal_length;
                } else {
                    // Fallback normal if vertex is exactly on capsule surface
                    surface_normal = Eigen::Vector3d(0, 0, 1);
                }
                
                CollisionContact contact(vertex, surface_normal, penetration_depth, -1);
                contacts.push_back(contact);
                
                current_stats_.contacts_found++;
                current_stats_.max_penetration_depth = std::max(current_stats_.max_penetration_depth, 
                                                               penetration_depth);
                
                // Limit number of contacts per capsule for performance
                if (contacts.size() >= static_cast<size_t>(max_contacts_per_capsule_)) {
                    break;
                }
            }
        }
    }
    
    // Calculate average penetration depth
    if (!contacts.empty()) {
        double total_depth = 0.0;
        for (const auto& contact : contacts) {
            total_depth += contact.penetration_depth;
        }
        current_stats_.avg_penetration_depth = total_depth / contacts.size();
    }
    
    return contacts;
}

std::vector<CollisionContact> MeshCollisionDetector::detect_capsule_triangulated_collision(
    const CapsuleData& capsule,
    const std::vector<Eigen::Vector3d>& vertices,
    const std::vector<Eigen::Vector3i>& triangles) {
    
    ScopedTimer normal_timer(normal_computation_time_ms_);
    
    std::vector<CollisionContact> contacts;
    contacts.reserve(max_contacts_per_capsule_);
    
    current_stats_.total_triangle_tests += static_cast<int>(triangles.size());
    
    // Get potential triangle candidates
    std::vector<int> candidate_triangles;
    std::vector<int> candidate_vertices;
    
    if (use_spatial_acceleration_ && spatial_grid_.is_built) {
        query_spatial_grid(capsule.start_point, capsule.end_point, capsule.radius,
                          candidate_vertices, candidate_triangles);
    } else {
        // Test all triangles
        candidate_triangles.resize(triangles.size());
        std::iota(candidate_triangles.begin(), candidate_triangles.end(), 0);
    }
    
    // Test each candidate triangle
    for (int tri_idx : candidate_triangles) {
        if (tri_idx < 0 || tri_idx >= static_cast<int>(triangles.size())) {
            continue;
        }
        
        const auto& triangle = triangles[tri_idx];
        
        // Validate triangle indices
        if (triangle.x() < 0 || triangle.x() >= static_cast<int>(vertices.size()) ||
            triangle.y() < 0 || triangle.y() >= static_cast<int>(vertices.size()) ||
            triangle.z() < 0 || triangle.z() >= static_cast<int>(vertices.size())) {
            continue;
        }
        
        const Eigen::Vector3d& v0 = vertices[triangle.x()];
        const Eigen::Vector3d& v1 = vertices[triangle.y()];
        const Eigen::Vector3d& v2 = vertices[triangle.z()];
        
        Eigen::Vector3d contact_point;
        Eigen::Vector3d surface_normal;
        double penetration_depth;
        
        if (capsule_triangle_collision(capsule.start_point, capsule.end_point, capsule.radius,
                                      v0, v1, v2, contact_point, surface_normal, penetration_depth)) {
            
            if (penetration_depth > penetration_tolerance_) {
                CollisionContact contact(contact_point, surface_normal, penetration_depth, -1);
                contacts.push_back(contact);
                
                current_stats_.contacts_found++;
                current_stats_.max_penetration_depth = std::max(current_stats_.max_penetration_depth, 
                                                               penetration_depth);
                
                // Limit contacts per capsule
                if (contacts.size() >= static_cast<size_t>(max_contacts_per_capsule_)) {
                    break;
                }
            }
        }
    }
    
    // Calculate average penetration depth
    if (!contacts.empty()) {
        double total_depth = 0.0;
        for (const auto& contact : contacts) {
            total_depth += contact.penetration_depth;
        }
        current_stats_.avg_penetration_depth = total_depth / contacts.size();
    }
    
    return contacts;
}

// =============================================================================
// PERFORMANCE AND DIAGNOSTICS
// =============================================================================

void MeshCollisionDetector::get_performance_timings(double& collision_time_ms, double& normal_time_ms) const {
    collision_time_ms = collision_computation_time_ms_;
    normal_time_ms = normal_computation_time_ms_;
}

void MeshCollisionDetector::reset_performance_timings() {
    collision_computation_time_ms_ = 0.0;
    normal_computation_time_ms_ = 0.0;
    current_stats_ = CollisionStats{};
}

MeshCollisionDetector::CollisionStats MeshCollisionDetector::get_collision_statistics() const {
    return current_stats_;
}

// =============================================================================
// INTERNAL COLLISION PRIMITIVES
// =============================================================================

bool MeshCollisionDetector::point_in_capsule_detailed(const Eigen::Vector3d& point,
                                                     const Eigen::Vector3d& capsule_start,
                                                     const Eigen::Vector3d& capsule_end,
                                                     double capsule_radius,
                                                     Eigen::Vector3d& closest_point,
                                                     double& penetration_depth) const {
    
    // Find closest point on capsule axis
    closest_point = closest_point_on_line_segment(point, capsule_start, capsule_end);
    
    // Calculate distance from point to closest point on axis
    Eigen::Vector3d to_point = point - closest_point;
    double distance = to_point.norm();
    
    // Check if point is inside capsule
    if (distance <= capsule_radius) {
        penetration_depth = capsule_radius - distance;
        
        // Move closest_point to surface of capsule
        if (distance > 1e-12) {
            closest_point += (to_point / distance) * capsule_radius;
        } else {
            // Point is exactly on axis - choose arbitrary direction
            closest_point += Eigen::Vector3d(capsule_radius, 0, 0);
        }
        
        return true;
    }
    
    penetration_depth = distance - capsule_radius; // Negative for outside
    return false;
}

Eigen::Vector3d MeshCollisionDetector::closest_point_on_line_segment(const Eigen::Vector3d& point,
                                                                    const Eigen::Vector3d& line_start,
                                                                    const Eigen::Vector3d& line_end) const {
    
    Eigen::Vector3d line_vec = line_end - line_start;
    double line_length_sq = line_vec.squaredNorm();
    
    if (line_length_sq < 1e-12) {
        // Degenerate line segment
        return line_start;
    }
    
    // Project point onto line
    double t = (point - line_start).dot(line_vec) / line_length_sq;
    
    // Clamp to line segment
    t = clamp(t, 0.0, 1.0);
    
    return line_start + t * line_vec;
}

bool MeshCollisionDetector::capsule_triangle_collision(const Eigen::Vector3d& capsule_start,
                                                      const Eigen::Vector3d& capsule_end,
                                                      double capsule_radius,
                                                      const Eigen::Vector3d& v0,
                                                      const Eigen::Vector3d& v1,
                                                      const Eigen::Vector3d& v2,
                                                      Eigen::Vector3d& contact_point,
                                                      Eigen::Vector3d& surface_normal,
                                                      double& penetration_depth) const {
    
    // Calculate triangle normal
    surface_normal = calculate_triangle_normal(v0, v1, v2);
    
    if (surface_normal.squaredNorm() < 1e-12) {
        // Degenerate triangle
        return false;
    }
    
    // Find closest point on triangle to capsule axis
    Eigen::Vector3d closest_on_capsule = closest_point_on_line_segment(v0, capsule_start, capsule_end);
    Eigen::Vector3d closest_on_triangle = closest_point_on_triangle(closest_on_capsule, v0, v1, v2);
    
    // Test capsule start point against triangle
    Eigen::Vector3d start_closest = closest_point_on_triangle(capsule_start, v0, v1, v2);
    double start_distance = (capsule_start - start_closest).norm();
    
    // Test capsule end point against triangle  
    Eigen::Vector3d end_closest = closest_point_on_triangle(capsule_end, v0, v1, v2);
    double end_distance = (capsule_end - end_closest).norm();
    
    // Find minimum distance
    double min_distance = std::min(start_distance, end_distance);
    contact_point = (start_distance < end_distance) ? start_closest : end_closest;
    
    // Also test middle of capsule for better accuracy
    Eigen::Vector3d capsule_mid = (capsule_start + capsule_end) * 0.5;
    Eigen::Vector3d mid_closest = closest_point_on_triangle(capsule_mid, v0, v1, v2);
    double mid_distance = (capsule_mid - mid_closest).norm();
    
    if (mid_distance < min_distance) {
        min_distance = mid_distance;
        contact_point = mid_closest;
    }
    
    // Check if collision occurs
    if (min_distance <= capsule_radius) {
        penetration_depth = capsule_radius - min_distance;
        
        // Ensure normal points away from capsule
        Eigen::Vector3d capsule_to_contact = contact_point - closest_on_capsule;
        if (surface_normal.dot(capsule_to_contact) < 0) {
            surface_normal = -surface_normal;
        }
        
        return true;
    }
    
    return false;
}

Eigen::Vector3d MeshCollisionDetector::calculate_triangle_normal(const Eigen::Vector3d& v0,
                                                                const Eigen::Vector3d& v1,
                                                                const Eigen::Vector3d& v2) const {
    
    Eigen::Vector3d edge1 = v1 - v0;
    Eigen::Vector3d edge2 = v2 - v0;
    Eigen::Vector3d normal = edge1.cross(edge2);
    
    double length = normal.norm();
    if (length > 1e-12) {
        normal /= length;
    }
    
    return normal;
}

Eigen::Vector3d MeshCollisionDetector::closest_point_on_triangle(const Eigen::Vector3d& point,
                                                                const Eigen::Vector3d& v0,
                                                                const Eigen::Vector3d& v1,
                                                                const Eigen::Vector3d& v2) const {
    
    // Check if point projects inside triangle
    Eigen::Vector3d edge0 = v1 - v0;
    Eigen::Vector3d edge1 = v2 - v1;
    Eigen::Vector3d edge2 = v0 - v2;
    
    Eigen::Vector3d v0_to_point = point - v0;
    Eigen::Vector3d v1_to_point = point - v1;
    Eigen::Vector3d v2_to_point = point - v2;
    
    // Check if point is in vertex regions
    if (edge0.dot(v0_to_point) <= 0 && edge2.dot(-v0_to_point) <= 0) {
        return v0;
    }
    
    if (edge1.dot(v1_to_point) <= 0 && edge0.dot(-v1_to_point) <= 0) {
        return v1;
    }
    
    if (edge2.dot(v2_to_point) <= 0 && edge1.dot(-v2_to_point) <= 0) {
        return v2;
    }
    
    // Check if point is in edge regions
    // Edge v0-v1
    double t01 = edge0.dot(v0_to_point) / edge0.squaredNorm();
    if (t01 > 0 && t01 < 1) {
        Eigen::Vector3d edge_point = v0 + t01 * edge0;
        Eigen::Vector3d edge_to_point = point - edge_point;
        Eigen::Vector3d triangle_normal = calculate_triangle_normal(v0, v1, v2);
        
        if (edge0.cross(triangle_normal).dot(edge_to_point) >= 0) {
            return edge_point;
        }
    }
    
    // Edge v1-v2
    double t12 = edge1.dot(v1_to_point) / edge1.squaredNorm();
    if (t12 > 0 && t12 < 1) {
        Eigen::Vector3d edge_point = v1 + t12 * edge1;
        Eigen::Vector3d edge_to_point = point - edge_point;
        Eigen::Vector3d triangle_normal = calculate_triangle_normal(v0, v1, v2);
        
        if (edge1.cross(triangle_normal).dot(edge_to_point) >= 0) {
            return edge_point;
        }
    }
    
    // Edge v2-v0
    double t20 = edge2.dot(v2_to_point) / edge2.squaredNorm();
    if (t20 > 0 && t20 < 1) {
        Eigen::Vector3d edge_point = v2 + t20 * edge2;
        Eigen::Vector3d edge_to_point = point - edge_point;
        Eigen::Vector3d triangle_normal = calculate_triangle_normal(v0, v1, v2);
        
        if (edge2.cross(triangle_normal).dot(edge_to_point) >= 0) {
            return edge_point;
        }
    }
    
    // Point projects inside triangle - project onto triangle plane
    Eigen::Vector3d triangle_normal = calculate_triangle_normal(v0, v1, v2);
    double distance_to_plane = triangle_normal.dot(point - v0);
    
    return point - distance_to_plane * triangle_normal;
}

bool MeshCollisionDetector::point_in_triangle(const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& v0,
                                             const Eigen::Vector3d& v1,
                                             const Eigen::Vector3d& v2) const {
    
    // Use barycentric coordinates
    Eigen::Vector3d v0v1 = v1 - v0;
    Eigen::Vector3d v0v2 = v2 - v0;
    Eigen::Vector3d v0p = point - v0;
    
    double dot00 = v0v2.dot(v0v2);
    double dot01 = v0v2.dot(v0v1);
    double dot02 = v0v2.dot(v0p);
    double dot11 = v0v1.dot(v0v1);
    double dot12 = v0v1.dot(v0p);
    
    double inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01);
    double u = (dot11 * dot02 - dot01 * dot12) * inv_denom;
    double v = (dot00 * dot12 - dot01 * dot02) * inv_denom;
    
    return (u >= 0) && (v >= 0) && (u + v <= 1);
}

// =============================================================================
// SPATIAL ACCELERATION
// =============================================================================

void MeshCollisionDetector::build_spatial_grid(const std::vector<Eigen::Vector3d>& vertices,
                                               const std::vector<Eigen::Vector3i>& triangles) const {
    
    if (vertices.empty()) {
        return;
    }
    
    // Calculate bounding box
    spatial_grid_.min_bounds = vertices[0];
    spatial_grid_.max_bounds = vertices[0];
    
    for (const auto& vertex : vertices) {
        for (int i = 0; i < 3; ++i) {
            spatial_grid_.min_bounds[i] = std::min(spatial_grid_.min_bounds[i], vertex[i]);
            spatial_grid_.max_bounds[i] = std::max(spatial_grid_.max_bounds[i], vertex[i]);
        }
    }
    
    // Add small padding
    Eigen::Vector3d padding = Eigen::Vector3d::Constant(0.1);
    spatial_grid_.min_bounds -= padding;
    spatial_grid_.max_bounds += padding;
    
    // Calculate cell size
    Eigen::Vector3d grid_size = spatial_grid_.max_bounds - spatial_grid_.min_bounds;
    spatial_grid_.cell_size = grid_size / spatial_grid_.grid_resolution;
    
    // Initialize grid
    spatial_grid_.grid.resize(spatial_grid_.grid_resolution);
    for (auto& x_layer : spatial_grid_.grid) {
        x_layer.resize(spatial_grid_.grid_resolution);
        for (auto& y_layer : x_layer) {
            y_layer.resize(spatial_grid_.grid_resolution);
        }
    }
    
    // Add vertices to grid
    for (size_t vertex_idx = 0; vertex_idx < vertices.size(); ++vertex_idx) {
        const auto& vertex = vertices[vertex_idx];
        
        // Calculate grid coordinates
        Eigen::Vector3i grid_coord;
        for (int i = 0; i < 3; ++i) {
            grid_coord[i] = static_cast<int>((vertex[i] - spatial_grid_.min_bounds[i]) / spatial_grid_.cell_size[i]);
            grid_coord[i] = clamp(grid_coord[i], 0, spatial_grid_.grid_resolution - 1);
        }
        
        spatial_grid_.grid[grid_coord.x()][grid_coord.y()][grid_coord.z()].vertex_indices.push_back(static_cast<int>(vertex_idx));
    }
    
    // Add triangles to grid (if provided)
    for (size_t tri_idx = 0; tri_idx < triangles.size(); ++tri_idx) {
        const auto& triangle = triangles[tri_idx];
        
        // Get triangle bounding box
        Eigen::Vector3d tri_min = vertices[triangle.x()];
        Eigen::Vector3d tri_max = vertices[triangle.x()];
        
        for (int i = 0; i < 3; ++i) {
            const Eigen::Vector3d& v = vertices[triangle[i]];
            for (int j = 0; j < 3; ++j) {
                tri_min[j] = std::min(tri_min[j], v[j]);
                tri_max[j] = std::max(tri_max[j], v[j]);
            }
        }
        
        // Add triangle to all overlapping cells
        Eigen::Vector3i min_cell, max_cell;
        for (int i = 0; i < 3; ++i) {
            min_cell[i] = static_cast<int>((tri_min[i] - spatial_grid_.min_bounds[i]) / spatial_grid_.cell_size[i]);
            max_cell[i] = static_cast<int>((tri_max[i] - spatial_grid_.min_bounds[i]) / spatial_grid_.cell_size[i]);
            min_cell[i] = clamp(min_cell[i], 0, spatial_grid_.grid_resolution - 1);
            max_cell[i] = clamp(max_cell[i], 0, spatial_grid_.grid_resolution - 1);
        }
        
        for (int x = min_cell.x(); x <= max_cell.x(); ++x) {
            for (int y = min_cell.y(); y <= max_cell.y(); ++y) {
                for (int z = min_cell.z(); z <= max_cell.z(); ++z) {
                    spatial_grid_.grid[x][y][z].triangle_indices.push_back(static_cast<int>(tri_idx));
                }
            }
        }
    }
    
    spatial_grid_.is_built = true;
}

void MeshCollisionDetector::query_spatial_grid(const Eigen::Vector3d& capsule_start,
                                               const Eigen::Vector3d& capsule_end,
                                               double capsule_radius,
                                               std::vector<int>& candidate_vertices,
                                               std::vector<int>& candidate_triangles) const {
    
    if (!spatial_grid_.is_built) {
        return;
    }
    
    // Calculate capsule bounding box
    Eigen::Vector3d min_bounds, max_bounds;
    calculate_capsule_bounds(capsule_start, capsule_end, capsule_radius, min_bounds, max_bounds);
    
    // Find overlapping grid cells
    Eigen::Vector3i min_cell, max_cell;
    for (int i = 0; i < 3; ++i) {
        min_cell[i] = static_cast<int>((min_bounds[i] - spatial_grid_.min_bounds[i]) / spatial_grid_.cell_size[i]);
        max_cell[i] = static_cast<int>((max_bounds[i] - spatial_grid_.min_bounds[i]) / spatial_grid_.cell_size[i]);
        min_cell[i] = clamp(min_cell[i], 0, spatial_grid_.grid_resolution - 1);
        max_cell[i] = clamp(max_cell[i], 0, spatial_grid_.grid_resolution - 1);
    }
    
    // Collect candidates from overlapping cells
    std::unordered_set<int> vertex_set;
    std::unordered_set<int> triangle_set;
    
    for (int x = min_cell.x(); x <= max_cell.x(); ++x) {
        for (int y = min_cell.y(); y <= max_cell.y(); ++y) {
            for (int z = min_cell.z(); z <= max_cell.z(); ++z) {
                const auto& cell = spatial_grid_.grid[x][y][z];
                
                for (int vertex_idx : cell.vertex_indices) {
                    vertex_set.insert(vertex_idx);
                }
                
                for (int triangle_idx : cell.triangle_indices) {
                    triangle_set.insert(triangle_idx);
                }
            }
        }
    }
    
    // Convert sets to vectors
    candidate_vertices.assign(vertex_set.begin(), vertex_set.end());
    candidate_triangles.assign(triangle_set.begin(), triangle_set.end());
}

// =============================================================================
// UTILITY METHODS
// =============================================================================

void MeshCollisionDetector::calculate_capsule_bounds(const Eigen::Vector3d& capsule_start,
                                                     const Eigen::Vector3d& capsule_end,
                                                     double capsule_radius,
                                                     Eigen::Vector3d& min_bounds,
                                                     Eigen::Vector3d& max_bounds) const {
    
    // Start with capsule endpoints
    min_bounds = capsule_start;
    max_bounds = capsule_start;
    
    for (int i = 0; i < 3; ++i) {
        min_bounds[i] = std::min(min_bounds[i], capsule_end[i]);
        max_bounds[i] = std::max(max_bounds[i], capsule_end[i]);
    }
    
    // Expand by capsule radius
    Eigen::Vector3d radius_expansion = Eigen::Vector3d::Constant(capsule_radius);
    min_bounds -= radius_expansion;
    max_bounds += radius_expansion;
}

} // namespace delta