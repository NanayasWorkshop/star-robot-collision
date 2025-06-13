#pragma once

#include "collision_types.hpp"
#include <vector>
#include <memory>

namespace delta {

// Forward declaration
struct CapsuleData;

/**
 * High-precision mesh collision detection for Layer 0
 * Handles capsule vs triangulated mesh collision with depth and normal calculation
 */
class MeshCollisionDetector {
private:
    // Performance tracking
    mutable double collision_computation_time_ms_;
    mutable double normal_computation_time_ms_;
    
    // Configuration
    double penetration_tolerance_;          // Minimum penetration depth to report
    int max_contacts_per_capsule_;         // Maximum contacts to calculate per capsule
    bool use_spatial_acceleration_;        // Use spatial indexing for large meshes
    
public:
    MeshCollisionDetector();
    ~MeshCollisionDetector();
    
    // =============================================================================
    // CONFIGURATION
    // =============================================================================
    
    /**
     * Set collision detection parameters
     * @param penetration_tolerance Minimum penetration depth to report (default: 1e-6)
     * @param max_contacts Maximum contacts per capsule (default: 10)
     * @param use_spatial_accel Use spatial acceleration structures (default: true)
     */
    void set_parameters(double penetration_tolerance = 1e-6, 
                       int max_contacts = 10,
                       bool use_spatial_accel = true);
    
    // =============================================================================
    // MAIN COLLISION DETECTION INTERFACE
    // =============================================================================
    
    /**
     * Detect collisions between a robot capsule and mesh vertices
     * @param capsule Robot capsule to test
     * @param vertices Mesh vertices in world space
     * @param triangles Optional triangle indices for triangulated mesh
     * @return Vector of collision contacts with depth and normals
     */
    std::vector<CollisionContact> detect_capsule_mesh_collision(
        const CapsuleData& capsule,
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<Eigen::Vector3i>& triangles = {});
    
    /**
     * Batch collision detection for multiple capsules against same mesh
     * More efficient than individual calls for large robot chains
     * @param capsules Vector of robot capsules
     * @param vertices Mesh vertices in world space  
     * @param triangles Optional triangle indices
     * @return Vector of all collision contacts found
     */
    std::vector<CollisionContact> detect_multi_capsule_mesh_collision(
        const std::vector<CapsuleData>& capsules,
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<Eigen::Vector3i>& triangles = {});
    
    // =============================================================================
    // SPECIALIZED COLLISION TESTS
    // =============================================================================
    
    /**
     * Fast point cloud collision (no triangulation)
     * Treats vertices as individual points and finds closest approaches
     * @param capsule Robot capsule
     * @param vertices Point cloud vertices
     * @return Collision contacts
     */
    std::vector<CollisionContact> detect_capsule_pointcloud_collision(
        const CapsuleData& capsule,
        const std::vector<Eigen::Vector3d>& vertices);
    
    /**
     * Triangulated mesh collision with proper surface normals
     * More accurate but slower than point cloud method
     * @param capsule Robot capsule
     * @param vertices Mesh vertices
     * @param triangles Triangle indices
     * @return Collision contacts with accurate surface normals
     */
    std::vector<CollisionContact> detect_capsule_triangulated_collision(
        const CapsuleData& capsule,
        const std::vector<Eigen::Vector3d>& vertices,
        const std::vector<Eigen::Vector3i>& triangles);
    
    // =============================================================================
    // PERFORMANCE AND DIAGNOSTICS
    // =============================================================================
    
    /**
     * Get performance timing information
     * @param collision_time_ms Time spent on collision detection
     * @param normal_time_ms Time spent computing normals
     */
    void get_performance_timings(double& collision_time_ms, double& normal_time_ms) const;
    
    /**
     * Reset performance counters
     */
    void reset_performance_timings();
    
    /**
     * Get collision detection statistics
     */
    struct CollisionStats {
        int total_vertex_tests;
        int total_triangle_tests;
        int contacts_found;
        double avg_penetration_depth;
        double max_penetration_depth;
    };
    
    CollisionStats get_collision_statistics() const;

private:
    // Statistics tracking
    mutable CollisionStats current_stats_;
    
    // =============================================================================
    // INTERNAL COLLISION PRIMITIVES
    // =============================================================================
    
    /**
     * Test if a point is inside a capsule
     * @param point Test point
     * @param capsule_start Capsule start point
     * @param capsule_end Capsule end point
     * @param capsule_radius Capsule radius
     * @param closest_point Output: closest point on capsule surface
     * @param penetration_depth Output: penetration depth (negative if outside)
     * @return true if point is inside capsule
     */
    bool point_in_capsule_detailed(const Eigen::Vector3d& point,
                                  const Eigen::Vector3d& capsule_start,
                                  const Eigen::Vector3d& capsule_end,
                                  double capsule_radius,
                                  Eigen::Vector3d& closest_point,
                                  double& penetration_depth) const;
    
    /**
     * Find closest point on line segment to a given point
     * @param point Test point
     * @param line_start Line segment start
     * @param line_end Line segment end
     * @return Closest point on line segment
     */
    Eigen::Vector3d closest_point_on_line_segment(const Eigen::Vector3d& point,
                                                 const Eigen::Vector3d& line_start,
                                                 const Eigen::Vector3d& line_end) const;
    
    /**
     * Test capsule against triangle for collision
     * @param capsule_start Capsule start point
     * @param capsule_end Capsule end point
     * @param capsule_radius Capsule radius
     * @param v0, v1, v2 Triangle vertices
     * @param contact_point Output: contact point if collision
     * @param surface_normal Output: surface normal at contact
     * @param penetration_depth Output: penetration depth
     * @return true if collision detected
     */
    bool capsule_triangle_collision(const Eigen::Vector3d& capsule_start,
                                   const Eigen::Vector3d& capsule_end,
                                   double capsule_radius,
                                   const Eigen::Vector3d& v0,
                                   const Eigen::Vector3d& v1,
                                   const Eigen::Vector3d& v2,
                                   Eigen::Vector3d& contact_point,
                                   Eigen::Vector3d& surface_normal,
                                   double& penetration_depth) const;
    
    /**
     * Calculate triangle normal
     * @param v0, v1, v2 Triangle vertices
     * @return Normalized triangle normal
     */
    Eigen::Vector3d calculate_triangle_normal(const Eigen::Vector3d& v0,
                                             const Eigen::Vector3d& v1,
                                             const Eigen::Vector3d& v2) const;
    
    /**
     * Find closest point on triangle to a given point
     * @param point Test point
     * @param v0, v1, v2 Triangle vertices
     * @return Closest point on triangle
     */
    Eigen::Vector3d closest_point_on_triangle(const Eigen::Vector3d& point,
                                             const Eigen::Vector3d& v0,
                                             const Eigen::Vector3d& v1,
                                             const Eigen::Vector3d& v2) const;
    
    /**
     * Test if point is inside triangle (barycentric coordinates)
     * @param point Test point (assumed to be on triangle plane)
     * @param v0, v1, v2 Triangle vertices
     * @return true if point is inside triangle
     */
    bool point_in_triangle(const Eigen::Vector3d& point,
                          const Eigen::Vector3d& v0,
                          const Eigen::Vector3d& v1,
                          const Eigen::Vector3d& v2) const;
    
    // =============================================================================
    // SPATIAL ACCELERATION (for large meshes)
    // =============================================================================
    
    /**
     * Simple spatial grid for accelerating collision queries
     * Only used for very large vertex sets (>1000 vertices)
     */
    struct SpatialGrid {
        struct GridCell {
            std::vector<int> vertex_indices;
            std::vector<int> triangle_indices;
        };
        
        std::vector<std::vector<std::vector<GridCell>>> grid;
        Eigen::Vector3d min_bounds;
        Eigen::Vector3d max_bounds;
        Eigen::Vector3d cell_size;
        int grid_resolution;
        bool is_built;
        
        SpatialGrid() : grid_resolution(16), is_built(false) {}
    };
    
    mutable SpatialGrid spatial_grid_;
    
    /**
     * Build spatial acceleration structure
     * @param vertices Mesh vertices
     * @param triangles Triangle indices (optional)
     */
    void build_spatial_grid(const std::vector<Eigen::Vector3d>& vertices,
                           const std::vector<Eigen::Vector3i>& triangles = {}) const;
    
    /**
     * Query spatial grid for potential collision candidates
     * @param capsule_start Capsule start point
     * @param capsule_end Capsule end point
     * @param capsule_radius Capsule radius
     * @param candidate_vertices Output: potential vertex indices
     * @param candidate_triangles Output: potential triangle indices
     */
    void query_spatial_grid(const Eigen::Vector3d& capsule_start,
                           const Eigen::Vector3d& capsule_end,
                           double capsule_radius,
                           std::vector<int>& candidate_vertices,
                           std::vector<int>& candidate_triangles) const;
    
    // =============================================================================
    // UTILITY METHODS
    // =============================================================================
    
    /**
     * Clamp value to range [min_val, max_val]
     */
    template<typename T>
    T clamp(T value, T min_val, T max_val) const {
        return std::max(min_val, std::min(value, max_val));
    }
    
    /**
     * Calculate bounding box of capsule
     * @param capsule_start Capsule start point
     * @param capsule_end Capsule end point
     * @param capsule_radius Capsule radius
     * @param min_bounds Output: minimum bounds
     * @param max_bounds Output: maximum bounds
     */
    void calculate_capsule_bounds(const Eigen::Vector3d& capsule_start,
                                 const Eigen::Vector3d& capsule_end,
                                 double capsule_radius,
                                 Eigen::Vector3d& min_bounds,
                                 Eigen::Vector3d& max_bounds) const;
};

} // namespace delta