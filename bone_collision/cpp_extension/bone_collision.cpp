// bone_collision.cpp - Clean implementation of bone-based collision system
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <chrono>
#include <memory>
#include <iostream>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

struct Vector3 {
    float x, y, z;
    
    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vector3 operator+(const Vector3& other) const {
        return Vector3(x + other.x, y + other.y, z + other.z);
    }
    
    Vector3 operator-(const Vector3& other) const {
        return Vector3(x - other.x, y - other.y, z - other.z);
    }
    
    Vector3 operator*(float scalar) const {
        return Vector3(x * scalar, y * scalar, z * scalar);
    }
    
    float dot(const Vector3& other) const {
        return x * other.x + y * other.y + z * other.z;
    }
    
    Vector3 cross(const Vector3& other) const {
        return Vector3(
            y * other.z - z * other.y,
            z * other.x - x * other.z,
            x * other.y - y * other.x
        );
    }
    
    float norm() const {
        return std::sqrt(x * x + y * y + z * z);
    }
    
    Vector3 normalized() const {
        float n = norm();
        if (n > 1e-8f) {
            return Vector3(x / n, y / n, z / n);
        }
        return Vector3(0, 0, 1);
    }
};

struct BoneCapsule {
    Vector3 start;
    Vector3 end;
    float radius;
    int bone_id;
    std::string bone_name;
    
    BoneCapsule() : radius(0.05f), bone_id(-1) {}
    BoneCapsule(Vector3 s, Vector3 e, float r, int id, const std::string& name) 
        : start(s), end(e), radius(r), bone_id(id), bone_name(name) {}
    
    float distance_to_point(const Vector3& point) const {
        Vector3 line_vec = end - start;
        Vector3 point_vec = point - start;
        
        float line_len_sq = line_vec.dot(line_vec);
        if (line_len_sq < 1e-8f) {
            return (point - start).norm();
        }
        
        float t = point_vec.dot(line_vec) / line_len_sq;
        t = std::max(0.0f, std::min(1.0f, t));
        
        Vector3 closest_point = start + line_vec * t;
        return (point - closest_point).norm();
    }
    
    bool intersects_sphere(const Vector3& sphere_center, float sphere_radius) const {
        return distance_to_point(sphere_center) <= (radius + sphere_radius);
    }
};

struct Triangle {
    Vector3 v1, v2, v3;
    
    Triangle() {}
    Triangle(const Vector3& a, const Vector3& b, const Vector3& c) : v1(a), v2(b), v3(c) {}
    
    Vector3 normal() const {
        Vector3 edge1 = v2 - v1;
        Vector3 edge2 = v3 - v1;
        return edge1.cross(edge2).normalized();
    }
    
    Vector3 center() const {
        return (v1 + v2 + v3) * (1.0f / 3.0f);
    }
};

struct CollisionResult {
    bool collision;
    Vector3 contact_point;
    Vector3 surface_normal;
    float penetration_distance;
    int triangle_id;
    
    CollisionResult() : collision(false), penetration_distance(0.0f), triangle_id(-1) {}
};

class BoneCollisionSystem {
private:
    std::vector<BoneCapsule> bone_capsules;
    std::vector<Triangle> triangles;
    std::vector<Vector3> vertices;
    std::unordered_map<int, std::vector<int>> bone_triangle_map;
    std::vector<double> timing_bone_filter;
    std::vector<double> timing_collision_check;
    
public:
    BoneCollisionSystem() {
        std::cout << "BoneCollisionSystem initialized" << std::endl;
    }
    
    void setup_bone_capsules(py::array_t<double> joint_positions, 
                           py::array_t<int> bone_connections,
                           py::array_t<double> bone_radii,
                           py::list bone_names) {
        
        auto joints_buf = joint_positions.request();
        auto connections_buf = bone_connections.request();
        auto radii_buf = bone_radii.request();
        
        if (joints_buf.ndim != 2 || joints_buf.shape[1] != 3) {
            throw std::runtime_error("Joint positions must be [n_joints, 3]");
        }
        if (connections_buf.ndim != 2 || connections_buf.shape[1] != 2) {
            throw std::runtime_error("Bone connections must be [n_bones, 2]");
        }
        
        double* joints_ptr = static_cast<double*>(joints_buf.ptr);
        int* connections_ptr = static_cast<int*>(connections_buf.ptr);
        double* radii_ptr = static_cast<double*>(radii_buf.ptr);
        
        int n_joints = joints_buf.shape[0];
        int n_bones = connections_buf.shape[0];
        
        bone_capsules.clear();
        bone_capsules.reserve(n_bones);
        
        for (int i = 0; i < n_bones; i++) {
            int joint1_idx = connections_ptr[i * 2 + 0];
            int joint2_idx = connections_ptr[i * 2 + 1];
            
            if (joint1_idx >= n_joints || joint2_idx >= n_joints) {
                throw std::runtime_error("Invalid joint index in bone connections");
            }
            
            Vector3 start(
                static_cast<float>(joints_ptr[joint1_idx * 3 + 0]),
                static_cast<float>(joints_ptr[joint1_idx * 3 + 1]),
                static_cast<float>(joints_ptr[joint1_idx * 3 + 2])
            );
            
            Vector3 end(
                static_cast<float>(joints_ptr[joint2_idx * 3 + 0]),
                static_cast<float>(joints_ptr[joint2_idx * 3 + 1]),
                static_cast<float>(joints_ptr[joint2_idx * 3 + 2])
            );
            
            float radius = static_cast<float>(radii_ptr[i]);
            std::string name = py::cast<std::string>(bone_names[i]);
            
            bone_capsules.emplace_back(start, end, radius, i, name);
        }
        
        std::cout << "Set up " << n_bones << " bone capsules" << std::endl;
    }
    
    void auto_tune_bone_radii(py::array_t<double> vertices_array,
                             py::array_t<double> skinning_weights,
                             double safety_margin = 1.15) {
        
        auto vertices_buf = vertices_array.request();
        auto weights_buf = skinning_weights.request();
        
        if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
            throw std::runtime_error("Vertices must be [n_vertices, 3]");
        }
        
        double* vertices_ptr = static_cast<double*>(vertices_buf.ptr);
        double* weights_ptr = static_cast<double*>(weights_buf.ptr);
        
        int n_vertices = vertices_buf.shape[0];
        int n_bones = weights_buf.shape[1];
        
        if (n_bones != static_cast<int>(bone_capsules.size())) {
            throw std::runtime_error("Number of bones in skinning weights doesn't match bone capsules");
        }
        
        std::cout << "Auto-tuning bone radii for " << n_vertices << " vertices..." << std::endl;
        
        for (int bone_id = 0; bone_id < n_bones; bone_id++) {
            float max_distance = 0.0f;
            int influenced_vertices = 0;
            
            for (int vertex_id = 0; vertex_id < n_vertices; vertex_id++) {
                double weight = weights_ptr[vertex_id * n_bones + bone_id];
                
                if (weight > 0.1) {
                    influenced_vertices++;
                    
                    Vector3 vertex(
                        static_cast<float>(vertices_ptr[vertex_id * 3 + 0]),
                        static_cast<float>(vertices_ptr[vertex_id * 3 + 1]),
                        static_cast<float>(vertices_ptr[vertex_id * 3 + 2])
                    );
                    
                    float distance = bone_capsules[bone_id].distance_to_point(vertex);
                    max_distance = std::max(max_distance, distance);
                }
            }
            
            if (influenced_vertices > 0) {
                bone_capsules[bone_id].radius = max_distance * static_cast<float>(safety_margin);
                std::cout << "Bone " << bone_id << " (" << bone_capsules[bone_id].bone_name 
                         << "): radius=" << bone_capsules[bone_id].radius 
                         << ", vertices=" << influenced_vertices << std::endl;
            } else {
                std::cout << "WARNING: Bone " << bone_id << " has no influenced vertices!" << std::endl;
            }
        }
        
        std::cout << "Bone radius auto-tuning complete" << std::endl;
    }
    
    void build_bone_triangle_mapping(py::array_t<double> vertices_array,
                                   py::array_t<int> faces_array,
                                   py::array_t<double> skinning_weights,
                                   double weight_threshold = 0.1) {
        
        auto vert_buf = vertices_array.request();
        auto face_buf = faces_array.request();
        auto weight_buf = skinning_weights.request();
        
        double* vert_ptr = static_cast<double*>(vert_buf.ptr);
        int* face_ptr = static_cast<int*>(face_buf.ptr);
        double* weight_ptr = static_cast<double*>(weight_buf.ptr);
        
        int n_verts = vert_buf.shape[0];
        int n_faces = face_buf.shape[0];
        int n_bones = weight_buf.shape[1];
        
        // Store vertices
        vertices.clear();
        vertices.reserve(n_verts);
        for (int i = 0; i < n_verts; i++) {
            vertices.emplace_back(
                static_cast<float>(vert_ptr[i * 3 + 0]),
                static_cast<float>(vert_ptr[i * 3 + 1]),
                static_cast<float>(vert_ptr[i * 3 + 2])
            );
        }
        
        // Store triangles
        triangles.clear();
        triangles.reserve(n_faces);
        for (int i = 0; i < n_faces; i++) {
            int v1_idx = face_ptr[i * 3 + 0];
            int v2_idx = face_ptr[i * 3 + 1];
            int v3_idx = face_ptr[i * 3 + 2];
            
            triangles.emplace_back(vertices[v1_idx], vertices[v2_idx], vertices[v3_idx]);
        }
        
        // Build bone to triangle mapping
        bone_triangle_map.clear();
        
        std::cout << "Building bone-triangle mapping..." << std::endl;
        
        for (int bone_id = 0; bone_id < n_bones; bone_id++) {
            std::vector<int> bone_triangles;
            
            for (int face_id = 0; face_id < n_faces; face_id++) {
                bool triangle_influenced = false;
                
                for (int vert_idx = 0; vert_idx < 3; vert_idx++) {
                    int vertex_id = face_ptr[face_id * 3 + vert_idx];
                    double weight = weight_ptr[vertex_id * n_bones + bone_id];
                    
                    if (weight > weight_threshold) {
                        triangle_influenced = true;
                        break;
                    }
                }
                
                if (triangle_influenced) {
                    bone_triangles.push_back(face_id);
                }
            }
            
            bone_triangle_map[bone_id] = bone_triangles;
            std::cout << "Bone " << bone_id << ": " << bone_triangles.size() << " triangles" << std::endl;
        }
        
        std::cout << "Bone-triangle mapping complete" << std::endl;
    }
    
    std::vector<int> find_bones_near_capsule(double x1, double y1, double z1,
                                           double x2, double y2, double z2,
                                           double radius) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        Vector3 capsule_start(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(z1));
        Vector3 capsule_end(static_cast<float>(x2), static_cast<float>(y2), static_cast<float>(z2));
        float capsule_radius = static_cast<float>(radius);
        
        std::vector<int> nearby_bones;
        
        for (const auto& bone_capsule : bone_capsules) {
            float dist_to_start = bone_capsule.distance_to_point(capsule_start);
            float dist_to_end = bone_capsule.distance_to_point(capsule_end);
            
            float combined_radius = bone_capsule.radius + capsule_radius;
            
            if (dist_to_start <= combined_radius || dist_to_end <= combined_radius) {
                nearby_bones.push_back(bone_capsule.bone_id);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        timing_bone_filter.push_back(duration.count() / 1000.0);
        
        return nearby_bones;
    }
    
    CollisionResult check_capsule_collision(double x1, double y1, double z1,
                                          double x2, double y2, double z2,
                                          double radius) {
        
        auto start = std::chrono::high_resolution_clock::now();
        
        CollisionResult result;
        
        // Phase 1: Find nearby bones
        std::vector<int> nearby_bones = find_bones_near_capsule(x1, y1, z1, x2, y2, z2, radius);
        
        if (nearby_bones.empty()) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            timing_collision_check.push_back(duration.count() / 1000.0);
            return result;
        }
        
        // Phase 2: Collect triangles from nearby bones
        std::vector<int> relevant_triangles;
        for (int bone_id : nearby_bones) {
            auto it = bone_triangle_map.find(bone_id);
            if (it != bone_triangle_map.end()) {
                const auto& bone_triangles = it->second;
                relevant_triangles.insert(relevant_triangles.end(), 
                                        bone_triangles.begin(), bone_triangles.end());
            }
        }
        
        // Remove duplicates
        std::sort(relevant_triangles.begin(), relevant_triangles.end());
        relevant_triangles.erase(std::unique(relevant_triangles.begin(), relevant_triangles.end()), 
                               relevant_triangles.end());
        
        // Phase 3: Check collision with relevant triangles
        Vector3 capsule_start(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(z1));
        Vector3 capsule_end(static_cast<float>(x2), static_cast<float>(y2), static_cast<float>(z2));
        float capsule_radius = static_cast<float>(radius);
        
        float min_distance = std::numeric_limits<float>::max();
        
        for (int triangle_id : relevant_triangles) {
            const Triangle& tri = triangles[triangle_id];
            
            // Simple triangle-capsule collision check
            Vector3 tri_center = tri.center();
            
            // Distance from triangle center to capsule line segment
            Vector3 line_vec = capsule_end - capsule_start;
            Vector3 point_vec = tri_center - capsule_start;
            
            float line_len_sq = line_vec.dot(line_vec);
            float t = 0.0f;
            if (line_len_sq > 1e-8f) {
                t = std::max(0.0f, std::min(1.0f, point_vec.dot(line_vec) / line_len_sq));
            }
            
            Vector3 closest_point_on_capsule = capsule_start + line_vec * t;
            float distance = (tri_center - closest_point_on_capsule).norm();
            
            if (distance < capsule_radius && distance < min_distance) {
                result.collision = true;
                result.contact_point = closest_point_on_capsule;
                result.surface_normal = tri.normal();
                result.penetration_distance = capsule_radius - distance;
                result.triangle_id = triangle_id;
                min_distance = distance;
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        timing_collision_check.push_back(duration.count() / 1000.0);
        
        return result;
    }
    
    std::vector<double> get_performance_stats() {
        double avg_bone_filter = 0.0;
        double avg_collision = 0.0;
        
        if (!timing_bone_filter.empty()) {
            for (double t : timing_bone_filter) avg_bone_filter += t;
            avg_bone_filter /= timing_bone_filter.size();
        }
        
        if (!timing_collision_check.empty()) {
            for (double t : timing_collision_check) avg_collision += t;
            avg_collision /= timing_collision_check.size();
        }
        
        return {avg_bone_filter, avg_collision};
    }
    
    std::vector<double> get_bone_capsule_info(int bone_id) {
        if (bone_id < 0 || bone_id >= static_cast<int>(bone_capsules.size())) {
            return {};
        }
        
        const BoneCapsule& capsule = bone_capsules[bone_id];
        return {
            capsule.start.x, capsule.start.y, capsule.start.z,
            capsule.end.x, capsule.end.y, capsule.end.z,
            capsule.radius
        };
    }
    
    int get_num_bones() const {
        return static_cast<int>(bone_capsules.size());
    }
    
    std::vector<std::string> get_bone_names() const {
        std::vector<std::string> names;
        for (const auto& capsule : bone_capsules) {
            names.push_back(capsule.bone_name);
        }
        return names;
    }
};

// Python bindings
PYBIND11_MODULE(bone_collision, m) {
    m.doc() = "High-performance bone-based collision detection system";
    
    py::class_<CollisionResult>(m, "CollisionResult")
        .def_readwrite("collision", &CollisionResult::collision)
        .def_readwrite("contact_point", &CollisionResult::contact_point)
        .def_readwrite("surface_normal", &CollisionResult::surface_normal)
        .def_readwrite("penetration_distance", &CollisionResult::penetration_distance)
        .def_readwrite("triangle_id", &CollisionResult::triangle_id);
    
    py::class_<Vector3>(m, "Vector3")
        .def(py::init<>())
        .def(py::init<float, float, float>())
        .def_readwrite("x", &Vector3::x)
        .def_readwrite("y", &Vector3::y)
        .def_readwrite("z", &Vector3::z)
        .def("norm", &Vector3::norm)
        .def("normalized", &Vector3::normalized);
    
    py::class_<BoneCollisionSystem>(m, "BoneCollisionSystem")
        .def(py::init<>())
        .def("setup_bone_capsules", &BoneCollisionSystem::setup_bone_capsules,
             "Set up bone capsules from joint positions and connections")
        .def("auto_tune_bone_radii", &BoneCollisionSystem::auto_tune_bone_radii,
             "Auto-tune bone capsule radii to cover all influenced vertices",
             py::arg("vertices_array"), py::arg("skinning_weights"), py::arg("safety_margin") = 1.15)
        .def("build_bone_triangle_mapping", &BoneCollisionSystem::build_bone_triangle_mapping,
             "Build mapping from bones to triangles they influence",
             py::arg("vertices_array"), py::arg("faces_array"), py::arg("skinning_weights"), py::arg("weight_threshold") = 0.1)
        .def("find_bones_near_capsule", &BoneCollisionSystem::find_bones_near_capsule,
             "Find bones near given capsule")
        .def("check_capsule_collision", &BoneCollisionSystem::check_capsule_collision,
             "Check collision between capsule and human mesh")
        .def("get_performance_stats", &BoneCollisionSystem::get_performance_stats,
             "Get performance statistics")
        .def("get_bone_capsule_info", &BoneCollisionSystem::get_bone_capsule_info,
             "Get bone capsule information")
        .def("get_num_bones", &BoneCollisionSystem::get_num_bones,
             "Get number of bones")
        .def("get_bone_names", &BoneCollisionSystem::get_bone_names,
             "Get bone names");
}