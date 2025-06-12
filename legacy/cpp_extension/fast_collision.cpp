// fast_collision.cpp - v4 with correct FCL addSubModel usage
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <fcl/fcl.h>
#include <vector>
#include <chrono>
#include <memory>
#include <iostream>

namespace py = pybind11;
using namespace fcl;

class FastBVHCollider {
private:
    std::shared_ptr<BVHModel<OBBRSSf>> bvh_model;
    std::shared_ptr<CollisionObject<float>> collision_object;
    bool initialized = false;
    int num_vertices = 0;
    int num_faces = 0;
    
public:
    FastBVHCollider() {
        bvh_model = std::make_shared<BVHModel<OBBRSSf>>();
    }
    
    void create_from_mesh(py::array_t<double> vertices_array, 
                         py::array_t<int> faces_array) {
        
        auto vertices_buf = vertices_array.request();
        auto faces_buf = faces_array.request();
        
        if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
            throw std::runtime_error("Vertices must be (N, 3) array");
        }
        if (faces_buf.ndim != 2 || faces_buf.shape[1] != 3) {
            throw std::runtime_error("Faces must be (M, 3) array");
        }
        
        num_vertices = vertices_buf.shape[0];
        num_faces = faces_buf.shape[0];
        
        double* vertices_ptr = static_cast<double*>(vertices_buf.ptr);
        int* faces_ptr = static_cast<int*>(faces_buf.ptr);
        
        // Create new BVH model
        bvh_model = std::make_shared<BVHModel<OBBRSSf>>();
        
        // Convert vertices to FCL format
        std::vector<Vector3f> fcl_vertices;
        fcl_vertices.reserve(num_vertices);
        for (int i = 0; i < num_vertices; i++) {
            fcl_vertices.emplace_back(
                static_cast<float>(vertices_ptr[i * 3 + 0]),
                static_cast<float>(vertices_ptr[i * 3 + 1]),
                static_cast<float>(vertices_ptr[i * 3 + 2])
            );
        }
        
        // Convert triangles to FCL format
        std::vector<Triangle> fcl_triangles;
        fcl_triangles.reserve(num_faces);
        for (int i = 0; i < num_faces; i++) {
            fcl_triangles.emplace_back(
                faces_ptr[i * 3 + 0],
                faces_ptr[i * 3 + 1], 
                faces_ptr[i * 3 + 2]
            );
        }
        
        // Begin model creation
        int result = bvh_model->beginModel(num_faces, num_vertices);
        if (result != BVH_OK) {
            std::cerr << "Failed to begin BVH model creation, error code: " << result << std::endl;
            throw std::runtime_error("Failed to begin BVH model creation");
        }
        
        // Use addSubModel for correct mesh construction with shared vertices
        result = bvh_model->addSubModel(fcl_vertices, fcl_triangles);
        if (result != BVH_OK) {
            std::cerr << "Failed to add sub model, error code: " << result << std::endl;
            throw std::runtime_error("Failed to add sub model to BVH model");
        }
        
        // End model creation - this builds the BVH tree
        result = bvh_model->endModel();
        if (result != BVH_OK) {
            std::cerr << "Failed to end BVH model creation, error code: " << result << std::endl;
            throw std::runtime_error("Failed to end BVH model creation");
        }
        
        // Create collision object
        Transform3f transform = Transform3f::Identity();
        collision_object = std::make_shared<CollisionObject<float>>(bvh_model, transform);
        
        initialized = true;
        
        std::cout << "Created BVH with " << num_vertices << " vertices, " << num_faces << " faces" << std::endl;
        std::cout << "BVH build state: " << static_cast<int>(bvh_model->build_state) << std::endl;
        std::cout << "BVH actual vertices: " << bvh_model->num_vertices << std::endl;
        std::cout << "BVH actual triangles: " << bvh_model->num_tris << std::endl;
    }
    
    double update_vertices(py::array_t<double> vertices_array) {
        if (!initialized) {
            throw std::runtime_error("BVH not initialized. Call create_from_mesh first.");
        }
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto vertices_buf = vertices_array.request();
        if (vertices_buf.ndim != 2 || vertices_buf.shape[1] != 3) {
            throw std::runtime_error("Vertices must be (N, 3) array");
        }
        if (vertices_buf.shape[0] != num_vertices) {
            std::cerr << "Vertex count mismatch: expected " << num_vertices 
                      << ", got " << vertices_buf.shape[0] << std::endl;
            throw std::runtime_error("Vertex count mismatch");
        }
        
        double* vertices_ptr = static_cast<double*>(vertices_buf.ptr);
        
        // Check current build state
        std::cout << "Current build state before update: " << static_cast<int>(bvh_model->build_state) << std::endl;
        std::cout << "BVH internal vertex count: " << bvh_model->num_vertices << std::endl;
        
        // Step 1: Begin update model (preserves topology)
        int result = bvh_model->beginUpdateModel();
        if (result != BVH_OK) {
            std::cerr << "Failed to begin BVH model update, error code: " << result << std::endl;
            std::cerr << "Current build state: " << static_cast<int>(bvh_model->build_state) << std::endl;
            throw std::runtime_error("Failed to begin BVH model update");
        }
        
        std::cout << "Build state after beginUpdateModel: " << static_cast<int>(bvh_model->build_state) << std::endl;
        
        // Step 2: Update vertices using updateSubModel for efficiency
        std::vector<Vector3f> new_vertices;
        new_vertices.reserve(num_vertices);
        
        for (int i = 0; i < num_vertices; i++) {
            new_vertices.emplace_back(
                static_cast<float>(vertices_ptr[i * 3 + 0]),
                static_cast<float>(vertices_ptr[i * 3 + 1]),
                static_cast<float>(vertices_ptr[i * 3 + 2])
            );
        }
        
        // Use updateSubModel instead of individual updateVertex calls
        result = bvh_model->updateSubModel(new_vertices);
        if (result != BVH_OK) {
            std::cerr << "Failed to update vertices in BVH model, error code: " << result << std::endl;
            throw std::runtime_error("Failed to update vertices in BVH model");
        }
        
        std::cout << "Build state after updateSubModel: " << static_cast<int>(bvh_model->build_state) << std::endl;
        
        // Step 3: End update with refitting (bottomup=true for speed)
        result = bvh_model->endUpdateModel(true, true);  // refit=true, bottomup=true
        if (result != BVH_OK) {
            std::cerr << "Failed to end BVH model update, error code: " << result << std::endl;
            std::cerr << "Build state: " << static_cast<int>(bvh_model->build_state) << std::endl;
            std::cerr << "Number of vertices in model: " << bvh_model->num_vertices << std::endl;
            std::cerr << "Expected vertices: " << num_vertices << std::endl;
            throw std::runtime_error("Failed to end BVH model update");
        }
        
        std::cout << "Build state after endUpdateModel: " << static_cast<int>(bvh_model->build_state) << std::endl;
        
        // Update collision object transform (usually stays identity)
        Transform3f transform = Transform3f::Identity();
        collision_object->setTransform(transform);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        return duration.count() / 1000.0; // Return time in milliseconds
    }
    
    std::vector<double> get_bounding_box() {
        if (!initialized) return {0, 0, 0, 0, 0, 0};
        
        AABB<float> aabb = bvh_model->aabb_local;
        return {
            aabb.min_[0], aabb.min_[1], aabb.min_[2],
            aabb.max_[0], aabb.max_[1], aabb.max_[2]
        };
    }
    
    bool check_collision_with_sphere(double x, double y, double z, double radius) {
        if (!initialized) return false;
        
        // Create sphere collision object
        auto sphere = std::make_shared<Sphere<float>>(static_cast<float>(radius));
        Transform3f sphere_transform = Transform3f::Identity();
        sphere_transform.translation() = Vector3f(
            static_cast<float>(x), 
            static_cast<float>(y), 
            static_cast<float>(z)
        );
        CollisionObject<float> sphere_obj(sphere, sphere_transform);
        
        // Perform collision check
        CollisionRequest<float> request;
        CollisionResult<float> result;
        
        return collide(collision_object.get(), &sphere_obj, request, result) > 0;
    }
    
    bool check_collision_with_capsule(double x1, double y1, double z1,
                                     double x2, double y2, double z2,
                                     double radius) {
        if (!initialized) return false;
        
        // Calculate capsule height and center
        Vector3f p1(static_cast<float>(x1), static_cast<float>(y1), static_cast<float>(z1));
        Vector3f p2(static_cast<float>(x2), static_cast<float>(y2), static_cast<float>(z2));
        Vector3f center = (p1 + p2) * 0.5f;
        float height = (p2 - p1).norm();
        
        // Create capsule
        auto capsule = std::make_shared<Capsule<float>>(static_cast<float>(radius), height);
        
        // Calculate rotation to align capsule with line segment
        Vector3f direction = (p2 - p1).normalized();
        Vector3f z_axis(0, 0, 1);
        
        Transform3f capsule_transform = Transform3f::Identity();
        capsule_transform.translation() = center;
        
        // If direction is not aligned with z-axis, rotate
        if (std::abs(direction.dot(z_axis)) < 0.999f) {
            Vector3f rotation_axis = z_axis.cross(direction).normalized();
            float dot_product = z_axis.dot(direction);
            dot_product = std::max(-1.0f, std::min(1.0f, dot_product)); // Manual clamp for C++14
            float angle = std::acos(dot_product);
            AngleAxisf rotation(angle, rotation_axis);
            capsule_transform.linear() = rotation.toRotationMatrix();
        }
        
        CollisionObject<float> capsule_obj(capsule, capsule_transform);
        
        // Perform collision check
        CollisionRequest<float> request;
        CollisionResult<float> result;
        
        return collide(collision_object.get(), &capsule_obj, request, result) > 0;
    }
    
    double compute_distance_to_sphere(double x, double y, double z, double radius) {
        if (!initialized) return -1.0;
        
        auto sphere = std::make_shared<Sphere<float>>(static_cast<float>(radius));
        Transform3f sphere_transform = Transform3f::Identity();
        sphere_transform.translation() = Vector3f(
            static_cast<float>(x), 
            static_cast<float>(y), 
            static_cast<float>(z)
        );
        CollisionObject<float> sphere_obj(sphere, sphere_transform);
        
        DistanceRequest<float> request;
        DistanceResult<float> result;
        
        distance(collision_object.get(), &sphere_obj, request, result);
        
        return static_cast<double>(result.min_distance);
    }
    
    std::vector<int> get_info() {
        if (!initialized) return {0, 0};
        return {num_vertices, num_faces};
    }
    
    bool is_initialized() const {
        return initialized;
    }
    
    // Debug method to get build state
    int get_build_state() const {
        if (!initialized) return -1;
        return static_cast<int>(bvh_model->build_state);
    }
    
    // Debug method to get BVH statistics
    std::vector<int> get_bvh_stats() const {
        if (!initialized) return {0, 0, 0};
        return {
            bvh_model->getNumBVs(),  // Number of BV nodes
            bvh_model->num_vertices, // Actual vertex count in BVH
            bvh_model->num_tris      // Actual triangle count in BVH
        };
    }
};

// Multi-object collision manager for robot parts
class CollisionManager {
private:
    std::vector<std::shared_ptr<FastBVHCollider>> objects;
    std::vector<std::string> object_names;
    
public:
    void add_object(const std::string& name, std::shared_ptr<FastBVHCollider> collider) {
        object_names.push_back(name);
        objects.push_back(collider);
    }
    
    std::vector<std::string> check_collisions_with_sphere(double x, double y, double z, double radius) {
        std::vector<std::string> colliding_objects;
        
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->check_collision_with_sphere(x, y, z, radius)) {
                colliding_objects.push_back(object_names[i]);
            }
        }
        
        return colliding_objects;
    }
    
    std::vector<std::string> check_collisions_with_capsule(double x1, double y1, double z1,
                                                          double x2, double y2, double z2,
                                                          double radius) {
        std::vector<std::string> colliding_objects;
        
        for (size_t i = 0; i < objects.size(); i++) {
            if (objects[i]->check_collision_with_capsule(x1, y1, z1, x2, y2, z2, radius)) {
                colliding_objects.push_back(object_names[i]);
            }
        }
        
        return colliding_objects;
    }
    
    size_t size() const {
        return objects.size();
    }
    
    std::vector<std::string> get_object_names() const {
        return object_names;
    }
};

// Python bindings
PYBIND11_MODULE(fast_collision, m) {
    m.doc() = "High-performance FCL collision detection with PROPER BVH refitting v4";
    
    py::class_<FastBVHCollider, std::shared_ptr<FastBVHCollider>>(m, "FastBVHCollider")
        .def(py::init<>())
        .def("create_from_mesh", &FastBVHCollider::create_from_mesh,
             "Create BVH from mesh vertices and faces")
        .def("update_vertices", &FastBVHCollider::update_vertices,
             "Update BVH vertices with PROPER FCL refitting. Returns update time in ms.")
        .def("get_bounding_box", &FastBVHCollider::get_bounding_box,
             "Get axis-aligned bounding box [min_x, min_y, min_z, max_x, max_y, max_z]")
        .def("check_collision_with_sphere", &FastBVHCollider::check_collision_with_sphere,
             "Check collision with sphere at (x, y, z) with given radius")
        .def("check_collision_with_capsule", &FastBVHCollider::check_collision_with_capsule,
             "Check collision with capsule from (x1,y1,z1) to (x2,y2,z2) with radius")
        .def("compute_distance_to_sphere", &FastBVHCollider::compute_distance_to_sphere,
             "Compute minimum distance to sphere")
        .def("get_info", &FastBVHCollider::get_info,
             "Get [num_vertices, num_faces]")
        .def("is_initialized", &FastBVHCollider::is_initialized,
             "Check if BVH is initialized")
        .def("get_build_state", &FastBVHCollider::get_build_state,
             "Get current BVH build state for debugging")
        .def("get_bvh_stats", &FastBVHCollider::get_bvh_stats,
             "Get BVH statistics [num_bvs, actual_vertices, actual_faces]");
    
    py::class_<CollisionManager, std::shared_ptr<CollisionManager>>(m, "CollisionManager")
        .def(py::init<>())
        .def("add_object", &CollisionManager::add_object,
             "Add named collision object to manager")
        .def("check_collisions_with_sphere", &CollisionManager::check_collisions_with_sphere,
             "Check sphere collision with all objects, return list of colliding object names")
        .def("check_collisions_with_capsule", &CollisionManager::check_collisions_with_capsule,
             "Check capsule collision with all objects, return list of colliding object names")
        .def("size", &CollisionManager::size,
             "Get number of objects in manager")
        .def("get_object_names", &CollisionManager::get_object_names,
             "Get list of all object names");
}