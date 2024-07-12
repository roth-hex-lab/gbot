// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include <filesystem/filesystem.h>
#include <m3t/generator.h>
#include <m3t/tracker.h>
#include <opencv2/core/eigen.hpp> 


#include <thread>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <restbed>
#include <nanoflann/KDTreeVectorOfVectorAdaptor.hpp>
#include "chrono"
#include "yolov8-pose.hpp"
 
using namespace std;
using namespace restbed;

#ifdef _WIN32
#define tcout std::wcout
#else
#define tcout std::cout
#endif

using namespace m3t;

int assembly_state = 1;
string json = "{ \"Hello\": \", World!\" }";
bool server_running = false;
m3t::Transform3fA link1_pose;
m3t::Transform3fA link2_pose;
m3t::Transform3fA link3_pose;
std::vector<std::string> object_names{"GearedCaliper"};
//std::filesystem::path p = std::filesystem::current_path();


std::filesystem::path dataset_directory{
    "../../data"};
std::filesystem::path external_directory{
"../../data/GearedCaliper/external"};
std::string engine_file_path =
    "../../data/GearedCaliper/train/weights";
std::string track_mode = "GBOT";
std::string yolo_mode = "single";
bool use_random_seed = false;
int n_vertices_evaluation = -1;
// Parameters for tracker configuration
bool use_shared_color_histograms = true;
bool use_region_checking = true;
bool use_silhouette_checking = true;
bool visualize_tracking = true;
bool save_images = true;

struct Assemblystate {
std::vector<std::string> tracker_string{};
std::vector<std::vector<std::string>> Ref{};
std::vector<bool> symmetry;
std::vector<Eigen::Matrix3f> R{};
std::vector<Eigen::Vector3f> T{};
std::vector<std::shared_ptr<m3t::Tracker>> tracker_ptrs{};
};

static constexpr size_t kNCurveValues = 100;

struct ExecutionTimes {
float calculate_correspondences = 0.0f;
float calculate_gradient_and_hessian = 0.0f;
float calculate_optimization = 0.0f;
float calculate_results = 0.0f;
float complete_cycle = 0.0f;
};

struct Result {
int frame_index = 0;
float add_auc = 0.0f;
float adds_auc = 0.0f;
std::array<float, kNCurveValues> add_curve{};
std::array<float, kNCurveValues> adds_curve{};
ExecutionTimes execution_times{};
std::vector<std::string> add_poses{};
};

cv::Matx33d cam_K(607.249, 0.0, 639.167, 0.0, 607.165, 364.762, 0.0, 0.0, 1.0);

std::vector<std::vector<cv::Point3f>> kpts_gt;

std::vector<YOLOv8_pose *> yolo_pool;
YOLOv8_pose* yolov8_pose;


  // Internal data objects
typedef KDTreeVectorOfVectorsAdaptor<std::vector<Eigen::Vector3f>, float>
    KDTreeVector3f;
std::array<float, kNCurveValues> thresholds_{};
std::map<std::string, int> body_name2idx_map_{};
std::vector<std::shared_ptr<m3t::RegionModel>> region_model_ptrs_{};
std::vector<std::shared_ptr<m3t::DepthModel>> depth_model_ptrs_{};
std::vector<std::vector<Eigen::Vector3f>> reduced_vertices_vector_{};
std::vector<std::unique_ptr<KDTreeVector3f>> kdtree_ptr_vector_{};
std::vector<std::vector<std::string>> ids_combined_bodies_{};
float error_threshold_;
float max_model_contour_length_;
float max_model_surface_area_;
std::map<std::string, Result> final_results_{};

  // Setters for object setters
std::function<void(std::shared_ptr<m3t::Tracker>)> tracker_setter_{[](auto) {}};
std::function<void(std::shared_ptr<m3t::Optimizer>)> optimizer_setter_{
    [](auto) {}};
std::function<void(std::shared_ptr<m3t::RegionModality>)>
    region_modality_setter_{[](auto) {}};
std::function<void(std::shared_ptr<m3t::ColorHistograms>)>
    color_histograms_setter_{[](auto) {}};
std::function<void(std::shared_ptr<m3t::RegionModel>)> region_model_setter_{
    [](auto) {}};
std::function<void(std::shared_ptr<m3t::DepthModality>)> depth_modality_setter_{
    [](auto) {}};
std::function<void(std::shared_ptr<m3t::DepthModel>)> depth_model_setter_{
    [](auto) {}};

void get_method_handler(const shared_ptr<Session> session) {
  session->close(OK, json,
                 {{"Connection", "close"}});
}

void get_xml_method_handler(const shared_ptr<Session> session) {
  const multimap<string, string> headers{{"Content-Length", "30"},
                                         {"Content-Type", "application/xml"}};

  session->close(200, "<hello><world></world></hello>", headers);
}

void get_json_method_handler(const shared_ptr<Session> session) {
  const multimap<string, string> headers{{"Content-Length", "23"},
                                         {"Content-Type", "application/json"}};

  session->close(200, "{ \"Hello\": \", World!\" }", headers);
}

void failed_filter_validation_handler(const shared_ptr<Session> session) {
  session->close(400);
}

void service_ready_handler(Service&) {
  fprintf(stderr, "Hey! The service is up and running.\n");
}


float calcRotationAngle(Eigen::Matrix3f R1, Eigen::Matrix3f R2) {
  Eigen::Matrix3f R =
      R2 * R1.transpose();    // Calculate the relative rotation matrix
  float trace_R = R.trace();  // Calculate the trace of R
  float cos_theta =
      (trace_R - 1.0) / 2.0;  // Calculate the cosine of the rotation angle
  float sin_theta = sqrt(
      1.0 - cos_theta * cos_theta);  // Calculate the sine of the rotation angle
  float theta = acos(cos_theta);     // Calculate the rotation angle in radians
  float pi = 2 * acos(0.0);
  theta =
      theta * 180.0 / pi;  // Convert the rotation angle from radians to degrees

  // Determine the sign of the rotation angle
  if (sin_theta >= 0.0) {
    return theta;
  } else {
    return -theta;
  }
}

float calcVectorDifference(Eigen::Vector3f v1, Eigen::Vector3f v2) {
  Eigen::Vector3f diff = v2 - v1;  // Calculate the difference vector
  float norm_diff = diff.norm();  // Calculate the norm of the difference vector
  return norm_diff;               // Return the norm of the difference vector
}

void cppserver() 
{
  auto resource = make_shared<Resource>();
  resource->set_path("/resource");
  resource->set_failed_filter_validation_handler(
      failed_filter_validation_handler);
  resource->set_method_handler("GET", get_method_handler);
  /*
  resource->set_method_handler(
      "GET",
      {{"Accept", "application/xml"}, {"Content-Type", "application/xml"}},
      &get_xml_method_handler);


  resource->set_method_handler(
      "GET",
      {{"Accept", "application/json"}, {"Content-Type", "application/json"}},
      &get_json_method_handler);
      */
  auto settings = make_shared<Settings>();
  settings->set_port(1984);
  settings->set_default_header("Connection", "close");


  auto service = make_shared<Service>();
  service->publish(resource);
  service->set_ready_handler(service_ready_handler);
  service->start(settings);

  //Service service;
  //service.publish(resource);
  //service.start(settings);


}

std::string toString(const Eigen::MatrixXf& mat) {
  std::stringstream ss;
  ss << mat;
  return ss.str();
}

std::string toString_v(const Eigen::VectorXf & mat) {
  std::stringstream ss;
  ss << mat;
  return ss.str();
}
    /**
 * Class that visualizes the drawing of a stabilo pen on a paper
 */



bool UpdatePublisher(const std::shared_ptr<m3t::Tracker> &tracker_ptr_) {
// Calculate pose between two assembly parts
json = "[";
for (const auto &[body_name, idx] : body_name2idx_map_) {
    const auto &body_ptr_ = *std::find_if(
        begin(tracker_ptr_->body_ptrs()), end(tracker_ptr_->body_ptrs()),
        [&](const auto &b) { return b->name() == body_name; });
    float x = body_ptr_->body2world_pose().translation().x();
    string trans1_x = to_string(x);
    float y = -body_ptr_->body2world_pose().translation().y();
    string trans1_y = to_string(y);
    float z = body_ptr_->body2world_pose().translation().z();
    string trans1_z = to_string(z);
    Eigen::Matrix3f rot = body_ptr_->world2body_pose().rotation();
    Eigen::Vector3f eulerAngle = rot.eulerAngles(2, 0, 1);
    float qx = eulerAngle[1] * 180 / 3.14159;
    string trans1_qx = to_string(qx);
    float qy = -eulerAngle[2] * 180 / 3.14159;
    string trans1_qy = to_string(qy);
    float qz = eulerAngle[0] * 180 / 3.14159 + 180;
    string trans1_qz = to_string(qz);
    string name = body_name;
    if (strstr(body_name.c_str(), "Copy") != NULL) {
      int index_Copy = body_name.find("Copy", 0);
      int index_num = stoi(body_name.substr(
          index_Copy + 4, body_name.length() - index_Copy - 4));
      name = body_name.substr(0, index_Copy + 4) + to_string(index_num);
    }
    if (z != 0) {
        json += "{  \"classId\": \"" + name + "\" , \"x\": " + trans1_x +
                "\ , \"y\": " + trans1_y + "\ , \"z\": " + trans1_z +
                "\ , \"qx\": " + trans1_qx + "\ , \"qy\": " + trans1_qy +
                "\ , \"qz\": " + trans1_qz + "\},";
    }

}
    json[json.size() - 1] = ']';
    thread server(cppserver);
    server.detach();

}


bool ConfigureAssemblyMetafile(
    std::filesystem::path& path,
    std::shared_ptr<Assemblystate>* assembly_ptr) {
  // Open file
  cv::FileStorage fs;
  try {
    fs.open(cv::samples::findFile(path.string()), cv::FileStorage::READ);
  } catch (cv::Exception e) {
  }
  if (!fs.isOpened()) {
    std::cerr << "Could not open file " << path << std::endl;
    return false;
  }

  // Iterate all assembly states
  for (int state_i = 0; true; ++state_i) {
    cv::FileNode fn_sequence{fs[std::to_string(state_i)]};
    if (fn_sequence.empty()) break;

    std::string path_string;
    fn_sequence["metafile_path"] >> path_string;
    (*assembly_ptr)->tracker_string.push_back(path_string);
    // std::filesystem::path metafile_path{path_string};
    // std::cout << metafile_path;

    std::vector<std::string> ref;
    fn_sequence["referenced_bodies"] >> ref;
    (*assembly_ptr)->Ref.push_back(ref);
    // std::cout << ref[0];

    bool symmetric;
    fn_sequence["Symmetric"] >> symmetric;
    // std::cout << symmetric;
    (*assembly_ptr)->symmetry.push_back(symmetric);

    // Extract pose
    Eigen::Matrix3f rot;
    Eigen::Vector3f trans;
    for (int i = 0; i < 9; ++i) fn_sequence["R"][i] >> rot(i / 3, i % 3);

    for (int i = 0; i < 3; ++i) {
      fn_sequence["T"][i] >> trans(i, 3);
    }
    (*assembly_ptr)->R.push_back(rot);
    (*assembly_ptr)->T.push_back(trans);
    // std::cout << trans;
  }
  return true;
}

bool LoadObjectBodies(
    const std::string &object_name,
    std::vector<std::shared_ptr<m3t::Body>> *body_ptrs) {
  body_ptrs->clear();
  std::filesystem::path configfile_path{dataset_directory / object_name /
                                        "model" / "tracker_config" /
                                        "config.yaml"};
  cv::FileStorage fs;
  if (!m3t::OpenYamlFileStorage(configfile_path, &fs)) return false;
  if (!m3t::ConfigureObjectsMetafileRequired<m3t::Body>(configfile_path, fs,
                                                        "Body", body_ptrs))
    return false;
  for (auto &body_ptr : *body_ptrs)
    if (!body_ptr->SetUp()) return false;
  return true;
}

bool GenerateModels(
    std::vector<std::shared_ptr<m3t::Body>> &body_ptrs,
    const std::string &object_name) {
  region_model_ptrs_.clear();
  depth_model_ptrs_.clear();
  max_model_contour_length_ = 0.0f;
  max_model_surface_area_ = 0.0f;

  std::filesystem::path configfile_path{dataset_directory / object_name /
                                        "model" / "tracker_config" /
                                        "config.yaml"};
  cv::FileStorage fs;
  if (!m3t::OpenYamlFileStorage(configfile_path, &fs)) return false;
  if (!m3t::ConfigureRegionModels(configfile_path, fs, body_ptrs,
                                  &region_model_ptrs_))
    return false;
  if (!m3t::ConfigureObjectsMetafileAndBodyRequired<m3t::DepthModel>(
          configfile_path, fs, "DepthModel", body_ptrs, &depth_model_ptrs_))
    return false;

  std::filesystem::path directory{external_directory / "models" / object_name};
  std::filesystem::create_directories(directory);
  for (auto &region_model_ptr : region_model_ptrs_) {
    const auto &body_ptr{region_model_ptr->body_ptr()};
    region_model_ptr = std::make_shared<m3t::RegionModel>(
        body_ptr->name() + "_region_model", body_ptr,
        directory / (body_ptr->name() + "_region_model.bin"), 1.5f, 4, 500,
        0.05f, 0.002f, false, 2000);
    region_model_setter_(region_model_ptr);
    if (!region_model_ptr->SetUp()) return false;
    // max_model_contour_length_ = std::max(max_model_contour_length_,
    // kObject2SizeMultiplier.at(object_name)
    // *region_model_ptr->max_contour_length());
  }
  for (auto &depth_model_ptr : depth_model_ptrs_) {
    const auto &body_ptr{depth_model_ptr->body_ptr()};
    depth_model_ptr = std::make_shared<m3t::DepthModel>(
        body_ptr->name() + "_depth_model", body_ptr,
        directory / (body_ptr->name() + "_depth_model.bin"), 1.5f, 4, 500,
        0.05f, 0.002f, false, 2000);
    depth_model_setter_(depth_model_ptr);
    if (!depth_model_ptr->SetUp()) return false;
    // max_model_surface_area_ = std::max(max_model_surface_area_,
    // kObject2SizeMultiplier.at(object_name)
    // *depth_model_ptr->max_surface_area());
  }
  return true;
}

void GenderateReducedVertices(
    std::vector<std::shared_ptr<m3t::Body>> &body_ptrs) {
  reduced_vertices_vector_.resize(body_ptrs.size());
#pragma omp parallel for
  for (int i = 0; i < body_ptrs.size(); ++i) {
    const auto &vertices{body_ptrs[i]->vertices()};
    if (n_vertices_evaluation <= 0 ||
        n_vertices_evaluation >= vertices.size()) {
      reduced_vertices_vector_[i] = vertices;
      continue;
    }

    std::mt19937 generator{7};
    if (use_random_seed)
      generator.seed(unsigned(
          std::chrono::system_clock::now().time_since_epoch().count()));

    std::vector<Eigen::Vector3f> reduced_vertices(n_vertices_evaluation);
    int n_vertices = vertices.size();
    for (auto &v : reduced_vertices) {
      int idx = int(generator() % n_vertices);
      v = vertices[idx];
    }
    reduced_vertices_vector_[i] = std::move(reduced_vertices);
  }
}

void GenerateKDTrees(
    std::vector<std::shared_ptr<m3t::Body>> &body_ptrs) {
  kdtree_ptr_vector_.resize(body_ptrs.size());
#pragma omp parallel for
  for (int i = 0; i < body_ptrs.size(); ++i) {
    const auto &vertices{body_ptrs[i]->vertices()};
    auto kdtree_ptr{std::make_unique<KDTreeVector3f>(3, vertices, 10)};
    kdtree_ptr->index->buildIndex();
    kdtree_ptr_vector_[i] = std::move(kdtree_ptr);
  }
}

bool LoadEvaluationData(const std::string &object_name) {
  std::filesystem::path path{dataset_directory / object_name / "model" /
                             "evaluation_data.yaml"};
  cv::FileStorage fs;
  if (!m3t::OpenYamlFileStorage(path, &fs)) return false;
  fs["ErrorThreshold"] >> error_threshold_;

  cv::FileNode fn = fs["IDsCombinedBodies"];
  ids_combined_bodies_.clear();
  for (const auto &fn_body : fn) {
    std::vector<std::string> ids_combined_body;
    fn_body >> ids_combined_body;
    ids_combined_bodies_.push_back(std::move(ids_combined_body));
    // std::cout << ids_combined_body[0] << std::endl;
  }
  return true;
}

bool SetUpTracker(
    const std::string &object_name,
    const std::shared_ptr<m3t::RendererGeometry> &renderer_geometry_ptr,
    std::shared_ptr<m3t::Tracker> *tracker_ptr,
    std::shared_ptr<Assemblystate> *assembly_ptr) {
    *tracker_ptr = std::make_shared<m3t::Tracker>("tracker");
    *assembly_ptr = std::make_shared<Assemblystate>();
    

    // Create file storage
    cv::FileStorage fs;
    std::filesystem::path configfile_path;

    configfile_path = dataset_directory / object_name / "model" /
                        "tracker_config" / "config.yaml";

    if (!m3t::OpenYamlFileStorage(configfile_path, &fs)) return false;
    // Init bodies
    std::vector<std::shared_ptr<m3t::Body>> body_ptrs;
    if (!LoadObjectBodies(object_name, &body_ptrs)) return false;

    // Init renderer geometry
    renderer_geometry_ptr->ClearBodies();
    for (const auto &body_ptr : body_ptrs)
        renderer_geometry_ptr->AddBody(body_ptr);

    // Set up cameras
    auto color_camera_ptr{
        std::make_shared<m3t::AzureKinectColorCamera>("azure_kinect_color")};
    //if (!color_camera_ptr->SetUp()) return false;
    auto depth_camera_ptr{
        std::make_shared<m3t::AzureKinectDepthCamera>("azure_kinect_depth")};
    //if (!depth_camera_ptr->SetUp()) return false;


    // Init viewers
    auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
        "color_viewer", color_camera_ptr, renderer_geometry_ptr)};
    (*tracker_ptr)->AddViewer(color_viewer_ptr);
    auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
          "depth_viewer", depth_camera_ptr, renderer_geometry_ptr, 0.3f, 1.0f)};
    (*tracker_ptr)->AddViewer(depth_viewer_ptr);

    // Init renderer
    auto silhouette_renderer_color_ptr{
        std::make_shared<m3t::FocusedSilhouetteRenderer>(
            "silhouette_renderer_color", renderer_geometry_ptr,
            color_camera_ptr, m3t::IDType::REGION, 400)};
    auto silhouette_renderer_depth_ptr{
        std::make_shared<m3t::FocusedSilhouetteRenderer>(
            "silhouette_renderer_depth", renderer_geometry_ptr,
            depth_camera_ptr, m3t::IDType::BODY, 400)};
    for (const auto &body_ptr : body_ptrs) {
        silhouette_renderer_color_ptr->AddReferencedBody(body_ptr);
        silhouette_renderer_depth_ptr->AddReferencedBody(body_ptr);
    }


    //if (!silhouette_renderer_color_ptr->SetUp()) return false;
    //if (!silhouette_renderer_depth_ptr->SetUp()) return false;
    if (track_mode == "GBOT") {
        std::filesystem::path assemblyfile_path{dataset_directory / object_name /
                                                "model" / "tracker_config" /
                                                "assembly_state.json"};

        // m3t::ReadOptionalValueFromYaml(fa, "AssemblyState", &state);
        ConfigureAssemblyMetafile(assemblyfile_path, assembly_ptr);

        // std::cout << assembly_ptr->tracker_string[0];
        for (auto it = (*assembly_ptr)->tracker_string.begin();
            it != (*assembly_ptr)->tracker_string.end(); ++it) {
        std::cout << "Set up " + *it << std::endl;
        configfile_path =
            dataset_directory / object_name / "model" / "tracker_config" / *it;
        if (!m3t::OpenYamlFileStorage(configfile_path, &fs)) return false;

        *tracker_ptr = std::make_shared<m3t::Tracker>("tracker");
        // Init viewers

        auto color_viewer_ptr{std::make_shared<m3t::NormalColorViewer>(
            "color_viewer", color_camera_ptr, renderer_geometry_ptr)};
        color_viewer_ptr->set_display_images(true);
        //if (!color_viewer_ptr->SetUp()) return false;
        (*tracker_ptr)->AddViewer(color_viewer_ptr);
        auto depth_viewer_ptr{std::make_shared<m3t::NormalDepthViewer>(
            "depth_viewer", depth_camera_ptr, renderer_geometry_ptr, 0.0f,
            2.0f)};
        depth_viewer_ptr->set_display_images(true);
        //if (!depth_viewer_ptr->SetUp()) return false;
        (*tracker_ptr)->AddViewer(depth_viewer_ptr);

        // Configure color histograms
        std::vector<std::shared_ptr<m3t::ColorHistograms>>
            color_histograms_ptrs;

        if (!m3t::ConfigureObjectsMetafileOptional<m3t::ColorHistograms>(
                configfile_path, fs, "ColorHistograms", &color_histograms_ptrs))
            return false;
        for (auto &color_histograms_ptr : color_histograms_ptrs) {
            color_histograms_setter_(color_histograms_ptr);
            //if (!color_histograms_ptr->SetUp()) return false;
        }

        // Init region modalities
        std::vector<std::shared_ptr<m3t::Modality>> modality_ptrs;

        if (!m3t::ConfigureObjects<m3t::RegionModality>(
                fs, "RegionModality",
                {"name", "body", "color_camera", "region_model"},
                [&](const auto &file_node, auto *region_modality_ptr) {
                    // Get objects required for region modality constructor
                    std::shared_ptr<m3t::Body> body_ptr;
                    std::shared_ptr<m3t::RegionModel> region_model_ptr;
                    if (!m3t::GetObject(file_node, "body", "RegionModality",
                                        body_ptrs, &body_ptr) ||
                        !m3t::GetObject(file_node, "region_model",
                                        "RegionModality", region_model_ptrs_,
                                        &region_model_ptr))
                    return false;

                    // Construct region modality
                    *region_modality_ptr = std::make_shared<m3t::RegionModality>(
                        m3t::Name(file_node), body_ptr, color_camera_ptr,
                        region_model_ptr);

                    // Add additional objects
                    if (use_shared_color_histograms) {
                    if (!file_node["use_shared_color_histograms"].empty()) {
                        if (!m3t::AddObject(
                                file_node["use_shared_color_histograms"],
                                "color_histograms", "RegionModality",
                                color_histograms_ptrs,
                                [&](const auto &color_histograms_ptr) {
                                (*region_modality_ptr)
                                    ->UseSharedColorHistograms(
                                        color_histograms_ptr);
                                return true;
                                }))
                        return false;
                    }
                    }
                    if (use_region_checking)
                    (*region_modality_ptr)
                        ->UseRegionChecking(silhouette_renderer_color_ptr);

                    // Set parameters
                    region_modality_setter_(*region_modality_ptr);
                    (*region_modality_ptr)->set_n_unoccluded_iterations(0);
                    (*region_modality_ptr)
                        ->set_reference_contour_length(max_model_contour_length_);
                    //return (*region_modality_ptr)->SetUp();
                },
                &modality_ptrs))
            return false;
        

        // Init depth modalities
        if (!m3t::ConfigureObjects<m3t::DepthModality>(
                fs, "DepthModality",
                {"name", "body", "depth_camera", "depth_model"},
                [&](const auto &file_node, auto *depth_modality_ptr) {
                    // Get objects required for depth modality constructor
                    std::shared_ptr<m3t::Body> body_ptr;
                    std::shared_ptr<m3t::DepthModel> depth_model_ptr;
                    if (!m3t::GetObject(file_node, "body", "DepthModality",
                                        body_ptrs, &body_ptr) ||
                        !m3t::GetObject(file_node, "depth_model", "DepthModality",
                                        depth_model_ptrs_, &depth_model_ptr))
                    return false;

                    // Construct depth modality
                    *depth_modality_ptr = std::make_shared<m3t::DepthModality>(
                        m3t::Name(file_node), body_ptr, depth_camera_ptr,
                        depth_model_ptr);

                    // Add additional objects
                    if (use_silhouette_checking)
                    (*depth_modality_ptr)
                        ->UseSilhouetteChecking(silhouette_renderer_depth_ptr);

                    // Set parameters
                    depth_modality_setter_(*depth_modality_ptr);
                    (*depth_modality_ptr)->set_n_unoccluded_iterations(0);
                    (*depth_modality_ptr)
                        ->set_reference_surface_area(max_model_surface_area_);
                    //return (*depth_modality_ptr)->SetUp();
                },
                &modality_ptrs))
            return false;

        // Init links
        std::vector<std::shared_ptr<m3t::Link>> link_ptrs;
        if (!m3t::ConfigureLinks(configfile_path, fs, body_ptrs, modality_ptrs,
                                    &link_ptrs))
            return false;
        //for (auto &link_ptr : link_ptrs)
         //   if (!link_ptr->SetUp()) return false;

        // Init constraints
        std::vector<std::shared_ptr<m3t::Constraint>> contraint_ptrs;
        contraint_ptrs.clear();
        if (!m3t::ConfigureConstraints(configfile_path, fs, link_ptrs,
                                        &contraint_ptrs))
            return false;
        //for (auto &constraint_ptr : contraint_ptrs)
            //if (!constraint_ptr->SetUp()) return false;

        // Init optimizers
        std::vector<std::shared_ptr<m3t::Optimizer>> optimizer_ptrs;
        if (!m3t::ConfigureOptimizers(
                configfile_path, fs, link_ptrs, contraint_ptrs,
                std::vector<std::shared_ptr<m3t::SoftConstraint>>{},
                &optimizer_ptrs))
            return false;
        for (auto &optimizer_ptr : optimizer_ptrs) {
            optimizer_setter_(optimizer_ptr);
            //if (!optimizer_ptr->SetUp()) return false;
        }

        // Set up detector

        std::filesystem::path detector_path{dataset_directory / object_name / "model" /
                        "tracker_config" / "static_detector.yaml"};
        auto detector_ptr{std::make_shared<m3t::StaticDetector>(
            "static_detector", detector_path, optimizer_ptrs[0])};
        (*tracker_ptr)->AddDetector(detector_ptr);

        // Init tracker
        (*tracker_ptr)->ClearOptimizers();
        for (auto &optimizer_ptr : optimizer_ptrs)
            (*tracker_ptr)->AddOptimizer(optimizer_ptr);
        tracker_setter_(*tracker_ptr);
        //if (!GenerateConfiguredTracker(configfile_path, &*tracker_ptr)) return -1;
        if (!(*tracker_ptr)->SetUp()) return -1;
        //if (!(*tracker_ptr)->RunTrackerProcess(false, false)) return -1;
        //if (!(*tracker_ptr)->SetUp(false)) return false;

        (*assembly_ptr)->tracker_ptrs.push_back(*tracker_ptr);
        }

        return true;
    } else {
        // Configure color histograms
        std::vector<std::shared_ptr<m3t::ColorHistograms>> color_histograms_ptrs;
        if (!m3t::ConfigureObjectsMetafileOptional<m3t::ColorHistograms>(
                configfile_path, fs, "ColorHistograms", &color_histograms_ptrs))
        return false;
        for (auto &color_histograms_ptr : color_histograms_ptrs) {
        color_histograms_setter_(color_histograms_ptr);
        //if (!color_histograms_ptr->SetUp()) return false;
        }

        // Init region modalities
        std::vector<std::shared_ptr<m3t::Modality>> modality_ptrs;
        if (!m3t::ConfigureObjects<m3t::RegionModality>(
                fs, "RegionModality",
                {"name", "body", "color_camera", "region_model"},
                [&](const auto &file_node, auto *region_modality_ptr) {
                // Get objects required for region modality constructor
                std::shared_ptr<m3t::Body> body_ptr;
                std::shared_ptr<m3t::RegionModel> region_model_ptr;
                if (!m3t::GetObject(file_node, "body", "RegionModality",
                                    body_ptrs, &body_ptr) ||
                    !m3t::GetObject(file_node, "region_model", "RegionModality",
                                    region_model_ptrs_, &region_model_ptr))
                    return false;

                // Construct region modality
                *region_modality_ptr = std::make_shared<m3t::RegionModality>(
                    m3t::Name(file_node), body_ptr, color_camera_ptr,
                    region_model_ptr);

                // Add additional objects
                if (use_shared_color_histograms) {
                    if (!file_node["use_shared_color_histograms"].empty()) {
                    if (!m3t::AddObject(
                            file_node["use_shared_color_histograms"],
                            "color_histograms", "RegionModality",
                            color_histograms_ptrs,
                            [&](const auto &color_histograms_ptr) {
                                (*region_modality_ptr)
                                    ->UseSharedColorHistograms(
                                        color_histograms_ptr);
                                return true;
                            }))
                        return false;
                    }
                }
                if (use_region_checking)
                    (*region_modality_ptr)
                        ->UseRegionChecking(silhouette_renderer_color_ptr);

                // Set parameters
                region_modality_setter_(*region_modality_ptr);
                (*region_modality_ptr)->set_n_unoccluded_iterations(0);
                (*region_modality_ptr)
                    ->set_reference_contour_length(max_model_contour_length_);
                //return (*region_modality_ptr)->SetUp();
                },
                &modality_ptrs))
        return false;

        // Init depth modalities
        if (!m3t::ConfigureObjects<m3t::DepthModality>(
                fs, "DepthModality",
                {"name", "body", "depth_camera", "depth_model"},
                [&](const auto &file_node, auto *depth_modality_ptr) {
                // Get objects required for depth modality constructor
                std::shared_ptr<m3t::Body> body_ptr;
                std::shared_ptr<m3t::DepthModel> depth_model_ptr;
                if (!m3t::GetObject(file_node, "body", "DepthModality",
                                    body_ptrs, &body_ptr) ||
                    !m3t::GetObject(file_node, "depth_model", "DepthModality",
                                    depth_model_ptrs_, &depth_model_ptr))
                    return false;

                // Construct depth modality
                *depth_modality_ptr = std::make_shared<m3t::DepthModality>(
                    m3t::Name(file_node), body_ptr, depth_camera_ptr,
                    depth_model_ptr);

                // Add additional objects
                if (use_silhouette_checking)
                    (*depth_modality_ptr)
                        ->UseSilhouetteChecking(silhouette_renderer_depth_ptr);

                // Set parameters
                depth_modality_setter_(*depth_modality_ptr);
                (*depth_modality_ptr)->set_n_unoccluded_iterations(0);
                (*depth_modality_ptr)
                    ->set_reference_surface_area(max_model_surface_area_);
                //return (*depth_modality_ptr)->SetUp();
                },
                &modality_ptrs))
        return false;

        // Init links

        std::vector<std::shared_ptr<m3t::Link>> link_ptrs;
        if (!m3t::ConfigureLinks(configfile_path, fs, body_ptrs, modality_ptrs,
                                &link_ptrs))
        return false;
        //for (auto &link_ptr : link_ptrs)
        //if (!link_ptr->SetUp()) return false;

        // Init constraints
        std::vector<std::shared_ptr<m3t::Constraint>> contraint_ptrs;
        if (!m3t::ConfigureConstraints(configfile_path, fs, link_ptrs,
                                        &contraint_ptrs))
        return false;
        //for (auto &constraint_ptr : contraint_ptrs)
        //if (!constraint_ptr->SetUp()) return false;

        // Init optimizers
        std::vector<std::shared_ptr<m3t::Optimizer>> optimizer_ptrs;
        if (!m3t::ConfigureOptimizers(
                configfile_path, fs, link_ptrs, contraint_ptrs,
                std::vector<std::shared_ptr<m3t::SoftConstraint>>{},
                &optimizer_ptrs))
        return false;
        for (auto &optimizer_ptr : optimizer_ptrs) {
        optimizer_setter_(optimizer_ptr);
        //if (!optimizer_ptr->SetUp()) return false;
        }

        // Init tracker
        for (auto &optimizer_ptr : optimizer_ptrs)
        (*tracker_ptr)->AddOptimizer(optimizer_ptr);
        tracker_setter_(*tracker_ptr);

        if (!(*tracker_ptr)->SetUp()) return -1;
        return true;
    }
}

void YOLOdetector(const std::shared_ptr<m3t::Tracker> &tracker_ptr,
                  std::vector<m3t::Transform3fA> &body2world_poses =
        std::vector<m3t::Transform3fA>(body_name2idx_map_.size())
                  ) {
  //std::vector<m3t::Transform3fA> body2world_poses(body_name2idx_map_.size());
  const auto &viewer_ptr = *std::find_if(
      begin(tracker_ptr->viewer_ptrs()), end(tracker_ptr->viewer_ptrs()),
      [&](const auto &b) { return b->name() == "color_viewer"; });
  cv::Mat image = viewer_ptr->camera_ptr()->image();
  cv::Mat res;
  cv::Size size = cv::Size{1280, 1280};
  int topk = 100;
  int num_labels = body_name2idx_map_.size();
  //int num_labels = 3;
  float score_thres = 0.35f;
  float iou_thres = 0.65f;
  Object obj_select;
  std::vector<Object> objs;
  std::vector<Object> objs_select;
  std::vector<Object> objs_sum;
  cam_K = {viewer_ptr->camera_ptr()->intrinsics().fu,
           0.0,
           viewer_ptr->camera_ptr()->intrinsics().ppu,
           0.0,
           viewer_ptr->camera_ptr()->intrinsics().fv,
           viewer_ptr->camera_ptr()->intrinsics().ppv,
           0.0,
           0.0,
           1.0};
  

  objs_sum.clear();
  auto start = std::chrono::system_clock::now();
  if (yolo_mode == "single") {
    yolov8_pose->copy_from_Mat(image, size);
    yolov8_pose->infer();
    yolov8_pose->postprocess(objs, score_thres, iou_thres, topk, num_labels);
  }
  int idx_back = 0;
  for (const auto &[body_name, idx] : body_name2idx_map_) {
    m3t::Transform3fA body_pose;
    memset(&body_pose, 0, sizeof(body_pose));
    if (yolo_mode == "multi") {
      auto yolov8_pose = yolo_pool[idx];
      objs.clear();
      yolov8_pose->copy_from_Mat(image, size);
      yolov8_pose->infer();
      yolov8_pose->postprocess(objs, score_thres, iou_thres, topk, num_labels);
      yolov8_pose->calculate6dpose(objs[0], kpts_gt[idx], cam_K, body_pose);
    } 
    else {
      objs_select.clear();
      int index_Copy = 0;
      if (strstr(body_name.c_str(), "Copy") != NULL) {
        index_Copy = body_name.find("Copy", 0);
        index_Copy = stoi(body_name.substr(index_Copy + 4, body_name.length()-index_Copy-4));
        idx_back++;
      }
      //std::cout << idx_back;
      for (auto &obj : objs) {
        if (obj.label == idx-idx_back) objs_select.push_back(obj);
      }
      //memset(&obj_select, 0, sizeof(obj_select));
      if (index_Copy < objs_select.size()) {
          obj_select = objs_select[index_Copy];
          yolov8_pose->calculate6dpose(obj_select, kpts_gt[idx], cam_K, body_pose);
      }
         
    }
    
    
    //std::cout << "demo";
    //objs_sum.reserve(objs_sum.size() + objs.size());
    //objs_sum.insert(objs_sum.end(), objs.begin(), objs.end());
   
    body2world_poses.push_back(body_pose);
    //body2world_poses.at(idx) = body_pose;

  }
}

bool LoadPoses(
    const std::filesystem::path &path, int start_index,
    std::vector<std::vector<m3t::Transform3fA>> *body2world_poses_sequence) {
  // Open file
  cv::FileStorage fs;
  try {
    fs.open(cv::samples::findFile(path.string()), cv::FileStorage::READ);
  } catch (cv::Exception e) {
  }
  if (!fs.isOpened()) {
    std::cerr << "Could not open file " << path << std::endl;
    return false;
  }

  // Iterate all sequences
  body2world_poses_sequence->clear();
  for (int i = start_index; true; ++i) {
    cv::FileNode fn_sequence{fs[std::to_string(i)]};
    if (fn_sequence.empty()) break;

    // Iterate all bodies
    std::vector<m3t::Transform3fA> body2world_poses(body_name2idx_map_.size());
    for (const auto &[body_name, idx] : body_name2idx_map_) {
      std::string body_id = body_name;
      // Find file node for body
      cv::FileNode fn_body;
      for (const auto &fn_obj : fn_sequence) {
        std::string obj_id;
        fn_obj["obj_id"] >> obj_id;
        if (obj_id == body_id) {
          fn_body = fn_obj;
          break;
        }
      }
      if (fn_body.empty()) {
        std::cerr << "obj_id " << body_id << " not found in sequence " << i
                  << std::endl;
        // return false;
      }

      // Extract pose
      m3t::Transform3fA body2world_pose;
      for (int i = 0; i < 9; ++i)
        fn_body["cam_R_m2c"][i] >> body2world_pose.matrix()(i / 3, i % 3);
      for (int i = 0; i < 3; ++i) {
        fn_body["cam_t_m2c"][i] >> body2world_pose.matrix()(i, 3);
        body2world_pose.matrix()(i, 3) *= 0.001;
      }

      body2world_poses[idx] = body2world_pose;
    }
    body2world_poses_sequence->push_back(std::move(body2world_poses));
  }
  return true;
}


void SetBodyAndJointPosesNC(
    const std::vector<m3t::Transform3fA> &body2world_poses,
    const std::shared_ptr<m3t::Link> &link_ptr,
    const std::shared_ptr<m3t::Link> &parent_link_ptr) {
  auto &body_ptr{link_ptr->body_ptr()};

  // Set body pose
  int idx = body_name2idx_map_.at(body_ptr->name());
  body_ptr->set_body2world_pose(body2world_poses[idx]);

  // Set joint pose
  if (parent_link_ptr) {
    auto &parent_body_ptr{parent_link_ptr->body_ptr()};
    link_ptr->set_joint2parent_pose(parent_body_ptr->world2body_pose() *
                                    body_ptr->body2world_pose() *
                                    link_ptr->body2joint_pose().inverse());
  }

  // Recursion
  for (const auto &child_link_ptr : link_ptr->child_link_ptrs())
    SetBodyAndJointPosesNC(body2world_poses, child_link_ptr, link_ptr);
}


void SetBodyAndJointPoses(
    const std::vector<m3t::Transform3fA> &body2world_poses,
    const std::shared_ptr<m3t::Tracker> &tracker_ptr) {
  for (auto &optimizer_ptr : tracker_ptr->optimizer_ptrs()) {

      SetBodyAndJointPosesNC(body2world_poses, optimizer_ptr->root_link_ptr(),
                           nullptr);
  }
}



void ExecuteMeasuredTrackingCycle(
    const std::shared_ptr<m3t::Tracker> &tracker_ptr, int iteration,
    ExecutionTimes *execution_times) {
  std::chrono::high_resolution_clock::time_point begin_time;

  // Update Cameras
  tracker_ptr->UpdateCameras(iteration);

  execution_times->calculate_correspondences = 0.0f;
  execution_times->calculate_gradient_and_hessian = 0.0f;
  execution_times->calculate_optimization = 0.0f;
  for (int corr_iteration = 0;
       corr_iteration < tracker_ptr->n_corr_iterations(); ++corr_iteration) {
    // Calculate correspondences
    begin_time = std::chrono::high_resolution_clock::now();
    tracker_ptr->CalculateCorrespondences(iteration, corr_iteration);

    // Visualize correspondences
    int corr_save_idx =
        iteration * tracker_ptr->n_corr_iterations() + corr_iteration;
    tracker_ptr->VisualizeCorrespondences(corr_save_idx);

    for (int update_iteration = 0;
         update_iteration < tracker_ptr->n_update_iterations();
         ++update_iteration) {
      // Calculate gradient and hessian
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateGradientAndHessian(iteration, corr_iteration,
                                               update_iteration);

      // Calculate optimization
      begin_time = std::chrono::high_resolution_clock::now();
      tracker_ptr->CalculateOptimization(iteration, corr_iteration,
                                         update_iteration);

      // Visualize pose update
      int update_save_idx =
          corr_save_idx * tracker_ptr->n_update_iterations() + update_iteration;
      tracker_ptr->VisualizeOptimization(update_save_idx);
    }
  }

  // Calculate results
  begin_time = std::chrono::high_resolution_clock::now();
  tracker_ptr->CalculateResults(iteration);

  // Visualize results and update viewers
  tracker_ptr->VisualizeResults(iteration);
  if (visualize_tracking || save_images)
    tracker_ptr->UpdateViewers(iteration);

  execution_times->complete_cycle =
      execution_times->calculate_correspondences +
      execution_times->calculate_gradient_and_hessian +
      execution_times->calculate_optimization +
      execution_times->calculate_results;
}

int main(int argc, char *argv[]) {

  for (const auto &object_name : object_names) {

    std::vector<std::shared_ptr<m3t::Body>> body_ptrs;
    if (!LoadObjectBodies(object_name, &body_ptrs)) return false;
    if (!GenerateModels(body_ptrs, object_name)) return false;
    GenderateReducedVertices(body_ptrs);
    GenerateKDTrees(body_ptrs);
    LoadEvaluationData(object_name);
    body_name2idx_map_.clear();
    for (int i = 0; i < body_ptrs.size(); ++i)
    body_name2idx_map_.insert({body_ptrs[i]->name(), i});

     // Read gt poses
    //std::vector<std::vector<m3t::Transform3fA>> gt_body2world_poses_sequence;
    //std::filesystem::path gt_path{dataset_directory / "HobbyCornerClamp" / "test" / "normal" / "scene_gt.json"};
    //if (!LoadPoses(gt_path, 0, &gt_body2world_poses_sequence)) return false;

    //std::cout << gt_body2world_poses_sequence[0].size() << std::endl;

    // Set up renderer geometry
    auto renderer_geometry_ptr{std::make_shared<m3t::RendererGeometry>("rg")};
    if (!renderer_geometry_ptr->SetUp()) return false;

    // Set Up Tracker
    std::shared_ptr<m3t::Tracker> tracker_ptr;
    std::shared_ptr<Assemblystate> assembly_ptr;
    // auto assembly_ptr{std::make_shared<Assemblystate>()};


    if (!SetUpTracker(object_name, renderer_geometry_ptr, &tracker_ptr,
                      &assembly_ptr))
      return false;


    /*
    // Read external poses
    std::vector<std::vector<m3t::Transform3fA>>
        external_body2world_poses_sequence;
    std::vector<float> external_execution_times;

    if (track_mode == "GBOT") {
      // Initialize poses and start modalities


      SetBodyAndJointPoses(gt_body2world_poses_sequence[0], assembly_ptr->tracker_ptrs[0]);

      assembly_ptr->tracker_ptrs[0]->StartModalities(0);
      // if (!assembly_ptr->tracker_ptrs[0]->SetUp()) return -1;
      // if (!assembly_ptr->tracker_ptrs[0]->RunTrackerProcess(true, false))
      // return -1;
    } else {
      // Initialize poses and start modalities
      SetBodyAndJointPoses(gt_body2world_poses_sequence[0], tracker_ptr);
      tracker_ptr->StartModalities(0);
    }
    */

      // Iterate over all frames
    int assembly_num = 0;
    m3t::Transform3fA body22body1_pose, body32body1_pose;
    Eigen::Vector3f body22body1_trans_gt, body22body1_trans,
        body32body1_trans_gt, body32body1_trans, body_trans;
    Eigen::Matrix3f body22body1_rotation_gt, body22body1_rotation,
        body32body1_rotation_gt, body32body1_rotation, body_rotation;
    float rot_diff, trans_diff;
    static constexpr float transOffset = 0.030f;
    static constexpr float transOffset_reinitial = 0.050f;
    static constexpr float rotOffset = 20.0f;
    // std::shared_ptr<m3t::Tracker> curr_tracker_ptr;
    // calculate relative pose between two objects

      // Load YOLOv8pose
    for (const auto &[body_name, idx] : body_name2idx_map_) {
      std::cout << body_name;
      std::cout << idx << std::endl;

      if (yolo_mode == "multi") {
          std::string stringpath = engine_file_path + "/" + body_name + ".engine";
          std::cout << stringpath << std::endl;
          auto yolov8_pose = new YOLOv8_pose(stringpath);
          yolov8_pose->make_pipe(true);
          yolo_pool.push_back(yolov8_pose);
      }

      // read key points data
      std::ifstream accfile;
      std::string path_kpts = engine_file_path + "/" + body_name + ".txt";
      std::cout << path_kpts << std::endl;
      accfile.open(path_kpts);
      std::vector<cv::Point3f> numbers;

      if (!accfile.is_open()) {
        std::cout << "open key point file failed" << std::endl;
      }

      for (int i = 0; !accfile.eof(); i++) {
        double x = -1, y = -1, z = -1;
        accfile >> x >> y >> z;
        if (x > -1) {
          numbers.push_back(cv::Point3f(x, y, z));
          // std::cout << x << std::endl;
        }
      }
      if (accfile.is_open()) {
        accfile.close();
      }

      kpts_gt.push_back(numbers);
      // std::cout << kpts_gt[0].size() << std::endl;
    }

    if (yolo_mode == "single") {
      std::string stringpath = engine_file_path + "/" + object_name + ".engine";
      std::cout << stringpath << std::endl;
      yolov8_pose = new YOLOv8_pose(stringpath);
      yolov8_pose->make_pipe(true);
    }
    std::vector<m3t::Transform3fA> body2world_poses;
    
    for (int idx = 0;; ++idx) {
        body2world_poses.clear();
        assembly_ptr->tracker_ptrs[0]->UpdateCameras(0);
        YOLOdetector(assembly_ptr->tracker_ptrs[0], body2world_poses);
        //std::cout << body2world_poses.size();
        SetBodyAndJointPoses(body2world_poses, assembly_ptr->tracker_ptrs[0]);
        //assembly_ptr->tracker_ptrs[0]->StartModalities(0);
        // Visualize results and update viewers
        assembly_ptr->tracker_ptrs[0]->VisualizeResults(0);
        if (visualize_tracking || save_images)
          assembly_ptr->tracker_ptrs[0]->UpdateViewers(0);
        /*
        auto linking_publisher_ptr{std::make_shared<LinkPublisher>(
                LinkPublisher("Linking Publisher", assembly_ptr->tracker_ptrs[0]))};*/
        if (idx % 10==0) UpdatePublisher(assembly_ptr->tracker_ptrs[0]);
        //std::cout << body2world_poses[0].matrix().isZero(0) << std::endl;

        /*
        bool is_empty = false;
        for (int id = 0; id < body_name2idx_map_.size(); id++) {
          if (body2world_poses[id].matrix().isZero(0)) {
            is_empty = true;
          }
        }
        if (!is_empty) break;
        */
        
    }
    std::cout << " Detection finished!" << std::endl;

    std::string body_name1, body_name2;
    for (int i = 0;; ++i) {
      Result result;
      result.frame_index = i;

      // Calculate relative pose
      if (track_mode == "GBOT") {
        // curr_tracker_ptr = assembly_ptr->tracker_ptrs[assembly_num];
        body_name1 = assembly_ptr->Ref[assembly_num][0];
        body_name2 = assembly_ptr->Ref[assembly_num][1];

        const auto &body1_ptr = *std::find_if(
            begin(assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()),
            end(assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()),
            [&](const auto &b) { return b->name() == body_name1; });

        const auto &body2_ptr = *std::find_if(
            begin(assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()),
            end(assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()),
            [&](const auto &b) { return b->name() == body_name2; });

        // std::cout << body_name1;
        body22body1_pose =
            body1_ptr->world2body_pose() * body2_ptr->body2world_pose();
        body22body1_trans = body22body1_pose.translation();
        body22body1_rotation = body22body1_pose.rotation();
        body22body1_trans_gt = assembly_ptr->T[assembly_num];
        body22body1_rotation_gt = assembly_ptr->R[assembly_num];

        rot_diff =
            calcRotationAngle(body22body1_rotation, body22body1_rotation_gt);
        trans_diff =
            calcVectorDifference(body22body1_trans, body22body1_trans_gt);

        // std::cout << body2_ptr->world2body_pose().translation();
        // std::cout << "Pose:";
        // std::cout << body22body1_trans << std::endl;
        // std::cout << "GT pose:";
        // std::cout << body22body1_trans_gt << std::endl;
        // std::cout << trans_diff << std::endl;
        if (trans_diff < transOffset &&
            (assembly_num + 1) < assembly_ptr->Ref.size()) {
          if (assembly_ptr->symmetry[assembly_num]) {
            assembly_num++;
            // SetBodyAndJointPoses(gt_body2world_poses_sequence[i],
            // assembly_ptr->tracker_ptrs[assembly_num]);
            // assembly_ptr->tracker_ptrs[assembly_num]->StartModalities(0);

          } else {
            if (rot_diff < rotOffset) {
              assembly_num++;
              // SetBodyAndJointPoses(gt_body2world_poses_sequence[i],assembly_ptr->tracker_ptrs[assembly_num]);
              // assembly_ptr->tracker_ptrs[assembly_num]->StartModalities(0);
            }
          }
        }
        //std::cout << assembly_num << std::endl;

        ExecuteMeasuredTrackingCycle(assembly_ptr->tracker_ptrs[assembly_num],
                                     i, &result.execution_times);
        UpdatePublisher(assembly_ptr->tracker_ptrs[assembly_num]);

        //Pose reinitialization
        if ((i+1)%10 ==0) {
            body2world_poses.clear();
          assembly_ptr->tracker_ptrs[assembly_num]->UpdateCameras(0);
            YOLOdetector(assembly_ptr->tracker_ptrs[assembly_num], body2world_poses);
            // std::cout << body2world_poses.size();
            if (body2world_poses.size() ==
                body_name2idx_map_.size() &&
                assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs().size() ==
                    body_name2idx_map_.size()) {
                      for (int id = 0; id < body_name2idx_map_.size(); id++) {
                      body_trans = assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()[id]->body2world_pose().translation();
                      body_rotation = assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()[id]->body2world_pose().rotation();
                      trans_diff = calcVectorDifference(body_trans, body2world_poses[id].translation());
                      rot_diff = calcRotationAngle(body_rotation, body2world_poses[id].rotation());
                      if (body2world_poses[id].matrix().isZero(0) ||
                          ((trans_diff < transOffset_reinitial) && (rot_diff < rot_diff))) {
                        body2world_poses[id] =
                            assembly_ptr->tracker_ptrs[assembly_num]->body_ptrs()[id]->body2world_pose();
                      }
                    }
                    SetBodyAndJointPoses(body2world_poses, assembly_ptr->tracker_ptrs[0]);
            }
            
        }
        

      } 
      else {
        ExecuteMeasuredTrackingCycle(tracker_ptr, i, &result.execution_times);
        UpdatePublisher(tracker_ptr);
      }
    }


  }

}