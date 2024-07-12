// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef M3T_INCLUDE_M3T_SILHOUETTE_RENDERER_H
#define M3T_INCLUDE_M3T_SILHOUETTE_RENDERER_H

#include <m3t/body.h>
#include <m3t/camera.h>
#include <m3t/common.h>
#include <m3t/renderer.h>
#include <m3t/renderer_geometry.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace m3t {

/**
 * \brief Class that implements the main functionality for a silhouette renderer
 * and is used by \ref FullSilhouetteRenderer and \ref
 * FocusedSilhouetteRenderer.
 */
class SilhouetteRendererCore {
 public:
  // Destructor and setup method
  SilhouetteRendererCore() = default;
  SilhouetteRendererCore(const SilhouetteRendererCore &) = delete;
  SilhouetteRendererCore &operator=(const SilhouetteRendererCore &) = delete;
  ~SilhouetteRendererCore();
  bool SetUp(const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
             int image_width, int image_height);

  // Main methods
  bool StartRendering(const Eigen::Matrix4f &projection_matrix,
                      const Transform3fA &world2camera_pose, IDType id_type);
  bool FetchSilhouetteImage(cv::Mat *silhouette_image);
  bool FetchDepthImage(cv::Mat *depth_image);

 private:
  // Helper methods
  void CreateBufferObjects();
  void DeleteBufferObjects();

  // Internal data
  std::shared_ptr<RendererGeometry> renderer_geometry_ptr_;
  int image_width_;
  int image_height_;

  // Shader code
  static std::string vertex_shader_code_;
  static std::string fragment_shader_code_;

  // OpenGL variables
  unsigned fbo_ = 0;
  unsigned rbo_silhouette_ = 0;
  unsigned rbo_depth_ = 0;
  unsigned shader_program_ = 0;

  // Internal state
  bool image_rendered_ = false;
  bool silhouette_image_fetched_ = false;
  bool depth_image_fetched_ = false;
  bool initial_set_up_ = false;
};

/**
 * \brief Renderer that extends the full depth renderer class with functionality
 * from \ref SilhouetteRendererCore to render both a depth image and a
 * silhouette image where the `silhouette_id` of \ref Body objects is used as
 * intensity value for pixels on the silhouette.
 *
 * \details Rendering is started using `StartRendering()`. Images are fetched
 * from the GPU using `FetchSilhouetteImage()` and `FetchDepthImage()`. They can
 * then be accessed using the `silhouette_image()` and `depth_image()` getter.
 * Setters and all main methods are thread-safe.
 *
 * @param id_type type of ID from \ref Body object that is rendered. BODY = 0,
 * REGION = 1.
 */
class FullSilhouetteRenderer : public FullDepthRenderer {
 public:
  // Constructors, destructors, and setup method
  FullSilhouetteRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
      IDType id_type = IDType::BODY, float z_min = 0.02f, float z_max = 10.0f);
  FullSilhouetteRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<Camera> &camera_ptr, IDType id_type = IDType::BODY,
      float z_min = 0.02f, float z_max = 10.0f);
  FullSilhouetteRenderer(
      const std::string &name, const std::filesystem::path &metafile_path,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<Camera> &camera_ptr);
  bool SetUp() override;

  // Setters
  void set_id_type(IDType id_type);

  // Main methods
  bool StartRendering() override;
  bool FetchSilhouetteImage();
  bool FetchDepthImage() override;

  // Getters
  IDType id_type() const;
  const cv::Mat &silhouette_image() const;

  // Getters that calculate values based on the rendered silhouette image
  uchar SilhouetteValue(const cv::Point2i &image_coordinate) const;

 private:
  // Helper methods
  bool LoadMetaData();
  void ClearSilhouetteImage();

  // Data
  IDType id_type_{};
  cv::Mat silhouette_image_{};
  SilhouetteRendererCore core_{};
};

/**
 * \brief Renderer that extends the focused depth renderer class with
 * functionality from \ref SilhouetteRendererCore to render images with a
 * defined size that is focused on referenced bodies.
 *
 * \details It is able  to render both a depth image and a silhouette image
 * where the `silhouette_id` of \ref Body objects is used as intensity value for
 * pixels on the silhouette. Rendering is started using `StartRendering()`.
 * Images are fetched from the GPU using `FetchSilhouetteImage()` and
 * `FetchDepthImage()`. They can then be accessed using the `silhouette_image()`
 * and `depth_image()` getter. Setters and all main methods are thread-safe.
 *
 * @param id_type type of ID from \ref Body object that is rendered. BODY = 0,
 * REGION = 1.
 */
class FocusedSilhouetteRenderer : public FocusedDepthRenderer {
 public:
  // Constructors, destructors, and setup method
  FocusedSilhouetteRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const Transform3fA &world2camera_pose, const Intrinsics &intrinsics,
      IDType id_type = IDType::BODY, int image_size = 200, float z_min = 0.02f,
      float z_max = 10.0f);
  FocusedSilhouetteRenderer(
      const std::string &name,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<Camera> &camera_ptr, IDType id_type = IDType::BODY,
      int image_size = 200, float z_min = 0.02f, float z_max = 10.0f);
  FocusedSilhouetteRenderer(
      const std::string &name, const std::filesystem::path &metafile_path,
      const std::shared_ptr<RendererGeometry> &renderer_geometry_ptr,
      const std::shared_ptr<Camera> &camera_ptr);
  bool SetUp() override;

  // Setters
  void set_id_type(IDType id_type);

  // Main methods
  bool StartRendering() override;
  bool FetchSilhouetteImage();
  bool FetchDepthImage() override;

  // Getters
  IDType id_type() const;
  const cv::Mat &focused_silhouette_image() const;

  // Getters that calculate values based on the rendered silhouette image
  uchar SilhouetteValue(const cv::Point2i &image_coordinate) const;

 private:
  // Helper methods
  bool LoadMetaData();
  void ClearSilhouetteImage();

  // Data
  IDType id_type_{};
  cv::Mat focused_silhouette_image_{};
  SilhouetteRendererCore core_{};
};

}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_SILHOUETTE_RENDERER_H
