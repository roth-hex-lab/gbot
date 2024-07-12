// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#ifndef M3T_INCLUDE_M3T_HOLOLENS_CAMERA_H_
#define M3T_INCLUDE_M3T_HOLOLENS_CAMERA_H_

#include <filesystem/filesystem.h>
#include <m3t/camera.h>
#include <m3t/common.h>

#include <chrono>
#include <iostream>
#include <m3t/hl2ss_lnm.h>
#include <m3t/hl2ss_mt.h>
#include <mutex>
#include <opencv2/opencv.hpp>

namespace m3t {

/**
 * \brief Singleton class that allows getting data from a single Azure Kinect
 * instance and that is used by \ref HololensColorCamera and \ref
 * HololensDepthCamera.
 *
 * \details The method `UpdateCapture()` updates the `capture` object if
 * `UpdateCapture()` was already called with the same `id` before. If
 * `UpdateCapture()` was not yet called by the `id`, the same capture is used.
 * If the capture is updated, all memory values except for the `id` that called
 * the function are reset. All methods that are required to operate multiple
 * \ref Camera objects are thread-safe.
 */
class Hololens {
 public:
  // Singleton instance getter
  static Hololens &GetInstance();
  Hololens(const Hololens &) = delete;
  Hololens &operator=(const Hololens &) = delete;
  ~Hololens();

  // Configuration and setup
  void UseColorCamera();
  void UseDepthCamera();
  void CloseClient();
  int RegisterID();
  bool UnregisterID(int id);
  bool SetUp(char const *host);
  char const *host_;

  // Main methods
  //bool UpdateCapture(int id, bool synchronized);

  // Getters
  bool use_color_camera() const;
  bool use_depth_camera() const;
  uint16_t &color_capture();
  uint16_t &depth_capture();
  const Transform3fA *color2depth_pose() const;
  const Transform3fA *depth2color_pose() const;

 private:
  Hololens() = default;

  // Private data
  std::map<int, bool> update_capture_ids_{};
  int next_id_ = 0;

  // Public data
  uint16_t port_color_{};
  uint16_t port_depth_{};
  Transform3fA color2depth_pose_{Transform3fA::Identity()};
  Transform3fA depth2color_pose_{Transform3fA::Identity()};


  // Internal state variables
  std::mutex mutex_;
  bool use_color_camera_ = true;
  bool use_depth_camera_ = true;
  bool initial_set_up_ = false;
};

/**
 * \brief \ref Camera that allows getting color images from an \ref Hololens
 * camera.
 *
 * @param image_scale scales images to avoid borders after rectification.
 * @param use_depth_as_world_frame specifies the depth camera frame as world
 * frame and automatically defines `camera2world_pose` as `color2depth_pose`.
 */
class HololensColorCamera : public ColorCamera {
 public:
  // Constructors, destructor, and setup method
  HololensColorCamera(char const *host, const std::string &name,
                      float image_scale = 1.0f,
                         bool use_depth_as_world_frame = false);
  HololensColorCamera(char const *host, const std::string &name,
                         const std::filesystem::path &metafile_path);
  HololensColorCamera(const HololensColorCamera &) = delete;
  HololensColorCamera &operator=(const HololensColorCamera &) = delete;
  ~HololensColorCamera();
  bool SetUp() override;
  // Setters
  void set_image_scale(float image_scale);
  void set_use_depth_as_world_frame(bool use_depth_as_world_frame);
  void CloseClient();

  // Main method
  bool UpdateImage(bool synchronized) override;

  // Getters
  float image_scale() const;
  bool use_depth_as_world_frame() const;
  const Transform3fA *color2depth_pose() const;
  const Transform3fA *depth2color_pose() const;

 private:
  // Helper methods
  bool LoadMetaData();
  void GetIntrinsicsAndDistortionMap();
  void SaveTrackingIfDesired();
  // Data
  char const *host_;
  Hololens &hololens_;
  int hololens_id_{};
  uint16_t port_color_{};
  std::unique_ptr<hl2ss::mt::source> source_ptr;
  std::unique_ptr<hl2ss::rx_pv> client_color_;
  std::shared_ptr<hl2ss::packet> data_color_{};
  std::shared_ptr<hl2ss::calibration_pv> calibration_color_{};
  float image_scale_ = 1.0f;
  bool use_depth_as_world_frame_ = false;
  cv::Mat distortion_map_;
  bool initial_set_up_ = false;
  uint16_t pv_width = 1280;
  uint16_t pv_height = 720;
  uint8_t pv_fps = 30;
  uint64_t buffer_size = 10;
  int64_t pv_frame_index = 0;
  int32_t pv_status;
  std::exception error;
  // wait value for cv::waitKey
  int wait_key_ms = 1;
};

/**
 * \brief \ref Camera that allows getting depth images from an \ref Hololens
 * camera.
 *
 * @param image_scale scales image to avoid borders after rectification.
 * @param use_color_as_world_frame specifies the color camera frame as world
 * frame and automatically define `camera2world_pose` as `depth2color_pose`.
 * @param depth_offset constant offset that is added to depth image values and
 * which is provided in meter.
 */
class HololensDepthCamera : public DepthCamera {
 public:
  // Constructors, destructor, and setup method
  HololensDepthCamera(char const *host, const std::string &name,
                      float image_scale = 1.0f,
                         bool use_color_as_world_frame = true,
                         float depth_offset = 0.0f);
  HololensDepthCamera(char const *host, const std::string &name,
                         const std::filesystem::path &metafile_path);
  HololensDepthCamera(const HololensDepthCamera &) = delete;
  HololensDepthCamera &operator=(const HololensDepthCamera &) = delete;
  ~HololensDepthCamera();
  bool SetUp() override;

  // Setters
  void set_image_scale(float image_scale);
  void set_use_color_as_world_frame(bool use_color_as_world_frame);
  void set_depth_offset(float depth_offset);

  // Main method
  bool UpdateImage(bool synchronized) override;

  // Getters
  float image_scale() const;
  bool use_color_as_world_frame() const;
  float depth_offset() const;
  const Transform3fA *color2depth_pose() const;
  const Transform3fA *depth2color_pose() const;

 private:
  // Helper methods
  bool LoadMetaData();
  void GetIntrinsicsAndDistortionMap();
  // Data
  char const *host_;
  Hololens &hololens_;
  int hololens_id_{};
  std::shared_ptr<hl2ss::packet> data_depth_{};
  std::shared_ptr<hl2ss::calibration_rm_depth_ahat> calibration_depth_{};
  float image_scale_ = 1.0f;
  bool use_color_as_world_frame_ = true;
  float depth_offset_ = 0.0f;
  cv::Mat distortion_map_;
  bool initial_set_up_ = false;
};

}  // namespace m3t

#endif  // M3T_INCLUDE_M3T_HOLOLENS_CAMERA_H_
