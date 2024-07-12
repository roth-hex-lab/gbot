// SPDX-License-Identifier: MIT
// Copyright (c) 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include <m3t/hololens_camera.h>


namespace m3t {

Hololens &Hololens::GetInstance() {
  static Hololens holo;
  return holo;
}

Hololens::~Hololens() {
}

void Hololens::UseColorCamera() { use_color_camera_ = true; }

void Hololens::UseDepthCamera() { use_depth_camera_ = true; }

int Hololens::RegisterID() {
  const std::lock_guard<std::mutex> lock{mutex_};
  update_capture_ids_.insert(std::pair<int, bool>{next_id_, true});
  return next_id_++;
}

bool Hololens::UnregisterID(int id) {
  const std::lock_guard<std::mutex> lock{mutex_};
  return update_capture_ids_.erase(id);
}

bool Hololens::SetUp(char const *host_) {
  const std::lock_guard<std::mutex> lock{mutex_};
  if (!initial_set_up_) {
    // Configure color and depth camera
    hl2ss::client::initialize();

    port_color_ = hl2ss::stream_port::PERSONAL_VIDEO;
    port_depth_ = hl2ss::stream_port::RM_DEPTH_AHAT;

    initial_set_up_ = true;

  }
  return true;
}

bool Hololens::use_color_camera() const { return use_color_camera_; }

bool Hololens::use_depth_camera() const { return use_depth_camera_; }

uint16_t &Hololens::color_capture() {
  return port_color_;
}

uint16_t &Hololens::depth_capture() {
  return port_depth_;
}


const Transform3fA *Hololens::color2depth_pose() const {
  if (initial_set_up_)
    return &color2depth_pose_;
  else
    return nullptr;
}

const Transform3fA *Hololens::depth2color_pose() const {
  if (initial_set_up_)
    return &depth2color_pose_;
  else
    return nullptr;
}

HololensColorCamera::HololensColorCamera(char const *host, const std::string &name,
                                               float image_scale,
                                               bool use_depth_as_world_frame)
    : host_{host},
      ColorCamera{name},
      image_scale_{image_scale},
      use_depth_as_world_frame_{use_depth_as_world_frame},
      hololens_{Hololens::GetInstance()} {
  hololens_.UseColorCamera();
  hololens_id_ = hololens_.RegisterID();
}

HololensColorCamera::HololensColorCamera(
    char const *host, const std::string &name, const std::filesystem::path &metafile_path)
    : 
      host_{host},
      ColorCamera{name, metafile_path},
      hololens_{Hololens::GetInstance()} {
  hololens_.UseColorCamera();
  hololens_id_ = hololens_.RegisterID();
}

HololensColorCamera::~HololensColorCamera() {
  hololens_.UnregisterID(hololens_id_);
}

bool HololensColorCamera::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  if (!initial_set_up_ && !hololens_.SetUp(host_)) return false;
  hl2ss::client::initialize();
  port_color_ = hl2ss::stream_port::PERSONAL_VIDEO;
  client_color_ =
      hl2ss::lnm::rx_pv(host_, port_color_, pv_width, pv_height, pv_fps);
  hl2ss::lnm::start_subsystem_pv(host_, port_color_, false);
  std::string port_name = hl2ss::get_port_name(port_color_);
  std::cout << "Downloading calibration for " << port_name << " ..."
            << std::endl;
  calibration_color_ = hl2ss::download_calibration_pv(host_, port_color_, pv_width, pv_height, pv_fps);
  std::cout << "Done." << std::endl;

  source_ptr = std::make_unique<hl2ss::mt::source>(buffer_size * pv_fps,
                                                   std::move(client_color_));
  source_ptr->start();
  _sleep(2000);
  //if (use_depth_as_world_frame_)
  //  set_camera2world_pose(*hololens_.color2depth_pose());
  GetIntrinsicsAndDistortionMap();
  SaveMetaDataIfDesired();
  set_up_ = true;
  initial_set_up_ = true;
  return UpdateImage(true);
}

void HololensColorCamera::CloseClient()
{
    // Stop capture
    source_ptr->stop();

    // Stop PV camera
    hl2ss::lnm::stop_subsystem_pv(host_, hl2ss::stream_port::PERSONAL_VIDEO);
}

void HololensColorCamera::set_image_scale(float image_scale) {
  image_scale_ = image_scale;
  set_up_ = false;
}

void HololensColorCamera::set_use_depth_as_world_frame(
    bool use_depth_as_world_frame) {
  use_depth_as_world_frame_ = use_depth_as_world_frame;
  set_up_ = false;
}

bool HololensColorCamera::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up Hololens color camera " << name_ << " first"
              << std::endl;
    return false;
  }

  //data_color_ = source_mt_.get_packet(frame_index, status);
  // Get and undistort image
  //client_color_->close();
  //client_color_->open();
  //data_color_ = client_color_->get_next_packet();

  if (!source_ptr->status(error)) {
    throw error;
  }

  data_color_ = source_ptr->get_packet(pv_frame_index, pv_status);
  if (pv_status < 0) {
    // Requested frame is too old and has been dropped from the buffer
    // (data_pv is null) Advance to next frame
    pv_frame_index+=6;
  }

  else if (pv_status == 0) {
    pv_frame_index+=6;
    uint8_t *image;
    hl2ss::pv_intrinsics *intrinsics;
    hl2ss::unpack_pv(data_color_->payload.get(), data_color_->sz_payload,
                     &image, &intrinsics);
    // Load intrinsics from camera

    intrinsics_.fu = intrinsics->fx;
    intrinsics_.fv = intrinsics->fy;
    intrinsics_.ppu = intrinsics->cx;
    intrinsics_.ppv = intrinsics->cy;

    // Scale intrinsics according to image scale
    intrinsics_.fu *= image_scale_;
    intrinsics_.fv *= image_scale_;

    /*
    cv::cvtColor(
        cv::Mat{cv::Size{intrinsics_.width, intrinsics_.height}, CV_8UC4,
                image,
                cv::Mat::AUTO_STEP},
        temp_image, cv::COLOR_RGBA2RGB);
    */
    image_ = cv::Mat(intrinsics_.height, intrinsics_.width, CV_8UC3, image);
    // cv::remap(temp_image, image_, distortion_map_, cv::Mat{},
    // cv::INTER_NEAREST,cv::BORDER_CONSTANT);
    SaveTrackingIfDesired();
    
  } 
  else  // pv_status > 0
  {
    // Requested frame has not been received from the server yet (data_pv is
    // null) Do not advance to next frame Wait 1 frame in ms
    wait_key_ms = 1000 / pv_fps;
  }

      // Stop when ESC is pressed
  if ((cv::waitKey(wait_key_ms) & 0xFF) == 27) {
    return false;
  }

  SaveImageIfDesired();
  
  
  
  return true;
}

void HololensColorCamera::SaveTrackingIfDesired() {
    if (save_images_) {
        auto syst = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        try
        {
            hl2ss::matrix_4x4* matrix;
            matrix = data_color_->pose.get();
            std::ofstream fout(save_directory_ / (name_ + "pose.txt"), std::ios_base::app);
            fout << syst << ",";
            fout << " " << intrinsics_.fu << " " << intrinsics_.fv << " " << intrinsics_.ppu << " " << intrinsics_.ppv << ",";
            //fout << "[" << std::endl;
            for (int row = 0; row < 4; ++row)
            {
                for (int col = 0; col < 4; ++col)
                {
                    fout << " " << matrix->m[col][row];
                }
                if (row <3) fout << ",";
            }
            //fout << "]" << std::endl;
            fout << "\n";
            fout.flush();
 
        }
            
        catch(...){
            std::cout << "None" << std::endl;
        }
        


    }
}

float HololensColorCamera::image_scale() const { return image_scale_; }

bool HololensColorCamera::use_depth_as_world_frame() const {
  return use_depth_as_world_frame_;
}

const Transform3fA *HololensColorCamera::color2depth_pose() const {
  return hololens_.color2depth_pose();
}

const Transform3fA *HololensColorCamera::depth2color_pose() const {
  return hololens_.depth2color_pose();
}

bool HololensColorCamera::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_scale", &image_scale_);
  ReadOptionalValueFromYaml(fs, "use_depth_as_world_frame",
                            &use_depth_as_world_frame_);
  fs.release();

  // Process parameters
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

void HololensColorCamera::GetIntrinsicsAndDistortionMap() {
  // Load intrinsics from camera

  intrinsics_.fu = calibration_color_->focal_length[0];
  intrinsics_.fv = calibration_color_->focal_length[1];
  intrinsics_.ppu = calibration_color_->principal_point[0];
  intrinsics_.ppv = calibration_color_->principal_point[1];
  intrinsics_.width = pv_width;
  intrinsics_.height = pv_height;
  
  // Scale intrinsics according to image scale
  intrinsics_.fu *= image_scale_;
  intrinsics_.fv *= image_scale_;

  // Calculate distortion map
  /*
  cv::Mat1f camera_matrix(3, 3);
  camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
      intrinsics_.ppv, 0, 0, 1;
  cv::Mat1f new_camera_matrix(3, 3);
  new_camera_matrix << intrinsics_.fu, 0, intrinsics_.ppu, 0, intrinsics_.fv,
      intrinsics_.ppv, 0, 0, 1;
  cv::Mat1f distortion_coeff(1, 8);
  distortion_coeff << hololens_.color_calibration()->radial_distortion[0],
      hololens_.color_calibration()->radial_distortion[1],
      hololens_.color_calibration()->tangential_distortion[0],
      hololens_.color_calibration()->tangential_distortion[1],
      hololens_.color_calibration()->radial_distortion[2];
  cv::Mat map1, map2, map3;
  cv::initUndistortRectifyMap(
      camera_matrix, distortion_coeff, cv::Mat{}, new_camera_matrix,
      cv::Size{intrinsics_.width, intrinsics_.height}, CV_32FC1, map1, map2);
  cv::convertMaps(map1, map2, distortion_map_, map3, CV_16SC2, true);
  */
}



HololensDepthCamera::HololensDepthCamera(char const *host,
                                         const std::string &name,
                                               float image_scale,
                                               bool use_color_as_world_frame,
                                               float depth_offset)
    : 
      host_{host},
      DepthCamera{name},
      image_scale_{image_scale},
      use_color_as_world_frame_{use_color_as_world_frame},
      depth_offset_{depth_offset},
      hololens_{Hololens::GetInstance()} {
  hololens_.UseDepthCamera();
  hololens_id_ = hololens_.RegisterID();
}

HololensDepthCamera::HololensDepthCamera(
    char const *host, const std::string &name,
    const std::filesystem::path &metafile_path)
    : DepthCamera{name, metafile_path},
      hololens_{Hololens::GetInstance()} {
  hololens_.UseDepthCamera();
  hololens_id_ = hololens_.RegisterID();
}

HololensDepthCamera::~HololensDepthCamera() {
  hololens_.UnregisterID(hololens_id_);
}

bool HololensDepthCamera::SetUp() {
  set_up_ = false;
  if (!metafile_path_.empty())
    if (!LoadMetaData()) return false;
  if (!initial_set_up_ && !hololens_.SetUp(host_)) return false;
  if (use_color_as_world_frame_)
    set_camera2world_pose(*hololens_.depth2color_pose());
  GetIntrinsicsAndDistortionMap();
  SaveMetaDataIfDesired();
  set_up_ = true;
  initial_set_up_ = true;
  return UpdateImage(true);
}

void HololensDepthCamera::set_image_scale(float image_scale) {
  image_scale_ = image_scale;
  set_up_ = false;
}

void HololensDepthCamera::set_use_color_as_world_frame(
    bool use_color_as_world_frame) {
  use_color_as_world_frame_ = use_color_as_world_frame;
  set_up_ = false;
}

void HololensDepthCamera::set_depth_offset(float depth_offset) {
  depth_offset_ = depth_offset;
}

bool HololensDepthCamera::UpdateImage(bool synchronized) {
  if (!set_up_) {
    std::cerr << "Set up azure holo depth camera " << name_ << " first"
              << std::endl;
    return false;
  }

  // Get and undistort image
  //std::shared_ptr<hl2ss::packet> data_depth_ = hololens_.depth_capture()->get_next_packet();
  uint16_t *depth;
  uint16_t *ab;
  hl2ss::unpack_rm_depth_ahat(data_depth_->payload.get(), &depth, &ab);
  cv::remap(
      cv::Mat{cv::Size{intrinsics_.width, intrinsics_.height}, CV_16UC1,
              depth,
              cv::Mat::AUTO_STEP},
      image_, distortion_map_, cv::Mat{}, cv::INTER_NEAREST,
      cv::BORDER_CONSTANT);

  // Add depth offset
  if (depth_offset_) {
    short depth_value_offset = depth_offset_ / depth_scale_;
    image_ += depth_value_offset;
  }

  SaveImageIfDesired();
  return true;
}

float HololensDepthCamera::image_scale() const { return image_scale_; }

bool HololensDepthCamera::use_color_as_world_frame() const {
  return use_color_as_world_frame_;
}

float HololensDepthCamera::depth_offset() const { return depth_offset_; }

const Transform3fA *HololensDepthCamera::color2depth_pose() const {
  return hololens_.color2depth_pose();
}

const Transform3fA *HololensDepthCamera::depth2color_pose() const {
  return hololens_.depth2color_pose();
}

bool HololensDepthCamera::LoadMetaData() {
  // Open file storage from yaml
  cv::FileStorage fs;
  if (!OpenYamlFileStorage(metafile_path_, &fs)) return false;

  // Read parameters from yaml
  ReadOptionalValueFromYaml(fs, "camera2world_pose", &camera2world_pose_);
  ReadOptionalValueFromYaml(fs, "save_directory", &save_directory_);
  ReadOptionalValueFromYaml(fs, "save_index", &save_index_);
  ReadOptionalValueFromYaml(fs, "save_image_type", &save_image_type_);
  ReadOptionalValueFromYaml(fs, "save_images", &save_images_);
  ReadOptionalValueFromYaml(fs, "image_scale", &image_scale_);
  ReadOptionalValueFromYaml(fs, "use_color_as_world_frame",
                            &use_color_as_world_frame_);
  ReadOptionalValueFromYaml(fs, "depth_offset", &depth_offset_);
  fs.release();

  // Process parameters
  if (save_directory_.is_relative())
    save_directory_ = metafile_path_.parent_path() / save_directory_;
  world2camera_pose_ = camera2world_pose_.inverse();
  return true;
}

void HololensDepthCamera::GetIntrinsicsAndDistortionMap() {
  // Load intrinsics from camera
  intrinsics_.fu = calibration_depth_->intrinsics[0];
  intrinsics_.fv = calibration_depth_->intrinsics[1];
  intrinsics_.ppu = calibration_depth_->intrinsics[2];
  intrinsics_.ppv = calibration_depth_->intrinsics[3];
  intrinsics_.width = hl2ss::parameters_rm_depth_ahat::WIDTH;
  intrinsics_.height = hl2ss::parameters_rm_depth_ahat::HEIGHT;
  depth_scale_ = 0.001f;

  // Scale intrinsics according to image scale
  intrinsics_.fu *= image_scale_;
  intrinsics_.fv *= image_scale_;
}

}  // namespace m3t
