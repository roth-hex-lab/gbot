// SPDX-License-Identifier: MIT
// GBOT
// Adapted from 2023 Manuel Stoiber, German Aerospace Center (DLR)

#include "gbot_evaluator.h"

int main() {
  // Directories
  std::filesystem::path dataset_directory{"D:/PhD/shiyu/KARVIMIO/GBOT_dataset"};
  std::filesystem::path external_directory{"D:/PhD/shiyu/KARVIMIO/GBOT_dataset/GBOT_dataset/external"};
  std::filesystem::path result_path{"D:/PhD/shiyu/KARVIMIO/GBOT_dataset/NanoViseV2/result.csv"};
  std::filesystem::path result_pose_path{
      "D:/PhD/shiyu/KARVIMIO/GBOT_dataset/NanoViseV2/test/YOLOrealcluster_assembly-test.csv"};
  std::string engine_file_path = "D:/PhD/shiyu/KARVIMIO/GBOT_dataset/NanoViseV2/model/yolo/weights";


  // Dataset configuration
  std::vector<std::string> object_names{"NanoViseV2"};
  std::vector<std::string> difficulty_levels{"test"};
  std::vector<std::string> depth_names{"depth"};
  std::vector<std::string> sequence_numbers{"realcluster"};

  // Run experiments
  GBOTEvaluator evaluator{
      "evaluator",     dataset_directory, engine_file_path, external_directory, result_pose_path,
                         object_names,    difficulty_levels, depth_names,
                         sequence_numbers};
  evaluator.set_region_modality_setter([&](auto r) {
    r->set_n_lines_max(300);
    r->set_use_adaptive_coverage(true);
    r->set_min_continuous_distance(3.0f);
    r->set_function_length(8);
    r->set_distribution_length(12);
    r->set_function_amplitude(0.43f);
    r->set_function_slope(0.5f);
    r->set_learning_rate(1.3f);
    r->set_scales({9, 7, 5, 2});
    r->set_standard_deviations({25.0f, 15.0f, 10.0f});
    r->set_unconsidered_line_length(0.5f);
    r->set_max_considered_line_length(20.0f);
    if (!r->use_shared_color_histograms()) {
      r->set_n_histogram_bins(16);
      r->set_learning_rate_f(0.2f);
      r->set_learning_rate_b(0.2f);
    }
  });
  evaluator.set_color_histograms_setter([&](auto h) {
    h->set_n_bins(16);
    h->set_learning_rate_f(0.2f);
    h->set_learning_rate_b(0.2f);
  });
  evaluator.set_depth_modality_setter([&](auto d) {
    d->set_n_points_max(300);
    d->set_use_adaptive_coverage(true);
    d->set_use_depth_scaling(true);
    d->set_stride_length(0.008f);
    d->set_considered_distances({0.1f, 0.08f, 0.05f});
    d->set_standard_deviations({0.05f, 0.03f, 0.02f});
  });
  evaluator.set_optimizer_setter([&](auto o) {
    o->set_tikhonov_parameter_rotation(100.0f);
    o->set_tikhonov_parameter_translation(1000.0f);
  });
  evaluator.set_tracker_setter([&](auto t) {
    t->set_n_update_iterations(2);
    t->set_n_corr_iterations(6);
  });
  evaluator.set_evaluation_mode(GBOTEvaluator::EvaluationMode::COMBINED);
  evaluator.set_track_mode(GBOTEvaluator::TrackMode::YOLO);
  evaluator.set_evaluate_external(false);
  evaluator.set_external_results_folder("dart");
  evaluator.set_run_sequentially(true);
  evaluator.set_use_random_seed(false);
  evaluator.set_n_vertices_evaluation(1000);
  evaluator.set_visualize_frame_results(false);
  evaluator.set_visualize_tracking(true);
  evaluator.set_use_shared_color_histograms(true);
  evaluator.set_use_region_checking(true);
  evaluator.set_use_silhouette_checking(true);
  evaluator.SetUp();
  evaluator.Evaluate();
  evaluator.SaveResults(result_path);
}
