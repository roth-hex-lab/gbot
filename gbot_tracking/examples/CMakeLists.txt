cmake_minimum_required(VERSION 3.5)

add_executable(evaluate_gbot_dataset evaluate_gbot_dataset.cpp gbot_evaluator.cpp gbot_evaluator.h yolov8-pose.hpp common.hpp)
target_link_libraries(evaluate_gbot_dataset PUBLIC m3t nvinfer nvinfer_plugin cudart)

add_executable(run_assembly_demo run_assembly_demo.cpp yolov8-pose.hpp common.hpp)
target_link_libraries(run_assembly_demo PUBLIC m3t nvinfer nvinfer_plugin cudart)
