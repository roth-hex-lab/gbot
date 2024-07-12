cmake_minimum_required(VERSION 3.5)

# GBOT: Graph-Based 3D Object Tracking for
 Augmented Reality-Assisted Assembly Guidance(GBOT)
# =============================================================================
project(M3T)


# Options
# =============================================================================
option(USE_AZURE_KINECT "Use Azure Kinect" ON)
option(USE_HOLOLENS "Use HOLOLENS" ON)
option(USE_REALSENSE "Use RealSense D435" ON)
option(USE_GTEST "Use gtest" ON)


# Libraries
# =============================================================================

# Build static on Windows
if (WIN32)
    set(GLEW_STATIC true)
    set(OpenCV_STATIC true)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -MTd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -MT")
endif ()

# Used libraries
list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/packages/glew_x64-windows-static")
list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/packages/glfw3_x64-windows")
list(APPEND CMAKE_PREFIX_PATH "C:/vcpkg/packages/fmt_x64-windows")

set (OpenCV_DIR C:/opencv/build)
set(TRT_DIR "C:/TensorRT-8.6.1.6")
set(TensorRT_INCLUDE_DIRS C:/TensorRT-8.6.1.6/include)
set(TensorRT_LIBRARIES C:/TensorRT-8.6.1.6/lib)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)


# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")
list(APPEND INCLUDE_DIRS
        ${CUDA_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${TensorRT_INCLUDE_DIRS}
        ./include
        )

list(APPEND ALL_LIBS
        ${CUDA_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${TensorRT_LIBRARIES}
        )

include_directories(${INCLUDE_DIRS})
find_package(Eigen3 3.3.2 REQUIRED)
find_package(GLEW REQUIRED)
find_package(fmt REQUIRED)
if (UNIX)
    set(OpenGL_GL_PREFERENCE LEGACY)
endif ()
find_package(OpenGL REQUIRED)
find_path(RESTBED_INCLUDE_DIRS "restbed")
find_library(RESTBED_LIBRARY NAMES "librestbed.a" "restbed.lib")
find_package(glfw3 3.1.2 REQUIRED)
find_package(OpenCV 4.3.0 REQUIRED COMPONENTS core imgproc highgui imgcodecs calib3d features2d xfeatures2d OPTIONAL_COMPONENTS cudafeatures2d)

if (USE_AZURE_KINECT)
    find_package(k4a 1.3.0 REQUIRED)
endif ()

if (USE_REALSENSE)
    find_package(realsense2 REQUIRED)
endif ()

find_package(OpenMP)
if (OPENMP_FOUND)
    if (WIN32)
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} -openmp:experimental")
        set(CMAKE_C_FLAGS_RELEASE
                "${CMAKE_C_FLAGS_RELEASE} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS_RELEASE
                "${CMAKE_CXX_FLAGS_RELEASE} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    else ()
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        set(CMAKE_EXE_LINKER_FLAGS
                "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
    endIf ()
endif ()


# Compiler Settings
# =============================================================================

# Default settings
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag(-march HAS_MARCH)
check_cxx_compiler_flag(-mtune HAS_MTUNE)
if (HAS_MARCH)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
endif ()
if (HAS_MTUNE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mtune=native")
endif ()

# Debug settings
check_cxx_compiler_flag(-ggdb HAS_GGDB)
check_cxx_compiler_flag(-Z7 HAS_Z7)
check_cxx_compiler_flag(-ftrapv HAS_FTRAPV)
check_cxx_compiler_flag(-Od HAS_OD)
if (HAS_GGDB)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ggdb")
elseif (HAS_Z7)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Z7")
endif ()
if (HAS_FTRAPV)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -ftrapv")
endif ()
if (HAS_OD)
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Od")
endif ()
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall")

# Release settings
check_cxx_compiler_flag(-Ob3 HAS_OB3)
check_cxx_compiler_flag(-O3 HAS_O3)
check_cxx_compiler_flag(-Ob2 HAS_OB2)
check_cxx_compiler_flag(-O2 HAS_O2)
if (HAS_OB3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob3")
elseif (HAS_O3)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif (HAS_OB2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ob2")
elseif (HAS_O2)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O2")
endif ()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG")


# Definitions
# =============================================================================
if (USE_AZURE_KINECT)
    add_definitions( -DUSE_AZURE_KINECT=TRUE )
endif ()
if (USE_REALSENSE)
    add_definitions( -DUSE_REALSENSE=TRUE )
endif ()
if(OpenCV_CUDA_VERSION)
    add_definitions( -DUSE_CUDA=TRUE ) 
endif ()
if (USE_GTEST)
    set (GTEST_LIBRARY C:/googletest/build/lib/Release/gtest.lib)
    set (GTEST_MAIN_LIBRARY C:/googletest/build/lib/Release/gtest_main.lib)
    set (GTEST_INCLUDE_DIR C:/googletest/googletest/include)
    add_definitions( -DUSE_GTEST=TRUE )
endif ()


# Directories
# =============================================================================
add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(examples)
add_subdirectory(doc)
