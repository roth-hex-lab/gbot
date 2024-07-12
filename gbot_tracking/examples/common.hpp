//
// Created by ubuntu on 4/7/23.
//

#ifndef POSE_NORMAL_COMMON_HPP
#define POSE_NORMAL_COMMON_HPP
#include "opencv2/opencv.hpp"
#include <sys/stat.h>
//#include <unistd.h>
#include "NvInfer.h"
#include <filesystem>

#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

class Loggers : public nvinfer1::ILogger
{
public:
	nvinfer1::ILogger::Severity reportableSeverity;

	explicit Loggers(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kINFO) :
		reportableSeverity(severity)
	{
	}

	void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
	{
		if (severity > reportableSeverity)
		{
			return;
		}
		switch (severity)
		{
		case nvinfer1::ILogger::Severity::kINTERNAL_ERROR:
			std::cerr << "INTERNAL_ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kERROR:
			std::cerr << "ERROR: ";
			break;
		case nvinfer1::ILogger::Severity::kWARNING:
			std::cerr << "WARNING: ";
			break;
		case nvinfer1::ILogger::Severity::kINFO:
			std::cerr << "INFO: ";
			break;
		default:
			std::cerr << "VERBOSE: ";
			break;
		}
		std::cerr << msg << std::endl;
	}
};

inline int get_size_by_dims(const nvinfer1::Dims& dims)
{
	int size = 1;
	for (int i = 0; i < dims.nbDims; i++)
	{
		size *= dims.d[i];
	}
	return size;
}

inline int type_to_size(const nvinfer1::DataType& dataType)
{
	switch (dataType)
	{
	case nvinfer1::DataType::kFLOAT:
		return 4;
	case nvinfer1::DataType::kHALF:
		return 2;
	case nvinfer1::DataType::kINT32:
		return 4;
	case nvinfer1::DataType::kINT8:
		return 1;
	case nvinfer1::DataType::kBOOL:
		return 1;
	default:
		return 4;
	}
}

inline static float clamp(float val, float min, float max)
{
	return val > min ? (val < max ? val : max) : min;
}

inline bool IsPathExist(const std::string& path)
{
	namespace fs = std::filesystem;

	// Check if the path exists
	fs::path pathObj(path);
	return fs::exists(pathObj);
}

inline bool IsFile(const std::string& path)
{
	if (!IsPathExist(path))
	{
		printf("%s:%d %s not exist\n");
		return false;
	}
	//struct stat buffer;
	return true;
}

inline bool IsFolder(const std::string& path)
{
	if (!IsPathExist(path))
	{
		return false;
	}
	//struct stat buffer;
	return true;
}

namespace pose
{
	struct Binding
	{
		size_t size = 1;
		size_t dsize = 1;
		nvinfer1::Dims dims;
		std::string name;
	};

	struct Object
	{
		cv::Rect_<float> rect;
		int label = 0;
		float prob = 0.0;
        std::vector<float> kps;
	};

	struct PreParam
	{
		float ratio = 1.0f;
		float dw = 0.0f;
		float dh = 0.0f;
		float height = 0;
		float width = 0;
	};
}
#endif //POSE_NORMAL_COMMON_HPP
