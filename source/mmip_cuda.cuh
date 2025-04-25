//
// Created by viu on 16/04/2025.
//

#ifndef MMIP_CUDA_CUH
#define MMIP_CUDA_CUH

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

namespace GpuOperations{
    cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat openingImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat closingImage(const cv::Mat& img, const cv::Mat& kernel);
}

#endif //MMIP_CUDA_CUH
