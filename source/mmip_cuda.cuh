//
// Created by viu on 16/04/2025.
//

#ifndef MMIP_CUDA_CUH
#define MMIP_CUDA_CUH

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

namespace GpuOperations{
    enum Implementation{
        Base,
        Halo,
        Extern
    };

    cv::Mat erodeImage(const Implementation i, const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat dilateImage(const Implementation i, const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat openingImage(const Implementation i, const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat closingImage(const Implementation i, const cv::Mat& img, const cv::Mat& kernel);
}

#endif //MMIP_CUDA_CUH
