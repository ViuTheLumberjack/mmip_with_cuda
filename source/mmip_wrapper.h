//
// Created by viu on 16/04/2025.
//

#ifndef MMIP_WRAPPER_H
#define MMIP_WRAPPER_H

#include <opencv2/core/mat.hpp>
#include <opencv2/opencv.hpp>

cv::Mat loadImage(const std::string& imagePath, const int flags);
void displayImage(const std::string& windowName, const cv::Mat& img);
void saveImage(const std::string& path, const cv::Mat& img);

#endif //MMIP_WRAPPER_H
