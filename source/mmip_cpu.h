//
// Created by viu on 15/04/2025.
//

#ifndef MMIP_H
#define MMIP_H

#include "mmip_wrapper.h"

cv::Mat morphologicalOperation(const cv::Mat& img, const int operation, const cv::Mat& kernel);
cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat openingImage(const cv::Mat& img, const cv::Mat& kernel);
cv::Mat closingImage(const cv::Mat& img, const cv::Mat& kernel);

#endif //MMIP_H
