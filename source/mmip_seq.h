//
// Created by viu on 17/04/2025.
//

#ifndef MMIP_SEQ_H
#define MMIP_SEQ_H

#include "mmip_wrapper.h"

namespace SeqOperations{
    cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat openingImage(const cv::Mat& img, const cv::Mat& kernel);
    cv::Mat closingImage(const cv::Mat& img, const cv::Mat& kernel);
}

#endif //MMIP_SEQ_H
