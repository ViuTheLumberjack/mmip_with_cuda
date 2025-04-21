//
// Created by viu on 16/04/2025.
//

#include "mmip_wrapper.h"

cv::Mat loadImage(const std::string& imagePath, const int flags) {
    cv::Mat img = cv::imread(imagePath, flags);
    if (img.empty()) {
        throw std::runtime_error("Could not open or find the image: " + imagePath);
    }
    return img;
}

void displayImage(const std::string& windowName, const cv::Mat& img) {
    cv::imshow(windowName, img);
    if (const int k = cv::waitKey(0); k == 27) { // ESC key
        cv::destroyWindow(windowName);
    }
}

void saveImage(const std::string& path, const cv::Mat& img) {
    cv::imwrite(path, img);
}

cv::Mat morphologicalOperation(const cv::Mat& img, const int operation, const cv::Mat& kernel) {
    cv::Mat result;
    cv::morphologyEx(img, result, operation, kernel);
    return result;
}