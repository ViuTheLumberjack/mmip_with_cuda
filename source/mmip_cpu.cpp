//
// Created by viu on 11/04/2025.
//

#include "mmip_cpu.h"

namespace CpuOperations{
    cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel) {
        const auto anchor = cv::Point(kernel.cols / 2, kernel.rows / 2);
        cv::Mat appImage;
        copyMakeBorder(img, appImage, anchor.y, anchor.y, anchor.x, anchor.x, cv::BORDER_CONSTANT, cv::Scalar(255));
        
        auto result = cv::Mat(img.rows, img.cols, CV_8UC1);
        
        const auto resultPtr = result.ptr<uchar>();
        const auto appImagePtr = appImage.ptr<uchar>();
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                uchar ref = 255;
                // inf_y in B |f(x+y) - b(y)|
                for (int i_k = 0; i_k < kernel.rows; i_k++) {
                    for (int j_k = 0; j_k < kernel.cols; j_k++) {
                        if (kernel.at<uchar>(i_k, j_k) == 1) {
                            const int actual_i = i + i_k;
                            const int actual_j = j + j_k;
                            
                            ref = std::min(appImagePtr[actual_i * appImage.cols + actual_j], ref);
                        }
                    }
                }
                resultPtr[i * img.cols + j] = ref;
            }
        }
        
        return result;
    }
    
    cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel) {
        const auto anchor = cv::Point(kernel.cols / 2, kernel.rows / 2);
        cv::Mat appImage;
        copyMakeBorder(img, appImage, anchor.y, anchor.y, anchor.x, anchor.x, cv::BORDER_CONSTANT, cv::Scalar(0));
        
        auto result = cv::Mat(img.rows, img.cols, CV_8UC1);
        
        const auto resultPtr = result.ptr<uchar>();
        const auto appImagePtr = appImage.ptr<uchar>();
        
        #pragma omp parallel for collapse(2) schedule(dynamic)
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                uchar ref = 0;
                // sup_y in B |f(x-y) + b(y)|
                for (int i_k = 0; i_k < kernel.rows; i_k++) {
                    for (int j_k = 0; j_k < kernel.cols; j_k++) {
                        if (kernel.at<uchar>(i_k, j_k) == 1) {
                            const int actual_i = i + i_k;
                            const int actual_j = j + j_k;
                            
                            ref = std::max( appImagePtr[actual_i * appImage.cols + actual_j], ref);
                        }
                    }
                }
                resultPtr[i * img.cols + j] = ref;
            }
        }
        
        return result;
    }
    
    cv::Mat closingImage(const cv::Mat& img, const cv::Mat& kernel) {
        auto result = dilateImage(img, kernel);
        result = erodeImage(result, kernel);
        
        return result;
    }
    
    cv::Mat openingImage(const cv::Mat& img, const cv::Mat& kernel) {
        auto result = erodeImage(img, kernel);
        result = dilateImage(result, kernel);
        
        return result;
    }   
}