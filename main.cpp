#include <iostream>
#include "source/mmip_wrapper.h"
#include "source/mmip_cpu.h"

cv::Mat compareResults(cv::Mat (*myOperation)(const cv::Mat&, const cv::Mat&), const cv::Mat &img, const int operation, const cv::Mat &kernel, const std::string &operationName) {
    auto start = std::chrono::high_resolution_clock::now();
    auto myResult = myOperation(img, kernel);
    std::cout << "Custom "<< operationName <<" Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    auto cvResult = morphologicalOperation(img, operation, kernel);
    std::cout << "OpenCV "<< operationName <<" Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
              << " ms\n";

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (cvResult.at<uchar>(i, j) != myResult.at<uchar>(i, j)) {
                std::cout << operationName <<" mismatch at (" << i << ", " << j << ")\n";
            }
        }
    }

    return myResult;
}

int main() {
    auto image = loadImage("images/enrico.png", cv::IMREAD_GRAYSCALE);
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE , cv::Size(33, 33));

    auto myErode = compareResults(erodeImage, image, cv::MORPH_ERODE, kernel, "Erosion");
    auto myDilate = compareResults(dilateImage, image, cv::MORPH_DILATE, kernel, "Dilation");
    auto myOpen = compareResults(openingImage, image, cv::MORPH_OPEN, kernel, "Opening");
    auto myClose = compareResults(closingImage, image, cv::MORPH_CLOSE, kernel, "Closing");

    displayImage("Original Image", image);
    displayImage("Eroded Image (Custom)", myErode);
    displayImage("Dilated Image (Custom)", myDilate);
    displayImage("Opened Image (Custom)", myOpen);
    displayImage("Closed Image (Custom)", myClose);
}
