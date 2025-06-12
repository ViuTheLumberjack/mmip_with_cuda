#include <iostream>
#include "source/mmip_wrapper.h"
#include "source/mmip_cpu.h"
#include "source/mmip_cuda.cuh"

cv::Mat compareResultsWithCV(cv::Mat (*myOperation)(const cv::Mat &, const cv::Mat &), const cv::Mat &img, const int operation, const cv::Mat &kernel, const std::string &operationName)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto myResult = myOperation(img, kernel);
    std::cout << "Custom " << operationName << " Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    auto cvResult = morphologicalOperation(img, operation, kernel);
    std::cout << "OpenCV " << operationName << " Time: "
              << std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::high_resolution_clock::now() - start).count()
              << " ms\n";

    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (cvResult.at<uchar>(i, j) != myResult.at<uchar>(i, j))
            {
                std::cout << operationName << " mismatch at (" << i << ", " << j << ")\n";
            }
        }
    }

    return myResult;
}

cv::Mat compareImplementation(cv::Mat (*cpuOp)(const cv::Mat &, const cv::Mat &), cv::Mat (*gpuOp)(const GpuOperations::Implementation i, const cv::Mat &, const cv::Mat &, const int tile_width), const GpuOperations::Implementation i, const cv::Mat &img, const cv::Mat &kernel, const std::string &operationName, const int tile_width = 16)
{
    auto start = std::chrono::high_resolution_clock::now();
    auto cpuRes = cpuOp(img, kernel);
    auto end = std::chrono::high_resolution_clock::now();

    auto cpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "CPU " << operationName << " Time: "
              << cpuTime
              << " ms\n";

    start = std::chrono::high_resolution_clock::now();
    auto gpuRes = gpuOp(i, img, kernel, tile_width);
    end = std::chrono::high_resolution_clock::now();
    auto gpuTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    std::cout << "GPU " << operationName << " Time: "
              << gpuTime
              << " ms\n";

    std::cout << "Speedup: " << (float)cpuTime / gpuTime << "x" << std::endl;

    int counter = 0;
    for (int i = 0; i < img.rows; i++)
    {
        for (int j = 0; j < img.cols; j++)
        {
            if (gpuRes.at<uchar>(i, j) != cpuRes.at<uchar>(i, j))
            {
                counter += 1;
            }
        }
    }
    std::cout << "# Errors: " << counter << std::endl;

    return gpuRes;
}

int main()
{
    auto image = loadImage("images/spiaggia.jpg", cv::IMREAD_GRAYSCALE);
    auto kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(33, 33));

    GpuOperations::Implementation implementation = GpuOperations::Implementation::Extern;
    cv::Mat myErode;
    cv::Mat myDilate;
    cv::Mat myOpen;
    cv::Mat myClose;

    if (false)
    {
        myErode = compareResultsWithCV(CpuOperations::erodeImage, image, cv::MORPH_ERODE, kernel, "Erosion");
        myDilate = compareResultsWithCV(CpuOperations::dilateImage, image, cv::MORPH_DILATE, kernel, "Dilation");
        myOpen = compareResultsWithCV(CpuOperations::openingImage, image, cv::MORPH_OPEN, kernel, "Opening");
        myClose = compareResultsWithCV(CpuOperations::closingImage, image, cv::MORPH_CLOSE, kernel, "Closing");
    }
    else
    {
        myErode = compareImplementation(CpuOperations::erodeImage, GpuOperations::erodeImage, implementation, image, kernel, "Erosion");
    }

    displayImage("Original Image", image);
    displayImage("Eroded Image (Custom)", myErode);
    /*
    displayImage("Dilated Image (Custom)", myDilate);
    displayImage("Opened Image (Custom)", myOpen);
    displayImage("Closed Image (Custom)", myClose);
    */
}
