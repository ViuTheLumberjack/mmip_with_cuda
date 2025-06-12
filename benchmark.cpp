#include <filesystem>
#include <vector>
#include <chrono>
#include <fstream>
#include "source/mmip_wrapper.h"
#include "source/mmip_seq.h"
#include "source/mmip_cpu.h"
#include "source/mmip_cuda.cuh"

constexpr int ITERATIONS = 10;
constexpr int MAX_DOWNSAMPLE = 10;

void writeResults(const std::filesystem::path &resultsPath, const std::vector<long> &results)
{
    std::ofstream file{};
    file.open(resultsPath);
    for (const auto &result : results)
    {
        file << result << std::endl;
    }

    file.close();

    std::cout << "Written: " << results.size() << " tests" << std::endl;
}

int main()
{
    auto image = loadImage("images/spiaggia.jpg", cv::IMREAD_GRAYSCALE);
    std::filesystem::path result_base_folder = std::filesystem::path() / "results";
    if (!std::filesystem::exists(result_base_folder))
    {
        std::filesystem::create_directory(result_base_folder);
    }

    std::cout << "Path : " << result_base_folder << std::endl;

    auto kernels = std::vector<std::pair<cv::Mat, std::string>>{
        /*
         */
        {cv::getStructuringElement(cv::MORPH_RECT, cv::Size(33, 33)), "rect33"},
        {cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(33, 33)), "cross33"},
        {cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(33, 33)), "ellipse33"},
        /*
         {cv::getStructuringElement(cv::MORPH_RECT, cv::Size(15, 15)), "rect15"},
         {cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(15, 15)), "cross15"},
         {cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(15, 15)), "ellipse15"},
         {cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)), "rect7"},
         {cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(7, 7)), "cross7"},
         {cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7)), "ellipse7"},
         {cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3)), "rect3"},
         {cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3)), "cross3"},
         {cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3)), "ellipse3"},
         */
    };

    auto tiles = std::vector<int>{
        4, 8, 16, 32, 64};

    std::vector<long> durationSeq, durationParallel, durationCudaBase, durationCudaHalo, durationCudaExtern;

    for (const auto &kernel : kernels)
    {
        for (int downsample = 1; downsample <= MAX_DOWNSAMPLE; ++downsample)
        {
            cv::Mat downsampledImage;
            cv::Mat kernelResized = kernel.first;
            cv::resize(image, downsampledImage, cv::Size(), 1.0 / downsample, 1.0 / downsample, cv::INTER_LINEAR);
            std::cout << "Downsampling image by factor: " << downsample << std::endl;
            std::cout << "Image: " << downsampledImage.size() << std::endl;

            for (int iteration = 0; iteration < ITERATIONS; ++iteration)
            {
                std::cout << "Processing kernel: " << kernel.second << ", downsample: " << downsample << ", iteration: " << iteration + 1 << std::endl;

                auto start = std::chrono::high_resolution_clock::now();
                SeqOperations::erodeImage(downsampledImage, kernelResized);
                auto end = std::chrono::high_resolution_clock::now();
                durationSeq.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

                start = std::chrono::high_resolution_clock::now();
                CpuOperations::erodeImage(downsampledImage, kernelResized);
                end = std::chrono::high_resolution_clock::now();
                durationParallel.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

                // write the results to files, one file for each operation
                auto result_folder = result_base_folder / kernel.second / ("downsample_" + std::to_string(downsample));
                if (!std::filesystem::exists(result_folder))
                {
                    std::filesystem::create_directories(result_folder);
                }

                std::cout << "Writing CPU results to: " << result_folder << std::endl;

                writeResults(result_folder / "seq_results.txt", durationSeq);
                writeResults(result_folder / "parallel_results.txt", durationParallel);

                // Clear the vectors for the next kernel
                durationSeq.clear();
                durationParallel.clear();

                for (const int tile : tiles)
                {
                    start = std::chrono::high_resolution_clock::now();
                    GpuOperations::erodeImage(GpuOperations::Base, downsampledImage, kernelResized, tile);
                    end = std::chrono::high_resolution_clock::now();
                    durationCudaBase.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

                    start = std::chrono::high_resolution_clock::now();
                    GpuOperations::erodeImage(GpuOperations::Halo, downsampledImage, kernelResized, tile);
                    end = std::chrono::high_resolution_clock::now();
                    durationCudaHalo.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

                    start = std::chrono::high_resolution_clock::now();
                    GpuOperations::erodeImage(GpuOperations::Extern, downsampledImage, kernelResized, tile);
                    end = std::chrono::high_resolution_clock::now();
                    durationCudaExtern.push_back(std::chrono::duration_cast<std::chrono::microseconds>(end - start).count());

                    std::cout << "Writing GPU results to: " << result_folder << std::endl;
                    writeResults(result_folder / ("cuda_base_results_" + std::to_string(tile) + ".txt"), durationCudaBase);
                    writeResults(result_folder / ("cuda_halo_results_" + std::to_string(tile) + ".txt"), durationCudaHalo);
                    writeResults(result_folder / ("cuda_extern_results_" + std::to_string(tile) + ".txt"), durationCudaExtern);
                    durationCudaBase.clear();
                    durationCudaHalo.clear();
                    durationCudaExtern.clear();
                }
            }
        }
    }

    return 0;
}