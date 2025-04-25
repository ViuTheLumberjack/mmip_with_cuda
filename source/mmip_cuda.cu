//
// Created by viu on 16/04/2025.
//

#include "mmip_cuda.cuh"

namespace GpuOperations{
    __global__ void erosionKernel(const unsigned char* input, unsigned char* output, const unsigned char* kernel, const int width, const int height, const int k_width, const int k_height) {
        /*
        __global__ void convolution_1D_basic_kernel(float *N, float *P, int Mask_Width, int Width) {
            int i = blockIdx.x*blockDim.x + threadIdx.x; 
            __shared__ float N_ds[TILE_SIZE];
            N_ds[threadIdx.x] = N[i];
            __syncthreads();
            int This_tile_start_point = blockIdx.x * blockDim.x;
            int Next_tile_start_point = (blockIdx.x + 1) * blockDim.x; 
            int N_start_point = i - (Mask_Width/2); 
            float Pvalue = 0;
            for (intj = 0; j < Mask_Width; j ++) { 
                int N_index = N_start_point + j; 
                if (N_index >= 0 && N_index < Width) { 
                    if ((N_index >= This_tile_start_point) && (
                        Pvalue += N_ds[threadIdx.x+j-(Mask_Width/2)]*M[j]; } 
                        else {
                            Pvalue += N[N_index] * M[j]; } } 
                            } P[i] = Pvalue; }
                            */
        int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
        int y = blockIdx.y * blockDim.y + threadIdx.y; // rows
             
        int anchor_x = k_width / 2;
        int anchor_y = k_height / 2;
        unsigned char minVal = 255;
        for (int i_k = -anchor_y; i_k <= anchor_y; i_k++) { // rows
            for (int j_k = -anchor_x; j_k <= anchor_x; j_k++) { // cols
                if (kernel[(i_k + anchor_y) * k_width + j_k + anchor_x] == 1) {
                    const int actual_x = x + j_k;
                    const int actual_y = y + i_k;

                    if((actual_x >= 0 && actual_y >=0) && (actual_x < width && actual_y < height)){
                        if(input[actual_y * width + actual_x] < minVal){
                            minVal = input[actual_y * width + actual_x];
                        }
                    }
                }
            }
        }

        if (x < width && y < height) {
            output[y * width + x] = minVal;
        }
    }

    cv::Mat erodeImage(const cv::Mat& img, const cv::Mat& kernel){
        int IMG_WIDTH = img.cols;
        int IMG_HEIGHT = img.rows;
        
        size_t imgByte = IMG_WIDTH * IMG_HEIGHT * sizeof(unsigned char);
        size_t kernelBytes = kernel.rows * kernel.cols * sizeof(unsigned char); 

        const unsigned char* h_input = img.ptr(0);
        unsigned char* h_output = new unsigned char[imgByte];
        const unsigned char* h_kernel = kernel.ptr(0);

        unsigned char *d_input, *d_output, *d_kernel;
        cudaMalloc(&d_input, imgByte);
        cudaMalloc(&d_output, imgByte);
        cudaMalloc(&d_kernel, kernelBytes);
    
        // Copia input da host a device
        cudaMemcpy(d_input, h_input, imgByte, cudaMemcpyHostToDevice);
        cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice);
    
        // Lancio del kernel
        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((IMG_WIDTH + 15) / 16, (IMG_HEIGHT + 15) / 16);
        erosionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, IMG_WIDTH, IMG_HEIGHT, kernel.cols, kernel.rows);
    
        cudaDeviceSynchronize();
        cudaError err = cudaGetLastError();

        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch erosionKernel (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        // Copia risultato da device a host
        err = cudaMemcpy(h_output, d_output, imgByte, cudaMemcpyDeviceToHost);
    
        if (err != cudaSuccess) {
            fprintf(stderr, "Failed to launch erosionKernel (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        // Pulizia
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_kernel);
    
        auto mat = cv::Mat(img.rows, img.cols, CV_8UC1, h_output);
        return mat;
    }
    
    cv::Mat dilateImage(const cv::Mat& img, const cv::Mat& kernel){
        
        return cv::Mat();
    }
    
    cv::Mat openingImage(const cv::Mat& img, const cv::Mat& kernel){
        
        return cv::Mat();
    }
    
    cv::Mat closingImage(const cv::Mat& img, const cv::Mat& kernel){

        return cv::Mat();
    }
}