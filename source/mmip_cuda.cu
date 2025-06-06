//
// Created by viu on 16/04/2025.
//

#include "mmip_cuda.cuh"

#define TILE_WIDTH 16
namespace GpuOperations
{
    void checkError(const cudaError err)
    {
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Failed to launch erosionKernel (error code %s)!\n",
                    cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
    }

    __global__ void dumbErosionKernel(const unsigned char *input, unsigned char *output, const unsigned char *kernel, const int width, const int height, const int k_width, const int k_height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
        int y = blockIdx.y * blockDim.y + threadIdx.y; // rows

        int anchor_x = k_width / 2;
        int anchor_y = k_height / 2;
        unsigned char minVal = 255;
        for (int i_k = -anchor_y; i_k <= anchor_y; i_k++)
        { // rows
            for (int j_k = -anchor_x; j_k <= anchor_x; j_k++)
            { // cols
                if (kernel[(i_k + anchor_y) * k_width + j_k + anchor_x] == 1)
                {
                    const int actual_x = x + j_k;
                    const int actual_y = y + i_k;

                    if ((actual_x >= 0 && actual_y >= 0) && (actual_x < width && actual_y < height))
                    {
                        if (input[actual_y * width + actual_x] < minVal)
                        {
                            minVal = input[actual_y * width + actual_x];
                        }
                    }
                }
            }
        }

        if (x < width && y < height)
        {
            output[y * width + x] = minVal;
        }
    }

    __global__ void cachedTiledHaloErosionKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y; // Global row index of the output pixel
        int x = blockIdx.x * blockDim.x + threadIdx.x; // Global col index of the output pixel
        
        // Shared memory for the tile, renamed for clarity
        __shared__ unsigned char tile_cache[TILE_WIDTH][TILE_WIDTH];

        // Load data into shared memory: each thread loads one pixel.
        if (y < height && x < width)
        {
            tile_cache[threadIdx.y][threadIdx.x] = input[y * width + x];
        }
        else
        {
            // Pad with max value (255 for erosion) if thread is outside image bounds
            // This ensures out-of-bounds areas don't wrongly become the minimum.
            tile_cache[threadIdx.y][threadIdx.x] = 255; 
        }

        __syncthreads(); // Synchronize to ensure all data is loaded into tile_cache

        // Proceed with erosion if the current thread is processing a valid pixel within the image
        if (x < width && y < height)
        {
            int anchor_y = k_height / 2; // Kernel anchor y-coordinate
            int anchor_x = k_width / 2; // Kernel anchor x-coordinate
            unsigned char min_pixel_val = 255; // Initialize with max value for erosion

            // Iterate over the structuring element (kernel)
            for (int k_row = 0; k_row < k_height; ++k_row) // Kernel row index
            {
                for (int k_col = 0; k_col < k_width; ++k_col) // Kernel column index
                {
                    if (kernel[k_row * k_width + k_col] == 1) // Process only active parts of the kernel
                    {
                        // Calculate offsets from the kernel anchor
                        int kernel_offset_y = k_row - anchor_y;
                        int kernel_offset_x = k_col - anchor_x;

                        // Coordinates of the neighbor pixel in the shared memory tile
                        int S_y = threadIdx.y + kernel_offset_y; // y-coord in tile_cache
                        int S_x = threadIdx.x + kernel_offset_x; // x-coord in tile_cache

                        unsigned char current_neighbor_val;

                        // Check if the neighbor pixel is within the cached tile
                        if (S_x >= 0 && S_x < TILE_WIDTH && S_y >= 0 && S_y < TILE_WIDTH)
                        {
                            current_neighbor_val = tile_cache[S_y][S_x];
                        }
                        else // Neighbor is in the halo region, requires global memory access
                        {
                            // Global coordinates of the neighbor pixel
                            int G_y = y + kernel_offset_y; // Global y-coord of neighbor
                            int G_x = x + kernel_offset_x; // Global x-coord of neighbor

                            // Boundary check for global memory access
                            if (G_x >= 0 && G_x < width && G_y >= 0 && G_y < height)
                            {
                                current_neighbor_val = input[G_y * width + G_x];
                            }
                            else
                            {
                                // If neighbor is outside image bounds, treat as max value for erosion
                                current_neighbor_val = 255; 
                            }
                        }
                        
                        // Update the minimum value found so far
                        if (current_neighbor_val < min_pixel_val)
                        {
                            min_pixel_val = current_neighbor_val;
                        }
                    }
                }
            }
            output[y * width + x] = min_pixel_val; // Write the result
        }
    }

    __global__ void externErosionKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        extern __shared__ unsigned char shared_mem[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;

        int anchor_x = k_width / 2;
        int anchor_y = k_height / 2;

        int shared_width = blockDim.x + k_width - 1;
        int shared_height = blockDim.y + k_height - 1;

        for (int s_load_y = ty; s_load_y < shared_height; s_load_y += blockDim.y)
        {
            for (int s_load_x = tx; s_load_x < shared_width; s_load_x += blockDim.x)
            {
                int global_load_x = blockIdx.x * blockDim.x - anchor_x + s_load_x;
                int global_load_y = blockIdx.y * blockDim.y - anchor_y + s_load_y;

                if (s_load_x < shared_width && s_load_y < shared_height)
                {
                    if (global_load_x >= 0 && global_load_x < width && global_load_y >= 0 && global_load_y < height)
                    {
                        shared_mem[s_load_y * shared_width + s_load_x] = input[global_load_y * width + global_load_x];
                    }
                    else
                    {
                        shared_mem[s_load_y * shared_width + s_load_x] = 255; // Border handling for erosion (max value)
                    }
                }
            }
        }

        __syncthreads();

        if (x >= width || y >= height)
            return;

        // Perform erosion
        unsigned char min_val = 255;

        for (int j = 0; j < k_height; ++j) // Kernel row index
        {
            for (int i = 0; i < k_width; ++i) // Kernel column index
            {
                if (kernel[j * k_width + i] != 0)
                {
                    // Calculate the coordinates in shared memory to read from:
                    int sm_access_y = ty + j;
                    int sm_access_x = tx + i;

                    unsigned char val = shared_mem[sm_access_y * shared_width + sm_access_x];
                    if (val < min_val)
                    {
                        min_val = val;
                    }
                }
            }
        }

        output[y * width + x] = min_val;
    }

    
    __global__ void dumbDilateKernel(const unsigned char *input, unsigned char *output, const unsigned char *kernel, const int width, const int height, const int k_width, const int k_height)
    {
        int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
        int y = blockIdx.y * blockDim.y + threadIdx.y; // rows

        int anchor_x = k_width / 2;
        int anchor_y = k_height / 2;
        unsigned char max_val = 0;
        for (int i_k = -anchor_y; i_k <= anchor_y; i_k++)
        { // rows
            for (int j_k = -anchor_x; j_k <= anchor_x; j_k++)
            { // cols
                if (kernel[(i_k + anchor_y) * k_width + j_k + anchor_x] == 1)
                {
                    const int actual_x = x + j_k;
                    const int actual_y = y + i_k;

                    if ((actual_x >= 0 && actual_y >= 0) && (actual_x < width && actual_y < height))
                    {
                        if (input[actual_y * width + actual_x] > max_val)
                        {
                            max_val = input[actual_y * width + actual_x];
                        }
                    }
                }
            }
        }

        if (x < width && y < height)
        {
            output[y * width + x] = max_val;
        }
    }

    __global__ void cachedTiledHaloDilateKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        __shared__ unsigned char N_ds[TILE_WIDTH][TILE_WIDTH];

        if (y < height && x < width)
        {
            N_ds[threadIdx.y][threadIdx.x] = input[y * width + x];
        }
        else
        {
            N_ds[threadIdx.y][threadIdx.x] = 255;
        }

        __syncthreads();

        if (x < width && y < height)
        {
            int anchor_y = k_height / 2;
            int anchor_x = k_width / 2;
            unsigned char max_val = 0;
            for (int x_k = 0; x_k < k_width; x_k++)
            {
                for (int y_k = 0; y_k < k_height; y_k++)
                {
                    if (kernel[(y_k)*k_width + x_k] == 1)
                    {
                        int actual_x = threadIdx.x - anchor_x + x_k;
                        int actual_y = threadIdx.y - anchor_y + y_k;

                        if (actual_x >= 0 && actual_y >= 0 && actual_x < TILE_WIDTH && actual_y < TILE_WIDTH)
                        {
                            if (N_ds[actual_y][actual_x] > max_val)
                            {
                                max_val = N_ds[actual_y][actual_x];
                            }
                        }
                        else
                        {
                            actual_x = x - anchor_x + x_k;
                            actual_y = y - anchor_y + y_k;
                            if (actual_y >= 0 && actual_y < height && actual_x >= 0 && actual_x < width)
                            {
                                if (input[actual_y * width + actual_x] > max_val)
                                {
                                    max_val = input[actual_y * width + actual_x];
                                }
                            }
                        }
                    }
                }
            }

            output[y * width + x] = max_val;
        }
    }

    __global__ void externDilateKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        extern __shared__ unsigned char shared_mem[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;

        int anchor_x = k_width / 2;
        int anchor_y = k_height / 2;

        int shared_width = blockDim.x + k_width - 1;
        int shared_height = blockDim.y + k_height - 1;

        for (int s_load_y = ty; s_load_y < shared_height; s_load_y += blockDim.y)
        {
            for (int s_load_x = tx; s_load_x < shared_width; s_load_x += blockDim.x)
            {
                int global_load_x = blockIdx.x * blockDim.x - anchor_x + s_load_x;
                int global_load_y = blockIdx.y * blockDim.y - anchor_y + s_load_y;

                if (s_load_x < shared_width && s_load_y < shared_height)
                {
                    if (global_load_x >= 0 && global_load_x < width && global_load_y >= 0 && global_load_y < height)
                    {
                        shared_mem[s_load_y * shared_width + s_load_x] = input[global_load_y * width + global_load_x];
                    }
                    else
                    {
                        shared_mem[s_load_y * shared_width + s_load_x] = 0; // Border handling for dilate (min value)
                    }
                }
            }
        }

        __syncthreads();

        if (x >= width || y >= height)
            return;

        // Perform erosion
        unsigned char max_val = 0;

        for (int j = 0; j < k_height; ++j) // Kernel row index
        {
            for (int i = 0; i < k_width; ++i) // Kernel column index
            {
                if (kernel[j * k_width + i] != 0)
                {
                    // Calculate the coordinates in shared memory to read from:
                    int sm_access_y = ty + j;
                    int sm_access_x = tx + i;

                    unsigned char val = shared_mem[sm_access_y * shared_width + sm_access_x];
                    if (val > max_val)
                    {
                        max_val = val;
                    }
                }
            }
        }

        output[y * width + x] = max_val;
    }

    cv::Mat erodeImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel)
    {
        size_t imgByte = img.rows * img.cols * sizeof(uchar);
        size_t kernelBytes = kernel.rows * kernel.cols * sizeof(uchar);

        const uchar *h_input = img.ptr();
        const uchar *h_kernel = kernel.ptr();
        uchar *h_output = new uchar[imgByte];

        uchar *d_input, *d_output, *d_kernel;
        checkError(cudaMalloc(&d_input, imgByte));
        checkError(cudaMalloc(&d_output, imgByte));
        checkError(cudaMalloc(&d_kernel, kernelBytes));

        checkError(cudaMemcpy(d_input, h_input, imgByte, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 numBlocks((img.cols + TILE_WIDTH - 1) / TILE_WIDTH, (img.rows + TILE_WIDTH - 1) / TILE_WIDTH);
        switch (i)
        {
        case Base:
            dumbErosionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Halo:
            cachedTiledHaloErosionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Extern:
            externErosionKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * (TILE_WIDTH + kernel.rows - 1) * (TILE_WIDTH + kernel.cols - 1)>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        default:
            exit(EXIT_FAILURE);
        }

        checkError(cudaDeviceSynchronize());

        checkError(cudaMemcpy(h_output, d_output, imgByte, cudaMemcpyDeviceToHost));
        checkError(cudaDeviceSynchronize());

        checkError(cudaFree(d_input));
        checkError(cudaFree(d_output));
        checkError(cudaFree(d_kernel));

        return cv::Mat(img.rows, img.cols, CV_8UC1, h_output);
    }

    cv::Mat dilateImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel)
    {
        size_t imgByte = img.rows * img.cols * sizeof(uchar);
        size_t kernelBytes = kernel.rows * kernel.cols * sizeof(uchar);

        const uchar *h_input = img.ptr();
        const uchar *h_kernel = kernel.ptr();
        uchar *h_output = new uchar[imgByte];

        uchar *d_input, *d_output, *d_kernel;
        checkError(cudaMalloc(&d_input, imgByte));
        checkError(cudaMalloc(&d_output, imgByte));
        checkError(cudaMalloc(&d_kernel, kernelBytes));

        checkError(cudaMemcpy(d_input, h_input, imgByte, cudaMemcpyHostToDevice));
        checkError(cudaMemcpy(d_kernel, h_kernel, kernelBytes, cudaMemcpyHostToDevice));

        dim3 threadsPerBlock(TILE_WIDTH, TILE_WIDTH);
        dim3 numBlocks((img.cols + TILE_WIDTH - 1) / TILE_WIDTH, (img.rows + TILE_WIDTH - 1) / TILE_WIDTH);
        switch (i)
        {
        case Base:
            dumbDilateKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Halo:
            cachedTiledHaloDilateKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Extern:
            externDilateKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * (TILE_WIDTH + kernel.rows - 1) * (TILE_WIDTH + kernel.cols - 1)>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        default:
            exit(EXIT_FAILURE);
        }

        checkError(cudaDeviceSynchronize());

        checkError(cudaMemcpy(h_output, d_output, imgByte, cudaMemcpyDeviceToHost));
        checkError(cudaDeviceSynchronize());

        checkError(cudaFree(d_input));
        checkError(cudaFree(d_output));
        checkError(cudaFree(d_kernel));

        return cv::Mat(img.rows, img.cols, CV_8UC1, h_output);
    }

    cv::Mat openingImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel)
    {
        auto result = erodeImage(i, img, kernel);
        result = dilateImage(i, result, kernel);

        return result;
    }

    cv::Mat closingImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel)
    {
        auto result = dilateImage(i, img, kernel);
        result = erodeImage(i, result, kernel);

        return result;
    }
}