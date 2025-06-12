// %%writefile mmip_with_cuda/source/mmip_cuda.cu
//
// Created by viu on 16/04/2025.
//

#include "mmip_cuda.cuh"

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
        unsigned char min_val = 255;
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
                        min_val = min(input[actual_y * width + actual_x], min_val);
                    }
                }
            }
        }

        if (x < width && y < height)
        {
            output[y * width + x] = min_val;
        }
    }

    __global__ void cachedTiledHaloErosionKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        // 2D array for tile+kernel cache
        extern __shared__ unsigned char smm[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;

        int pdx = (k_width - 1) / 2;
        int pdy = (k_height - 1) / 2;

        int smm_width = blockDim.x + 2 * pdx;
        int smm_height = blockDim.y + 2 * pdy;
        int smm_y = ty + pdy;
        int smm_x = tx + pdx;
        int fill_value = 255; // Value to fill the padding pixels

        // Central pixel

        smm[smm_y * smm_width + smm_x] = (y < height && x < width) ? input[y * width + x] : fill_value;

        __syncthreads();

        // Load padding pixels
        if (tx < pdx)
        {
            smm[smm_y * smm_width + (smm_x - pdx)] = (y < height && x - pdx >= 0 && x - pdx < width) ? input[y * width + x - pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx)
        {
            smm[smm_y * smm_width + (smm_x + pdx)] = (y < height && x + pdx < width) ? input[y * width + x + pdx] : fill_value;
        }
        if (ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + smm_x] = (y - pdy >= 0 && y - pdy < height && x < width) ? input[(y - pdy) * width + x] : fill_value;
        }
        if (ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + smm_x] = (y + pdy < height && x < width) ? input[(y + pdy) * width + x] : fill_value;
        }

        // Load corner pixels
        if (tx < pdx && ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + (smm_x - pdx)] = (y - pdy >= 0 && y - pdy < height && x - pdx >= 0 && x - pdx < width) ? input[(y - pdy) * width + x - pdx] : fill_value;
        }
        if (tx < pdx && ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + (smm_x - pdx)] = (y + pdy < height && x - pdx >= 0 && x - pdx < width) ? input[(y + pdy) * width + x - pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx && ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + (smm_x + pdx)] = (y - pdy >= 0 && y - pdy < height && x + pdx < width) ? input[(y - pdy) * width + x + pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx && ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + (smm_x + pdx)] = (y + pdy < height && x + pdx < width) ? input[(y + pdy) * width + x + pdx] : fill_value;
        }

        __syncthreads();

        if (x < width && y < height)
        {
            unsigned char min_val = fill_value;

            for (int k_row = 0; k_row < k_height; ++k_row)
            {
                for (int k_col = 0; k_col < k_width; ++k_col)
                {
                    if (kernel[k_row * k_width + k_col] != 0)
                    {
                        // Calculate the coordinates in shared memory to read from:
                        int sm_access_y = smm_y + k_row - pdy;
                        int sm_access_x = smm_x + k_col - pdx;
                        unsigned char val = smm[sm_access_y * smm_width + sm_access_x];

                        min_val = min(val, min_val);
                    }
                }
            }

            output[y * width + x] = min_val;
        }
    }

    __global__ void externErosionKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        // This has size of (TILE_WIDTH) * (TILE_WIDTH)
        extern __shared__ unsigned char smm[];

        int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
        int y = blockIdx.y * blockDim.y + threadIdx.y; // rows
        int tile_width = max(blockDim.x, blockDim.y);

        smm[threadIdx.y * (tile_width) + threadIdx.x] = (x < width && y < height) ? input[y * width + x] : 255;

        __syncthreads();

        if (x < width && y < height)
        {
            int this_tile_start_x = blockIdx.x * blockDim.x;
            int this_tile_start_y = blockIdx.y * blockDim.y;
            int next_tile_start_x = this_tile_start_x + blockDim.x;
            int next_tile_start_y = this_tile_start_y + blockDim.y;

            int N_start_x = x - k_width / 2;
            int N_start_y = y - k_height / 2;
            // Perform erosion
            unsigned char min_val = 255;

            for (int j = 0; j < k_height; ++j) // Kernel row index
            {
                for (int i = 0; i < k_width; ++i) // Kernel column index
                {
                    if (kernel[j * k_width + i] != 0)
                    {
                        int N_index_x = (N_start_x + i);
                        int N_index_y = (N_start_y + j);

                        unsigned char val;
                        if (N_index_x >= 0 && N_index_x < width && N_index_y >= 0 && N_index_y < height)
                        {
                            if ((N_index_x >= this_tile_start_x) && (N_index_x < next_tile_start_x) &&
                                (N_index_y >= this_tile_start_y) && (N_index_y < next_tile_start_y))
                            {
                                // Calculate the coordinates in shared memory to read from:
                                int sm_access_y = N_index_y - this_tile_start_y;
                                int sm_access_x = N_index_x - this_tile_start_x;

                                val = smm[sm_access_y * (tile_width) + sm_access_x];
                            }
                            else
                            {
                                // Out of bounds, read from global memory
                                val = input[N_index_y * width + N_index_x];
                            }

                            min_val = min(val, min_val);
                        }
                    }
                }
            }

            output[y * width + x] = min_val;
        }
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
                        max_val = max(input[actual_y * width + actual_x], max_val);
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
        // 2D array for tile+kernel cache
        extern __shared__ unsigned char smm[];

        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int x = blockIdx.x * blockDim.x + tx;
        int y = blockIdx.y * blockDim.y + ty;

        int pdx = (k_width - 1) / 2;
        int pdy = (k_height - 1) / 2;

        int smm_width = blockDim.x + k_width - 1;
        int smm_height = blockDim.y + k_height - 1;
        int smm_y = ty + pdy;
        int smm_x = tx + pdx;
        int fill_value = 0; // Value to fill the padding pixels

        // Central pixel

        smm[smm_y * smm_width + smm_x] = (y < height && x < width) ? input[y * width + x] : fill_value;

        __syncthreads();

        // Load padding pixels
        if (tx < pdx)
        {
            smm[smm_y * smm_width + (smm_x - pdx)] = (y < height && x - pdx >= 0 && x - pdx < width) ? input[y * width + x - pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx)
        {
            smm[smm_y * smm_width + (smm_x + pdx)] = (y < height && x + pdx < width) ? input[y * width + x + pdx] : fill_value;
        }
        if (ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + smm_x] = (y - pdy >= 0 && y - pdy < height && x < width) ? input[(y - pdy) * width + x] : fill_value;
        }
        if (ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + smm_x] = (y + pdy < height && x < width) ? input[(y + pdy) * width + x] : fill_value;
        }

        // Load corner pixels
        if (tx < pdx && ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + (smm_x - pdx)] = (y - pdy >= 0 && y - pdy < height && x - pdx >= 0 && x - pdx < width) ? input[(y - pdy) * width + x - pdx] : fill_value;
        }
        if (tx < pdx && ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + (smm_x - pdx)] = (y + pdy < height && x - pdx >= 0 && x - pdx < width) ? input[(y + pdy) * width + x - pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx && ty < pdy)
        {
            smm[(smm_y - pdy) * smm_width + (smm_x + pdx)] = (y - pdy >= 0 && y - pdy < height && x + pdx < width) ? input[(y - pdy) * width + x + pdx] : fill_value;
        }
        if (tx >= blockDim.x - pdx && ty >= blockDim.y - pdy)
        {
            smm[(smm_y + pdy) * smm_width + (smm_x + pdx)] = (y + pdy < height && x + pdx < width) ? input[(y + pdy) * width + x + pdx] : fill_value;
        }

        __syncthreads();

        if (x < width && y < height)
        {
            unsigned char max_value = fill_value;

            for (int k_row = 0; k_row < k_height; ++k_row)
            {
                for (int k_col = 0; k_col < k_width; ++k_col)
                {
                    if (kernel[k_row * k_width + k_col] != 0)
                    {
                        // Calculate the coordinates in shared memory to read from:
                        int sm_access_y = smm_y + k_row - pdy;
                        int sm_access_x = smm_x + k_col - pdx;
                        unsigned char val = smm[sm_access_y * smm_width + sm_access_x];

                        max_value = max(val, max_value);
                    }
                }
            }

            output[y * width + x] = max_value;
        }
    }

    __global__ void externDilateKernel(const unsigned char *input, unsigned char *output, const unsigned char *__restrict__ kernel, const int width, const int height, const int k_width, const int k_height)
    {
        // This has size of (TILE_WIDTH) * (TILE_WIDTH)
        extern __shared__ unsigned char smm[];

        int x = blockIdx.x * blockDim.x + threadIdx.x; // cols
        int y = blockIdx.y * blockDim.y + threadIdx.y; // rows
        int tile_width = max(blockDim.x, blockDim.y);

        smm[threadIdx.y * (tile_width) + threadIdx.x] = (x < width && y < height) ? input[y * width + x] : 0;

        __syncthreads();

        if (x < width && y < height)
        {
            int this_tile_start_x = blockIdx.x * blockDim.x;
            int this_tile_start_y = blockIdx.y * blockDim.y;
            int next_tile_start_x = this_tile_start_x + blockDim.x;
            int next_tile_start_y = this_tile_start_y + blockDim.y;

            int N_start_x = x - k_width / 2;
            int N_start_y = y - k_height / 2;
            // Perform erosion
            unsigned char max_val = 0;

            for (int j = 0; j < k_height; ++j) // Kernel row index
            {
                for (int i = 0; i < k_width; ++i) // Kernel column index
                {
                    if (kernel[j * k_width + i] != 0)
                    {
                        int N_index_x = (N_start_x + i);
                        int N_index_y = (N_start_y + j);

                        unsigned char val;
                        if (N_index_x >= 0 && N_index_x < width && N_index_y >= 0 && N_index_y < height)
                        {
                            if ((N_index_x >= this_tile_start_x) && (N_index_x < next_tile_start_x) &&
                                (N_index_y >= this_tile_start_y) && (N_index_y < next_tile_start_y))
                            {
                                // Calculate the coordinates in shared memory to read from:
                                int sm_access_y = N_index_y - this_tile_start_y;
                                int sm_access_x = N_index_x - this_tile_start_x;

                                val = smm[sm_access_y * (tile_width) + sm_access_x];
                            }
                            else
                            {
                                // Out of bounds, read from global memory
                                val = input[N_index_y * width + N_index_x];
                            }

                            max_val = max(max_val, val);
                        }
                    }
                }
            }

            output[y * width + x] = max_val;
        }
    }

    cv::Mat erodeImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel,
                       const int tile_width)
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

        dim3 threadsPerBlock(tile_width, tile_width);
        dim3 numBlocks((img.cols + tile_width - 1) / tile_width, (img.rows + tile_width - 1) / tile_width);
        switch (i)
        {
        case Base:
            dumbErosionKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Halo:
            cachedTiledHaloErosionKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * (tile_width + kernel.rows - 1) * (tile_width + kernel.cols - 1)>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Extern:
            externErosionKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * tile_width * tile_width>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
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

    cv::Mat dilateImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel,
                        const int tile_width)
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

        dim3 threadsPerBlock(tile_width, tile_width);
        dim3 numBlocks((img.cols + tile_width - 1) / tile_width, (img.rows + tile_width - 1) / tile_width);
        switch (i)
        {
        case Base:
            dumbDilateKernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Halo:
            cachedTiledHaloDilateKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * (tile_width + kernel.rows - 1) * (tile_width + kernel.cols - 1)>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
            break;
        case Extern:
            externDilateKernel<<<numBlocks, threadsPerBlock, sizeof(uchar) * tile_width * tile_width>>>(d_input, d_output, d_kernel, img.cols, img.rows, kernel.cols, kernel.rows);
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

    cv::Mat openingImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel,
                         const int tile_width)
    {
        auto result = erodeImage(i, img, kernel, tile_width);
        result = dilateImage(i, result, kernel, tile_width);

        return result;
    }

    cv::Mat closingImage(const Implementation i, const cv::Mat &img, const cv::Mat &kernel,
                         const int tile_width)
    {
        auto result = dilateImage(i, img, kernel, tile_width);
        result = erodeImage(i, result, kernel, tile_width);

        return result;
    }
}