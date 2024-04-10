/*****************************************************************************
 * File:        main.cu
 *
 * Run:         ./conv2D
 *****************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common/common.h"
#include "common/common_string.h"
#include "kernel.cu"
#include "support.h"


bool compute(int width, int height, int channels, int kernel_width, const char* kernel_type)
{
    unsigned int bytes = width * height * channels * sizeof(float);
    float* h_in, *h_out, *h_kernel;
    float* d_in, *d_out, *d_kernel;

    printf("bytes: %d\n", bytes);
    // allocate host memory
    h_in = (float*)malloc(bytes);
    h_out = (float*)malloc(bytes);
    h_kernel = (float*)malloc(kernel_width*kernel_width*sizeof(float));

    // init inputs
    for (int c = 0; c < channels; c++)
        for (int i = 0; i < height; i++)
            for (int j = 0; j < width; j++)
                h_in[c*height*width + i*width + j] = 1 ;// rand() / (float)RAND_MAX;

    for (int i = 0; i < kernel_width; i++)
        for (int j = 0; j < kernel_width; j++)
            h_kernel[i*kernel_width + j] = 1 ;//rand() / (float)RAND_MAX;

    // allocate device memory
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_kernel, kernel_width*kernel_width*sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    //remove this later
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel, kernel_width*kernel_width*sizeof(float), cudaMemcpyHostToDevice));
    //CUDA_CHECK(cudaMemcpyToSymbol(M, h_kernel, kernel_width*kernel_width*sizeof(float)));

    // launch Kernel
    printf("\nLaunch Kernel...\n");

    Timer timer;
    if (kernel_type == "vanilla") {
        // basic 2d conv
         
        //todo: define block and grid size
        int gx = (width + 32 - 1)/32;
        int gy = (height + 8 - 1)/8;

        dim3 dimGrid(gx , gy, channels);
        //dim3 dimBlock(width, height);
        dim3 dimBlock(32, 8, 1);

        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);
        
        //todo: warmup
        cudaError_t cudaStatus = cudaGetLastError();
        printf("launced vanilla\n");
        //convolution2D<<<1,1>>>(h_in, h_out, width, height, channels, h_kernel, kernel_width);
        convolution2D<<< dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, d_kernel, kernel_width);

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "in vanilla: %s\n", cudaGetErrorString(cudaStatus));
        }  
        //cudaDeviceSynchronize(); 

        //CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        startTime(&timer);
        //todo: launch kernel on device and test performance, get results
        convolution2D<<< dimGrid, dimBlock >>>(d_in, d_out, width, height, channels, d_kernel, kernel_width);
        //cudaDeviceSynchronize(); 

        stopTime(&timer);

        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    }
    else if (kernel_type == "shared_mem") {
        //  2d conv on shared mem
        //todo: define block and grid size

        int gx = (width + 32 - 1)/32;
        int gy = (height + 8 - 1)/8;

        dim3 dimGrid(gx , gy, channels);
        dim3 dimBlock(32, 8);
        //dim3 dimBlock(32, 4);

        printf("block dim: %d x %d x %d\n", dimBlock.x, dimBlock.y, dimBlock.z);
        printf("grid dim: %d x %d x %d\n", dimGrid.x, dimGrid.y, dimGrid.z);

        cudaEvent_t start, stop;
        //todo: warmup

        printf("launced shared\n");

        // complete later
        convolution2D_sharedmem<<< dimGrid, dimBlock>>>(d_in, d_out, width, height, channels, d_kernel, kernel_width);
        cudaError_t cudaStatus = cudaGetLastError();

        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "in shared memory: %s\n", cudaGetErrorString(cudaStatus));
        }  
        //cudaDeviceSynchronize(); 

        //CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        //todo: launch kernel on device and test performance, get results
        cudaEventRecord(start,0);

    
        CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
    }

    // todo: bonus optimization 
    /*
    else if (kernel_type == "optimized") {
    }
    */


    // result in CPU
    float* cpu_out = (float*)malloc(bytes);
    printf("\nCalculating in CPU...\n");
    verify(h_in, cpu_out, h_kernel, width, height, channels, kernel_width);

    int precision = 8;
    double threshold = 1e-8 * channels*width*height;
    double diff = 0.0;
    //todo: compare kernel result with CPU result
    double s=0.0;
    double throughput=0.0;

    for(int i=0;i<channels*height*width;i++){
        diff+= (cpu_out[i]-h_out[i]);
        s+=cpu_out[i];
        if (cpu_out[i]!=h_out[i]){
            printf("\nerror pos : %d \n",i);
            break;
        }
    }
    
    //for(int i=0;i<height*width;i++){
    //    printf("cpu %f gpu %f \n", cpu_out[i], h_out[i]);
    //}
    printf("diff sum: %f \n", diff);
  

    //todo: getting result
    /* uncomment this later*/
    //long long int th = (height*width*height*width*kernel_width*kernel_width);///(double)(elapsedTime(timer));
    //printf("\n th %lld %d %d %d\n", th, height, width, kernel_width);
    printf("[Kernel %s] Throughput = %f GB/s, Time = %.10f ms\n",
    kernel_type, throughput , elapsedTime(timer) );
    printf("Error : %.*f (threshold: %f)\n", precision, (double)diff, threshold); 

    // todo: 
    // free memory (both device and host mem)
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_kernel);

    free(h_in);
    free(h_out);
    free(h_kernel);

    return (diff < threshold);
}

/*
 * Argument:
 *      "--width=<N>"       : Specify the number of width of input image (default: 1024)
 *      "--height=<N>"      : Specify the number of height of input image (default: 2048)
 *      "--channel=<N>"     : Specify the number of channels of input image (default: 1, <= 3)
 *      "--filter=<N>"      : Specify the number of filter width for convolution (default: 5)
*/

int main(int argc, char** argv)
{
    printf("[2D Convolution...]\n\n");

    //int height = 4;//24;
    //int width = 4;//24;//4096;

    int height = 1024;//24;
    int width = 8192;//24;//4096;

    int channels = 3;
    int kernel_width = 5;

    if (checkCmdLineFlag(argc, (const char **)argv, "width")) {
        width = getCmdLineArgumentInt(argc, (const char **)argv, "width");
        printf("inside main: %d\n", width);
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "height")) {
        height = getCmdLineArgumentInt(argc, (const char **)argv, "height");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "channels")) {
        channels = getCmdLineArgumentInt(argc, (const char **)argv, "channels");
    }
    if (checkCmdLineFlag(argc, (const char **)argv, "filter")) {
        kernel_width = getCmdLineArgumentInt(argc, (const char **)argv, "filter");
    }

    printf("Kernel Size: %d x %d\n", kernel_width, kernel_width);
    printf("Input Size: %d x %d x %d\n", width, height, channels);

    int dev = 0;
    cudaSetDevice(dev);
    
    bool result;

    result = compute(width, height, channels, kernel_width, "vanilla");
    printf(result ? "Test PASSED\n" : "Test FAILED!\n");


    result = compute(width, height, channels, kernel_width, "shared_mem");
    printf(result ? "Test PASSED\n" : "Test FAILED!\n");


    //todo: bonus
    //result = compute(width, height, channels, kernel_width, "optimized");
    //printf(result ? "Test PASSED\n" : "Test FAILED!\n");
    cudaDeviceReset();

    return 0;
}


