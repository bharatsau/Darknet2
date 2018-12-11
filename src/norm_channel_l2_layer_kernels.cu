#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "norm_channel_l2_layer.h"
#include "cuda.h"
}

__global__ void forward_norm_channel_l2_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index, in_index;
    double L2, sum_c;

    sum_c = 0;
    for(i = 0; i < h*w; ++i){
        in_index = i + h*w*(k + b*c);
        sum_c += input[in_index] * input[in_index];
    }
    L2 = sqrt(sum_c);
    for(i = 0; i < h*w; ++i){
        in_index = i + h*w*(k + b*c);
        out_index = in_index;
        output[out_index] = input[in_index]/L2;
    }

}

__global__ void backward_norm_channel_l2_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % c;
    id /= c;
    int b = id;

    int i;
    int out_index, in_index;
    double L2, sum_c;

    sum_c = 0;
    for(i = 0; i < h*w; ++i){
        out_index = i + h*w*(k + b*c);
        sum_c += out_delta[out_index] * out_delta[out_index];
    }
    L2 = sqrt(sum_c);
    for(i = 0; i < h*w; ++i){
        out_index = i + h*w*(k + b*c);
        in_index = out_index;
        in_delta[in_index] += out_delta[out_index]/L2;
    }
}

extern "C" void forward_norm_channel_l2_layer_gpu(norm_channel_l2_layer layer, network net)
{
    size_t n = layer.c * layer.batch;

    forward_norm_channel_l2_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_norm_channel_l2_layer_gpu(norm_channel_l2_layer layer, network net)
{
    size_t n = layer.c * layer.batch;

    backward_norm_channel_l2_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}

