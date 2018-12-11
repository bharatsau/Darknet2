#include "cuda_runtime.h"
#include "curand.h"
#include "cublas_v2.h"

extern "C" {
#include "channelwise_sum_layer.h"
#include "cuda.h"
}


__global__ void forward_channelwise_sum_layer_kernel(int n, int w, int h, int c, float *input, float *output)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % (h*w);
    id /= (h*w);
    int b = id;

    int i, in_index, out_index;
    
    out_index = (b*h*w + k);
    output[out_index] = 0;
    for(i = 0; i < c; ++i){
        in_index = h*w*(b*c + i) + k;
        output[out_index] += input[in_index];
    }
    output[out_index] /= c;
}


__global__ void backward_channelwise_sum_layer_kernel(int n, int w, int h, int c, float *in_delta, float *out_delta)
{
    int id = (blockIdx.x + blockIdx.y*gridDim.x) * blockDim.x + threadIdx.x;
    if(id >= n) return;

    int k = id % (h*w);
    id /= (h*w);
    int b = id;

    int i, in_index, out_index;
    out_index = (b*h*w + k);
    for(i = 0; i < c; ++i){
        in_index = h*w*(b*c + i) + k;
        in_delta[in_index] += out_delta[out_index] / c;
    }
}


extern "C" void forward_channelwise_sum_layer_gpu(channelwise_sum_layer layer, network net)
{
    size_t n = layer.out_h*layer.out_w*layer.batch;

    forward_channelwise_sum_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.input_gpu, layer.output_gpu);
    check_error(cudaPeekAtLastError());
}

extern "C" void backward_channelwise_sum_layer_gpu(channelwise_sum_layer layer, network net)
{
    size_t n = layer.h*layer.w*layer.batch;

    backward_channelwise_sum_layer_kernel<<<cuda_gridsize(n), BLOCK>>>(n, layer.w, layer.h, layer.c, net.delta_gpu, layer.delta_gpu);
    check_error(cudaPeekAtLastError());
}

