#include "norm_channel_l2_layer.h"
#include "cuda.h"
#include <stdio.h>

norm_channel_l2_layer make_norm_channel_l2_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "norm_channel_l2         %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  w, h, c, w, h, c);
    norm_channel_l2_layer l = {0};
    l.type = NORM_CHANNEL_L2;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = c;
    l.outputs = l.out_h*l.out_w*l.out_c;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_norm_channel_l2_layer;
    l.backward = backward_norm_channel_l2_layer;
    #ifdef GPU
    l.forward_gpu = forward_norm_channel_l2_layer_gpu;
    l.backward_gpu = backward_norm_channel_l2_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    return l;
}

void forward_norm_channel_l2_layer(const norm_channel_l2_layer l, network net)
{
    int b,i,k;
    int out_index, in_index;
    double L2, sum_c;
    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            sum_c = 0;
            for(i = 0; i < l.h*l.w; ++i){
                in_index = i + l.h*l.w*(k + b*l.c);
                sum_c += net.input[in_index] * net.input[in_index];
            }
            L2 = sqrt(sum_c);
            for(i = 0; i < l.h*l.w; ++i){
                in_index = i + l.h*l.w*(k + b*l.c);
                out_index = in_index;
                l.output[out_index] = net.input[in_index]/L2;
            }
        }
    }
}

void backward_norm_channel_l2_layer(const norm_channel_l2_layer l, network net)
{
    int b,i,k;
    int out_index, in_index;
    double L2, sum_c;    

    for(b = 0; b < l.batch; ++b){
        for(k = 0; k < l.c; ++k){
            sum_c = 0;
            for(i = 0; i < l.h*l.w; ++i){
                out_index = i + l.h*l.w*(k + b*l.c);
                sum_c += l.delta[out_index] * l.delta[out_index];
            }
            L2 = sqrt(sum_c);
                
            for(i = 0; i < l.h*l.w; ++i){
                in_index = i + l.h*l.w*(k + b*l.c);
                out_index = in_index;
                net.delta[in_index] += l.delta[out_index]/L2;
            }
        }
    }
}


