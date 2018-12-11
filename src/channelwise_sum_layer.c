#include "channelwise_sum_layer.h"
#include "cuda.h"
#include <stdio.h>

channelwise_sum_layer make_channelwise_sum_layer(int batch, int w, int h, int c)
{
    fprintf(stderr, "channelwise_sum         %4d x%4d x%4d   ->  %4d x%4d x%4d\n",  w, h, c, w, h, 1);
    channelwise_sum_layer l = {0};
    l.type = CHANNELWISE_SUM;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.out_w = w;
    l.out_h = h;
    l.out_c = 1;
    l.outputs = l.out_h * l.out_w;
    l.inputs = h*w*c;
    int output_size = l.outputs * batch;
    l.output =  calloc(output_size, sizeof(float));
    l.delta =   calloc(output_size, sizeof(float));
    l.forward = forward_channelwise_sum_layer;
    l.backward = backward_channelwise_sum_layer;
    
    #ifdef GPU
    l.forward_gpu = forward_channelwise_sum_layer_gpu;
    l.backward_gpu = backward_channelwise_sum_layer_gpu;
    l.output_gpu  = cuda_make_array(l.output, output_size);
    l.delta_gpu   = cuda_make_array(l.delta, output_size);
    #endif
    
    return l;
}

void resize_channelwise_sum_layer(channelwise_sum_layer *l, int w, int h)
{
    l->w = w;
    l->h = h;
    l->inputs = h*w*l->c;
    
    l->out_w = w;
    l->out_h = h;
    l->outputs = l->out_w * l->out_h;
    int output_size = l->outputs * l->batch;

    l->output = realloc(l->output, output_size * sizeof(float));
    l->delta = realloc(l->delta, output_size * sizeof(float));
    
    #ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu  = cuda_make_array(l->output, output_size);
    l->delta_gpu   = cuda_make_array(l->delta,  output_size);
    #endif
    
}


void forward_channelwise_sum_layer(const channelwise_sum_layer l, network net)
{
    int b,i,j,k;
    int out_index, in_index;
    
    for(b = 0; b < l.batch; b++){
        k = b*l.out_h*l.out_w;
        for(i = 0; i < l.h*l.w; i++){
            out_index = k + i;
            l.output[out_index] = 0;
            for(j = 0; j < l.c; j++){
                in_index = l.h*l.w*(b*l.c + j) + i;
                l.output[out_index] += net.input[in_index];
            }
            l.output[out_index] /= l.c;
        }
    }
}

void backward_channelwise_sum_layer(const channelwise_sum_layer l, network net)
{

    int b,i,j,k;
    int out_index, in_index;
    
    for(b = 0; b < l.batch; b++){
        k = b*l.out_h*l.out_w;
        for(i = 0; i < l.h*l.w; i++){
            out_index = k + i;
            for(j = 0; j < l.c; j++){
                in_index = l.h*l.w*(b*l.c + j) + i;
                net.delta[in_index] += l.delta[out_index] / l.c;
            }
        }
    }

}


