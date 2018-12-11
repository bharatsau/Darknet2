#ifndef CHANNELWISE_SUM_LAYER_H
#define CHANNELWISE_SUM_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer channelwise_sum_layer;

channelwise_sum_layer make_channelwise_sum_layer(int batch, int w, int h, int c);
void resize_channelwise_sum_layer(channelwise_sum_layer *l, int w, int h);
void forward_channelwise_sum_layer(const channelwise_sum_layer l, network net);
void backward_channelwise_sum_layer(const channelwise_sum_layer l, network net);

#ifdef GPU
void forward_channelwise_sum_layer_gpu(channelwise_sum_layer l, network net);
void backward_channelwise_sum_layer_gpu(channelwise_sum_layer l, network net);
#endif

#endif

