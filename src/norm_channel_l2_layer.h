#ifndef NORM_CHANNEL_L2_LAYER_H
#define NORM_CHANNEL_L2_LAYER_H

#include "image.h"
#include "cuda.h"
#include "layer.h"
#include "network.h"

typedef layer norm_channel_l2_layer;

norm_channel_l2_layer make_norm_channel_l2_layer(int batch, int w, int h, int c);
void forward_norm_channel_l2_layer(const norm_channel_l2_layer l, network net);
void backward_norm_channel_l2_layer(const norm_channel_l2_layer l, network net);

#ifdef GPU
void forward_norm_channel_l2_layer_gpu(norm_channel_l2_layer l, network net);
void backward_norm_channel_l2_layer_gpu(norm_channel_l2_layer l, network net);
#endif

#endif

