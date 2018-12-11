#ifndef ELTWISE_LAYER_H
#define ELTWISE_LAYER_H

#include "layer.h"
#include "network.h"

layer make_eltwise_layer(int batch, int index, int w, int h, int c, int w2, int h2, int c2);
void forward_eltwise_layer(const layer l, network net);
void backward_eltwise_layer(const layer l, network net);
void resize_eltwise_layer(layer *l, int w, int h);

#ifdef GPU
void forward_eltwise_layer_gpu(const layer l, network net);
void backward_eltwise_layer_gpu(const layer l, network net);
#endif

#endif

