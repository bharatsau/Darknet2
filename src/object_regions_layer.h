#ifndef OBJECT_REGIONS_LAYER_H
#define OBJECT_REGIONS_LAYER_H
#include "layer.h"
#include "network.h"

layer make_object_regions_layer(int batch, int inputs);
void forward_object_regions_layer(const layer l, network net);
void backward_object_regions_layer(const layer l, network net);

#ifdef GPU
void backward_object_regions_layer_gpu(const layer l, network net);
#endif

#endif

