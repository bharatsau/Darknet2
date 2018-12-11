#ifndef FG_BG_COST_LAYER_H
#define FG_BG_COST_LAYER_H
#include "layer.h"
#include "network.h"

layer make_fg_bg_cost_layer(int batch, int w, int h, int c, float scale);
void forward_fg_bg_cost_layer(const layer l, network net);
void backward_fg_bg_cost_layer(const layer l, network net);
void resize_fg_bg_cost_layer(layer *l, int w, int h);

#ifdef GPU
void backward_fg_bg_cost_layer_gpu(const layer l, network net);
#endif

#endif
