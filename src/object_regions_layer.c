#include "object_regions_layer.h"
#include "blas.h"
#include "cuda.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

layer make_object_regions_layer(int batch, int inputs)
{
    fprintf(stderr, "object_regions                                         %4d\n",  inputs);
    layer l = {0};
    l.type = OBJECT_REGIONS;
    l.batch = batch;
    l.inputs = inputs;
    l.outputs = inputs;
    l.max_boxes = 90;
    l.truths = l.max_boxes*(4 + 1);
    
    l.output = calloc(inputs*batch, sizeof(float));
    l.delta = calloc(inputs*batch, sizeof(float));

    l.forward = forward_object_regions_layer;
    l.backward = backward_object_regions_layer;
    #ifdef GPU
    l.forward_gpu = forward_object_regions_layer;  // cpu implementation only
    l.backward_gpu = backward_object_regions_layer_gpu;

    l.output_gpu = cuda_make_array(l.output, inputs*batch); 
    l.delta_gpu = cuda_make_array(l.delta, inputs*batch); 
    #endif
    return l;
}


void forward_object_regions_layer(const layer l, network net)
{
    int i,j,b,t;
    int left_x, right_x, top_y, bottom_y, out_index;
    float left_xf, right_xf, top_yf, bottom_yf, extra_h, extra_w;

    for (b = 0; b < l.batch; ++b) {
        float *map_2d = calloc(l.h * l.w, sizeof(float));
        for(t = 0; t < l.max_boxes; ++t){
            box bbox = float_to_box(net.truth + t*(4 + 1) + b*l.truths, 1);
            if(!bbox.x) break;
            // Bounding box scaling
            left_xf = (bbox.x-bbox.w/2)*l.w;
            right_xf = (bbox.x+bbox.w/2)*l.w;
            top_yf = (bbox.y-bbox.h/2)*l.h;
            bottom_yf = (bbox.y+bbox.h/2)*l.h;
            // Margin
            extra_h = (bottom_yf - top_yf + 1)*l.total_margin*0.5;
            extra_w = (right_xf - left_xf + 1)*l.total_margin*0.5;
            // Final bbox
            left_x = (int)round(left_xf - extra_h);
            right_x = (int)round(right_xf + extra_h);
            top_y = (int)round(top_yf - extra_w);
            bottom_y = (int)round(bottom_yf + extra_w);
            // Boundary conditions
            if(left_x < 1) left_x = 1;
            if(right_x > l.w) right_x = l.w;
            if(top_y < 1) top_y = 1;
            if(bottom_y > l.h) bottom_y = l.h;
            // Assign 1 to bbox windows
            for(i=top_y-1; i<bottom_y; i++)
                for(j=left_x-1; j<right_x; j++)
                    *(map_2d + i*l.w + j) = 1;
        }
        
        // Copy to all layers
        for(i=0; i<l.c; i++)
            for(j=0; j<l.h*l.w; j++)
            {
                out_index = b*l.outputs + i*l.h*l.w + j;
                l.output[out_index] = map_2d[j];
            }
    }
    
    #ifdef GPU
    cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);
    #endif    
            
}


void backward_object_regions_layer(const layer l, network net)
{
    // No backward for this
}

#ifdef GPU

void backward_object_regions_layer_gpu(const layer l, network net)
{
    // No backward for this
}

#endif

