#include "fg_bg_cost_layer.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>


layer make_fg_bg_cost_layer(int batch, int w, int h, int c, float scale)
{
    fprintf(stderr, "fg_bg_cost              %4d x%4d x%4d   ->  %4d\n",  w, h, c, c);
    layer l = {0};
    l.type = FG_BG_COST;
    
    l.h = h;
    l.w = w;
    l.c = c;
    l.scale = scale;
    l.batch = batch;
    l.inputs = l.h*l.w*l.c;
    l.outputs = l.c;
    l.max_boxes = 90;
    l.truths = l.max_boxes*(4 + 1);    

    l.forward = forward_fg_bg_cost_layer;
    l.backward = backward_fg_bg_cost_layer;
    l.delta = calloc(l.inputs*batch, sizeof(float));
    l.output = calloc(l.outputs*batch, sizeof(float));
    l.cost = calloc(1, sizeof(float));
    
    #ifdef GPU
    l.forward_gpu = forward_fg_bg_cost_layer;  // CPU implementation only
    l.backward_gpu = backward_fg_bg_cost_layer_gpu;
    l.delta_gpu = cuda_make_array(l.delta, l.inputs*batch);
    l.output_gpu = cuda_make_array(l.output, l.outputs*batch);
    #endif
    return l;
}

void resize_fg_bg_cost_layer(layer *l, int w, int h)
{
    l->h = h;
    l->w = w;
    l->inputs = l->h*l->w*l->c;

    l->delta = realloc(l->inputs*l->batch, sizeof(float));
    l->output = realloc(l->outputs*l->batch, sizeof(float));
    
    #ifdef GPU
    cuda_free(l->delta_gpu);
    cuda_free(l->output_gpu);
    l->delta_gpu = cuda_make_array(l->delta, l->inputs*l->batch);
    l->output_gpu = cuda_make_array(l->output, l->outputs*l->batch);
    #endif
}



void forward_fg_bg_cost_layer(const layer l, network net)
{
    #ifdef GPU
    cuda_pull_array(net.input_gpu, net.input, l.batch*l.inputs);
    #endif            

    int i,j,b,t;
    int left_x, right_x, top_y, bottom_y, in_index, out_index, bg_count, fg_count;
    float left_xf, right_xf, top_yf, bottom_yf, extra_h, extra_w;
    float delta_bg, delta_fg, avg_sqr_bg, avg_sqr_fg;

    for (b = 0; b < l.batch; ++b) {

        // Union ground truth boxes
        int *map_2d = calloc(l.h * l.w, sizeof(int));
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
        
        // Get no of bg and fg pixels
        bg_count=0;
        fg_count=0;
        for(i=0; i<l.h*l.w; i++)
        {
            if(map_2d[i]==1) fg_count++;
            else bg_count++;
        }
        if(fg_count==0) error("fg_count can't be 0.");
        
        // Calculate avg_sqr_bg and avg_sqr_fg and delta for each channel
        for(i=0; i<l.c; i++)
        {            
            avg_sqr_bg=0;
            avg_sqr_fg=0;
            for(j=0; j<l.h*l.w; j++)
            {
                in_index = (b*l.c + i)*l.h*l.w + j;
                if(map_2d[j]==1) avg_sqr_fg += net.input[in_index]*net.input[in_index];
                else avg_sqr_bg += net.input[in_index]*net.input[in_index];
            }
            avg_sqr_bg = avg_sqr_bg/bg_count;
            avg_sqr_fg = (avg_sqr_fg + 1e-9)/fg_count;

            // Delta
            delta_bg = (-1)*avg_sqr_bg/avg_sqr_fg; // Negative to reduce the values
            delta_fg = avg_sqr_bg/avg_sqr_fg;      // Positive to increase the values
            for(j=0; j<l.h*l.w; j++)
            {
                out_index = (b*l.c + i)*l.h*l.w + j;
                if(map_2d[j]==1) l.delta[out_index] = delta_fg;
                else l.delta[out_index] = delta_bg;
            }
            
            // Error i.e. l.output
            l.output[b*l.c+i] = delta_fg;
        }
        free(map_2d);
    }

    l.cost[0] = sum_array(l.output, l.batch*l.outputs);
    
    #ifdef GPU
    cuda_push_array(l.delta_gpu, l.delta, l.inputs*l.batch);
    cuda_push_array(l.output_gpu, l.output, l.outputs*l.batch);    
    #endif
            
}

void backward_fg_bg_cost_layer(const layer l, network net)
{
    axpy_cpu(l.batch*l.inputs, l.scale, l.delta, 1, net.delta, 1);
}

#ifdef GPU
void backward_fg_bg_cost_layer_gpu(const layer l, network net)
{
    axpy_gpu(l.batch*l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1);
}
#endif


