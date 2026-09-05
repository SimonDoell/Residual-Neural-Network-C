#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include "matrix.h"

#ifndef LAYER_H
#define LAYER_H

typedef struct {
    Matrix parameter;
    Matrix gradient;
} Parameter;


typedef struct {
    void* context;

    Matrix activation; // activation in forward propagation
    Matrix gradient;   // stored gradient in backpropagation

    uint32_t* in;  // Used for backward propagating the gradients
    uint32_t* out; // Used for forward propagation
    uint32_t in_count;
    uint32_t out_count;
    
    void(*free)    (void* context);
    void(*forward) (void* context, const Matrix* activation, Matrix* activation_output);
    void(*backward)(void* context, const Matrix* gradient, const Matrix* activation /*own activation from the forward pass*/, Matrix* gradient_output);
} Layer;

Layer layer_new(
    void* context,
    void(*free)(void* context),
    void(*forward)(void* context, const Matrix* activation, Matrix* activation_output),
    void(*backward)(void* context, const Matrix* gradient, const Matrix* activation, Matrix* gradient_output))
{
    return (Layer){
        .context = context,
        .activation = mat_matrix(1, 1),
        .gradient   = mat_matrix(1, 1),
        .in_count   = 0,
        .out_count  = 0,
        .in         = NULL,
        .out        = NULL,
        .free       = free,
        .forward    = forward,
        .backward   = backward,
    };
}

void layers_connect(Layer* from_layer, uint32_t from, Layer* to_layer, uint32_t to) {
    assert(from_layer);
    assert(to_layer);

    from_layer->out = realloc(from_layer->out, ++from_layer->out_count);
    from_layer->out[from_layer->out_count - 1] = to;

    to_layer->in = realloc(to_layer->in, ++to_layer->in_count);
    to_layer->in[to_layer->in_count - 1] = from;
}


#endif