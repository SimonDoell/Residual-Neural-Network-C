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

typedef void(*free_function)    (void* context);
typedef void(*forward_function) (void* context, const Matrix* activation, Matrix* activation_output);
typedef void(*backward_function)(void* context, const Matrix* gradient, const Matrix* activation /*own activation from the forward pass*/, Matrix* gradient_output, float lr);

typedef struct {
    void* context;

    Matrix input;           // input in forward propagation
    Matrix output;          // output in forward propagation
    Matrix gradient_input;  // input gradient in backpropagation
    Matrix gradient_output; // output / produced gradient in backpropagation

    uint32_t* in;  // Used for backward propagating the gradients
    uint32_t* out; // Used for forward propagation
    uint32_t in_count;
    uint32_t out_count;
    
    free_function     free;
    forward_function  forward;
    backward_function backward;
} Layer;

static inline Layer layer_new(void* context, free_function free, forward_function forward, backward_function backward) {
    return (Layer){
        .context   = context,
        .input     = mat_matrix(1, 1),
        .output    = mat_matrix(1, 1),
        .gradient_input  = mat_matrix(1, 1),
        .gradient_output  = mat_matrix(1, 1),
        .in_count  = 0,
        .out_count = 0,
        .in        = NULL,
        .out       = NULL,
        .free      = free,
        .forward   = forward,
        .backward  = backward,
    };
}

static inline void layer_assert(Layer* layer) {
    assert(layer);
    assert(layer->forward);
    assert(layer->backward);
}

static inline void layers_free(Layer* layer) {
    layer_assert(layer);

    if (layer->free)
        layer->free(layer->context);

    mat_free(&layer->input);
    mat_free(&layer->output);
    mat_free(&layer->gradient_input);
    mat_free(&layer->gradient_output);

    if (layer->in)  free(layer->in);
    if (layer->out) free(layer->out);

    layer->in_count  = 0;
    layer->out_count = 0;
    layer->context   = NULL;
    layer->in        = NULL;
    layer->out       = NULL;
    layer->free      = NULL;
    layer->forward   = NULL;
    layer->backward  = NULL;
}

static inline void layers_connect(Layer* from_layer, uint32_t from, Layer* to_layer, uint32_t to) {
    layer_assert(from_layer);
    layer_assert(to_layer);

    from_layer->out = realloc(from_layer->out, ++from_layer->out_count * sizeof(uint32_t));
    from_layer->out[from_layer->out_count - 1] = to;

    to_layer->in = realloc(to_layer->in, ++to_layer->in_count * sizeof(uint32_t));
    to_layer->in[to_layer->in_count - 1] = from;
}

static inline void layer_forward(Layer* layer) {
    layer_assert(layer);
    layer->forward(layer->context, &layer->input, &layer->output);
}

static inline void layer_backward(Layer* layer, float lr) {
    layer_assert(layer);
    layer->backward(layer->context, &layer->gradient_input, &layer->input, &layer->gradient_output, lr);
}

#endif