#include <stdint.h>
#include <stdlib.h>
#include "layer.h"

#ifndef NETWORK_H
#define NETWORK_H

typedef struct {
    float learning_rate;
    
    Layer* layers;
    uint32_t layer_count;

    uint32_t* graph_order;
} NeuralNetwork;


static inline NeuralNetwork nn_init() {
    return (NeuralNetwork){
        .learning_rate = 0.01f,
        .layers        = NULL,
        .layer_count   = 0,
        .graph_order   = NULL,
    };
}

static inline void nn_free(NeuralNetwork* network) {
    assert(network);

    for (int i = 0; i < network->layer_count; ++i) {
        layers_free(&network->layers[i]);
    }

    network->layer_count = 0;
    
    free(network->layers);
    network->layers = NULL;

    free(network->graph_order);
    network->graph_order = NULL;
}

static inline void nn_connect_layers(NeuralNetwork* network, uint32_t from, uint32_t to) {
    assert(network);
    assert(from < network->layer_count);
    assert(to   < network->layer_count);

    layers_connect(&network->layers[from], from, &network->layers[to], to);
}

static inline uint32_t nn_add_layer(NeuralNetwork* network, Layer layer, bool do_default_connect) {
    assert(network);
    layer_assert(&layer);

    network->layers = realloc(network->layers, ++network->layer_count * sizeof(Layer));

    // Free the topological ordering when present
    if (network->graph_order) {
        free(network->graph_order);
        network->graph_order = NULL;
    }

    uint32_t index = network->layer_count - 1;
    network->layers[index] = layer;

    if (do_default_connect && network->layer_count > 1) {
        nn_connect_layers(network, index - 1, index);
    }
    
    return index;
}

static inline void nn_order_layers(NeuralNetwork* network) {
    assert(network);
    if (network->graph_order) return; // Already computed

    int n = network->layer_count;
    
    network->graph_order = calloc(n, sizeof(uint32_t));
    uint32_t* graph_order = network->graph_order;
    uint32_t* in_degrees = calloc(n, sizeof(uint32_t));
    uint32_t* queu       = calloc(n, sizeof(uint32_t));

    uint32_t graph_index = 0;
    uint32_t queu_index  = 0;

    for (int i = 0; i < n; ++i) {
        in_degrees[i] = network->layers[i].in_count;
    }

    // Add empty nodes to the front of the graph
    for (int i = 0; i < n; ++i) {
        if (in_degrees[i] == 0)
            queu[queu_index++] = i;
    }

    // Kahn's Algorithm
    while (queu_index != 0) {
        uint32_t top = queu[--queu_index];
        graph_order[graph_index++] = top;

        for (int i = 0; i < network->layers[top].out_count; ++i) {
            uint32_t next = network->layers[top].out[i];
            in_degrees[next]--;

            if (in_degrees[next] == 0) {
                queu[queu_index++] = next;
            }
        }
    }

    free(queu);
    free(in_degrees);

    // -----
    assert(network->graph_order[ 0 ] ==  0 );
    assert(network->graph_order[n-1] == n-1);
}

static inline Matrix nn_forward(NeuralNetwork* network, const Matrix* activation) {
    assert(network);
    assert(network->layers);
    assert(network->graph_order);  // First call "nn_order_layers()" on the neural network!
    int n = network->layer_count;

    Layer* layers = network->layers;
    uint32_t* graph_order = network->graph_order;

    mat_copy_into(&layers[0].input, activation);
    layer_forward(&layers[0]);
    
    for (int i = 1; i < n; ++i) {
        Layer* layer = &layers[graph_order[i]];

        mat_copy_into(&layer->input, &layers[layer->in[0]].output);

        if (layer->in_count > 1) {
            for (int l = 1; l < layer->in_count; ++l) {
                mat_add_inplace(&layer->input, &layers[layer->in[l]].output);
            }
        }

        layer_forward(layer);
    }
    
    return mat_copy(&layers[n-1].output);
}

static inline void nn_train(NeuralNetwork* network, const Matrix* activation, const Matrix* desired_output) {
    assert(network);
    assert(network->layers);
    assert(network->graph_order);  // First call "nn_order_layers()" on the neural network!
    int n = network->layer_count;
    float lr = network->learning_rate;

    Layer* layers = network->layers;
    uint32_t* graph_order = network->graph_order;

    Matrix output_to_grad = nn_forward(network, activation);
    assert(mat_same_shape(&output_to_grad, desired_output));

    mat_sub_inplace(&output_to_grad, desired_output); // MSE derivative

    // --- Backpropagation ---
    mat_copy_into(&layers[n-1].gradient_input, &output_to_grad);
    layer_backward(&layers[n-1], lr);
    mat_free(&output_to_grad);

    for (int i = n-2; i >= 0; --i) {
        Layer* layer = &layers[graph_order[i]];

        mat_copy_into(&layer->gradient_input, &layers[layer->out[0]].gradient_output);

        if (layer->out_count > 1) {
            for (int l = 1; l < layer->out_count; ++l) {
                mat_add_inplace(&layer->gradient_input, &layers[layer->out[l]].gradient_output);
            }
        }

        layer_backward(layer, lr);
    }
}


#endif