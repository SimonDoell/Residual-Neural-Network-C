#include <time.h>
#include <stdio.h>
#include "layer.h"
#include "linear_adam.h"
#include "activation_layer.h"
#include "network.h"

#define countof(arr) (sizeof(arr) / sizeof(arr[0]))

int main() {srand(time(0));
    NeuralNetwork network = nn_init();

    uint32_t layers[] = {
        /* 0*/ nn_add_layer(&network, linear_adam_new_default(1 , 16), true),
        /* 1*/ nn_add_layer(&network, ReLU_new(), true),
        /* 2*/ nn_add_layer(&network, linear_adam_new_default(16, 16), true),
        /* 3*/ nn_add_layer(&network, ReLU_new(), true),
        /* 4*/ nn_add_layer(&network, linear_adam_new_default(16, 16), true),
        /* 5*/ nn_add_layer(&network, ReLU_new(), true),
        /* 6*/ nn_add_layer(&network, linear_adam_new_default(16, 16), true),
        /* 7*/ nn_add_layer(&network, ReLU_new(), true),
        /* 8*/ nn_add_layer(&network, linear_adam_new_default(16, 16), true),
        /* 9*/ nn_add_layer(&network, ReLU_new(), true),
        /*10*/ nn_add_layer(&network, linear_adam_new_default(16, 1 ), true),
        /*11*/ nn_add_layer(&network, Tanh_new(), true),
    };

    nn_connect_layers(&network, layers[0], layers[2]);
    nn_connect_layers(&network, layers[2], layers[4]);
    nn_connect_layers(&network, layers[4], layers[6]);
    nn_connect_layers(&network, layers[6], layers[8]);
    nn_connect_layers(&network, layers[8], layers[10]);

    nn_order_layers(&network); // Finalizing the NN structure

    printf("Ordering: ");
    for (int i = 0; i < countof(layers); ++i)
        printf("%i ", network.graph_order[i]);
    printf("\n");

    Matrix input = mat_vector(1);
    input.values[0] = 1.0f;

    Matrix output = mat_vector(1);
    output.values[0] = 0.5f;

    for (int i = 0; i < 10000; ++i)
        nn_train(&network, &input, &output);

    mat_free(&input);
    mat_free(&output);

    nn_free(&network);
    return 0;
}