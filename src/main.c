#include <time.h>
#include <stdio.h>
#include "layer.h"
#include "linear_adam.h"
#include "activation_layer.h"
#include "network.h"

#define countof(arr) (sizeof(arr) / sizeof(arr[0]))

int main() {srand(time(0));
    NeuralNetwork network = nn_init();
    uint32_t l1, l2, l3, l4, l5, l6;
    
    l1 = nn_add_layer(&network, linear_adam_new_default(1 , 16), true);
         nn_add_layer(&network, ReLU_new(), true);
    l2 = nn_add_layer(&network, linear_adam_new_default(16, 16), true);
         nn_add_layer(&network, ReLU_new(), true);
    l3 = nn_add_layer(&network, linear_adam_new_default(16, 16), true);
         nn_add_layer(&network, ReLU_new(), true);
    l4 = nn_add_layer(&network, linear_adam_new_default(16, 16), true);
         nn_add_layer(&network, ReLU_new(), true);
    l5 = nn_add_layer(&network, linear_adam_new_default(16, 16), true);
         nn_add_layer(&network, ReLU_new(), true);
    l6 = nn_add_layer(&network, linear_adam_new_default(16, 1 ), true);
         // nn_add_layer(&network, Tanh_new(), true); // <-- Breaks for some reason!

    nn_connect_layers(&network, l1, l2);
    nn_connect_layers(&network, l2, l3);
    nn_connect_layers(&network, l3, l4);
    nn_connect_layers(&network, l4, l5);
    nn_connect_layers(&network, l5, l6);
    

    nn_order_layers(&network); // Finalizing the NN structure

    printf("Ordering: ");
    for (int i = 0; i < network.layer_count; ++i)
        printf("%i ", network.graph_order[i]);
    printf("\n");

    Matrix input = mat_vector(1);
    input.values[0] = 1.0f;

    Matrix output = mat_vector(1);
    output.values[0] = 0.5706f;

    for (int i = 0; i < 10000; ++i)
        nn_train(&network, &input, &output);

    mat_free(&output);
    output = nn_forward(&network, &input);

    printf("Output %ix%i: %f\n", output.rows, output.cols, output.values[0]);

    mat_free(&input);
    mat_free(&output);

    nn_free(&network);
    return 0;
}