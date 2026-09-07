#include <time.h>
#include <stdio.h>
#include "layer.h"
#include "linear_adam.h"
#include "activation_layer.h"
#include "network.h"

#define countof(arr) (sizeof(arr) / sizeof(arr[0]))

float learn_function(float x) {
    return sin(x * 3.14159265358979f * 2.0f);
}

int main() {srand(time(0));
    NeuralNetwork network = nn_init();
    network.learning_rate = 0.001f;
    float scaling = 0.05f;

    nn_add_layer(&network, linear_adam_new_default(1, 16, scaling), true);
    nn_add_layer(&network, ReLU_new(), true);
    
    for (int i = 0; i < 40; ++i) {
        nn_add_layer(&network, linear_adam_new_default(16, 16, scaling), true);
        nn_add_layer(&network, ReLU_new(), true);
    }

    nn_add_layer(&network, linear_adam_new_default(16, 1, scaling), true);
    nn_add_layer(&network, leakyTanh_new(), true);

    for (int i = 1; i < network.layer_count-4; i+=2) {
        nn_connect_layers(&network, i, i+2);
    }

    nn_order_layers(&network); // Finalizing the NN structure

    // printf("Ordering: ");
    // for (int i = 0; i < network.layer_count; ++i)
    //     printf("%i ", network.graph_order[i]);
    // printf("\n");

    Matrix input  = mat_vector(1);
    Matrix output = mat_vector(1);

    // --- TRAINING ---
    uint32_t train_iter = 100000;
    for (int i = 0; i < train_iter; ++i) {
        input .values[0] = rand_float(0, 1);
        output.values[0] = learn_function(input.values[0]);
        nn_train(&network, &input, &output);

        if (i % (train_iter / 10) == 0 && i != 0) {
            printf("Percentage: %i Percent\n", i / (train_iter / 100));
        }
    }

    // --- TESTING ---
    float error = 0.0f;
    uint32_t iter = 100000;
    
    for (int i = 0; i < iter; ++i) {
        input.values[0] = rand_float(0, 1);

        mat_free(&output);
        output = nn_forward(&network, &input);

        error += fabsf(learn_function(input.values[0]) - output.values[0]);
    }

    error = error / (float)iter;
    printf("Error (amortized): %f\n", error);

    mat_free(&input);
    mat_free(&output);

    nn_free(&network);
    return 0;
}