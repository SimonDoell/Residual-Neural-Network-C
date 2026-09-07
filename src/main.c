// #include <time.h>
// #include <stdio.h>
// #include "layer.h"
// #include "linear_adam.h"
// #include "activation_layer.h"
// #include "network.h"

// #define countof(arr) (sizeof(arr) / sizeof(arr[0]))

// float learn_function(float x) {
//     return sin(x * 3.14159265358979f * 2.0f);
// }

// int main() {srand(time(0));
//     NeuralNetwork network = nn_init();
//     network.learning_rate = 0.001f;
//     float scaling = 0.05f;

//     nn_add_layer(&network, linear_adam_new_default(1, 16, scaling), true);
//     nn_add_layer(&network, ReLU_new(), true);
    
//     for (int i = 0; i < 40; ++i) {
//         nn_add_layer(&network, linear_adam_new_default(16, 16, scaling), true);
//         nn_add_layer(&network, ReLU_new(), true);
//     }

//     nn_add_layer(&network, linear_adam_new_default(16, 1, scaling), true);
//     nn_add_layer(&network, linearTanh_new(), true);

//     for (int i = 1; i < network.layer_count-4; i+=2) {
//         nn_connect_layers(&network, i, i+2);
//     }

//     nn_order_layers(&network); // Finalizing the NN structure

//     // printf("Ordering: ");
//     // for (int i = 0; i < network.layer_count; ++i)
//     //     printf("%i ", network.graph_order[i]);
//     // printf("\n");

//     Matrix input  = mat_vector(1);
//     Matrix output = mat_vector(1);

//     // --- TRAINING ---
//     uint32_t train_iter = 100000;
//     for (int i = 0; i < train_iter; ++i) {
//         input .values[0] = rand_float(0, 1);
//         output.values[0] = learn_function(input.values[0]);
//         nn_train(&network, &input, &output);

//         if (i % (train_iter / 10) == 0 && i != 0) {
//             printf("Percentage: %i Percent\n", i / (train_iter / 100));
//         }
//     }

//     // --- TESTING ---
//     float error = 0.0f;
//     uint32_t iter = 100000;
    
//     for (int i = 0; i < iter; ++i) {
//         input.values[0] = rand_float(0, 1);

//         mat_free(&output);
//         output = nn_forward(&network, &input);

//         error += fabsf(learn_function(input.values[0]) - output.values[0]);
//     }

//     error = error / (float)iter;
//     printf("Error (amortized): %f\n", error);

//     mat_free(&input);
//     mat_free(&output);

//     nn_free(&network);
//     return 0;
// }


// AI generated:
#include <time.h>
#include <stdio.h>
#include "layer.h"
#include "linear_adam.h"
#include "activation_layer.h"
#include "rmsnorm.h"
#include "network.h"

float learn_function(float x) {
    return sin(x * 3.14159265358979f * 2.0f);
}

static NeuralNetwork build_network(float scaling) {
    NeuralNetwork network = nn_init();
    network.learning_rate = 0.001f;

    nn_add_layer(&network, linear_adam_new_default(1, 16, scaling), true);
    nn_add_layer(&network, RMSNorm_new(16), true);
    nn_add_layer(&network, leakyReLU_new(), true);

    for (int i = 0; i < 10; ++i) {
        nn_add_layer(&network, linear_adam_new_default(16, 16, scaling), true);
        nn_add_layer(&network, RMSNorm_new(16), true);
        nn_add_layer(&network, leakyReLU_new(), true);
    }

    nn_add_layer(&network, linear_adam_new_default(16, 1, scaling), true);
    nn_add_layer(&network, leakyTanh_new(), true);

    for (int i = 0; i < (int)network.layer_count - 6; i += 3) {
        nn_connect_layers(&network, i, i + 3);
    }

    nn_order_layers(&network);
    return network;
}

// Frobenius norm of a weight matrix
static float fro_norm(Matrix* m) {
    double s = 0.0;
    for (int i = 0; i < (int)mat_value_count(m); ++i) s += (double)m->values[i] * m->values[i];
    return (float)sqrt(s);
}

static void diagnostics(NeuralNetwork* network, int pre_out_idx, int step, int n_test) {
    Matrix input = mat_vector(1);
    float max_abs = 0.0f;
    int saturated = 0; // |deriv| < 0.01
    for (int i = 0; i < n_test; ++i) {
        input.values[0] = rand_float(0.0f, 1.0f);
        Matrix o = nn_forward(network, &input);
        float p = network->layers[pre_out_idx].output.values[0];
        if (fabsf(p) > max_abs) max_abs = fabsf(p);
        if (leakyTanh_derivative(p) < 0.01f) saturated++;
        mat_free(&o);
    }
    mat_free(&input);
    printf("  step %6d: max|pre-tanh|=%9.4f   saturated(deriv<0.01)=%3d/%d\n", step, max_abs, saturated, n_test);
    // printf("Layer gradient magnitudes:\n");

    // Gradient magnitude
    for (int i = 0; i < network->layer_count; ++i) {
        float grad_sum = 0.0f;
        float max_grad = 0.0f;

        for (int j = 0; j < network->layers[i].gradient_output.rows; ++j) {
            float grad = mat_at_const(&network->layers[i].gradient_output, j, 0);
            grad_sum += powf(grad, 2);
            max_grad = fmaxf(max_grad, grad);
        }

        grad_sum = sqrtf(grad_sum);

        // printf("L%i: %f (max: %f) | ", i, grad_sum, max_grad);

        // if (i % 5 == 0) printf("\n");
    }
    
    printf("\n");
}

static void run(float scaling, unsigned seed, int steps, int log_every) {
    printf("\n=== scaling = %.3f (seed=%u, steps=%d) ===\n", scaling, seed, steps);
    srand(seed);
    NeuralNetwork network = build_network(scaling);
    int pre_out_idx = network.layer_count - 2;

    diagnostics(&network, pre_out_idx, 0, 200);

    Matrix input = mat_vector(1);
    Matrix output = mat_vector(1);

    for (int i = 0; i < steps; ++i) {
        input.values[0] = rand_float(0, 1);
        output.values[0] = learn_function(input.values[0]);
        nn_train(&network, &input, &output);

        if ((i + 1) % log_every == 0) {
            diagnostics(&network, pre_out_idx, i + 1, 200);
        }
    }

    // final test error, like the repo's own test loop
    double err = 0.0;
    int iter = 20000;
    for (int i = 0; i < iter; ++i) {
        input.values[0] = rand_float(0, 1);
        Matrix o = nn_forward(&network, &input);
        err += fabs(learn_function(input.values[0]) - o.values[0]);
        mat_free(&o);
    }
    printf("  final amortized test error over %d samples: %f\n", iter, err / iter);

    mat_free(&input);
    mat_free(&output);
    nn_free(&network);
}

int main() {
    unsigned seed = 42;
    int steps = 100000;
    int log_every = 5000;

    run(0.30f, seed, steps, log_every);
    run(0.10f, seed, steps, log_every);
    run(0.05f, seed, steps, log_every);
    run(0.01f, seed, steps, log_every);

    return 0;
}