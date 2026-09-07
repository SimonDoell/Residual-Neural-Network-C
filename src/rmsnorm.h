#include <math.h>
#include <stdlib.h>
#include <stdint.h>
#include "layer.h"

#ifndef RMSNORM_H
#define RMSNORM_H

const float rms_eps = 1e-5f;

typedef struct {
    Matrix parameters;          // gamma (scale)
    Matrix normalized_vector;   // x / RMS(x)   (saved for backward)
    float  inv_rms;             // 1 / RMS(x)   (saved for backward)
} RMSNorm_Context;

static inline void rmsnorm_free(void* context) {
    RMSNorm_Context* rms = context;
    assert(rms);

    mat_free(&rms->parameters);
    mat_free(&rms->normalized_vector);
    free(rms);
}

static inline void rmsnorm_forward(void* context, const Matrix* activation, Matrix* activation_output) {
    RMSNorm_Context* rms = context;
    assert(rms);


    float mean_square = 0.0f;
    for (int i = 0; i < activation->rows; ++i) {
        float v = activation->values[i];
        mean_square += v * v;
    }
    mean_square = mean_square / (float)activation->rows + rms_eps;

    float root_mean_square = sqrtf(mean_square);
    rms->inv_rms = 1.0f / root_mean_square;

    mat_copy_into(activation_output, activation);

    for (int i = 0; i < activation_output->rows; ++i) {
        activation_output->values[i] *= rms->inv_rms;
    }

    mat_copy_into(&rms->normalized_vector, activation_output);

    for (int i = 0; i < activation_output->rows; ++i) {
        activation_output->values[i] *= rms->parameters.values[i];
    }
}

static inline void rmsnorm_backward(void* context, const Matrix* gradient, const Matrix* activation, Matrix* gradient_output, float lr) {
    RMSNorm_Context* rms = context;
    assert(rms);
    assert(gradient->rows == rms->parameters.rows);

    const int n = gradient->rows;
    
    Matrix dy_gamma = mat_vector(n);
    for (int i = 0; i < n; ++i) {
        dy_gamma.values[i] = gradient->values[i] * rms->parameters.values[i];
    }

    float mean_term = 0.0f;
    for (int i = 0; i < n; ++i) {
        mean_term += dy_gamma.values[i] * rms->normalized_vector.values[i];
    }
    mean_term /= (float)n;

    // dL/dx = inv_rms * (dy*gamma - x_hat * mean_term)
    Matrix d_x = mat_vector(n);
    for (int i = 0; i < n; ++i) {
        d_x.values[i] = rms->inv_rms *
            (dy_gamma.values[i] - rms->normalized_vector.values[i] * mean_term);
    }
    
    // Gradient of gamma and parameter update:
    // dL/dgamma_i = sum( dy_i * x_hat_i )
    for (int i = 0; i < n; ++i) {
        float dgamma = gradient->values[i] * rms->normalized_vector.values[i];
        rms->parameters.values[i] -= lr * dgamma;   // SGD update
    }
    
    mat_free(&dy_gamma);
    mat_move_into(gradient_output, &d_x);
}

static inline Layer RMSNorm_new(uint32_t in) {
    RMSNorm_Context* context = malloc(sizeof(RMSNorm_Context));
    context->parameters        = mat_vector(in);
    context->normalized_vector = mat_vector(in);
    context->inv_rms           = 0.0f;

    for (int i = 0; i < context->parameters.rows; ++i) {
        context->parameters.values[i] = 1.0f;
    }
    
    return layer_new(context, rmsnorm_free, rmsnorm_forward, rmsnorm_backward);
}

#endif