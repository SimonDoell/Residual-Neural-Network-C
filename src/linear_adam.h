#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "layer.h"

#ifndef LINEAR_ADAM_H
#define LINEAR_ADAM_H

static inline float rand_float(float min, float max) {
    return (float)rand() / (float)RAND_MAX * (max - min) + min;
}

// --- Linear class + Adam optimizer ---
typedef struct {
    Matrix weights;
    Matrix biases;

    Matrix mean_weights;
    Matrix mean_biases;

    Matrix variance_weights;
    Matrix variance_biases;

    float betta_1;
    float betta_2;
    uint32_t t;
} LinearAdam_Context;

static inline void linear_adam_free(void* context) {
    LinearAdam_Context* linear = context;
    assert(linear);
    mat_free(&linear->weights);
    mat_free(&linear->biases);
    mat_free(&linear->mean_weights);
    mat_free(&linear->mean_biases);
    mat_free(&linear->variance_weights);
    mat_free(&linear->variance_biases);
    free(linear);
}

static inline void linear_adam_forward(void* context, const Matrix* activation, Matrix* activation_output) {
    LinearAdam_Context* linear = context;
    assert(linear);
    assert(activation);
    assert(activation_output);

    Matrix res = matmul(&linear->weights, activation);
    mat_add_bias_inplace(&res, &linear->biases);
    mat_move_into(activation_output, &res);
}

static inline void linear_adam_backward(void* context, const Matrix* batch_gradient, const Matrix* activation, Matrix* gradient_output, float lr) {
    LinearAdam_Context* linear = context;
    assert(linear);
    assert(activation);
    assert(gradient_output);

    const float eps = 1e-8f;

    Matrix weights_T = mat_transposed(&linear->weights);
    Matrix prev_grad = matmul(&weights_T, batch_gradient);
    mat_free(&weights_T);

    Matrix gradient = mat_sum_columns(batch_gradient);

    float betta_1_t = powf(linear->betta_1, linear->t);
    float betta_2_t = powf(linear->betta_2, linear->t);

    for (int i = 0; i < linear->biases.rows; ++i) {
        float* param    = mat_at(&linear->biases,          i, 0);
        float* mean     = mat_at(&linear->mean_biases,     i, 0);
        float* variance = mat_at(&linear->variance_biases, i, 0);
        float  grad     = mat_at_const(&gradient,          i, 0);

        *mean     = *mean     * linear->betta_1 + (1.0f - linear->betta_1) * grad;
        *variance = *variance * linear->betta_2 + (1.0f - linear->betta_2) * grad * grad;

        float corr_mean     = (*mean)     / (1.0f - betta_1_t);
        float corr_variance = (*variance) / (1.0f - betta_2_t);

        *param -= (corr_mean / (sqrtf(corr_variance) + eps)) * lr;
    }

    for (int i = 0; i < linear->weights.rows; ++i) {
        for (int j = 0; j < linear->weights.cols; ++j) {
            float* param    = mat_at(&linear->weights,          i, j);
            float* mean     = mat_at(&linear->mean_weights,     i, j);
            float* variance = mat_at(&linear->variance_weights, i, j);
            float  grad     = mat_at_const(activation, j, 0) * mat_at_const(&gradient, i, 0);

            *mean     = *mean     * linear->betta_1 + (1.0f - linear->betta_1) * grad;
            *variance = *variance * linear->betta_2 + (1.0f - linear->betta_2) * grad * grad;

            float corr_mean     = (*mean)     / (1.0f - betta_1_t);
            float corr_variance = (*variance) / (1.0f - betta_2_t);

            *param -= (corr_mean / (sqrtf(corr_variance) + eps)) * lr;
        }
    }

    linear->t++;
    mat_free(&gradient);
    mat_move_into(gradient_output, &prev_grad);
}

static inline Layer linear_adam_new(uint32_t fan_in, uint32_t fan_out, float betta_1, float betta_2) {
    LinearAdam_Context* context = malloc(sizeof(LinearAdam_Context));

    context->betta_1 = betta_1;
    context->betta_2 = betta_2;
    context->t       = 1;

    context->weights = mat_matrix(fan_out, fan_in);
    context->biases  = mat_vector(fan_out);
    context->mean_weights = mat_matrix(fan_out, fan_in);
    context->mean_biases  = mat_vector(fan_out);
    context->variance_weights = mat_matrix(fan_out, fan_in);
    context->variance_biases  = mat_vector(fan_out);

    // Xavier initialization
    float range = sqrtf(6.0f / (float)(fan_in + fan_out));

    for (int i = 0; i < context->weights.rows; ++i) {
        *mat_at(&context->biases, i, 0) = rand_float(-range, range);
        for (int j = 0; j < context->weights.cols; ++j) {
            *mat_at(&context->weights, i, j) = rand_float(-range, range);
        }
    }

    return layer_new(
        context,
        linear_adam_free,
        linear_adam_forward,
        linear_adam_backward
    );
}

static inline Layer linear_adam_new_default(uint32_t fan_in, uint32_t fan_out) {
    return linear_adam_new(fan_in, fan_out, 0.9f, 0.999f);
}


#endif