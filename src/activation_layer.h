#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include "layer.h"

#ifndef ACTIVATION_LAYER_H
#define ACTIVATION_LAYER_H

static inline float ReLU_forward   (float x) {return (x > 0.0f ? x : 0.0f);}
static inline float ReLU_derivative(float x) {return (x > 0.0f ? 1 : 0.0f);}

static const float relu_leakage = 0.2f;
static inline float leakyReLU_forward   (float x) {return (x > 0.0f ? x    : x * relu_leakage);}
static inline float leakyReLU_derivative(float x) {return (x > 0.0f ? 1.0f : relu_leakage);}

static inline float Tanh_forward   (float x) {return tanhf(x);}
static inline float Tanh_derivative(float x) {return 1.0f - powf(tanhf(x), 2);}

static const float tanh_leakage = 1.15f;
static inline float leakyTanh_forward   (float x) {return tanhf(x) * tanh_leakage;}
static inline float leakyTanh_derivative(float x) {return (1.0f - powf(tanhf(x), 2)) * tanh_leakage;}

static const float liniear_factor = 0.1f;
static inline float linearTanh_forward   (float x) {return tanhf(x) + liniear_factor * x;}
static inline float linearTanh_derivative(float x) {return (1.0f - powf(tanhf(x), 2)) + liniear_factor;}

typedef struct {
    float(*forward)(float);
    float(*derivative)(float);
} ActivationLayer_Context;

static inline void activation_layer_free(void* context) {
    assert(context);
    free(context);
}

static inline void activation_layer_forward(void* context, const Matrix* activation, Matrix* activation_output) {
    ActivationLayer_Context* activation_layer = context;
    assert(activation_layer);

    mat_copy_into(activation_output, activation);

    for (int i = 0; i < mat_value_count(activation_output); ++i) {
        activation_output->values[i] = activation_layer->forward(activation_output->values[i]);
    }
}

static inline void activation_layer_backward(void* context, const Matrix* gradient, const Matrix* activation, Matrix* gradient_output, float lr) {
    ActivationLayer_Context* activation_layer = context;
    assert(activation_layer);

    mat_copy_into(gradient_output, gradient);

    for (int i = 0; i < mat_value_count(gradient_output); ++i) {
        gradient_output->values[i] *= activation_layer->derivative(activation->values[i]);
    }
}

static inline Layer activation_layer_new(float(*forward)(float), float(*derivative)(float)) {
    ActivationLayer_Context* context = malloc(sizeof(ActivationLayer_Context));
    context->forward    = forward;
    context->derivative = derivative;
    return layer_new(context, activation_layer_free, activation_layer_forward, activation_layer_backward);
}

static inline Layer ReLU_new() {return activation_layer_new(ReLU_forward, ReLU_derivative);}
static inline Layer Tanh_new() {return activation_layer_new(Tanh_forward, Tanh_derivative);}
static inline Layer leakyReLU_new() {return activation_layer_new(leakyReLU_forward, leakyReLU_derivative);}
static inline Layer leakyTanh_new() {return activation_layer_new(leakyTanh_forward, leakyTanh_derivative);}
static inline Layer linearTanh_new() {return activation_layer_new(linearTanh_forward, linearTanh_derivative);}


#endif