#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <stdbool.h>

#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
    float* values;
    uint32_t rows;
    uint32_t cols;
} Matrix;


Matrix mat_matrix(uint32_t rows, uint32_t cols) {
    assert(rows > 0 && cols > 0);
    Matrix result;
    result.rows = rows;
    result.cols = cols;
    result.values = calloc(rows * cols, sizeof(float));
    return result;
}

Matrix mat_vector(uint32_t rows) {
    assert(rows > 0);
    Matrix result;
    result.rows = rows;
    result.cols = 1;
    result.values = calloc(rows * 1, sizeof(float));
    return result;
}

void mat_assert(const Matrix* matrix) {
    assert(matrix != NULL);
    
    // Use-after-free  OR  Free-after-free  OR  False initilisation
    assert(matrix->rows > 0);
    assert(matrix->cols > 0);
    assert(matrix->values != NULL);
}

void mat_free(Matrix* matrix) {
    mat_assert(matrix);
    free(matrix->values);
    
    matrix->rows   = 0;
    matrix->cols   = 0;
    matrix->values = NULL;
}

bool mat_same_shape(const Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    return (a->rows == b->rows && a->cols == b->cols);
}

uint32_t mat_value_count(const Matrix* matrix) {
    mat_assert(matrix);
    return matrix->rows * matrix->cols;
}

Matrix mat_copy(const Matrix* matrix) {
    mat_assert(matrix);
    
    Matrix copy = mat_matrix(matrix->rows, matrix->cols);
    
    memcpy(copy.values, matrix->values, sizeof(float) * mat_value_count(matrix));

    return copy;
}

void mat_copy_into(Matrix* dest, const Matrix* src) {
    mat_assert(src);
    mat_assert(dest);
    assert(dest != src);

    if (mat_same_shape(dest, src)) {
        memcpy(dest->values, src->values, sizeof(float) * mat_value_count(src));
    } else {
        mat_free(dest);
        *dest = mat_copy(src);
    }
}

void mat_move_into(Matrix* dest, Matrix* src) {
    mat_assert(src);
    mat_assert(dest);
    assert(dest != src);

    mat_free(dest);
    *dest = *src;

    src->cols   = 0;
    src->rows   = 0;
    src->values = NULL;
}

float* mat_at(Matrix* matrix, uint32_t row, uint32_t col) {
    mat_assert(matrix);
    assert(row < matrix->rows);
    assert(col < matrix->cols);

    return &matrix->values[row + col * matrix->rows]; // Column-major
}

float mat_at_const(const Matrix* matrix, uint32_t row, uint32_t col) {
    mat_assert(matrix);
    assert(row < matrix->rows);
    assert(col < matrix->cols);

    return matrix->values[row + col * matrix->rows]; // Column-major
}


Matrix mat_transposed(const Matrix* matrix) {
    mat_assert(matrix);
    
    Matrix result = mat_matrix(matrix->cols, matrix->rows);

    for (int j = 0; j < matrix->cols; ++j)
        for (int i = 0; i < matrix->rows; ++i)
            *mat_at(&result, j, i) = mat_at_const(matrix, i, j);
    
    return result;
}

Matrix mat_column(const Matrix* matrix, uint32_t column) {
    mat_assert(matrix);
    assert(column < matrix->cols);
    
    Matrix result = mat_matrix(matrix->rows, 1);

    for (int i = 0; i < matrix->rows; ++i)
        *mat_at(&result, i, 0) = mat_at_const(matrix, i, column);
    
    return result;
}

Matrix mat_sum_columns(const Matrix* matrix) {
    mat_assert(matrix);

    Matrix result = mat_vector(matrix->rows);

    for (int j = 0; j < matrix->cols; ++j)
        for (int i = 0; i < matrix->rows; ++i)
            result.values[i] += mat_at_const(matrix, i, j);

    return result;
}

Matrix mat_add(const Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    assert(mat_same_shape(a, b));

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] + b->values[i];
    
    return result;
}

Matrix mat_sub(const Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    assert(mat_same_shape(a, b));

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] - b->values[i];
    
    return result;
}

void mat_add_inplace(Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    assert(mat_same_shape(a, b));

    for (int i = 0; i < mat_value_count(a); ++i)
        a->values[i] += b->values[i];
}

void mat_sub_inplace(Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    assert(mat_same_shape(a, b));

    for (int i = 0; i < mat_value_count(a); ++i)
        a->values[i] -= b->values[i];
}

Matrix mat_fadd(const Matrix* a, const float b) {
    mat_assert(a);

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] + b;
    
    return result;
}

Matrix mat_fsub(const Matrix* a, const float b) {
    mat_assert(a);

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] - b;
    
    return result;
}

Matrix mat_fmul(const Matrix* a, const float b) {
    mat_assert(a);

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] * b;
    
    return result;
}

Matrix mat_fdiv(const Matrix* a, const float b) {
    mat_assert(a);
    assert(b != 0);

    Matrix result = mat_matrix(a->rows, a->cols);

    for (int i = 0; i < mat_value_count(a); ++i)
        result.values[i] = a->values[i] / b;
    
    return result;
}

void mat_fmul_inplace(Matrix* a, const float b) {
    mat_assert(a);

    for (int i = 0; i < mat_value_count(a); ++i)
        a->values[i] *= b;
}

void mat_fdiv_inplace(Matrix* a, const float b) {
    mat_assert(a);
    assert(b != 0);

    for (int i = 0; i < mat_value_count(a); ++i)
        a->values[i] /= b;
}


void mat_push_vector(Matrix* matrix, const Matrix* vector) {
    mat_assert(matrix);
    mat_assert(vector);

    assert(vector->cols == 1);
    assert(vector->rows == matrix->rows);

    matrix->values = realloc(matrix->values, sizeof(float) * matrix->rows * (matrix->cols + 1));
    assert(matrix->values);
    matrix->cols++;

    for (int i = 0; i < matrix->rows; ++i)
        *mat_at(matrix, i, matrix->cols-1) = mat_at_const(vector, i, 0);
}

Matrix matmul(const Matrix* a, const Matrix* b) {
    mat_assert(a);
    mat_assert(b);
    assert(a->cols == b->rows);

    Matrix result = mat_matrix(a->rows, b->cols);

    for (int i = 0; i < a->rows; ++i) {
        for (int k = 0; k < b->cols; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < a->cols; ++j) {
                sum += mat_at_const(a, i, j) * mat_at_const(b, j, k);
            }
            *mat_at(&result, i, k) = sum;
        }
    }

    return result;
}


// --- Source Clanker: ChatGPT ---
#define BLOCK 32

Matrix f_matmul(const Matrix *a, const Matrix *b) {
    mat_assert(a);
    mat_assert(b);
    assert(a->cols == b->rows);

    const int m = a->rows;
    const int n = a->cols;
    const int p = b->cols;

    Matrix result = mat_matrix(m, p);

    for (int kk = 0; kk < n; kk += BLOCK) {
        int kend = kk + BLOCK;
        if (kend > n) kend = n;

        for (int k = 0; k < p; ++k) {
            float *r_col = result.values + k * m;
            const float *b_col = b->values + k * n;

            for (int jj = kk; jj < kend; ++jj) {
                const float bkj = b_col[jj];
                const float *a_col = a->values + jj * m;

                for (int i = 0; i < m; ++i) {
                    r_col[i] += a_col[i] * bkj;
                }
            }
        }
    }

    return result;
}
// --- Clanker slop ending ---

void mat_add_bias_inplace(Matrix* matrix, const Matrix* vector) {
    mat_assert(matrix);
    mat_assert(vector);
    assert(vector->cols == 1);
    assert(matrix->rows == vector->rows);

    for (int j = 0; j < matrix->cols; ++j)
        for (int i = 0; i < matrix->rows; ++i)
            *mat_at(matrix, i, j) += mat_at_const(vector, i, 0);
}

#endif