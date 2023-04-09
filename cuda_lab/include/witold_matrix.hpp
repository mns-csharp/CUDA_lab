#ifndef CUDA_LAB_WITOLD_MATRIX_H
#define CUDA_LAB_WITOLD_MATRIX_H

#include <chrono>
#include <iostream>
#include <cmath>
#include <cstdarg>
#include <cstdlib>

void MyMalloc(float**A, int N)
{
    *A = (float*) malloc(N * N * sizeof(float));
    memset(*A, 0, N * N * sizeof(float));
}

void CreateMatrix(float**A, int count, ...)
{
    *A = (float*) malloc(count * count * sizeof(float));
    va_list args;
    va_start(args, count);
    for (int i = 0; i < count; i++) {
        for (int j = 0; j < count; j++) {
            (*A)[i * count + j] = va_arg(args, double);
        }
    }
    va_end(args);
}

void Transpose(float *A, float**At, int N)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // copy the value at (i,j) to (j,i) in At
            (*At)[j*N + i] = A[i*N + j];
        }
    }
}

void Multiply(float* A, float* B, float**C, int N)
{
    // B is already transposed
    int loc_a, loc_b, loc_c;
    printf("In Multiply\n");
    for (int i =0; i< N; i++) {
        for (int j = 0; j<N;j++) {
            loc_c = i*N+j;
            loc_a = i*N;
            loc_b = j*N;
            (*C)[loc_c] = 0.0f;
            for (int k=0;k<N;k++) {
                float temp = A[loc_a]*B[loc_b];
                (*C)[loc_c] += temp;
                loc_a++;
                loc_b++;
            }
        }
    }
}

void OuterProduct(float* A, float* B, float** C, int N)
{
    for(int r=0 ; r<N ; r++)
    {
        for(int c=0 ; c<N ; c++)
        {
            for(int cc=0 ; cc<N ; cc++)
            {
                (*C)[r * N + c] += A[r * N + cc] * B[cc * N + c];
            }
        }
    }
}

void PrintMat(float* A, int row, int ext_row, int col, int ext_col, int N)
{
    int cur_row;
    int loc;
    cur_row = row;
    for (int i = 0; i< ext_row; i++)
    {
        loc = cur_row*N +col;
        for (int j=0; j< ext_col;j++)
        {
            printf("%f  ",A[loc+j]);
        }
        printf("\n");
        cur_row++;
    }
}

#endif //CUDA_LAB_WITOLD_MATRIX_H
