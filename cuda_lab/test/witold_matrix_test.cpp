#include "gtest/gtest.h"
#include <cstdio>
#include "witold_matrix.hpp"

TEST(matrix_hpp_test, create_matrix_test)
{
    float *A = nullptr;  MyMalloc(&A, 2);
    CreateMatrix(&A, 2,    1.0, 2.0,
                                 3.0, 4.0);
    ASSERT_EQ(A[0], 1);
    ASSERT_EQ(A[1], 2);
    ASSERT_EQ(A[2], 3);
    ASSERT_EQ(A[3], 4);

    free(A);
}

TEST(matrix_hpp_test, matrix_tanspose_test)
{
    float *A = nullptr; MyMalloc(&A, 3);
    CreateMatrix(&A, 3,  1.0, 2.0, 3.0,
                               4.0, 5.0, 6.0,
                               7.0, 8.0, 9.0);
    float * At = nullptr; MyMalloc(&At, 3);
    Transpose(A, &At, 3);

    ASSERT_EQ(At[0], 1);
    ASSERT_EQ(At[1], 4);
    ASSERT_EQ(At[2], 7);
    ASSERT_EQ(At[3], 2);
    ASSERT_EQ(At[4], 5);
    ASSERT_EQ(At[5], 8);
    ASSERT_EQ(At[6], 3);
    ASSERT_EQ(At[7], 6);
    ASSERT_EQ(At[8], 9);

    free(A);
    free(At);
}

TEST(matrix_hpp_test, matrix_multiply_with_tanspose)
{
    float *A = nullptr; MyMalloc(&A, 3);
    float *B = nullptr; MyMalloc(&B, 3);
    float *C = nullptr; MyMalloc(&C, 3);
    CreateMatrix(&A, 3, 1.0, 2.0, 3.0,
                              4.0, 5.0, 6.0,
                              7.0, 8.0, 9.0);
    CreateMatrix(&B, 3, 1.0, 4.0, 7.0,
                                 2.0, 5.0, 8.0,
                                 3.0, 6.0, 9.0);
    Multiply(A, B, &C, 3);
    ASSERT_EQ(C[0], 30);
    ASSERT_EQ(C[1], 36);
    ASSERT_EQ(C[2], 42);

    ASSERT_EQ(C[3], 66);
    ASSERT_EQ(C[4], 81);
    ASSERT_EQ(C[5], 96);

    ASSERT_EQ(C[6], 102);
    ASSERT_EQ(C[7], 126);
    ASSERT_EQ(C[8], 150);

    free(A);
    free(B);
    free(C);
}

TEST(matrix_hpp_test, matrix_multiply_with_outer_product)
{
    float *A = nullptr; MyMalloc(&A, 3);
    float *B = nullptr; MyMalloc(&B, 3);
    float *C = nullptr; MyMalloc(&C, 3);
    CreateMatrix(&A, 3,   1.0, 2.0, 3.0,
                                4.0, 5.0, 6.0,
                                7.0, 8.0, 9.0);
    CreateMatrix(&B, 3, 1.0, 2.0, 3.0,
                                 4.0, 5.0, 6.0,
                                 7.0, 8.0, 9.0);
    OuterProduct(A, B, &C, 3);
    ASSERT_EQ(C[0], 30);
    ASSERT_EQ(C[1], 36);
    ASSERT_EQ(C[2], 42);
    ASSERT_EQ(C[3], 66);
    ASSERT_EQ(C[4], 81);
    ASSERT_EQ(C[5], 96);
    ASSERT_EQ(C[6], 102);
    ASSERT_EQ(C[7], 126);
    ASSERT_EQ(C[8], 150);

    free(A);
    free(B);
    free(C);
}

