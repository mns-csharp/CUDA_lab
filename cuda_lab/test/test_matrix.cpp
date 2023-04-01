/*
#include "gtest/gtest.h"

#include "matrix.hpp"

TEST(MatrixHppTest, instantiate_object_test)
{
    Matrix v;
    v.set_dimensions(3, 2);

    v[0][0] = 3;
    v[0][1] = 6;
    v[1][0] = 4;
    v[1][1] = 8;
    v[2][0] = 5;
    v[2][1] = 10;

    ASSERT_EQ(v.get_columns(), 2);
    ASSERT_EQ(v.get_rows(), 3);
    ASSERT_EQ(v[0][0], 3);
    ASSERT_EQ(v[0][1], 6);
    ASSERT_EQ(v[1][0], 4);
    ASSERT_EQ(v[1][1], 8);
    ASSERT_EQ(v[2][0], 5);
    ASSERT_EQ(v[2][1], 10);
}

TEST(MatrixHppTest, getColumn_i)
{
    Matrix mat;
    mat.set_dimensions(3,2);

    mat[0][0] = 3; mat[0][1] = 6;
    mat[1][0] = 4; mat[1][1] = 8;
    mat[2][0] = 5; mat[2][1] = 10;

    Vector col0 = mat.get_column(0);
    Vector col1 = mat.get_column(1);

    ASSERT_EQ(col0.get_length(), 3);
    ASSERT_EQ(col1.get_length(), 3);

    ASSERT_EQ(col0[0], 3);
    ASSERT_EQ(col0[1], 4);
    ASSERT_EQ(col0[2], 5);

    ASSERT_EQ(col1[0], 6);
    ASSERT_EQ(col1[1], 8);
    ASSERT_EQ(col1[2], 10);
}

TEST(MatrixHppTest, assignment_operator)
{
    Matrix mat1;
    mat1.set_dimensions(1,1);
    mat1[0][0] = -9;

    Matrix mat2;
    mat2.set_dimensions(1,1);
    mat2[0][0] = -99;

    Matrix mat3;
    mat3.set_dimensions(2,2);
    mat3[0][0] = 1; mat3[0][1] = 2;
    mat3[1][0] = 3; mat3[1][1] = 4;

    Matrix mat4;
    mat4.set_dimensions(2,2);
    mat4[0][0] = -1; mat4[0][1] = -2;
    mat4[1][0] = -3; mat4[1][1] = -4;

    //assign a bigger matrix to a smaller matrix
    mat1 = mat3;

    ASSERT_EQ(mat1[0][0], 1);    ASSERT_EQ(mat1[0][1], 2);
    ASSERT_EQ(mat1[1][0], 3);    ASSERT_EQ(mat1[1][1], 4);

    //assign a smaller matrix to a bigger matrix
    mat4 = mat2;
    ASSERT_EQ(mat4[0][0], -99);
}


TEST(MatrixHppTest, transpose)
{
    Matrix m;
    m.set_dimensions(2, 3);

    m[0][0] = 1; m[0][1] = 2; m[0][2] = 3;
    m[1][0] = 4; m[1][1] = 5; m[1][2] = 6;

    m.transpose();

    ASSERT_EQ(m.get_rows(), 3);
    ASSERT_EQ(m.get_columns(), 2);
//     1  4
//     2  5
//     3  6
    ASSERT_EQ(m[0][0], 1); ASSERT_EQ(m[0][1], 4);
    ASSERT_EQ(m[1][0], 2); ASSERT_EQ(m[1][1], 5);
    ASSERT_EQ(m[2][0], 3); ASSERT_EQ(m[2][1], 6);
}

/*
TEST(MatrixHppTest, multiplication)
{
    //result = u * v
    //u has a dimension : m x n (m = rows , n = cols)
    //v has a dimension : n x p (n = rows , p = cols)
    const int m = 1;
    const int n = 3;
    const int p = 2;
    Matrix u;
    u.set_rows(m);
    u.set_columns(n);

    u[0][0] = 1; u[0][1] = 2; u[0][2] = 3;

    Matrix v;
    v.set_rows(n);
    v.set_columns(p);

    v[0][0] = 1; v[0][1] = 2;
    v[1][0] = 3; v[1][1] = 4;
    v[2][0] = 5; v[2][1] = 6;

    Matrix result = u.multiply(v);

    ASSERT_EQ(result.get_rows(), 1);
    ASSERT_EQ(v.get_columns(), 2);
    ASSERT_EQ(v[0][0], 22);
    ASSERT_EQ(v[0][1], 28);
}*/

/*
TEST(MatrixHppTest, outer_product_test)
{
    Matrix mat1;
    mat1.set_dimensions(2, 3);
    mat1[0][0] = 1; mat1[0][1] = 2; mat1[0][2] = 3;
    mat1[1][0] = 4; mat1[1][1] = 5; mat1[1][2] = 6;

    Matrix mat2;
    mat2.set_dimensions(3, 3);
    mat2[0][0] = 7; mat2[0][1] = 8; mat2[0][2] = 9;
    mat2[1][0] = 10; mat2[1][1] = 11; mat2[1][2] = 12;
    mat2[2][0] = 13; mat2[2][1] = 14; mat2[2][2] = 15;

    Matrix result = mat1.multiply_outer(mat2);

    ASSERT_EQ(result.get_rows(), 2);
    ASSERT_EQ(result.get_columns(), 3);

    ASSERT_EQ(result[0][0], 66); ASSERT_EQ(result[0][1], 72); ASSERT_EQ(result[0][2], 78);
    ASSERT_EQ(result[1][0], 156); ASSERT_EQ(result[1][1], 171); ASSERT_EQ(result[1][2], 186);
}
 */