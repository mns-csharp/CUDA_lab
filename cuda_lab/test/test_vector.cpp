#include "gtest/gtest.h"

#include "vector.hpp"
/*
TEST(VectorHppTest, hadamard_product_test)
{
    Vector u;
    u.set_length(2);
    u[0] = 1;
    u[1] = 2;

    Vector v;
    v.set_length(3);
    v.set_row_vector(true);
    v[0] = 3;
    v[1] = 4;
    v[2] = 5;

    Vector * products1;
    u.hadamard_product(v, &products1);
    ASSERT_EQ(products1[0][0], 3);
    ASSERT_EQ(products1[0][1], 4);
    ASSERT_EQ(products1[0][2], 5);
    ASSERT_EQ(products1[1][0], 6);
    ASSERT_EQ(products1[1][1], 8);
    ASSERT_EQ(products1[1][2], 10);

    v.set_row_vector(false);
    u.set_row_vector(true);

    Vector * products2;
    v.hadamard_product(u, &products2);
    ASSERT_EQ(products2[0][0], 3);
    ASSERT_EQ(products2[0][1], 6);
    ASSERT_EQ(products2[1][0], 4);
    ASSERT_EQ(products2[1][1], 8);
    ASSERT_EQ(products2[2][0], 5);
    ASSERT_EQ(products2[2][1], 10);
}
*/

TEST(VectorHppTest_, c_hadamard_product_test)
{
    float * u = new float[2];
    u[0] = 1;
    u[1] = 2;

    float * v = new float[3];
    v[0] = 3;
    v[1] = 4;
    v[2] = 5;

    float * products1 = new float[2 * 3];
    hadamard_product_(u, 2, v, 3, products1);
    /**
 * vector1 = | 1 |
 *           | 2 |
 *
 * vector2 = |3  4  5|
 *
 * result = |3  4  5|
 *          |6  8 10|
 *
 * i=0, j=1, index = 3
 *
 *                0 1 2  3 4 5
 * flat_result = |3 4 5  6 7 8|
 **/
    ASSERT_EQ(get_value_(products1, 3, 0, 0), 3); ASSERT_EQ(get_value_(products1, 3, 0, 1), 4); ASSERT_EQ(
            get_value_(products1, 3, 0, 2), 5);
    ASSERT_EQ(get_value_(products1, 3, 1, 0), 6); ASSERT_EQ(get_value_(products1, 3, 1, 1), 8); ASSERT_EQ(
            get_value_(products1, 3, 1, 2), 10);
}

/*
TEST(VectorHppTest, instantiateObject)
{
    Vector v;
    v.set_length(3);
    v[0] = 10;
    v[1] = 11;
    v[2] = 12;

    ASSERT_EQ(v.get_length(), 3);
    ASSERT_EQ(v.is_column_vector(), true);
    ASSERT_EQ(v[0], 10);
    ASSERT_EQ(v[1], 11);
    ASSERT_EQ(v[2], 12);
}
*/






