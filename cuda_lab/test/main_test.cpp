#include "gtest/gtest.h"
//#include "vector.hpp"
//#include "matrix.hpp"

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
//    Matrix m;
//    m.set_rows(2);
//    m.set_rows(3);
//    m[0][0] = 1; m[0][1] = 2; m[0][2] = 3;
//    m[1][0] = 4; m[1][1] = 5; m[1][2] = 6;
//    //m.transpose();
//    m.display();
}
