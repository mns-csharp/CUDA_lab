#ifndef FLAT_MATRIX
#define FLAT_MATRIX

#include <cstring> // for memcpy

class FlatMatrix
{
private:
    float* elements;
    int rows;
    int cols;

public:
    FlatMatrix() : elements(nullptr), rows(0), cols(0) {}

    FlatMatrix(int rows, int cols) : rows(rows), cols(cols)
    {
        this->elements = new float[rows * cols];
    }

    FlatMatrix(const FlatMatrix& rhs) : rows(rhs.rows), cols(rhs.cols)
    {
        this->elements = new float[rows * cols];
        std::memcpy(this->elements, rhs.elements, rows * cols * sizeof(float));
    }


    FlatMatrix& operator=(const FlatMatrix& rhs) // copy assignment operator
    {
        if (this == &rhs)
            return *this; // self-assignment check

        delete[] this->elements;

        this->rows = rhs.rows;
        this->cols = rhs.cols;
        this->elements = new float[rows * cols];
        std::memcpy(this->elements, rhs.elements, rows * cols * sizeof(float));

        return *this;
    }

    ~FlatMatrix()
    {
        delete[] elements;
    }

    void set_dimensions(int rows, int cols)
    {
        if (this->rows == rows && this->cols == cols)
            return; // nothing to do

        delete[] elements;

        this->rows = rows;
        this->cols = cols;
        this->elements = new float[rows * cols];
    }

    float get(int row, int col) const
    {
        return elements[row * cols + col];
    }

    void set(int row, int col, float val)
    {
        elements[row * cols + col] = val;
    }

    void dot_product(const FlatMatrix& rhs)
    {
        if (cols != rhs.rows)
        {
            // error: incompatible dimensions
            return;
        }

        FlatMatrix result(rows, rhs.cols);

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < rhs.cols; j++)
            {
                float sum = 0.0f;
                for (int k = 0; k < cols; k++)
                {
                    sum += get(i, k) * rhs.get(k, j);
                }
                result.set(i, j, sum);
            }
        }

        *this = result;
    }

    /* //must be implemented in CUDA
    void hadamard_product(const FlatMatrix& rhs)
    {
        if (rows != rhs.rows || cols != rhs.cols)
        {
            // error: incompatible dimensions
            return;
        }

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                set(i, j, get(i, j) * rhs.get(i, j));
            }
        }
    }*/

    FlatMatrix transpose() const
    {
        FlatMatrix result(cols, rows);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.set(j, i, get(i, j));
            }
        }
        return result;
    }

    void get_column(int col_no, float* column)
    {
        for (int i = 0; i < rows; i++) {
            column[i] = elements[i * cols + col_no];
        }
    }
};
#endif