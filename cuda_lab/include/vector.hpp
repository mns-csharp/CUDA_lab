#ifndef VECTOR_HPP
#define VECTOR_HPP

#include <cstdlib>
#include <ctime>
#include <iostream>

float get_value_(float * vector, int cols, int row, int col)
{
    return vector[row * cols + col];
}

void get_column_(int col_no, float * column, float * flat_matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        column[i] = flat_matrix[i * cols + col_no];
    }
}

void scalar_product_(float * vector, int length, float scalar)
{
    for(int i=0 ; i<length; i++)
    {
        vector[i] = vector[i] * scalar;
    }
}

float dot_product_(float * vector1, float * vector2, int len)
{
    float result =0;
    for(int i=0 ; i<len; i++)
    {
        result += vector1[i] * vector2[i];
    }
    return result;
}
/**
 * vector1 = | 1 |
 *           | 2 |
 *
 * vector2 = |3  4  5|
 *
 * result = |3  4  5|
 *          |6  8 10|
 * i=0, j=1     index = 3
 * 0 x 2 +
 * //             0 1 2  3 4 5
 * flat_result = |3 4 5  6 7 8|
 **/
void hadamard_product_(float* vector1, int rows, float* vector2, int cols, float* &product_mat)
{
    product_mat = new float[rows * cols];
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            product_mat[i * cols + j] = vector1[i] * vector2[j];
        }
    }
}

/*
class Vector {
private:
    int length;
    float * elements;
    bool is_column; // column-vector = single column, multiple rows.
    // row-vector = singl row, multiple columns.
    //bool is_memory_allocated;
    bool is_constructor_called;
public:
    Vector()
    {
        this->length = 0;
        this->elements = nullptr;
        this->is_column = true;
        //this->is_memory_allocated = false;
        this->is_constructor_called = true;
    }
    Vector(int length)
    {
        this->set_length(length);
        is_column = true;
        this->is_constructor_called = true;
    }
    void set_length(int len)
    {
        this->length = len;
        if(elements!= nullptr)
        {
            delete [] elements;
        }
        elements = new float[len];
        //is_memory_allocated = true;
    }
    void set_row_vector(bool is_row)
    {
        is_column = !is_row;
    }
    int get_length() const
    {
        return length;
    }
    // Copy constructor
    Vector(const Vector& other)
    {
        this->length = other.length;
        this->elements = new float[length];
        //this->is_memory_allocated = true;
        for (int i = 0; i < length; i++) {
            this->elements[i] = other.elements[i];
        }
        this->is_column = true;
        this->is_constructor_called = true;
    }
    // Destructor
    ~Vector()
    {
        if(this->is_constructor_called) {
            this->length = 0;
            if (this->elements != nullptr)
            {
                delete[] this->elements;
            }
            this->is_constructor_called = false;
        }
    }

    Vector clone() const
    {
        Vector v(*this);
        return v;
    }
    // Overloaded assignment operator
    Vector& operator=(const Vector& other)
    {
        if (this != &other)
        {
            this->length = other.length;
            if(this->elements != nullptr)
            {
                delete[] this->elements;
            }
            this->elements = new float[length];
            for (int i = 0; i < length; i++) {
                this->elements[i] = other.elements[i];
            }
        }
        return *this;
    }
    float& operator[](int index) // will modify the state of the object
    // when assigned a value
    {
        return elements[index];
    }
    bool operator==(const Vector& other) const // this function will not modify the state
    // of the object on which it is called
    {
        if (this->length != other.length) {
            return false;
        }
        for (int i = 0; i < this->length; i++) {
            if (this->elements[i] != other.elements[i]) {
                return false;
            }
        }
        return true;
    }
    void scalar_product(float scalar)
    {
        for (int i = 0; i < length; i++) {
            this->elements[i] = this->elements[i] * scalar;
        }
    }
    friend Vector operator*(float scalar, const Vector& vec)
    {
        Vector result(vec);
        result.scalar_product(scalar);
        return result;
    }
    void display() const
    {
        if (length == 0) {
            std::cout << "[]" << std::endl;
            return;
        }

        if (is_column) {
            std::cout << "[";
            for (int i = 0; i < length - 1; i++) {
                std::cout << elements[i] << std::endl;
            }
            std::cout << elements[length - 1] << "]" << std::endl;
        } else {
            std::cout << "[";
            for (int i = 0; i < length - 1; i++) {
                std::cout << elements[i] << " ";
            }
            std::cout << elements[length - 1] << "]" << std::endl;
        }
    }
    bool is_column_vector() const
    {
        return is_column;
    }
    float dot_product(const Vector& other) const // inner product
    {
        if (!(!this->is_column && other.is_column_vector())) {
            throw std::runtime_error("Error: lhs must be row vector, rhs must be column vector");
        }

        if (this->length != other.length) {
            throw std::runtime_error("Error: Vectors must have the same length to perform Dot product");
        }
        float result = 0.0;
        for (int i = 0; i < this->length; i++) {
            result += this->elements[i] * other.elements[i];
        }
        return result;
    }
    void hadamard_product(const Vector& rhs, Vector** returns)
    {
        if (!(this->is_column && !rhs.is_column_vector())) {
            throw std::runtime_error("Error: rhs must be a row-vector");
        }
        *returns = new Vector[this->length];
        int index = 0;
        for (int i = 0; i < this->length; i++) {
            Vector temp = rhs.clone();
            float val = this->elements[i];
            temp.scalar_product(val);
            (*returns)[index] = temp;
            index++;
        }
    }
};
*/

#endif
