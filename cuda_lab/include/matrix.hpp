//#ifndef MATRIX_HPP
//#define MATRIX_HPP
//
//#include "vector.hpp"
//
//class Matrix
//{
//private:
//    int cols; // width or vector-length
//    int rows; // height or vectors-count or how many vector-obejects
//    Vector * vectors;
//    //bool is_memory_allocated;
//    bool is_constructor_called;
//public:
//    Matrix()
//    {
//        this->cols = 0;
//        this->rows = 0;
//        this->vectors = nullptr;
//        //this->is_memory_allocated = false;
//        this->is_constructor_called = true;
//    }
//    Matrix(const Matrix& other)// Copy constructor
//    {
//        this->cols = other.cols;
//        this->rows = other.rows;
//        this->vectors = new Vector[rows];
//        //this->is_memory_allocated = true;
//        for (int i = 0; i < rows; i++)
//        {
//            this->vectors[i] = other.vectors[i];
//        }
//        this->is_constructor_called = true;
//    }
//    Matrix& operator=(const Matrix& other)// Overloaded assignment operator
//    {
//        if (this != &other)
//        {
//            this->cols = other.cols;
//            this->rows = other.rows;
//            delete[] this->vectors;
//            this->vectors = new Vector[rows];
//            for (int i = 0; i < rows; i++)
//            {
//                this->vectors[i] = other.vectors[i];
//            }
//        }
//        return *this;
//    }
//    // Destructor
//    ~Matrix()
//    {
//        if (this->is_constructor_called)
//        {
//            this->cols = 0;
//            this->rows = 0;
//
//            delete[] this->vectors;
//
//            this->is_constructor_called = false;
//        }
//    }
//    void set_dimensions(int row, int col)
//    {
//        this->rows = row;
//        delete[] this->vectors;
//        this->vectors = new Vector[row];
//        this->cols = col;
//        for (int i = 0; i < row; i++)
//        {
//            this->vectors[i].set_length(col);
//        }
//    }
//
//    // Overloaded [] operator
//    Vector& operator[](int index)
//    {
//        if (index < 0 || index >= rows) {
//            throw std::out_of_range("Index out of bounds");
//        }
//        return vectors[index];
//    }
//
//    Vector get_column(int col_index) const
//    {
//        if (col_index < 0 || col_index >= cols)
//        {
//            std::cout << "Error: Invalid column index" << std::endl;
//        }
//        Vector col;
//        col.set_length(rows);
//        for (int i = 0; i < rows; i++)
//        {
//            col[i] = vectors[i][col_index];
//        }
//        return col;
//    }
//    int get_rows() const
//    {
//        return this->rows;
//    }
//
//    int get_columns() const
//    {
//        return this->cols;
//    }
//
//    void display() const
//    {
//        for(int i=0 ; i<rows ; i++)
//        {
//            vectors[i].display();
//            std::cout<<"\n";
//        }
//    }
//    void transpose()
//    {
//        Matrix transposed;
//        transposed.set_dimensions(this->cols, this->rows);
//        for (int i = 0; i < cols; i++)
//        {
//            Vector col = this->get_column(i);
//            for (int j = 0; j < rows; j++)
//            {
//                transposed[i][j] = col[j];
//            }
//        }
//        *this = transposed;
//    }
//    Matrix multiply_outer(const Matrix& other)
//    {
//        Matrix lhs(*this);
//        Matrix rhs(other);
//        rhs.transpose();
//
//        Matrix result;
//        result.set_dimensions(lhs.get_rows(), lhs.get_columns());
//        for(int i=0 ; i<lhs.get_rows() ; i++)
//        {
//            Vector lhs_vec = lhs[i];
//            for(int j=0 ; j<rhs.get_rows() ; j++)
//            {
//                Vector rhs_vec = rhs[j];
//                lhs_vec.set_row_vector(true);
//                float val = lhs_vec.dot_product(rhs_vec);
//                result[i][j] = val;
//            }
//        }
//        return result;
//    }
//};
//#endif