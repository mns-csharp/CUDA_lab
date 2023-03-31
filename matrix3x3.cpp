#include <iostream>
#include <cuda_runtime.h>

class Matrix {
public:
    Matrix(int _m, int _n) : m(_m), n(_n) {
        data = new int[m * n];
    }

    ~Matrix() {
        delete[] data;
    }

    void setData(int *values) {
        for (int i = 0; i < m * n; i++) {
            data[i] = values[i];
        }
    }

    void multiply(Matrix &a, Matrix &b) {
        if (a.n != b.m) {
            std::cerr << "Error: Invalid matrix dimensions!" << std::endl;
            return;
        }

        m = a.m;
        n = b.n;
        data = new int[m * n];

        int *dev_a, *dev_b, *dev_c;

        cudaMalloc((void**)&dev_a, a.m * a.n * sizeof(int));
        cudaMalloc((void**)&dev_b, b.m * b.n * sizeof(int));
        cudaMalloc((void**)&dev_c, m * n * sizeof(int));

        cudaMemcpy(dev_a, a.data, a.m * a.n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b.data, b.m * b.n * sizeof(int), cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16);
        dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, (m + threadsPerBlock.y - 1) / threadsPerBlock.y);

        matrixMultiplicationKernel<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c, a.m, a.n, b.n);

        cudaMemcpy(data, dev_c, m * n * sizeof(int), cudaMemcpyDeviceToHost);

        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }

    void print() const {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                std::cout << data[i * n + j] << " ";
            }
            std::cout << std::endl;
        }
    }

private:
    int *data;
    int m, n;

    static __global__ void matrixMultiplicationKernel(int *a, int *b, int *c, int m, int n, int p) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < m && col < p) {
            int sum = 0;
            for (int i = 0; i < n; i++) {
                sum += a[row * n + i] * b[i * p + col];
            }
            c[row * p + col] = sum;
        }
    }
};

int main() {
    const int m1 = 3;
    const int n1 = 4;
    int a[m1][n1] = {{1, 2, 3, 4},
                     {5, 6, 7, 8},
                     {9, 10, 11, 12}};

    const int m2 = 4;
    const int n2 = 2;
    int b[m2][n2] = {{1, 2},
                     {3, 4},
                     {5, 6},
                     {7, 8}};

    Matrix matrix1(m1, n1);
    matrix1.setData(&a[0][0]);

    Matrix matrix2(m2, n2);
    matrix2.setData(&b[0][0]);

    Matrix result(m1, n2);
    result.multiply(matrix1, matrix2);

    std::cout << "Matrix 1:" << std::endl;
    matrix1.print();

    std::cout << "Matrix 2:" << std::endl;
    matrix2.print();

    std::cout << "Result:" << std::endl;
    result.print();

    return 0;
}
