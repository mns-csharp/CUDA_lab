#include <iostream>
#include <cstdlib>
#include <ctime>

void create_mat(float ** a, int m, int n)
{
    *a = (float *) malloc(sizeof(float) * m * n);
    int index = 0;
    for(int i=0; i<m ; i++)
    {
        for(int j=0 ; j<n ; j++)
        {
            (*a)[index] = rand() % 100;
            index++;
        }
    }
}

void delete_mat(float * a)
{
    free(a);
}

void add_mat(float * a, float * b, int m, int n)
{
    int index = 0;
    for(int i=0 ; i<m ; i++)
    {
        for(int j=0 ; j<n ; j++)
        {
            b[index] = a[index] + b[index];
            index++;
        }
    }
}

void matrix_multiply(float * a, float * b, int m, int n)
{
    // allocate memory for the result matrix
    float *c = (float *) malloc(sizeof(float) * m * n);

    // multiply the matrices
    for(int i=0 ; i<m ; i++)
    {
        for(int j=0 ; j<n ; j++)
        {
            c[i*n+j] = 0;
            for(int k=0 ; k<n ; k++)
            {
                c[i*n+j] += a[i*n+k] * b[k*n+j];
            }
        }
    }

    // copy the result matrix to b
    for(int i=0 ; i<m*n ; i++)
    {
        b[i] = c[i];
    }

    // free the memory allocated for c
    free(c);
}


void print_mat(float * a, int m, int n)
{
    int index = 0;
    for(int i=0; i<m ; i++)
    {
        for(int j=0 ; j<n ; j++)
        {
            printf("%f ", a[index]);
            index++;
        }
        printf("\n");
    }
}

int main()
{
    const int m = 3;
    const int n = 3;

    float * a;
    float * b;

    // seed the random number generator with the current time
    srand(time(0));

    create_mat(&a, m, n);
    create_mat(&b, m, n);

    add_mat(a, b, m, n);

    print_mat(b, m, n);

    delete_mat(a);
    delete_mat(b);
    return 0;
}
