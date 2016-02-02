#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include <stdlib.h>
#include <string.h>
#include "timers.h"

size_t const N = 10000;

int print_matrix = 0;
int validation = 0;

void mat_mul(float *a, float *b, float *c,
    size_t *dim, size_t *global_size, size_t *local_size);

/************************** DO NOT TOUCH BELOW HERE ******************************/

void check_mat_mul(float *a, float *b, float *c, size_t *dim)
{
    int i, j, k;
    float sum;
    int validated = 1;

    printf("Validating the result..\n");

    // C = AB
    for( i = 0; i < N; i++ )
    {
        for( j = 0; j < N; j++ )
        {
            sum = 0;
            for( k = 0; k < N; k++ )
            {
                sum += a[i * dim[2] + k] * b[k * dim[0] + j];
            }

            if( c[i * dim[0] + j] != sum )
            {
                printf("c[%d][%d] is differ(value=%lf correct_value=%lf)!!\n",
                    i, j, c[i * dim[0] + j], sum );
                validated = 0;
            }
        }
    }

    printf("Validation : ");
    if( validated )
        printf("SUCCESSFUL.\n");
    else
        printf("FAILED.\n");
}

void print_mat(float *mat, size_t *dim)
{
    int i, j;

    for( i = 0; i < N; i++ )
    {
        for( j = 0; j < N; j++ )
        {
            printf("%8.2lf ", mat[i * dim[0] + j]);
        }
        printf("\n");
    }
}

void print_help(const char* prog_name)
{
    printf("Usage: %s [-pvh]\n", prog_name );
    printf("\n");
    printf("OPTIONS\n");
    printf("  -p : print matrix data.\n");
    printf("  -v : validate matrix multiplication.\n");
    printf("  -h : print this page.\n");
}

void parse_opt(int argc, char** argv)
{
    int opt;

    while( (opt = getopt(argc, argv, "pvhikjs:")) != -1 )
    {
        switch(opt)
        {
            case 'p':
                // print matrix data.
                print_matrix = 1;
                break;

            case 'v':
                // validation
                validation = 1;
                break;

            case 'h':
            default:
                print_help(argv[0]);
                exit(0);
                break;
        }
    }
}

int main(int argc, char** argv)
{
    parse_opt( argc, argv );

    // initialize matrices
    float *a, *b, *c;
    size_t global_size[3] = {4096, 4096, 4096};
    size_t local_size[2] = {16, 16};
    size_t dim[3] = {N, N, N};
    for (int i = 0; i < 3; ++i)
        dim[i] = (dim[i] + global_size[i] - 1) / global_size[i] * global_size[i];
    a = (float*)calloc(dim[1] * dim[2], sizeof(float));
    b = (float*)calloc(dim[2] * dim[0], sizeof(float));
    c = (float*)calloc(dim[1] * dim[0], sizeof(float));
    float k = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            a[i * dim[0] + j] = b[i * dim[0] + j] = k;
            k += 1.f;
        }

    timer_start(1);
    mat_mul(a, b, c, dim, global_size, local_size);
    timer_stop(1);

    printf("Time elapsed : %lf sec\n", timer_read(1));


    if( validation )
        check_mat_mul(a, b, c, dim);

    if( print_matrix )
    {
        printf("MATRIX A: \n");
        print_mat(a, dim);

        printf("MATRIX B: \n");
        print_mat(b, dim);

        printf("MATRIX C: \n");
        print_mat(c, dim);
    }

    return 0;
}
