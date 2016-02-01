#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>
#include <sys/time.h>
#include <unistd.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;

char *get_source_code(const char *file_name, size_t *len) {
    char *source_code;
    size_t length;
    FILE *file = fopen(file_name, "r");
    if (file == NULL) {
        printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
        exit(EXIT_FAILURE);
    }

    fseek(file, 0, SEEK_END);
    length = (size_t)ftell(file);
    rewind(file);

    source_code = (char *)malloc(length + 1);
    fread(source_code, length, 1, file);
    source_code[length] = '\0';

    fclose(file);

    *len = length;
    return source_code;
}

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1e-6 * tv.tv_usec;
}

int main() {
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);

    const char *source_code;
    size_t source_size;
    source_code = get_source_code("kernel.cl", &source_size);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source_code, &source_size, &err);
    CHECK_ERROR(err);

    err = clBuildProgram(program, 1, &device, "", NULL, NULL);
    if (err == CL_BUILD_PROGRAM_FAILURE) {
        char *log;
        size_t log_size;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        log = (char*)malloc(log_size + 1);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);
        log[log_size] = 0;
        printf("Compile error:\n%s\n", log);
        free(log);
    }
    CHECK_ERROR(err);

    cl_kernel kernel;
    kernel = clCreateKernel(program, "vec_add", &err);
    CHECK_ERROR(err);

    // BEGIN

    double t0[8][8], t1[8][8], t2[8][8], t3[8][8];
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            int n = 1 << (8 + x * 2), mod = 65536, i;
            size_t global_size = n;
            size_t local_size = 1 << (y + 1);
            double t;

            int *a = (int*)malloc(sizeof(int) * n);
            int *b = (int*)malloc(sizeof(int) * n);
            int *c = (int*)malloc(sizeof(int) * n);
            for (i = 0; i < n; ++i) {
                a[i] = rand() % mod;
                b[i] = rand() % mod;
            }

            cl_mem memA, memB, memC;
            t = get_time();
            memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, a, &err);
            CHECK_ERROR(err);
            memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * n, b, &err);
            CHECK_ERROR(err);
            t0[x][y] = get_time() - t;
            memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * n, NULL, &err);
            CHECK_ERROR(err);

            err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
            CHECK_ERROR(err);
            err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
            CHECK_ERROR(err);

            t = get_time();
            err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_size, &local_size, 0, NULL, NULL);
            CHECK_ERROR(err);
            err = clFinish(queue);
            CHECK_ERROR(err);
            t1[x][y] = get_time() - t;

            t = get_time();
            err = clEnqueueReadBuffer(queue, memC, CL_TRUE, 0, sizeof(int) * n, c, 0, NULL, NULL);
            CHECK_ERROR(err);
            t2[x][y] = get_time() - t;

            for (i = 0; i < n; ++i)
                if (a[i] + b[i] != c[i])
                    break;

            if (i < n) {
                printf("Verification failed!\n");
                printf("a[%d] = %d, b[%d] = %d, c[%d] = %d\n", i, a[i], i, b[i], i, c[i]);
                exit(EXIT_FAILURE);
            }
            printf("Verification success!\n");

            t = get_time();
            for (i = 0; i < n; ++i)
                c[i] = a[i] + b[i];
            t3[x][y] = get_time() - t;

            free(a);
            free(b);
            free(c);
            clReleaseMemObject(memA);
            clReleaseMemObject(memB);
            clReleaseMemObject(memC);
        }
    }

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y)
            printf("%f ", t0[x][y]);
        printf("\n");
    }
    printf("\n");

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y)
            printf("%f ", t1[x][y]);
        printf("\n");
    }
    printf("\n");

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y)
            printf("%f ", t2[x][y]);
        printf("\n");
    }
    printf("\n");

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y)
            printf("%f ", t3[x][y]);
        printf("\n");
    }
    printf("\n");

    // END

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Finished!\n");
    return 0;
}
