#include <stdio.h>
#include <stdlib.h>
#include "timers.h"
#include <CL/cl.h>

#define CHECK_ERROR(err) \
  if (err != CL_SUCCESS) { \
    printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
    exit(EXIT_FAILURE); \
  }

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

void mat_mul(float *a, float *b, float *c,
    size_t *dim, size_t *global_size, size_t *local_size) {
    cl_int err;

    timer_start(12);
    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);
    timer_stop(12);
    printf("platform : %lf sec\n", timer_read(12));

    timer_start(13);
    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);
    timer_stop(13);
    printf("device : %lf sec\n", timer_read(13));

    timer_start(14);
    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);
    timer_stop(14);
    printf("context : %lf sec\n", timer_read(14));

    timer_start(15);
    cl_command_queue queue;
    queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    timer_stop(15);
    printf("create queue : %lf sec\n", timer_read(15));

    timer_start(11);
    const char *source_code;
    size_t source_size;
    source_code = get_source_code("kernel.cl", &source_size);

    cl_program program;
    program = clCreateProgramWithSource(context, 1, &source_code, &source_size, &err);
    CHECK_ERROR(err);
    timer_stop(11);
    printf("create program : %lf sec\n", timer_read(11));

    timer_start(7);
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
    timer_stop(7);
    printf("Compile : %lf sec\n", timer_read(7));


    timer_start(8);
    cl_kernel kernel;
    kernel = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);
    timer_stop(8);
    printf("kernel create : %lf sec\n", timer_read(8));

    timer_start(9);
    cl_mem memA = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[1] * global_size[2], NULL, &err);
    CHECK_ERROR(err);
    cl_mem memB = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[2] * global_size[0], NULL, &err);
    CHECK_ERROR(err);
    cl_mem memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * global_size[1] * global_size[0], NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &global_size[2]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &global_size[0]);
    CHECK_ERROR(err);
    timer_stop(9);
    printf("buffer create : %lf sec\n", timer_read(9));

    float *bufA = (float*)malloc(sizeof(float) * global_size[1] * global_size[2]);
    float *bufB = (float*)malloc(sizeof(float) * global_size[2] * global_size[0]);
    float *bufC = (float*)malloc(sizeof(float) * global_size[1] * global_size[0]);
    for (int i = 0; i < dim[1]; i += global_size[1]) {
        for (int k = 0; k < dim[2]; k += global_size[2]) {
            timer_start(2);
            for (int x = 0; x < global_size[1]; ++x) {
                for (int y = 0; y < global_size[2]; ++y) {
                    bufA[x * global_size[2] + y] = a[(i + x) * dim[2] + (k + y)];
                }
            }
            err = clEnqueueWriteBuffer(queue, memA, CL_FALSE, 0,
                sizeof(float) * global_size[1] * global_size[2],
                bufA, 0, NULL, NULL);
            CHECK_ERROR(err);
            clFinish(queue);
            timer_stop(2);
            printf("A copy : %lf sec\n", timer_read(2));
            timer_clear(2);
            for (int j = 0; j < dim[0]; j += global_size[0]) {
                timer_start(3);
                for (int x = 0; x < global_size[2]; ++x) {
                    for (int y = 0; y < global_size[0]; ++y) {
                        bufB[x * global_size[0] + y] = b[(k + x) * dim[0] + (j + y)];
                    }
                }
                err = clEnqueueWriteBuffer(queue, memB, CL_FALSE, 0,
                    sizeof(float) * global_size[2] * global_size[0],
                    bufB, 0, NULL, NULL);
                CHECK_ERROR(err);
                clFinish(queue);
                timer_stop(3);
                printf("B copy : %lf sec\n", timer_read(3));
                timer_clear(3);
                timer_start(4);
                err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL,
                    global_size, local_size, 0, NULL, NULL);
                CHECK_ERROR(err);
                clFinish(queue);
                timer_stop(4);
                printf("Mult : %lf sec\n", timer_read(4));
                timer_clear(4);
                timer_start(5);
                err = clEnqueueReadBuffer(queue, memC, CL_TRUE, 0,
                    sizeof(float) * global_size[0] * global_size[1],
                    bufC, 0, NULL, NULL);
                CHECK_ERROR(err);
                timer_stop(5);
                printf("C copy : %lf sec\n", timer_read(5));
                timer_clear(5);
                timer_start(6);
                for (int x = 0; x < global_size[1]; ++x) {
                    for (int y = 0; y < global_size[0]; ++y) {
                        c[(i + x) * dim[0] + (j + y)] += bufC[x * global_size[0] + y];
                    }
                }
                timer_stop(6);
                printf("C add : %lf sec\n", timer_read(6));
                timer_clear(6);
            }
        }
    }

    timer_start(10);
    free(bufA);
    free(bufB);
    free(bufC);
    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    clReleaseMemObject(memC);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    timer_stop(10);
    printf("release : %lf sec\n", timer_read(10));
}
