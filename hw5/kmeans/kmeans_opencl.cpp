#include "kmeans.h"

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <string.h>
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

void kmeans(int iteration_n, int class_n, int data_n, Point* centroids, Point* data, int* partitioned)
{
    cl_int err;

    cl_platform_id platform;
    err = clGetPlatformIDs(1, &platform, NULL);
    CHECK_ERROR(err);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    CHECK_ERROR(err);

    cl_context context;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_ERROR(err);

    cl_command_queue queueIO;
    queueIO = clCreateCommandQueue(context, device, 0, &err);
    CHECK_ERROR(err);
    cl_command_queue queueSM;
    queueSM = clCreateCommandQueue(context, device, 0, &err);
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
    kernel = clCreateKernel(program, "classify", &err);
    CHECK_ERROR(err);

    size_t global_size = 65536, local_size = 256;
    int n = (data_n + global_size - 1) / global_size * global_size;
    float *D = (float*)malloc(sizeof(float) * 2 * data_n);
    float *C = (float*)malloc(sizeof(float) * 2 * class_n);
    cl_uchar *E = (cl_uchar*)malloc(sizeof(cl_uchar) * n);
    int *F = (int*)malloc(sizeof(int) * class_n);
    for (int i = 0; i < data_n; ++i) {
        D[i * 2] = data[i].x;
        D[i * 2 + 1] = data[i].y;
    }
    for (int i = 0; i < class_n; ++i) {
        C[i * 2] = centroids[i].x;
        C[i * 2 + 1] = centroids[i].y;
    }

    cl_mem memD;
    memD = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(cl_float2) * n, NULL, &err);
    CHECK_ERROR(err);
    cl_mem memC;
    memC = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(cl_float2) * class_n, NULL, &err);
    CHECK_ERROR(err);
    cl_mem memE;
    memE = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(cl_uchar) * n, NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memD);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memC);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memE);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uchar), &class_n);
    CHECK_ERROR(err);

    err = clEnqueueWriteBuffer(queueIO, memD, CL_TRUE, 0,
        sizeof(cl_float2) * data_n, D, 0, NULL, NULL);
    CHECK_ERROR(err);
    for (int iter = 0; iter < iteration_n; ++iter) {
        printf("iter %d begin\n", iter);

        err = clEnqueueWriteBuffer(queueIO, memC, CL_TRUE, 0,
            sizeof(cl_float2) * class_n, C, 0, NULL, NULL);
        CHECK_ERROR(err);
        memset(C, 0, sizeof(cl_float2) * class_n);
        memset(F, 0, sizeof(int) * class_n);

        int xSM = -1, xHost = -1;
        cl_uint num_events = 0;
        cl_event event[2];
        for (int i = 0; i < n; i += global_size) {
            if (num_events > 0) {
                err = clWaitForEvents(num_events, event);
                CHECK_ERROR(err);
            }

            if (xSM != -1) {
                err = clEnqueueReadBuffer(queueIO, memE, CL_FALSE,
                    sizeof(cl_uchar) * xSM, sizeof(cl_uchar) * global_size,
                    &E[xSM], 0, NULL, &event[1]);
                CHECK_ERROR(err);
                xHost = xSM;
            }

            num_events = 0;
            size_t global_offset = i;
            err = clEnqueueNDRangeKernel(queueSM, kernel, 1, &global_offset,
                &global_size, &local_size, 0, NULL, &event[num_events++]);
            CHECK_ERROR(err);
            xSM = i;

            if (xHost != -1) {
                err = clWaitForEvents(1, &event[1]);
                CHECK_ERROR(err);
                for (size_t x = 0; x < global_size; ++x) {
                    int idx = xHost + x;
                    C[E[idx] * 2] += D[idx * 2];
                    C[E[idx] * 2 + 1] += D[idx * 2 + 1];
                    ++F[E[idx]];
                }
            }
        }

        if (num_events > 0) {
            err = clWaitForEvents(num_events, event);
            CHECK_ERROR(err);
        }

        if (xSM != -1) {
            err = clEnqueueReadBuffer(queueIO, memE, CL_FALSE,
                sizeof(cl_uchar) * xSM, sizeof(cl_uchar) * global_size,
                &E[xSM], 0, NULL, &event[1]);
            CHECK_ERROR(err);
            xHost = xSM;
        }

        if (xHost != -1) {
            err = clWaitForEvents(1, &event[1]);
            CHECK_ERROR(err);
            for (size_t x = 0; x < global_size; ++x) {
                int idx = xHost + x;
                if (idx >= n) break;
                C[E[idx] * 2] += D[idx * 2];
                C[E[idx] * 2 + 1] += D[idx * 2 + 1];
                ++F[E[idx]];
            }
        }

        for (int x = 0; x < class_n; ++x) {
            if (F[x] > 0) {
                C[x * 2] /= F[x];
                C[x * 2 + 1] /= F[x];
            }
        }
    }

    for (int i = 0; i < class_n; ++i) {
        centroids[i].x = C[i * 2];
        centroids[i].y = C[i * 2 + 1];
    }
    for (int i = 0; i < data_n; ++i) {
        partitioned[i] = E[i];
    }

    free(D);
    free(C);
    free(E);
    free(F);
    clReleaseMemObject(memD);
    clReleaseMemObject(memC);
    clReleaseMemObject(memE);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queueIO);
    clReleaseCommandQueue(queueSM);
    clReleaseContext(context);
}

