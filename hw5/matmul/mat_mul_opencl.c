#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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

void in2buf(float *in, float *buf, size_t n, size_t m, size_t l, int sx, int sy) {
    for (int x = 0; x < n; ++x)
        memcpy(&buf[x * m], &in[(sx + x) * l + sy], sizeof(float) * m);
}

void mat_mul(float *a, float *b, float *c,
    size_t *dim, size_t *global_size, size_t *local_size) {
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
    kernel = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);

    cl_mem memA[2];
    memA[0] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[1] * global_size[2], NULL, &err);
    CHECK_ERROR(err);
    memA[1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[1] * global_size[2], NULL, &err);
    CHECK_ERROR(err);
    cl_mem memB[2];
    memB[0] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[2] * global_size[0], NULL, &err);
    CHECK_ERROR(err);
    memB[1] = clCreateBuffer(context, CL_MEM_READ_ONLY,
        sizeof(float) * global_size[2] * global_size[0], NULL, &err);
    CHECK_ERROR(err);
    cl_mem memC[2];
    memC[0] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * global_size[1] * global_size[0], NULL, &err);
    CHECK_ERROR(err);
    memC[1] = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * global_size[1] * global_size[0], NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC[0]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_ulong), &global_size[2]);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_ulong), &global_size[0]);
    CHECK_ERROR(err);

    float *bufA, *bufB, *bufC;
    bufA = (float*)malloc(sizeof(float) * global_size[1] * global_size[2]);
    bufB = (float*)malloc(sizeof(float) * global_size[2] * global_size[0]);
    bufC = (float*)malloc(sizeof(float) * global_size[1] * global_size[0]);

    int swA = 0, swB = 0, swC = 0;
    cl_uint num_events = 0;
    cl_event event[3];
    int xIO, yIO, xSM, ySM, xHost, yHost;
    xIO = yIO = xSM = ySM = xHost = yHost = -1;
    for (int i = 0; i < dim[1]; i += global_size[1]) {
        for (int k = 0; k < dim[2]; k += global_size[2]) {
            for (int j = 0; j < dim[0]; j += global_size[0]) {
                if (num_events > 0) {
                    err = clWaitForEvents(num_events, event);
                    CHECK_ERROR(err);
                }

                if (xSM != -1) {
                    err = clEnqueueReadBuffer(queueIO, memC[swC], CL_FALSE, 0,
                        sizeof(float) * global_size[0] * global_size[1],
                        bufC, 0, NULL, &event[1]);
                    CHECK_ERROR(err);
                    swC ^= 1;
                    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC[swC]);
                    CHECK_ERROR(err);
                    xHost = xSM; yHost = ySM;
                }

                num_events = 0;
                if (xIO != -1) {
                    err = clEnqueueNDRangeKernel(queueSM, kernel, 2, NULL,
                        global_size, local_size, 0, NULL, &event[num_events++]);
                    CHECK_ERROR(err);
                    xSM = xIO; ySM = yIO;
                }

                if (xHost != -1) {
                    err = clWaitForEvents(1, &event[1]);
                    CHECK_ERROR(err);
                }

                if (j == 0) {
                    in2buf(a, bufA, global_size[1], global_size[2], dim[2], i, k);
                    err = clEnqueueWriteBuffer(queueIO, memA[swA], CL_FALSE, 0,
                        sizeof(float) * global_size[1] * global_size[2],
                        bufA, 0, NULL, &event[num_events++]);
                    CHECK_ERROR(err);
                    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA[swA]);
                    CHECK_ERROR(err);
                    swA ^= 1;
                }

                in2buf(b, bufB, global_size[2], global_size[0], dim[0], k, j);
                err = clEnqueueWriteBuffer(queueIO, memB[swB], CL_FALSE, 0,
                    sizeof(float) * global_size[2] * global_size[0],
                    bufB, 0, NULL, &event[num_events++]);
                CHECK_ERROR(err);
                err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB[swB]);
                CHECK_ERROR(err);
                swB ^= 1;

                xIO = i; yIO = j;

                if (xHost != -1) {
                    for (int x = 0; x < global_size[1]; ++x) {
                        for (int y = 0; y < global_size[0]; ++y) {
                            c[(xHost + x) * dim[0] + (yHost + y)] += bufC[x * global_size[0] + y];
                        }
                    }
                }
            }
        }
    }

    if (num_events > 0) {
        err = clWaitForEvents(num_events, event);
        CHECK_ERROR(err);
    }

    if (xSM != -1) {
        err = clEnqueueReadBuffer(queueIO, memC[swC], CL_FALSE, 0,
            sizeof(float) * global_size[0] * global_size[1],
            bufC, 0, NULL, &event[1]);
        CHECK_ERROR(err);
        swC ^= 1;
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC[swC]);
        CHECK_ERROR(err);
        xHost = xSM; yHost = ySM;
    }

    num_events = 0;
    if (xIO != -1) {
        err = clEnqueueNDRangeKernel(queueSM, kernel, 2, NULL,
            global_size, local_size, 0, NULL, &event[num_events++]);
        CHECK_ERROR(err);
        xSM = xIO; ySM = yIO;
    }

    if (xHost != -1) {
        err = clWaitForEvents(1, &event[1]);
        CHECK_ERROR(err);
        for (int x = 0; x < global_size[1]; ++x) {
            for (int y = 0; y < global_size[0]; ++y) {
                c[(xHost + x) * dim[0] + (yHost + y)] += bufC[x * global_size[0] + y];
            }
        }
    }

    if (num_events > 0) {
        err = clWaitForEvents(num_events, event);
        CHECK_ERROR(err);
    }

    if (xSM != -1) {
        err = clEnqueueReadBuffer(queueIO, memC[swC], CL_FALSE, 0,
            sizeof(float) * global_size[0] * global_size[1],
            bufC, 0, NULL, &event[1]);
        CHECK_ERROR(err);
        swC ^= 1;
        err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC[swC]);
        CHECK_ERROR(err);
        xHost = xSM; yHost = ySM;
    }

    if (xHost != -1) {
        err = clWaitForEvents(1, &event[1]);
        CHECK_ERROR(err);
        for (int x = 0; x < global_size[1]; ++x) {
            for (int y = 0; y < global_size[0]; ++y) {
                c[(xHost + x) * dim[0] + (yHost + y)] += bufC[x * global_size[0] + y];
            }
        }
    }

    free(bufA);
    free(bufB);
    free(bufC);
    clReleaseMemObject(memA[0]);
    clReleaseMemObject(memA[1]);
    clReleaseMemObject(memB[0]);
    clReleaseMemObject(memB[1]);
    clReleaseMemObject(memC[0]);
    clReleaseMemObject(memC[1]);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queueIO);
    clReleaseCommandQueue(queueSM);
    clReleaseContext(context);
}
