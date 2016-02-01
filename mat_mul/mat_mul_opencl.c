#include <stdio.h>
#include <stdlib.h>
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

void mat_mul_opencl(float *A, float *B, float *C,
                    int ROW_A, int COL_A, int COL_B) {
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
    kernel = clCreateKernel(program, "mat_mul", &err);
    CHECK_ERROR(err);

    cl_mem memA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * ROW_A * COL_A, A, &err);
    CHECK_ERROR(err);
    cl_mem memB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * COL_A * COL_B, B, &err);
    CHECK_ERROR(err);
    cl_mem memC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
        sizeof(float) * ROW_A * COL_B, NULL, &err);
    CHECK_ERROR(err);

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memA);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memB);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memC);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 3, sizeof(cl_int), &ROW_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 4, sizeof(cl_int), &COL_A);
    CHECK_ERROR(err);
    err = clSetKernelArg(kernel, 5, sizeof(cl_int), &COL_B);
    CHECK_ERROR(err);

    size_t global_size[2] = {COL_B, ROW_A};
    size_t local_size[2] = {16, 16};
    for (int i = 0; i < 2; ++i)
        global_size[i] = (global_size[i] + local_size[i] - 1) / local_size[i]
            * local_size[i];

    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_size, local_size,
        0, NULL, NULL);
    CHECK_ERROR(err);

    err = clEnqueueReadBuffer(queue, memC, CL_TRUE, 0, sizeof(float) * ROW_A * COL_B,
        C, 0, NULL, NULL);
    CHECK_ERROR(err);

    clReleaseMemObject(memA);
    clReleaseMemObject(memB);
    clReleaseMemObject(memC);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
}
