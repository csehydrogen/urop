#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

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
    kernel = clCreateKernel(program, "my_kernel", &err);
    CHECK_ERROR(err);

    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    printf("Finished!\n");
    return 0;
}
