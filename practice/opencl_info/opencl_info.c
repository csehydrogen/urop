#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define CHECK_ERROR(err) \
    if (err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

cl_int err;

void PrintPlatformInfo(cl_platform_id platform) {
    size_t buf_size;
    char *buf;

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, NULL, &buf_size);
    CHECK_ERROR(err);
    buf = (char*)malloc(buf_size);
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, buf_size, buf, NULL);
    CHECK_ERROR(err);
    printf("- CL_PLATFORM_NAME      : %s\n", buf);
    free(buf);

    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, NULL, &buf_size);
    CHECK_ERROR(err);
    buf = (char*)malloc(buf_size);
    err = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, buf_size, buf, NULL);
    CHECK_ERROR(err);
    printf("- CL_PLATFORM_VENDOR    : %s\n", buf);
    free(buf);
}

void PrintDeviceInfo(cl_device_id device) {
    size_t buf_size;
    char *buf;

    cl_device_type device_type;
    err = clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(device_type), &device_type, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_TYPE                :");
    if (device_type & CL_DEVICE_TYPE_CPU) printf(" CL_DEVICE_TYPE_CPU");
    if (device_type & CL_DEVICE_TYPE_GPU) printf(" CL_DEVICE_TYPE_GPU");
    if (device_type & CL_DEVICE_TYPE_ACCELERATOR) printf(" CL_DEVICE_TYPE_ACCELERATOR");
    if (device_type & CL_DEVICE_TYPE_DEFAULT) printf(" CL_DEVICE_TYPE_DEFAULT");
    if (device_type & CL_DEVICE_TYPE_CUSTOM) printf(" CL_DEVICE_TYPE_CUSTOM");
    printf("\n");

    err = clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &buf_size);
    CHECK_ERROR(err);
    buf = (char*)malloc(buf_size);
    err = clGetDeviceInfo(device, CL_DEVICE_NAME, buf_size, buf, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_NAME                : %s\n", buf);
    free(buf);

    size_t max_work_group_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_MAX_WORK_GROUP_SIZE : %u\n", max_work_group_size);

    cl_ulong global_mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_GLOBAL_MEM_SIZE     : %u\n", global_mem_size);

    cl_ulong local_mem_size;
    err = clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), &local_mem_size, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_LOCAL_MEM_SIZE      : %u\n", local_mem_size);

    cl_ulong max_mem_alloc_size;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_mem_alloc_size, NULL);
    CHECK_ERROR(err);
    printf("- CL_DEVICE_MAX_MEM_ALLOC_SIZE  : %u\n", max_mem_alloc_size);
}

int main() {
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err);
    printf("Number of platforms: %u\n\n", num_platforms);

    cl_platform_id *platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err);

    for (cl_uint i = 0; i < num_platforms; ++i) {
        printf("platform: %u\n", i);
        cl_platform_id platform = platforms[i];
        PrintPlatformInfo(platform);
        printf("\n");

        cl_uint num_devices;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        CHECK_ERROR(err);
        printf("Number of devices: %u\n\n", num_devices);

        cl_device_id *devices = (cl_device_id*)malloc(sizeof(cl_device_id) * num_devices);
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
        CHECK_ERROR(err);

        for (cl_uint j = 0; j < num_devices; ++j) {
            printf("device: %u\n", j);
            cl_device_id device = devices[j];
            PrintDeviceInfo(device);
            printf("\n");
        }
        free(devices);
    }
    free(platforms);
    return 0;
}
