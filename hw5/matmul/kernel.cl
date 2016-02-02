__kernel void mat_mul(__global float *A, __global float *B, __global float *C,
    ulong COL_A, ulong COL_B) {
    ulong i = get_global_id(0);
    ulong j = get_global_id(1);
    float s = 0.0f;
    for (ulong k = 0; k < COL_A; ++k)
        s += A[k + j * COL_A] * B[i + k * COL_B];
    C[i + j * COL_B] = s;
}
