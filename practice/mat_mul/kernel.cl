__kernel void mat_mul(__constant float *A, __constant float *B, __global float *C, int ROW_A, int COL_A, int COL_B) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    float s = 0.0f;
    if (i < COL_B && j < ROW_A) {
        for (int k = 0; k < COL_A; ++k)
            s += A[k + j * COL_A] * B[i + k * COL_B];
        C[i + j * COL_B] = s;
    }
}
