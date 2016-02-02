__kernel void classify(__global float2 *D, __global float2 *C, __global uchar *E,
    uchar cn) {
    int i = get_global_id(0);
    float m = INFINITY;
    uchar mj;
    for (uchar j = 0; j < cn; ++j) {
        float t = fast_distance(D[i], C[j]);
        if (m > t) {
            m = t;
            mj = j;
        }
    }
    E[i] = mj;
}
