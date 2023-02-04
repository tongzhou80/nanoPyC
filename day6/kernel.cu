extern "C" __global__
void kernel(int M, int N, float* a, float* c) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;

    if (m > M) {
        return;
    }

    // Each thread works on an entire row with row id `m`.
    float sum = 0;
    for (int n = 0; n < N; n++) {
        sum += exp(a[m*N + n]);
    }

    for (int n = 0; n < N; n++) {
        c[m*N+n] = exp(a[m*N + n]) / sum;
    }
}