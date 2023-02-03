extern "C" __global__
void kernel(int M, int N, float* a, float* b, float* c) {
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (m > M || n > N) {
        return;
    }

    // Neighboring threads will access contiguous memory locations now.
    c[m*N+n] = exp(a[m*N+n]) / b[m];
}