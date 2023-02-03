extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    if (m > M) {
        return;
    }
    
    for (int i = 0; i < N; i++) {
        b[m*N + i] = exp(a[m*N + i]);
    }
}