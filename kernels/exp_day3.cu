extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > M*N) {
        return;
    }

    int nblocks_per_row = N / blockDim.x;
    int m = blockIdx.x / nblocks_per_row;
    int n = (blockIdx.x % nblocks_per_row) * blockDim.x + threadIdx.x;
    b[m*N+n] = exp(a[m*N+n]);
}