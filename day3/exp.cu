extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i > M*N) {
        return;
    }
    
    // Each threads works on only one element, and we launch `M*N/blockDim.x` blocks
    // Calculate the row id and the col id using an 1D block id
    int nblocks_per_row = N / blockDim.x;
    int m = blockIdx.x / nblocks_per_row;
    int n = (blockIdx.x % nblocks_per_row) * blockDim.x + threadIdx.x;
    b[m*N+n] = exp(a[m*N+n]);
}