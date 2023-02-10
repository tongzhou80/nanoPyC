extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    // In CUDA, you can do 2D grid and 2D blocks (and even 3D), and this is how we do it!
    // Notice that `x` is mapped to the column dimension
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (m > M || n > N) {
        return;
    }

    // Neighboring threads will access contiguous memory locations now.
    b[m*N+n] = exp(a[m*N+n]);
}