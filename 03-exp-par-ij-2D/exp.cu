/* Code Logic
# original code:
for i in range(M):
    for j in range(N):
    	b[i,j] = exp(a[i,j])

# parallelized code:
for i_y in range(M // BM):  # parallelized among thread blocks (y dim)
    for i_x in range(N // BN):  # parallelized among thread blocks (x dim)
        for m in range(0, BM):  # parallelized among threads (y dim)
            for n in range(0, BN):  # parallelized among threads (x dim)
                ...
*/

extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    // In CUDA, you can do 2D grid and 2D blocks (and even 3D)! This can 
    // be convenient if you need to do some loop tiling, like in matmuls.
    // Notice that `x` is mapped to the column dimension
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (m > M || n > N) {
        return;
    }

    // Neighboring threads will access contiguous memory locations now.
    b[m*N+n] = exp(a[m*N+n]);
}