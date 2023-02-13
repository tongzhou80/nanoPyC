/* Code Logic
# original code:
for i in range(M):
    for j in range(N):
    	b[i,j] = exp(a[i,j])

# parallelized code:
for i in range(M):  # parallelized among thread blocks
    for j in range(N):  # parallelized among threads
    	b[i,j] = exp(a[i,j])

Mapping an entire row to a block is helpful when we need to reduce the row.
*/

extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    // In Cuda, we specify the behavior only for a single scalar thread.
    // Each thread block work on an entire row, and each thread works on 
    // N/nthreads elements in the row.
    int m = blockIdx.x;
    if (m > M) {
        return;
    }
    
    // blockDim.x is `nthreads`
    // The key here is strided access pattern to guarantee contiguous memory
    // access among the threads
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        b[m*N + n] = exp(a[m*N + n]);
    }
}