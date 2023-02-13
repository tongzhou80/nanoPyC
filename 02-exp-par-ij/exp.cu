/* Logic
# original code:
for i in range(M):
    for j in range(N):
    	b[i,j] = exp(a[i,j])

# parallelized code:
nthreads = 128
for i in range(M):  # parallelized among thread blocks
    for j in range(N):  # parallelized among threads
    	b[i,j] = exp(a[i,j])

*/



extern "C" __global__
void kernel(int M, int N, float* a, float* b) {
    int m = blockDim.x * blockIdx.x + threadIdx.x;
    if (m > M) {
        return;
    }
    
    // Each thread works on an entire row
    for (int i = 0; i < N; i++) {
        b[m*N + i] = exp(a[m*N + i]);
    }
}