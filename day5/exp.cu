extern "C" __global__
void kernel(int M, int N, float* a, float* b, float* c) {
    int m = blockDim.y * blockIdx.y + threadIdx.y;
    int n = blockDim.x * blockIdx.x + threadIdx.x;

    if (m > M || n > N) {
        return;
    }

    // A simple fused kernel. Note that the intermediate results exp(a[m*N+n]) are not stored 
    // back to global memory, and reside in just registers. This is how fusion improves performance.
    c[m*N+n] = exp(a[m*N+n]) / b[m];
}