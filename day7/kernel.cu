extern "C" __global__
void kernel(int M, int N, float* a, float* c) {
    int m = blockIdx.x;
    int tid = threadIdx.x;

    if (m > M) {
        return;
    }

    /* First attempt
    // Now we have `nthreads` threads for each row to do the reduction
    float sum = 0;
    for (int n = 0; n < N; n += blockDim.x) {
        if (n < N) {
            sum += exp(a[m*N + n]);
        }
    }

    // Now we want to sum up the partial sums in each thread. But how do we do this?
    // ???
    // Actually we'll need some mechanism for the threads to communicate with each other
    */

    __shared__ float psums[1024];
    int psum_size = blockDim.x;
    float sum = 0;
    for (int n = tid; n < N; n += blockDim.x) {
        if (n < N) {
            sum += exp(a[m*N + n]);
        }
    }
    psums[tid] = sum;
    __syncthreads();
    
    // Optional: do parallel reduction again
    // sum = 0;
    // for (int i = tid; i < blockDim.x; i += blockDim.x/2) {
    //     sum += psums[i];
    // }
    // if (tid < blockDim.x/2) {
    //     psums[tid] = sum;
    // }
    // psum_size = blockDim.x / 2;
    // __syncthreads();
    
    sum = 0;
    for (int i = 0; i < psum_size; i++) {
        sum += psums[i];
    }
        
    for (int n = tid; n < N; n += blockDim.x) {
        if (n < N) {
            c[m*N+n] = exp(a[m*N + n]) / sum;
        }
    }
}