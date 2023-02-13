extern "C" __global__
void kernel(int M, int N, float* a, float* c) {
    int m = blockIdx.x;
    int tid = threadIdx.x;

    if (m > M) {
        return;
    }

    const int ROW_SIZE = 1024*4;
    assert(N <= ROW_SIZE);

    /** Load array a into the shared memory (later reused)
     */
    __shared__ float exps[ROW_SIZE];
    for (int n = tid; n < N; n += blockDim.x) {
        if (n < N) {
            exps[n] = exp(a[m*N + n]);
        }
    }

    /** Accumulate the partial sums into the shared memory 
     */
    __shared__ float psums[1024];
    float sum = 0;
    for (int n = tid; n < N; n += blockDim.x) {
        if (n < N) {
            sum += exps[n];
        }
    }
    psums[tid] = sum;
    __syncthreads();

    /** Perform reduction across threads
    In the 1st iteration, half of the threads do the work. In the 2nd 
    iteration, 1/4 of the threads do the work etc.
    In the last iteration, only thread 0 does the work (adding two numbers)
    */
    for (int step = blockDim.x / 2; step > 0; step = step / 2) {
        if (tid < step) {
            psums[tid] += psums[tid + step];
        }
        __syncthreads();
    }

    sum = psums[0];
        
    for (int n = tid; n < N; n += blockDim.x) {
        if (n < N) {
            c[m*N+n] = exps[n] / sum;
        }
    }
}