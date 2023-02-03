import torch
import cupy as cp
from cupyx.profiler import benchmark
import nanopyc_day1 as nanopyc


@nanopyc.jit
def softmax(a):
    b = torch.exp(a)
    return b

# def softmax(a):
#     M, N = a.shape
#     b = torch.empty_like(a)

    # _a = cp.asarray(a)
    # _b = cp.asarray(b)

    # src = open('v1_kernel.cu').read()
    # kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
    # nthreads = 128
    # nblocks = ceil_div(M, nthreads)
    # kernel(
    #     (nblocks,), 
    #     (nthreads,), 
    #     (M, N, _a, _b)
    # )

    # print(benchmark(lambda: kernel(
    #     (nblocks,), 
    #     (nthreads,), 
    #     (M, N, _a, _b)
    # )))
    return b


for shape in [(1024, 1024)]:
    M, N = shape
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = softmax(a)
    print(torch.allclose(a+1, b))

    print(benchmark(lambda: softmax(a)))
