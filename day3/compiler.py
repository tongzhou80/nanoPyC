import sys
import torch
import cupy as cp
from cupyx.profiler import benchmark

compiled = {}

def ceil_div(a, b):
    return (a + b - 1) // b

def compile(fn, args):
    print(f'[jit] Compile function {fn.__name__}')
    src = open('exp.cu').read()
    kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
    def compiled_kernel(a):
        M, N = a.shape
        b = torch.empty_like(a)
        _a = cp.asarray(a)
        _b = cp.asarray(b)
        nthreads = 256
        
        # Each threads works on only one element, and we launch `M*N/nthreads` blocks
        # with `nthreads` in each block 
        nblocks = ceil_div(M*N, nthreads)
        kernel(
            (nblocks,), 
            (nthreads,), 
            (M, N, _a, _b)
        )
        return b
    compiled[fn] = compiled_kernel
    return compiled_kernel

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args=None)
        return compiled[fn](*args)
    
    return inner
    
