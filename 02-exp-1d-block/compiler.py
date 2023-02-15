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
        nblocks = M
        nthreads = 128  # 128 is a common starting point for thread number per block
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
    
