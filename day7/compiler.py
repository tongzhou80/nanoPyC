import sys
import torch
import cupy as cp
from cupyx.profiler import benchmark

compiled = {}

def ceil_div(a, b):
    return (a + b - 1) // b

def compile(fn, args):
    print(f'[jit] Compile function {fn.__name__}')
    src = open('kernel.cu').read()
    kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
    def compiled_kernel(a):
        M, N = a.shape
        c = torch.empty_like(a)
        _a = cp.asarray(a)
        _c = cp.asarray(c)
        
        # 2D grid, with M blocks and `nthreads` threads per block
        # Each block (`nthreads` threads) works on a row
        nthreads = 256
        kernel(
            (M,), 
            (nthreads,),
            (M, N, _a, _c)
        )
        return c
    compiled[fn] = compiled_kernel
    return compiled_kernel

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args=None)
        return compiled[fn](*args)
    
    return inner
    
