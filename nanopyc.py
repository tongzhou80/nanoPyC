import sys
import torch
import cupy as cp
from cupyx.profiler import benchmark

compiled = {}

def ceil_div(a, b):
    return (a + b - 1) // b

def compile(fn, args):
    print(f'[jit] Compile function {fn.__name__}')
    src = open('v1_kernel.cu').read()
    kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
    def my_softmax(a):
        M, N = a.shape
        b = torch.empty_like(a)
        _a = cp.asarray(a)
        _b = cp.asarray(b)
        nthreads = 128
        nblocks = ceil_div(M, nthreads)
        kernel(
            (nblocks,), 
            (nthreads,), 
            (M, N, _a, _b)
        )
        return b
    compiled[fn] = my_softmax
    return my_softmax

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args=None)
        return compiled[fn](*args)
    
    return inner
    
