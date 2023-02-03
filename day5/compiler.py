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
    def compiled_kernel(a, b):
        M, N = a.shape
        c = torch.empty_like(a)
        _a = cp.asarray(a)
        _b = cp.asarray(b)
        _c = cp.asarray(c)
        
        # 2D block and 2D grid
        # Key: make x the col dim and y the row dim
        BM = 8  # y dim
        BN = 16 # x dim
        kernel(
            # Direction: (x, y)
            (ceil_div(N, BN), ceil_div(M, BM)), 
            (BN, BM), 
            (M, N, _a, _b, _c)
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
    
