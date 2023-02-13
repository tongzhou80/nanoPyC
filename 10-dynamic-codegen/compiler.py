import sys
import torch
import cupy as cp
from cupyx.profiler import benchmark

compiled = {}

shape_to_fn = {}

def ceil_div(a, b):
    return (a + b - 1) // b

def compile(fn, args):
    '''
    Shape specific dynamic code generation.
    '''
    print(f'[jit] Compile function {fn.__name__}')
    
    
    def compiled_kernel(a):
        M, N = a.shape
        c = torch.empty_like(a)
        _a = cp.asarray(a)
        _c = cp.asarray(c)

        nthreads = 256
        if (M, N) not in shape_to_fn:
            # Generate code for this shape set
            src = open('kernel.cu').read()
            src = src.replace('{NANOPYC_ROWSIZE}', str(N)).replace('{NANOPYC_NTHREADS}', str(nthreads))
            # print(src) # dump the generated code
            kernel = cp.RawKernel(src, 'kernel', backend='nvcc', options=('-O3',))
            shape_to_fn[(M, N)] = kernel
        kernel = shape_to_fn[(M, N)]
        
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
    
