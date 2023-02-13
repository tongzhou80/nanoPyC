import torch
import torch.utils.benchmark as torchbench
import compiler

@compiler.jit
def my_exp(a, b):
    c = torch.exp(a) / b[:, None]
    return c

def torch_exp(a, b):
    c = torch.exp(a) / b[:, None]
    return c

def bench(fn):
    t0 = torchbench.Timer(
    stmt='fn()',
    globals={'fn': fn})
    return t0.timeit(20).mean * 1000

for shape in [(10240, 2048)]:
    M, N = shape
    a = torch.randn(M, N, device='cuda', dtype=torch.float32)
    b = torch.randn(M, device='cuda', dtype=torch.float32)
    c_torch = torch_exp(a, b)
    c_my = my_exp(a, b)
    print('allclose:', torch.allclose(c_my, c_torch))
    print('torch runtime:', bench(lambda: torch_exp(a, b)), 'ms')
    print('  our runtime:', bench(lambda: my_exp(a, b)), 'ms')
