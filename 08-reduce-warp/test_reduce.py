import torch
import torch.utils.benchmark as torchbench
import compiler

@compiler.jit
def my_kernel(a):
    b = torch.exp(a)
    c = b / torch.sum(b, axis=1)[:,None]
    return c

def torch_kernel(a):
    b = torch.exp(a)
    c = b / torch.sum(b, axis=1)[:,None]
    return c

def bench(fn):
    t0 = torchbench.Timer(
        stmt='fn()',
        globals={'fn': fn},
        num_threads=torch.get_num_threads()
    )
    return t0.timeit(20).mean * 1000

for shape in [(10240, 2048), (10240, 2048*2)]:
    M, N = shape
    a = torch.ones(M, N, device='cuda', dtype=torch.float32)
    a_cpu = a.cpu()
    out_torch = torch_kernel(a)
    out_my = my_kernel(a)
    print('allclose:', torch.allclose(out_my, out_torch))
    print('torch runtime:', bench(lambda: torch_kernel(a)), 'ms')
    print('torch (cpu) runtime:', bench(lambda: torch_kernel(a_cpu)), 'ms')
    print('  our runtime:', bench(lambda: my_kernel(a)), 'ms')

