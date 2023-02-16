This repo serves as an introduction to compiling numerical Python programs (A Python compiler). 
The source code is kept as simple as possible and each day is a small step forward based on the previous day.
We will start from hand-coding simple functions, and gradually try to automatally generate/optimize them
using a compiler.

Following the tutorial, you will learn:

* Basic CUDA programming
* How to accelerate your numerical code using parallelization, fusion and tiling
* How a compiler automatically generates and optimizes numerical code/loops

The end result is, given a function like the following:

```python
def foo(a):
    b = exp(a)
    c = b.sum(axis=1)
    d = b / c
    return d
```

You will know how to make this code run much faster on CPU/GPU by just adding one line, like this

```python
@compiler.jit
def foo(a):
    b = exp(a)
    c = b.sum(axis=1)
    d = b / c
    return d
```

and understand why it runs faster!

# Prerequisite

* pytorch
* cupy

You will also need a Nvidia GPU to run the code. For now we generate CUDA code and compute on the GPU by default.

# Day 1
Implement a JIT compiler using Python decorator!

# Day 2
Implement a simple matrix `exp` function in CUDA!

# Day 3
Make the `exp` kernel more efficient by using more parallelism! Now the performance already matches cuBLAS.

# Day 4
Simplify the kernel code by using 2D partitioning. The pitfall is partitioning the rows to x dim.

# Day 5
First taste of fusion by creating a fused exp-div kernel!

# Day 6
Introducing reduction by a simple implementation of `softmax`.

# Day 7
Use parallel reduction and shared memory to make the `softmax` implementation more efficient.

# Day 8
More parallel reduction.

# Day 9
Make the `softmax` even more efficient by storing `exp(a)` in fast memory and eliminate reloading from global memory.

# Day 10
Trying out template-based dynamic code generation.