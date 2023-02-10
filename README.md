This repo serves as an introduction to compiling numerical Python programs (A Python compiler). 
The code is kept as simple as possible and each day is a small step forward based on the previous day.

# Prerequisite

* pytorch
* cupy

You will also need a NVidia GPU to run the code.

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