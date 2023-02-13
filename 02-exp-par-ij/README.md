# Day 2

Implement a simple matrix `exp` function in CUDA! This also helps us get familiar with CuPy's rawkernel capability. 

For a `MxN` matrix, we will lanch `M` threads in total, and each thread will work on an entire row.