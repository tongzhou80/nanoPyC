# Day 3

Sometimes it's helpful to make a block and a grid 2D, as in matmuls. Today we are gonna do 2D grid and 2D thread block, Bam! Note that in CUDA, when a thread id is 3D, like `(x, y, z)`, `x` changes the fastest. Therefore for a row-major matrix, we'd want to make the column be the `x` dimension for memory coalescing (neighboring threads are accessing contiguous memory).