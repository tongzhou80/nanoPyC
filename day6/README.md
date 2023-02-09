# Day 6

Today we will start doing reduction, e.g. row-wise summation of a matrix, and have a simple implementation of `softmax`. We'll start off by row-wise partitioning, i.e. each thread works on an entire row.

Performance is not great, but we've got a working fused `softmax` CUDA kernel using less than 20 lines of CUDA C++!

Let's now see why this implementation is not optimal. 

* Each thread works on an entire row, when `M` is relatively small, and `N` is large, this may not utilize all the cores on the GPU, because the row reduction is not parallelized.
* Adjacent threads are accessing strided memory locations (with stride being `N`).
