# Day 9

In day 7 and 8, we got a more efficient reduction implementation. However, in terms of the softmax kernel as a whole, there's still a memory inefficiency, i.e. the `exp(a)` is loaded twice from the global memory. Computing the `exp` twice is fine, as computing is much faster than global memory accesses. This extra global memory access acn be eliminated if an entire row can fit in shared memory.
