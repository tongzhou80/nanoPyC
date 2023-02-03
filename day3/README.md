# Day 3

Make the `exp` kernel more efficient by using more parallelism! Now we launch `MxN` threads, and make a thread work on one element. Notice how we calculate the row id and col id by using a 1D `blockIdx.x`!

Now performance matches PyTorch (cuBLAS) for this test!
