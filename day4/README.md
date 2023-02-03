# Day 4

If you think the row/col id calculation is laborious in day 3, then today is for you! We are gonna do 2D grid and 2D thread block today, Bam! Note that in CUDA, (x, y, z), x changes the fastest. There for a row-major matrix, we'd want to make the column be the x dimension for memory coalescing (neighboring threads are accessing contiguous memory).