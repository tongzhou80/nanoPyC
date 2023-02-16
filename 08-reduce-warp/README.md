# Day 8 (Continued)
In fact we can perform reduction across threads in a warp without using shared memory, by using the shuffle `__shfl__xxx__sync` instructions. 