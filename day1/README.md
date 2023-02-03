# Day 1

Implement a JIT compiler using Python decorator!

A decorator takes a Python function as input and return a new function object. The trick here to achieve JIT compilation is that we return an `inner` function which JIT compiles on the first invocation and remembers the compiled function for later use!

Bam!