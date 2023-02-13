compiled = {}

def compile(fn, args):
    # Here we can intercept the arguments and compile based on the
    # function code and its arguments. Based on the arguments, we 
    # may compile the function multiple times for different args
    print(f'[jit] Compile function {fn.__name__} with args {args}')
    return fn

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args=None)
        return compiled[fn](*args)
    
    return inner