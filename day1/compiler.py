compiled = {}

def compile(fn, args):
    print(f'[jit] Compile function {fn.__name__}')
    return fn

def jit(fn):
    def inner(*args):
        if fn not in compiled:
            compiled[fn] = compile(fn, args=None)
        return compiled[fn](*args)
    
    return inner
    
