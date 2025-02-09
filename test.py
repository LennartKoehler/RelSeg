import numpy as np
import torch

a = np.arange(1,5)

def func(x):
    for i in x:
        print("test")
        yield i,2

def test(gen):
    print(next(gen))

gen = func(a)
test(gen)
