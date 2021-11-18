import numpy as np

def columns(*args):
    output  = [str(a).split('\n') for a in args]
    for x in zip(*output):
        for a in x:
            print(a, "    ", end="")
        print()    

columns(np.array([[3,4,5,6]]*10), np.array([[1,2,3,4]]*10), np.array([[33,44,54,64]]*10))