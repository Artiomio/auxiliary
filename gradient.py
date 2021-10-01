""" Scalar function gradient of n-dimensional argument
    Вычисление градиента функции n-мерного аргумента
"""
import numpy as np

"""Function gradient
"""
def func_gradient(f, x0, dx):
    our_gradient = []
    x = list(x0)
    f1 = f(*x)
    for i in range(0, len(x)):
        x[i] += dx
        f2 = f(*x)
        x[i] -= dx
        our_gradient.append((f2-f1)/dx)
        
    return our_gradient

""" Function gradient using numpy arrays
    Strangely, takes more time: 52.8 us per loop
    whereas the first version takes only :  10.1 us per loop
"""       
def func_gradient1(f, x0, dx):
    gradient = np.zeros_like(x0, dtype=np.float)
    x = np.array(x0, dtype=np.float)
    f1 = f(*x)
    for i in range(0, len(x)):
        x[i] += dx
        f2 = f(*x)
        x[i] -= dx
        gradient[i] = (f2-f1)/dx
    return gradient
    


if __name__ == "__main__":

    from biddy import Biddy
    here = Biddy()
    
    
    def g(x, y, z):
        return x*x +2*x*y + z*y


    here.start()
    for i in range(1,100000):
        func_gradient(g, (1, 2, 1), 0.01)

    print("Time spent:", here.end()) 
    print(func_gradient(g, (1, 2, 1), 0.01))
    