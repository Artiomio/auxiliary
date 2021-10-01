import numpy as np
from gradient import func_gradient, func_gradient1
import math
import biddy
    
def simple_find_min(func_param, x0, err=0.00001, dx=0.001, learning_rate=0.01):
    """ Simple gradient descent using numpy, which quite unexpectedly
        turned out to be slower than the function basic_find_min which
        uses nothing else than pure Python
    """
    x = np.array(x0)
    #delta_x = np.ones_like(x) * dx
  
    grad_norm = err * 2
    while (grad_norm > err):
        grad_f = np.array(func_gradient(func_param, x, dx))
        x = x - learning_rate * grad_f
        grad_norm = max(abs(grad_f))
        #print("Now f(x)=", func_param(*x), "and x =",x, "   grad=", grad_f,"grad norm=",grad_norm )

    return x

    
def basic_find_min(func_param, x, err=0.00001, dx=0.001, learning_rate=0.01):
    #x = x0
    #delta_x = np.ones_like(x) * dx
  
    grad_norm = err * 2
    while (grad_norm > err):
        grad_f = func_gradient(func_param, x, dx)

        # x = x - learning_rate * grad_f
        x = [ a - b * learning_rate for (a,b) in zip(x, grad_f)]

        # grad_norm = max(abs(grad_f))
        grad_norm = max([abs(x) for x in grad_f])
        #print("Now f(x)=", func_param(*x), "and x =",x, "   grad=", grad_f,"grad norm=",grad_norm )

    return x
   

    
 
def f1(x):
    return (x*(x+123))


def g(x,y,z):
    return (x-3)**2+(y-1)**2+(z-6)**2
    
#print("Minimum of f at:", simple_find_min(f1, (3,)))

if __name__ == "__main__":
    #print("Minimum of g at:", simple_find_min(g, (41,52,634), learning_rate=0.001, err=0.0001, dx=0.001))
 
    def f(x,y):
        return  (2-(x*1+y))**2 +(4-(x*2+y))**2 + (6-(x*3+y))**2
        
    def f1(x,y,z):
        return x**2+y**2+(z**2)
    
    here = biddy.Biddy()
    here.start()
    print (basic_find_min(f, (1,1), err=0.0001, learning_rate = 0.01, dx = 0.0001))        
    for i in range(0, 100):
        basic_find_min(f, (1,1), err=0.0001, learning_rate = 0.01, dx = 0.0001)
    print("Time elapsed:", here.end())        

    #print(func_gradient(f1,(1,1), dx=0.0001))
    #неужели градиент неправильно считается?
