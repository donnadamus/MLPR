import numpy as np, numpy
from scipy.optimize import fmin_l_bfgs_b

def func_v1(x):
    return (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2

def func_vfinal(x):
    return (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2, np.array([2*(x[0]+3) + np.cos(x[0]), 2*(x[1]+1)])


if __name__ == '__main__':
    x, f, d = fmin_l_bfgs_b(func_v1, [0,0], approx_grad=True)
    print(x)
    print(f)
    # print(d)

    x, f, d = fmin_l_bfgs_b(func_vfinal, [0,0])

    print(x)
    print(f)
    # print(d)
