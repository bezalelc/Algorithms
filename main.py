import numpy as np
import sympy as sp

if __name__ == '__main__':
    x = sp.symbols('x')
    f = 1 - x + x ** 2 - x ** 3
    f = sp.lambdify(x, f, 'numpy')
    y = np.array([-1, 0, 1, 2])
    print(f(y))
    print(np.sin(np.pi))
    # print(np.sin(np.pi))
