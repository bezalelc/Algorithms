import root
import sympy as sp
import numpy as np


def newton(x0, f_1, f_2, max_iter=100, eps=1e-7, landa=1e-7):
    """
    newton method for find min fo multiple variable

    :param
        f_1: derivative of function that get x=[x0,x1,x2,...,xn] array and return
             [f1([x0,x1,x2,...,xn]),...,fn([x0,x1,x2,...,xn])]
        f_2: derivative of function that get x=[x0,x1,x2,...,xn] array and return
             [f1([x0,x1,x2,...,xn]),...,fn([x0,x1,x2,...,xn])]
        landa: error for |f{x(n+1)}-f{x(n)}|
        eps: error for |x(n+1)-x(n)|
        max_iter: max iteration if the minimization not success

    :return: x_history:

    :efficiency: O(iter*n^3) when iter=number of iteration and n=length of x
    """
    return root.newton_raphson(f_1, f_2, x0, landa=landa, epsilon=eps, max_iter=max_iter)


def cross(f, l, u, landa=1e-10, eps=1e-10, max_iter=100):
    """
    cross method for find min fo one variable using sympy function

    :param f: sympy function
           l,u: border for search
           landa: error for |f{x(n+1)}-f{x(n)}|
           epsilon: error for |x(n+1)-x(n)|
           max_iter: max iteration if the minimization not success

    :return: x_history:

    :efficiency: O(iter*n) when iter=number of iteration and n=time to eval f(x)
    """
    x_history = []
    x1, x2 = l + (u - l) * 0.382, l + (u - l) * 0.618
    f_x1, f_x2 = f(x1), f(x2)

    for i in range(max_iter):
        x_history.append((x1, x2))

        if np.abs(f_x1 - f_x2) < landa and np.abs(x1 - x2) < eps:
            break

        if f_x2 > f_x1:
            u = x2
            x1, x2, f_x2 = l + (u - l) * 0.382, x1, f_x1
            f_x1 = f(x1)
        else:
            l = x1
            x1, x2, f_x1 = x2, l + (u - l) * 0.618, f_x2
            f_x2 = f(x2)

    return x_history


def gradient_descent(grad, x0, alpha=1e-2, max_iter=100):
    try:
        _ = iter(x0)
        x0 = np.array(x0, dtype=np.float64)
    except TypeError:
        x0 = np.array([x0, ], dtype=np.float64)
    x_history = [x0]

    for i in range(max_iter):
        x0 -= alpha * grad(*x0).reshape((-1,))
        x_history.append(x0)

    return x_history


if __name__ == '__main__':
    x = sp.symbols('x')
    f, x0 = 0.1 * x ** 2 - 2 * sp.sin(x), 30
    f_1, f_2 = sp.lambdify(x, sp.diff(f, x, 1), 'numpy'), sp.lambdify(x, sp.diff(f, x, 2), 'numpy')
    print(newton(x0, f_1, f_2, max_iter=10)[-1])
    print(cross(sp.lambdify(x, f, 'numpy'), 0, 4, max_iter=1500)[-1])
    print('--------------------------  gd  ------------------------')
    x, y = sp.symbols('x,y')
    f, x0 = sp.Matrix([2 * y ** 2 + x ** 2 - 2 * x * y - 2 * x]), [-1, 1]
    df = f.jacobian(list(f.free_symbols))
    # from sympy.vector import gradient
    f_ = sp.lambdify(list(f.free_symbols), f, 'numpy')
    x_history = gradient_descent(sp.lambdify(list(df.free_symbols), df, 'numpy'), x0, max_iter=500, alpha=0.002)
    print(f_(*x_history[-1]))
    print(x_history)
