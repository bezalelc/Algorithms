def f(t):
    t[0] = t[0] ** 2
    # print(t)


if __name__ == '__main__':
    # x = sp.symbols('x')
    # f = 1 - x + x ** 2 - x ** 3
    # f = sp.lambdify(x, f, 'numpy')
    # y = np.array([-1, 0, 1, 2])
    # print(f(y))
    # print(np.sin(np.pi))
    # print(np.sin(np.pi))
    a = {'k': 1, 'v': 0}
    b = {'k': 9, 'l': 8}
    # b.update(a)
    print({**a, **b})
    print({**b, **a})
    print({**b, **None})
