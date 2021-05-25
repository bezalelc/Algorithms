import numpy as np
import matplotlib as plot
import sympy

# x = np.arange(9).reshape((3, -1))
# y = np.arange(3)
# print(np.hstack((x, y[:, None])))


import numpy as np
import Algorithms.crypto_.prime as prime


def keys():
    p, q = 23, 29

    # n,l
    n, l = p * q, (p - 1) * (q - 1)
    # print(f'p,q={p, q}, n={n}, l={l}')
    e = 3
    _, _, y = prime.extended_gcd(l, e)
    d = y % l
    print(f'd,x,y={prime.extended_gcd(n, 3)},d={d}')

    # P,S
    P, S = (n, e), (n, d)
    return P, S


def pow_mod_large(x, p, n):
    """
    calculate x^p % n for large numbers

    :param
        x: base number
        p: pow number
        n: modulo number

    :return: x^p % n

    :complexity: O(log(n))
    """
    mask = np.array([int(i) for i in bin(p)[2:]], dtype=np.bool_)[::-1]
    m = mask.shape[0]
    pow_a = np.empty((m,), dtype=np.int_)
    # print(f'm={m}, mask={mask}, pow_a={pow_a}')
    for i in range(m):
        pow_a[i] = x
        x = x ** 2 % n

    print(f'm={m}, mask={mask}, pow_a={pow_a}')
    pow_a[~mask] = 1

    for i in range(1, m):
        pow_a[i] = (pow_a[i - 1] * pow_a[i]) % n

    return pow_a[-1]


P, S = keys()
# import Algorithms.crypto_.rsa as rsa
C = (100 ** P[1]) % P[0]
print((C ** S[1]) % S[0])
