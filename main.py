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


def rabin_karp_match(T, P, q=101):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string_ to search P inside
        @P: short string_ to search inside T
        @q: Prime number
        @d: Counting base by default is 256 -> the range of ascii table

    :return: list of indexes where P found in T, index of the first char mach

    :complexity: where n,m,t,(n/q) = len(T),len(P),Sub-shows number,shows where tha hash was equal but not the Sub-show
        Average-case: O(n + m*t + m*(n/q))
    """
    n, m = len(T), len(P)

    # -----
    chars = np.unique(list(T))
    d = chars.shape[0]
    map_char_val = {c: i for c, i in zip(chars, range(d))}
    print(map_char_val)
    # -----

    # h = (d ** (m - 1)) % q in base d
    h, p, t = 1, 0, 0
    for i in range(m - 1):
        h = (d * h) % q
    for c1, c2, i in zip(T, P, range(m)):
        t = (t * d + map_char_val[c1]) % q
        p = (p * d + map_char_val[c2]) % q

    res_indexes = []
    for i in range(n - m + 1):
        print(t)
        if t == p:
            found = True
            for c1, c2 in zip(P, T[i:m + i]):
                if c1 != c2:
                    found = False
                    break
            if found:
                res_indexes.append(i)

        if i < n - m:
            t = (d * (t - h * map_char_val[T[i]]) + map_char_val[T[m + i]]) % q
            t = t if t >= 0 else t + q

    return res_indexes


def KMP(T, P):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string_ to search P inside
        @P: short string_ to search inside T

    :return: list of indexes where P found in T, index of the first char mach

    :complexity:
        worst-case:O(2m+n)
    """
    n, m, s, q, pi = len(T), len(P), 0, 0, PI(P)
    print(pi)

    res_indexes = []
    while s <= n - m:
        if T[s + q] == P[q]:
            if q < m - 1:
                q += 1
            else:
                res_indexes.append(s)
                s += (m - pi[q - 1])
                q = pi[q - 1]
        else:
            if q > 0:
                s += (q - pi[q - 1])
                q = pi[q - 1]
            else:
                s += 1

    return res_indexes


def PI(P):
    """
    find for all sub-string_ in P the max length of prefix that is also suffix

    :return
        pi: list that contain for each char in P the corresponding max length

    :complexity: O(n)
    """
    n, pi, k, q = len(P), [0], 0, 1

    while q < n:
        if P[k] == P[q]:
            k, q = k + 1, q + 1
            pi.append(k)
        else:
            if k > 0:
                k = pi[k - 1]
            else:
                pi.append(0)
                q += 1

    return pi


P, S = keys()
# import Algorithms.crypto_.rsa as rsa
C = (100 ** P[1]) % P[0]
print((C ** S[1]) % S[0])
print('---------------  ex  ----------------')
# print(KMP("aaaabcabcabcaaa", "abcabc"))
print(KMP("ABCABCABCABCABCABC", "ABCABCD"))
