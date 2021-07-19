import numpy as np
import matplotlib as plot
import sympy

# x = np.arange(9).reshape((3, -1))
# y = np.arange(3)
# print(np.hstack((x, y[:, None])))


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
    # print(map_char_val)
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
        print(i + 1, ": ", t)
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

    res_indexes = []
    while s <= n - m:
        if T[s + q] == P[q]:
            if q < m - 1:
                q += 1
            else:
                res_indexes.append(s)
                s += (m - pi[m - 1])
                q = pi[m - 1]
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


def fft_match(T, P):
    """
    find Sub-shows of P inside T using fft

    @param:
        @T: long string_ to search P inside
        @P: short string_ to search inside T

    :return: list of indexes where P found in T, index of the first char mach

    :complexity: O(m*log(n)) where n,m=len(T),len(P)
    """
    char_list_t, char_list_p = list(T), list(P)
    char_unique = np.unique(char_list_t + char_list_p)  # sorted unique chars

    n = char_unique.shape[0]
    if n & (n - 1) and n != 0:
        n = 1 << (n - 1).bit_length()

    import Algorithms.string_.pattern_match
    roots = Algorithms.string_.pattern_match.unity_roots(n)
    char_roots = {c: root for c, root in zip(char_unique, roots)}
    T_, P_ = np.array([char_roots[c] for c in T]), np.array([char_roots[c] for c in P])[::-1]
    import multiply

    # dft_t, dft_p = fft.fft(T_), fft.fft(P_)
    # res = fft.fft_reverse(dft_p * dft_t)
    res = multiply.mult_fft(T_, P_)
    res = np.around(np.real(res), decimals=1)
    res = np.argwhere(res == len(P)) - len(P) + 1
    # res = res[res >= 0]
    # res = res[len(T) - len(P) >= res]
    return res.reshape((-1,))


def f(A, B):
    i, j = 0, 0
    while i < len(A) and j < len(B):
        if A[i] == B[j]:
            return True
    if A[i] < B[j]:
        i += 1
    else:
        j += 1

    return False


# P, S = keys()
# # import Algorithms.crypto_.rsa as rsa
# C = (100 ** P[1]) % P[0]
# print((C ** S[1]) % S[0])
# print('---------------  ex  ----------------')
# # print(KMP("aaaabcabcabcaaa", "abcabc"))
# print(KMP("aaaaaaaaaa", "aaaa"))
# PI('ababbaab')
# print(prime.is_prime(213))
# print(prime.is_prime_naive(213))
# print('----------------------')
# T, P = 'aaaaaaaa', 'aaa'
# print(KMP(T, P))
# print(rabin_karp_match(T, P))
# print('--------------')
# T, P = 'bab', 'ab'
# print(fft_match(T, P))
# # print(rabin_karp_match('aaababaabbbaaababaab', 'abaab', 11))
# print(-205%616)
# T, P = 'abcabcabcd', 'abcabc'
# print(fft_match(T, P))

import fractions
from fractions import Fraction as Fr

# def print(x):
#     print(fractions.Fraction(x).limit_denominator())


p, q = Fr(5, 10), Fr(4, 10)
p1, p2 = Fr(4, 10), Fr(3, 10)
x = 3 ** 2 / 6 ** 3
print(x)
print(x.__float__())
print(Fr(x).limit_denominator())
print(np.linalg.pinv([[2, 0], [4, 8]]))
