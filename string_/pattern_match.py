import numpy as np


def naive_match(T, P):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string_ to search P inside
        @P: short string_ to search inside T

    :return: list of indexes where P found in T, index of the first char mach

    :efficiency: where n,m = len(T),len(P)
        worst-case: O(n*m)
        Average-case: O(n) (=for a random P)
    """
    res_indexes = []

    for c, i in zip(T, range(len(T) - len(P) + 1)):
        found = True
        for c_p, c_t in zip(P, T[i:]):
            if c_p != c_t:
                found = False
                break
        if found:
            res_indexes.append(i)

    return res_indexes


def rabin_karp_match(T, P, q=101, d=256):
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
    h, p, t = (d ** (m - 1)) % q, 0, 0  #
    h = 1
    for i in range(m - 1):
        h = (d * h) % q
    for c1, c2, i in zip(T, P, range(m)):
        t = (t * d + ord(c1)) % q
        p = (p * d + ord(c2)) % q

    res_indexes = []
    for i in range(n - m + 1):
        if t == p:
            found = True
            for c1, c2 in zip(P, T[i:m + i]):
                if c1 != c2:
                    found = False
                    break
            if found:
                res_indexes.append(i)

        if i < n - m:
            t = (d * (t - h * ord(T[i])) + ord(T[m + i])) % q
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
    n = np.unique(list("oiuhni")).shape[0]
    
    pass


if __name__ == '__main__':

    T, P = 'aabababbababaaaababbbabaaabababab', 'abab'
    print(naive_match(T, P))
    print(rabin_karp_match(T, P, q=10))
    print(KMP(T, P))
    mach1, mach2, mach3 = naive_match(T, P), rabin_karp_match(T, P), KMP(T, P)
    print(np.array_equal(mach1, mach2))
    print(np.array_equal(mach3, mach2))
    P = 'abaabcabaaba'
    pi = PI(P)
    print(PI('abaabab'))

    print('-----------------')
    T, P = 'ababbaaabbbaaa', 'bbaa'
    print(naive_match(T, P))
    print(rabin_karp_match(T, P, q=10))
    print(KMP(T, P))

    P = 'ababbabbabbababbabb'
    print(PI(P))
