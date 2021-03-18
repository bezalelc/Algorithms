# import pip._vendor.msgpack.fallback
# from pip._vendor.msgpack.fallback import xrange


def naive_match(T, P):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string to search P inside
        @P: short string to search inside T

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
        @T: long string to search P inside
        @P: short string to search inside T
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


def KMP_(T, P):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string to search P inside
        @P: short string to search inside T

    :return: list of indexes where P found in T, index of the first char mach

    :complexity:
        worst-case:O(m+n)
    """
    n, m, s, q = len(T), len(P), 0, 0

    res_indexes = []
    while s <= n - m:
        if T[s + q] == P[q]:
            if q < m - 1:
                q += 1
            else:
                res_indexes.append(s)
                s += 1
        else:
            s += q + 1
            q = 0

    return res_indexes


def KMP(T, P):
    """
    find Sub-shows of P inside T

    @param:
        @T: long string to search P inside
        @P: short string to search inside T

    :return: list of indexes where P found in T, index of the first char mach

    :complexity:
        worst-case:O(2m+n)
    """
    n, m, s, q, pi = len(T), len(P), 0, 0, [0, 0, 1, 2]

    debug_s, debug_q = [s], [q]

    res_indexes = []
    while s <= n - m:
        if T[s + q] == P[q]:
            if q < m - 1:
                q += 1
                debug_q.append(q)
            else:
                res_indexes.append(s)
                s += (m - pi[m - 1])
                q = pi[m - 1]
                debug_s.append(s)
                debug_q.append(q)
        else:
            if q > 0:
                s += (q - pi[q - 1])
                q = pi[q - 1]
                debug_s.append(s)
                debug_q.append(q)
            else:
                s += 1
                debug_s.append(s)

    return res_indexes, debug_s, debug_q


def f(P):
    n, pi, k, q = len(P), [0], 0, 1

    debug_k, debug_q = [k], [q]

    while q <= n - 1:

        if P[k] == P[q]:
            k, q = k + 1, q + 1
            pi.append(k)
            debug_k.append(k), debug_q.append(q)
        else:
            if k > 0:
                k = pi[k]
                debug_k.append(k)
            else:
                pi.append(0)
                q += 1
                debug_q.append(q)

    # --------------- debug ------------------
    print('  P=[', end='')
    for c, i in zip(P, range(len(P))):
        print(c, end=", " if i < len(P) - 1 else "]\n")
    print("pi=", pi, "\ndebug_q=", debug_q, "\ndebug_k=", debug_k)
    # --------------- debug ------------------

    return pi, debug_q, debug_k


if __name__ == '__main__':
    # import gmpy

    T, P = 'aabababbababa', 'abab'
    print(naive_match(T, P))
    print(rabin_karp_match(T, P, q=10))
    # print(KMP_(T, P))
    # print(KMP(T, P))
    # res_indexes, debug_s, debug_q = KMP(T, P)
    # print(res_indexes, "\ndebug_s=", debug_s, "\ndebug_q=", debug_q)
    P = 'abaabcabaaba'
    pi, debug_q, debug_k = f(P)
    print('  P=[', end='')
    for c, i in zip(P, range(len(P))):
        print(c, end=", " if i < len(P) - 1 else "]\n")
    print("pi=", pi, "\ndebug_q=", debug_q, "\ndebug_k=", debug_k)
