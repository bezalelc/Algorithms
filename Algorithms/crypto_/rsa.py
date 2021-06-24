import numpy as np
import prime


def keys():
    # p,q
    while True:
        p = np.random.randint(1000, 1e+7)
        if prime.is_prime(p):
            break
    while True:
        q = np.random.randint(1000, 1e+7)
        if prime.is_prime(q):
            break
    # n,l
    n, l = p * q, (p - 1) * (q - 1)

    while True:
        e = np.random.randint(1, l - 1)
        if prime.gcd(l, e) == 1:
            break
    _, _, y = prime.extended_gcd(l, e)
    d = y % l

    # P,S
    P, S = (n, e), (n, d)
    return P, S


def encrypt(M, n, e):
    return prime.pow_mod_large(M, e, n)


def decrypt(C, n, d):
    return prime.pow_mod_large(C, d, n)


if __name__ == '__main__':
    M = np.array([0, 0, 1, 1, 0])
    k = np.array([1, 1, 0, 1, 0])
    C = M ^ k
    M = C ^ k
    print(M)

    M = 0b00110
    k = 0b11010
    print(bin(M ^ k ^ k))
    print('------------------------  generate keys  ------------------------------')
    P, S = keys()
    print(f'P: n={P[0]}, e={P[1]}')
    print(f'S: n={S[0]}, d={S[1]}')
    # C = encrypt(5, *P)
    # print(C)
    # print(decrypt(C, *S))
    # print(f'S: n={(e*d)%(())}')
    # print(prime.gcb(13950055084032, 401612))
    print('----------------------')
    # print(prime.extended_gcd(192, 5))
    # print(encrypt(5, 13, 5))
    # print(decrypt(108, 221, 77))
    print(prime.pow_mod_large(26, 37, 77))
    print(9 % 5)
