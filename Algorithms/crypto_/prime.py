import numpy as np


# ******************************************   gcb   ********************************************
def gcb(a, b):
    """

    :param a:
    :param b:

    :return:

    :complexity: O(log(a+b)) -> worst case
    """

    def gcb_(a, b):
        if b == 0:
            return a
        return gcb_(b, a % b)

    a, b = max(a, b), min(a, b)
    return gcb_(a, b)


def extended_gcb(a, b):
    """

    :param a:
    :param b:

    :return:

    :complexity: O(log(a+b))
    """

    def extended_gcb_(a, b):
        if b == 0:
            return a, 1, 0
        d, x, y = extended_gcb_(b, a % b)
        return d, y, x - (a // b) * y

    a, b = max(a, b), min(a, b)
    return extended_gcb_(a, b)


# ******************************************   check prime   ********************************************
def is_prime_naive(n):
    """
    check if number is prime

    :param n:

    :return: True if num is prime False else

    :efficiency: O(n)
    """
    for i in range(2, n):
        if n % i == 0:
            return False
    return True


def is_prime(n):
    """
    check if number is prime

    :param n:

    :return: True if num is prime False else

    :algorithm: - pick random a in range 2<=a<=n-1
                - calculate a^n-1
                - if a^(n-1) mod n ==1 probably n is prime
                  else n is not prime for sure

    :complexity: O(log(n))

    :accuracy: about 99.4 %
    """
    return pow_mod_large(np.random.randint(2, n - 1), n - 1, n) == 1


# ******************************************   pow_mod_large   *****************************************************
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

    for i in range(m):
        pow_a[i] = x
        x = x ** 2 % n

    pow_a[~mask] = 1

    for i in range(1, m):
        pow_a[i] = (pow_a[i - 1] * pow_a[i]) % n

    return pow_a[-1]


# ******************************************   main test   *****************************************************
if __name__ == '__main__':
    print('--------------------  gcb  ------------------')
    print(gcb(34, 21))
    a, b = 51, 21
    print(extended_gcb(a, b))
    d, x, y = extended_gcb(a, b)
    print(d == x * a + y * b)
    a, b = 18, 7
    print(extended_gcb(a, b))
    d, x, y = extended_gcb(a, b)
    print(d == x * a + y * b)
    a, b = 73, 40
    print(extended_gcb(a, b))
    d, x, y = extended_gcb(a, b)
    print(d == x * a + y * b)
    print('-----------------  check  ---------------------')
    # print(naive_check(6967))
    print(is_prime(6967))
    print(is_prime(345) == is_prime_naive(345))
    print('--------------  pow mod  --------------')
    x, p, n = 31, 7, 541
    print((x ** p) % n == pow_mod_large(x, p, n))
