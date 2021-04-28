import numpy as np


def naive_check(n):
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


def check(n):
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
    a = np.random.randint(2, n - 1)
    x = a
    mask = np.array([int(x) for x in bin(n - 1)[2:]], dtype=np.bool_)[::-1]
    m = mask.shape[0]
    pow_a = np.empty((m,), dtype=np.int_)

    for i in range(m):
        pow_a[i] = x
        x = x ** 2 % n

    pow_a[~mask] = 1

    for i in range(1, m):
        pow_a[i] = (pow_a[i - 1] * pow_a[i]) % n

    return pow_a[-1] == 1


if __name__ == '__main__':
    print(naive_check(6967))
    print(check(6967))
    n = 10000
    test = np.array([naive_check(i) == check(i) for i in range(4, n)])
    print(np.mean(test))
    # for i in range(4, 29):
    #     print(f'{i}: {naive_check(i) == check(i)}')
    # a = 2
    # x = a
    # print(f'k={0}:', 1)
    # for i in range(1, 15):
    #     print(f'k={2 ** i}:', x ** 2 % 6967)
    #     x **= 2
    #
    # print(bin(6967))
    # print(np.array([int(x) for x in bin(6967)[2:]]))
    # print(np.array([int(x) for x in bin(6967)[2:]], dtype=np.bool_)[::-1])
