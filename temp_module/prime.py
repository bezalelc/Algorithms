def gcb(a, b):
    """

    :param a:
    :param b:

    :return:

    :complexity: O(log(a+b))
    """
    if b == 0:
        return a
    return gcb(b, a % b)


def extended_gcb(a, b):
    """

    :param a:
    :param b:

    :return:

    :complexity: O(log(a+b))
    """
    if b == 0:
        return a, 1, 0
    d, x, y = extended_gcb(b, a % b)
    return d, y, x - (a // b) * y


if __name__ == '__main__':
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
