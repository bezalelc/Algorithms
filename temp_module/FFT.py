import numpy, cmath, math



if __name__ == '__main__':
    pass
    z = 1 - 1j
    z2 = complex(7, 8)
    z3 = complex(7, 8)
    print(type(z))
    print(z2 != z3)
    print(z2.conjugate())

    alpha = cmath.phase(1 + 1j)
    print(alpha, numpy.degrees(alpha))
    print(numpy.array([1 + 1j, 3j], dtype=numpy.complex256).dtype)
    # import gmpy
