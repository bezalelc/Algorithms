import numpy as np


def fft(coeff, round=0):
    """
    FFT algorithm to calculate the DFT of vector of polynomial in Coefficients representation


    :param coeff: polynomial in Coefficients representation

    :return: polynomial in values representation

    :complexity: O(n*log(n))
    """

    def fft_rec(coeff):
        """
        FFT recursive algorithm


        :param coeff: polynomial in Coefficients representation

        :return:

        :complexity: O(n) => for each recursion
        """
        if coeff.shape[0] == 2:
            return np.array([coeff[0] + coeff[1], coeff[0] - coeff[1]])

        # if coeff.shape[0] == 1:
        #     return coeff

        # coeff = coeff if coeff.shape[0] % 2 == 0 else np.append(coeff, 0)
        n = coeff.shape[0]
        y, y0, y1 = np.empty((n,), dtype=np.complex256), fft_rec(coeff[::2]), fft_rec(coeff[1::2])
        roots = unity_roots(n)
        for k in range(n):
            y[k] = y0[k % (n // 2)] + y1[k % (n // 2)] * roots[k]

        if round != 0:
            y = np.around(y, decimals=round)

        return y

    def unity_roots(n):
        """
        compute the unity toots fo Rank(n)

        :param n: rank of complex polynomial x^n=1

        :return: unity roots

        :complexity: O(n)
        """
        k = np.arange(n)
        theta = (2 * np.pi * k) / n

        roots = np.cos(theta) + np.sin(theta) * 1j
        # roots = np.cos(theta) + np.sin(theta) * 1j  # +
        # print(np.around(roots))
        return roots

    coeff = np.array(coeff, dtype=np.complex256)
    n = coeff.shape[0]
    if n & (n - 1) and n != 0:
        m = 1 << (n - 1).bit_length()
        coeff = np.append(coeff, np.zeros(m - n, ))

    return fft_rec(coeff)


def fft_reverse(DFT, round_=0):
    """
    FFT^-1 algorithm to calculate the Coefficients polynomial in values representation

    :param DFT: polynomial in DFT representation
    :param round_: <int between 1 to max digits> round the solution with <round> digits
                  example: if x=[1.0001, 2.3456] and round =2 then x will be [1., 2.34]

    :return: polynomial in Coefficients representation

    :complexity: O(n*log(n))
    """
    b = fft(DFT, round=round_)
    print(b)
    b[1:] = b[1:][::-1]
    return b / DFT.shape[0]


if __name__ == '__main__':
    import multiply

    # print(123 * 456, multiply.eval_coefficient(multiply.mult_fft([3, 2, 1], [6, 5, 4]), 10))
    # print(np.round(multiply.mult_fft([1, 1, -1], [-1, 2, 0, 1]), decimals=2))
    # print(np.unique(list("oiuhni")))
    # print(fft([],[-1,-1,2]))
    print('--------------  test 1  ------------------')
    q, p = [2, 1, 0, 0], [-1, -1, 2, 0]
    # DFT1, DFT2 = fft(q), fft(p)
    # print('q=', np.around(DFT1), ' p=', np.around(DFT2))
    # print(DFT2 * DFT1)
    print(np.around(multiply.mult_fft(p, q)))
    print(np.around(multiply.mult_coefficient(p, q)))
    # print(np.around(multiply.m(p1, p2)))
    print(np.around(fft([0, -5 - 5j, 2, -5 + 5j])))

    print('---------------  test 2 convolution  ---------------------')
    # print(conv([1, 2, 3], [-1, 3, 2]))
    # print(multiply.mult_coefficient([1, 2, 3], [-1, 3, 2]))
    print(np.convolve([1, 2, 3], [-1, 3, 2]))
    fft_conv = np.around(multiply.mult_fft([1, 2, 3], [-1, 3, 2]))
    print(np.array(fft_conv[fft_conv != 0], dtype=np.int_), np.real(fft_conv[fft_conv != 0]))
    # print(fft(np.array([81, 9 + 40j, 1, 9 - 40j], dtype=np.complex256)))
    # print(fft_reverse(np.array([81, 9 + 40j, 1, 9 - 40j], dtype=np.complex256)))
