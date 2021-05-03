import numpy as np
import multiply



def fft(coeff):
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

        coeff = coeff if coeff.shape[0] % 2 == 0 else np.append(coeff, 0)
        n = coeff.shape[0]

        y, y0, y1 = np.zeros((n,), dtype=np.complex256), fft_rec(coeff[::2]), fft_rec(coeff[1::2])
        roots = unity_roots(n)
        for k in range(n):
            # n_ = n // 2
            # k_ = k % (n // 2)
            y[k] = y0[k % (n // 2)] + y1[k % (n // 2)] * roots[k]

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
        # roots = np.around(np.cos(theta) - np.sin(theta) * 1j, decimals=10)#+
        roots = np.cos(theta) + np.sin(theta) * 1j  # +
        return roots

    coeff = np.array(coeff, dtype=np.complex256)
    n = coeff.shape[0]
    if n & (n - 1) and n != 0:
        m = 1 << (n - 1).bit_length()
        coeff = np.append(coeff, np.zeros(m - n, ))

    return fft_rec(coeff)


def fft_reverse(DFT):
    """
    FFT^-1 algorithm to calculate the Coefficients polynomial in values representation

    :param DFT: polynomial in DFT representation

    :return: polynomial in Coefficients representation

    :complexity: O(n*log(n))
    """
    b = fft(DFT)
    b[1:] = b[1:][::-1]
    return b / DFT.shape[0]


# def


if __name__ == '__main__':
    print(123 * 456, multiply.eval_coefficient(multiply.mult_fft([3, 2, 1], [6, 5, 4]), 10))
    print(np.round(multiply.mult_fft([1, 1, -1], [-1, 2, 0, 1]), decimals=2))
    print(np.unique(list("oiuhni")))
