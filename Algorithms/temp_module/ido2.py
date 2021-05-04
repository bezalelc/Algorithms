from Algorithms.polynomial import cubic_spline
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    y = np.sin(x) + np.cos(x)
    return np.sign(y) * np.abs(y) ** (1 / 3)


if __name__ == '__main__':
    real_points = np.linspace(-6, 6, num=1000)
    points2 = np.linspace(-6, 6, num=3)
    points6 = np.linspace(-6, 6, num=7)
    points12 = np.linspace(-6, 6, num=13)
    points90 = np.linspace(-6, 6, num=30)

    points2 = [(x, f(x)) for x in points2]
    points6 = [(x, f(x)) for x in points6]
    points12 = [(x, f(x)) for x in points12]
    points90 = [(x, f(x)) for x in points90]
    points = [points2, points6, points12, points90]
    points = [np.array(p) for p in points]

    # splines2 = cubic_spline.cubic_spline4(points2)
    # splines6 = cubic_spline.cubic_spline4(points6)
    # splines12 = cubic_spline.cubic_spline4(points12)
    # splines90 = cubic_spline.cubic_spline4(points90)
    spline = [cubic_spline.cubic_spline4(p) for p in points]

    # matrix splines
    # splines2_mat = cubic_spline.cubic_spline4_matrix(points2)
    # splines6_mat = cubic_spline.cubic_spline4_matrix(points6)
    # splines12_mat = cubic_spline.cubic_spline4_matrix(points12)
    # splines90_mat = cubic_spline.cubic_spline4_matrix(points90)
    spline_mat = [cubic_spline.cubic_spline4_matrix(p) for p in points]

    plt.style.use(
        'seaborn')  # seaborn,dark_background,seaborn-bright,grayscale,ggplot,fivethirtyeight,bmh,seaborn-poster
    fig = plt.figure(figsize=(14, 10))
    # f, ax = fig.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
    for p, sp, sp_mat, i in zip(points, spline, spline_mat, range(221, 225, 1)):
        fig_i = fig.add_subplot(i)
        fig_i.plot(real_points, f(real_points), label='f', c='b', alpha=0.5)
        fig_i.plot(real_points, sp(real_points), label='spline', c='k', alpha=1)
        fig_i.plot(real_points, sp_mat(real_points), label='spline mat', c='g', alpha=0.8)
        fig_i.scatter(p[:, 0], sp(p[:, 0]), marker='+', c='r', linewidths=2)
        fig_i.legend(loc="lower left")
        fig_i.set_title(f'{len(p) - 1} splines')
        fig_i.set_xlabel('x')
        fig_i.set_ylabel('f(x)')
    plt.show()

    # fig221 = fig.add_subplot(221)
    # fig221.plot(real_points, f(real_points), label='f', c='b')
    # fig221.plot(real_points, splines2(real_points), label='spline')
    # fig221.scatter(points2, splines2(points2), marker='o')
    # fig221.legend(loc="lower left")
    # fig221.set_title('2 splines')
    # fig221.set_xlabel('x')
    # fig221.set_ylabel('f(x)')
    #
    # fig222 = fig.add_subplot(222)
    # fig222.plot(real_points, f(real_points), label='f')
    # fig222.plot(real_points, splines6(real_points), label='spline')
    # fig222.legend(loc="lower left")
    # fig222.set_title('6 splines')
    # fig222.set_xlabel('x')
    # fig222.set_ylabel('f(x)')
    #
    # fig223 = fig.add_subplot(223)
    # fig223.plot(real_points, f(real_points), label='f')
    # fig223.plot(real_points, splines12(real_points), label='spline')
    # fig223.legend(loc="lower left")
    # fig223.set_title('12 splines')
    # fig223.set_xlabel('x')
    # fig223.set_ylabel('f(x)')
    #
    # fig224 = fig.add_subplot(224)
    # fig224.plot(real_points, f(real_points), label='f')
    # fig224.plot(real_points, splines90(real_points), label='spline')
    # fig224.legend(loc="lower left")
    # fig224.set_title('90 splines')
    # fig224.set_xlabel('x')
    # fig224.set_ylabel('f(x)')
    #
    # plt.show()
