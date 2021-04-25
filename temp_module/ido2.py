from polynomial import cubic_spline
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
    points90 = np.linspace(-6, 6, num=91)
    # points = np.vstack((points, f(points)))

    points2 = [(x, f(x)) for x in points2]
    points6 = [(x, f(x)) for x in points6]
    points12 = [(x, f(x)) for x in points12]
    points90 = [(x, f(x)) for x in points90]

    splines2 = cubic_spline.cubic_spline4(points2)
    splines6 = cubic_spline.cubic_spline4(points6)
    splines12 = cubic_spline.cubic_spline4(points12)
    splines90 = cubic_spline.cubic_spline4(points90)

    fig = plt.figure(figsize=(10, 10))

    fig221 = fig.add_subplot(221)
    fig221.plot(real_points, f(real_points), label='f')
    fig221.plot(real_points, splines2(real_points), label='spline')
    fig221.set_title('2')
    fig221.set_xlabel('x')
    fig221.set_ylabel('f(x)')

    fig222 = fig.add_subplot(222)
    fig222.plot(real_points, f(real_points))
    fig222.plot(real_points, splines6(real_points))
    fig222.set_title('6')
    fig222.set_xlabel('x')
    fig222.set_ylabel('f(x)')

    fig223 = fig.add_subplot(223)
    fig223.plot(real_points, f(real_points))
    fig223.plot(real_points, splines12(real_points))
    fig223.set_title('12')
    fig223.set_xlabel('x')
    fig223.set_ylabel('f(x)')

    fig224 = fig.add_subplot(224)
    fig224.plot(real_points, f(real_points))
    fig224.plot(real_points, splines90(real_points))
    fig224.set_title('90')
    fig224.set_xlabel('x')
    fig224.set_ylabel('f(x)')

    plt.show()
