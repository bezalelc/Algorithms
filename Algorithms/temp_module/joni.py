import numpy as np
import matplotlib as plot

# x = y = range(10)
# z
def init(x, y, n, h, b):

    for i in range(n):
        h[i] = x[i+1]-x[i]
        b[i] = (6/h[i]) * (y[i+1] - y[i])

def rowReduction:
    u[1] = 2(h[1] + h[0]), v[1] = b[1] - b[0], z[0] = z[n] = 0
    for i in range(2, n):
        u[i] = 2 * (h[i] + h[i-1]) - ((h[i-1])**2)/u[i-1]
        v[i] = b[i] - b[i-1] - (((h[i-1])**2)/u[i-1]) * v[i-1]

def solution:
    for i in range(n-1, 0, -1):
        z[i] = (v[i] - (h[i-1] * z[i+1])) / u[i]




def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')