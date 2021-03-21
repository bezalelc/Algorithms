import numpy as np

if __name__ == '__main__':
    x = np.array([2, 3])
    np.insert(x,0,[0,], axis=0)
    print(x)

