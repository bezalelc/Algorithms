import numpy as np


def load_from_file(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1].split(',')
            data.append([float(x) for x in line])
    return np.array(data)
