import numpy as np


def load_from_file(file_name, delime=','):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            line = line[:-1].split(delime)
            data.append([float(x) for x in line])
    return np.array(data, dtype=np.float64)
