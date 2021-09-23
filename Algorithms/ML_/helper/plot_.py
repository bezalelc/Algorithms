import numpy as np


def img_show_data(images: list[np.ndarray], labels: list[str], hspace=0.4, wspace=0.3) -> None:
    import matplotlib.pyplot as plt
    n = int(len(images) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    fig.subplots_adjust(hspace=hspace, wspace=wspace)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel('Label : %s' % labels[i])
    plt.show()


def scatter_data(data_set):
    import matplotlib.pyplot as plt
    for col in data_set.columns:
        print(f'--------------- {col} ----------------')
        print(f'range: [{data_set[col].min()},{data_set[col].max()}], unique: {data_set[col].unique().shape}')
        plt.scatter(range(data_set[col].shape[0]), data_set[col])
        plt.show()


def visualize_grid(Xs, ubound=255.0, padding=1):
    """
    Reshape a 4D tensor of image data to a grid for easy visualization.

    Inputs:
    - Xs: Data of shape (N, H, W, C)
    - ubound: Output grid will have values scaled to the range [0, ubound]
    - padding: The number of blank pixels between elements of the grid
    """
    (N, H, W, C) = Xs.shape
    from math import ceil, sqrt
    grid_size = int(ceil(sqrt(N)))
    grid_height = H * grid_size + padding * (grid_size - 1)
    grid_width = W * grid_size + padding * (grid_size - 1)
    grid = np.zeros((grid_height, grid_width, C))
    next_idx = 0
    y0, y1 = 0, H
    for y in range(grid_size):
        x0, x1 = 0, W
        for x in range(grid_size):
            if next_idx < N:
                img = Xs[next_idx]
                low, high = np.min(img), np.max(img)
                grid[y0:y1, x0:x1] = ubound * (img - low) / (high - low)
                # grid[y0:y1, x0:x1] = Xs[next_idx]
                next_idx += 1
            x0 += W + padding
            x1 += W + padding
        y0 += H + padding
        y1 += H + padding
    # grid_max = np.max(grid)
    # grid_min = np.min(grid)
    # grid = ubound * (grid - grid_min) / (grid_max - grid_min)
    return grid


def show_model_weights(W):
    import matplotlib.pyplot as plt
    W1 = W.reshape(32, 32, 3, -1).transpose(3, 0, 1, 2)
    plt.imshow(visualize_grid(W1, padding=3).astype('uint8'))
    plt.gca().axis('off')
    plt.show()


