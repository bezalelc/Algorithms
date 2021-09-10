import numpy as np


def img_show_data(images: list[np.ndarray], labels: list[str]) -> None:
    import matplotlib.pyplot as plt
    n = int(len(images) ** 0.5)
    fig, axes = plt.subplots(n, n, figsize=(n * 2, n * 2))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i])
        ax.set_xlabel('Label : %s' % labels[i])
    plt.show()
