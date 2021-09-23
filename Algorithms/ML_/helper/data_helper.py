import numpy as np
from pandas import DataFrame


def one_hot_encode(data_set):
    """
    one hot encode for pandas dataset

    :param data_set:
    :return:
    """
    import pandas as pd
    columns = list(data_set.select_dtypes(['object', 'category']))
    # print(columns)
    for col in columns:
        dummies = pd.get_dummies(data_set[col], prefix=col)
        data_set = pd.concat([data_set, dummies], axis=1)

    data_set.drop(columns, axis=1, inplace=True)
    return data_set


def describe_pandas_df(df: DataFrame):
    print('---------------   head   ------------------')
    print(df.head())
    print('---------------   classes   ------------------')
    print(df.columns)
    print('---------------   shape   ------------------')
    print(df.shape)
    print('---------------   describe   ------------------')
    print(df.describe())
    print('---------------   info   ------------------')
    print(df.info())
    print('---------------   null count   ------------------')
    print(df.isnull().sum().sum())
    print('---------------   correlation   ------------------')
    print(df.corr())
    print('---------------   tail   ------------------')
    print(df.tail())
    print('---------------   sample   ------------------')
    print(df.sample(5))


def resize_images(path_read: str, formats: list[str], new_size: tuple, path_write: str):
    """

    :param path_read:
    :param formats:
    :param new_size:
    :param path_write:
    :return:
    """
    import glob
    import cv2
    import os

    generator = (f_img for f_img in [glob.glob(path_read + '/*.' + format_) for format_ in formats])
    for file_list in generator:
        for file_img in file_list:
            img = cv2.imread(file_img)
            img = cv2.resize(img, new_size)
            cv2.imwrite(os.path.join(path_write, os.path.split(file_img)[-1]), img)


def split_train_val_test(X: np.ndarray, y: np.ndarray, val_size: float = 0.2, test_size: float = 0.2) -> tuple:
    from sklearn.model_selection import train_test_split
    train_val_size = 1. - test_size
    X, Xt, y, Yt = train_test_split(X, y, train_size=train_val_size, random_state=42, stratify=y)
    train_size = train_val_size - val_size
    train_size = train_size / train_val_size
    X, Xv, y, Yv = train_test_split(X, y, train_size=train_size, random_state=42, stratify=y)
    return X, Xv, y, Yv, Xt, Yt
