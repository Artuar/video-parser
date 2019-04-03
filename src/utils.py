import numpy as np
import os
from statistics import mode, StatisticsError


def mse(image_a, image_b):
    # the 'Mean Squared Error' between the two images is the
    # sum of the squared difference between the two images;
    # NOTE: the two images must have the same dimension
    err = np.sum((image_a.astype("float") - image_b.astype("float")) ** 2)
    err /= float(image_a.shape[0] * image_a.shape[1])

    # return the MSE, the lower the error, the more "similar"
    # the two images are
    return err


def clean_directory(directory):
    # clean directory
    file_list = [f for f in os.listdir(directory)]
    for f in file_list:
        remove_file(directory, f)


def remove_file(path, name):
    os.remove(os.path.join(path, name))


def get_most_often_value(arr):
    try:
        return mode(arr)
    except StatisticsError:
        return arr[0] or ''
