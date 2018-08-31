# -*- coding:utf-8 -*-


import numpy as np
import pickle
import os

########################################################################

data_path = "/home/hechen/Datasets/ImageNet_64x64/python2"

# Width and height of each image.
img_size = 64

# Number of channels in each image, 3 channels: Red, Green, Blue.
num_channels = 3

# Length of an image when flattened to a 1-dim array.
img_size_flat = img_size * img_size * num_channels

# Number of classes.
num_classes = 1000

########################################################################
# Various constants used to allocate arrays of the correct size.

# Total number of images in the training-set.
# This is used to pre-allocate arrays for efficiency.
_num_images_train = 1281167


########################################################################
# Private functions for downloading, unpacking and loading data-files.


def _get_file_path(filename=""):
    """
    Return the full path of a data-file for the data-set.
    If filename=="" then return the directory of the files.
    """

    return os.path.join(data_path, filename)


def _unpickle(filename):
    """
    Unpickle the given file and return the data.
    Note that the appropriate dir-name is prepended the filename.
    """

    # Create full path for the file.
    file_path = _get_file_path(filename)

    print("Loading data: " + file_path)

    import sys
    if sys.version_info <= (3, 3):
        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file)
    else:
        with open(file_path, mode='rb') as file:
            # In Python 3.X it is important to set the encoding,
            # otherwise an exception is raised here.
            data = pickle.load(file, encoding='bytes')

    return data


def _load_mean():

    data = _unpickle("mean")

    mean = data[b'mean']

    return mean


def _convert_images(raw, mean):
    """
    Convert images from the CIFAR-10 format and
    return a 4-dim array with shape: [image_number, height, width, channel]
    where the pixels are floats between 0.0 and 1.0.
    """

    raw_minus_mean = raw - mean

    # Convert the raw images from the data-files to floating-points.
    raw_float = np.array(raw_minus_mean, dtype=float) / 255.0

    # Reshape the array to 4-dimensions.
    images = raw_float.reshape([-1, num_channels, img_size, img_size])

    # Reorder the indices of the array.
    images = images.transpose([0, 2, 3, 1])

    return images


_mean = _load_mean()

def convert_images(raw, mean=_mean):

    return _convert_images(raw, mean)

def _load_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Get the class-numbers for each image. Convert to numpy-array.
    cls = np.array(data[b'labels']) - 1

    # Convert the images.
    images = _convert_images(raw_images, _mean)

    return images, cls, raw_images


def _load_train_data(filename):
    """
    Load a pickled data-file from the CIFAR-10 data-set
    and return the converted images (see above) and the class-number
    for each image.
    """

    # Load the pickled data-file.
    data = _unpickle(filename)

    # Get the raw images.
    raw_images = data[b'data']

    # Convert the images.
    images = _convert_images(raw_images, _mean)

    return images, raw_images


def load_train_data(category_idx, flip):
    """

    :param category_idx: assume it starts at 0 here!
    :param flip: only flip the mean-subtracted normalized images
    :return:
    """

    # Load the images and class-numbers from the data-file.
    images, raw_images = _load_train_data(filename="classes/class_" + str(category_idx + 1))

    # flip
    if flip:
        flipped_images = images.transpose((0, 3, 1, 2))[:, :, :, ::-1].transpose((0, 2, 3, 1))
        images = np.concatenate((images, flipped_images))

    return images, raw_images


def load_test_data():

    images, cls, raw_images = _load_data(filename="val_data")

    return images, cls, one_hot_encoded(class_numbers=cls, num_classes=num_classes), raw_images


def one_hot_encoded(class_numbers, num_classes=None):
    """
    Generate the One-Hot encoded class-labels from an array of integers.
    For example, if class_number=2 and num_classes=4 then
    the one-hot encoded label is the float array: [0. 0. 1. 0.]
    :param class_numbers:
        Array of integers with class-numbers.
        Assume the integers are from zero to num_classes-1 inclusive.
    :param num_classes:
        Number of classes. If None then use max(class_numbers)+1.
    :return:
        2-dim array of shape: [len(class_numbers), num_classes]
    """

    # Find the number of classes if None is provided.
    # Assumes the lowest class-number is 1.
    if num_classes is None:
        num_classes = np.max(class_numbers)

    return np.eye(num_classes, dtype=float)[class_numbers]

########################################################################