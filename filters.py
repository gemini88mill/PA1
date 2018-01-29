import numpy as np
import cv2

def box(kernel_size, image):
    """
    box - box filter function, takes the
    :param image:
    """
    new_image = np.zeros((image.shape[0], image.shape[1]))

    # for x in range(1, image.shape[0] - 1):
    #     for y in range(1, image.shape[1] - 1):
    curr = np.matrix(image[0:3], image[0:3])
    print(curr)

    cv2.imshow('image', image)
    cv2.waitKey(0)

    return 1


def median(kernel_size, image):
    return 2


def gaussian(kernel_size, image):
    return 3


def gradient(kernel_size, image):
    return 4


def sobel(kernel_size, image):
    return 5


def fast_gaussian(kernel_size, image):
    return 6


def histogram(kernel_size, image):
    return 7


def thesholding(kernel_size, image):
    return 8