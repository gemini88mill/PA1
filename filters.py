import numpy as np
import cv2

def box(kernel_size, image):
    """
    box - box filter function, takes the
    :param image:
    """

    kernel = np.ones((int(kernel_size), int(kernel_size)), np.float32)/(int(kernel_size)**2)
    processed_image = cv2.filter2D(image, -1, kernel)

    cv2.imshow('image', processed_image)
    cv2.imshow('original_image', image)
    cv2.waitKey(0)

    return 1


def median(kernel_size, image):
    """
    median filter
    """
    median_index = (int(kernel_size)**2 + 1)/2
    print(median_index)

    


    return 2


def gaussian(kernel_size, image):
    return 3


def gradient(kernel_size, image):
    return 4


def sobel(image):
    kernel_horizontal = np.array(([-1,-2,-1],
                                  [0, 0, 0],
                                  [1, 2, 1]), dtype=int)

    kernel_vertical = np.array(([-1,0,1],
                                [-2,0,2],
                                [-1,0,1]), dtype=int)

    processed_image_h = cv2.filter2D(image, -1, kernel_horizontal)
    processed_image_v = cv2.filter2D(image, -1, kernel_vertical)

    cv2.imshow('horizontal', processed_image_h)
    cv2.imshow('vertical', processed_image_v)
    cv2.imshow('original', image)
    cv2.waitKey(0)
    return 5


def fast_gaussian(kernel_size, image):
    return 6


def histogram(kernel_size, image):
    return 7


def thesholding(kernel_size, image):
    return 8