import numpy as np
import cv2
import scipy.stats as st
import matplotlib.pyplot as plt


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


def median(image):
    """
    median filter
    """
    # median_index = (int(kernel_size)**2 + 1)/2
    # print(median_index)

    # kernel = np.zeros((int(kernel_size), int(kernel_size)), dtype=np.float32)
    # kernel = image[0:3, 0:3]
    # median = np.median(kernel)
    # print(median)

    new_image = np.zeros((image.shape))
    # print(new_image)

    for x in range(1, image.shape[0]):
        for y in range(1, image.shape[1]):
            kernel = image[x:(x+3), y:(y+3)]
            new_image[x,y] = np.median(kernel)

    cv2.imshow('median', new_image)
    cv2.imshow('original', image)
    cv2.waitKey(0)

    return 2


def gaussian(kernel_size, sigma, image):
    """
    gaussian filter - implements a 2d gaussian filter and iterates over the image using the built in convolution
    """

    interval = (2 * int(sigma) + 1.) / (int(kernel_size))
    x = np.linspace(-int(sigma) - interval / 2., int(sigma) + interval / 2., int(kernel_size) + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    kernel.reshape(int(kernel_size), int(kernel_size))


    processed_image = cv2.filter2D(image, -1, kernel)

    cv2.imshow('image', processed_image)
    cv2.imshow('original_image', image)
    cv2.waitKey(0)
    return 3


def gradient(image):
    """
    gradient filter - collects the image and applies a gradient filter over the image,
    produces two images, one is a filter in the x direction, the other is a filter in the y direction
    """


    new_image_x = np.diff(image, axis=1)
    new_image_y = np.diff(image, axis=0)


    grad_x = np.power(new_image_x, 2)
    grad_y = np.power(new_image_y, 2)

    cv2.imshow('x', new_image_x)
    cv2.imshow('y', new_image_y)
    cv2.waitKey(0)

    return 4


def sobel(image):
    """
    sobel filter - implements two sobel filters, one in a vertical position and one in a horizontal position
    convultion using the standard method linear filter.
    """

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


def fast_gaussian(kernel_size, image, sigma):


    return 6


def histogram(bins, image):
    """
    histogram - creates a histogram without the histogram function built-in
    function iterates over the array and assigns a value that would represent the
    histogram bin, then a value in a histgram array is tallied and displayed using
    the matplotlib functions.
    """

    hist_array = np.zeros((int(bins) + 1), dtype=int)
    print(hist_array.shape)
    divisor = round((np.amax(image) / int(bins)))
    print(divisor)


    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            value = image[x,y] / int(divisor)
            hist_array[int(value)] += 1

    x_bar = np.arange(len(hist_array))

    plt.bar(x_bar, hist_array)
    # plt.xticks(len(hist_array), hist_array)
    plt.show()

    return 7


def thresholding(t_val, image):
    """
    Thresholding - the thresholding function implements a t_value in order to
    create a threshold binary image. 1 for values above the t_val and 0 for values
    under that threshold.
    """


    bin_image = np.zeros(image.shape)

    for x in range(0, image.shape[0]):
        for y in range(0, image.shape[1]):
            if image[x,y] > int(t_val):
                bin_image[x,y] = 1

    cv2.imshow('bin image', bin_image)
    cv2.imshow('original', image)
    cv2.waitKey(0)

    return 8