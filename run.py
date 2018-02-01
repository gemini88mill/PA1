"""
Programming assignment #1
Created by: Raphael Miller

PA1 - Due Feb 1st.

Purpose:

To demonstrate the common filters used in Computer Vision applications.
This program is designed to implement various filters for pictures
including but not limited to, box, gaussian, and sobel filters.
"""

"""
run.py - run handles the running of the program, and handles the arguments coming from the command line. 
"""

import sys
import filters
import cv2

# check for args incoming from cli
if len(sys.argv) > 5:
    print("ERROR: too many arguments correct use is run.py [filter_name][kernel_size][image_path]")


def main(filter, kernel_size, image, sigma):
    """
    main function
    reads the image coming from the image path cli arg

    handles the switch of argument/ perams for the PA1 program.
    accepts two arguments, filter and kernel_size

    :argument filter[dict], kernel_size[integer value]
    :returns success or failure value from dict filter

    """

    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # my_dict = {
    #     'box': filters.box(kernel_size, image),
    #     'median': filters.median(kernel_size, image),
    #     'guassian': filters.gaussian(kernel_size, image),
    #     'gradient': filters.gradient(kernel_size, image),
    #     'sobel': filters.sobel(image),
    #     'fast_gaussian': filters.fast_gaussian(kernel_size, image),
    #     'histogram': filters.histogram(kernel_size, image),
    #     'thresholding': filters.thesholding(kernel_size, image)
    # }
    # return my_dict[filter](kernel_size, image)

    if filter == 'box':
        return filters.box(kernel_size, image)
    elif filter == 'median':
        return filters.median(image)
    elif filter == 'gaussian':
        return filters.gaussian(kernel_size, sigma, image)
    elif filter == 'gradient':
        return filters.gradient(image)
    elif filter == 'sobel':
        return filters.sobel(image)
    elif filter == 'fast_gaussian':
        return filters.fast_gaussian(kernel_size, image, sigma)
    elif filter == 'histogram':
        return filters.histogram(kernel_size, image)
    elif filter == 'thresholding':
        return filters.thresholding(kernel_size, image)
    else:
        print("function not recognized")
        return 0

# collect value from main()
res = main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
# print("Status Code: ", res)
sys.exit(res)