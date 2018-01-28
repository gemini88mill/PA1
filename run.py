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

# check for args incoming from cli
if len(sys.argv) > 3:
    print("ERROR: too many arguments correct use is run.py [filter_name][kernel_size][image_path]")


def main(filter, kernel_size):
    """
    main function
    handles the switch of argument/ perams for the PA1 program.
    accepts two arguments, filter and kernel_size

    :argument filter[dict], kernel_size[integer value]
    :returns success or failure value from dict filter

    """
    return {
        'box': filters.box(kernel_size),
        'median': filters.median(kernel_size),
        'guassian': filters.gaussian(kernel_size),
        'gradient': filters.gradient(kernel_size),
        'sobel': filters.sobel(kernel_size),
        'fast_gaussian': filters.fast_gaussian(kernel_size),
        'histogram': filters.histogram(kernel_size),
        'thresholding': filters.thesholding(kernel_size)
    }[filter]

# collect value from main()
res = main(sys.argv[1], sys.argv[2])
# print("Status Code: ", res)
sys.exit(res)