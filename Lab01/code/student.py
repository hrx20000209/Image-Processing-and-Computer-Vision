# Spring 2022: Image Processing and Computer Vision
# Beihang Univeristy
# Homework set 1
# Lu Sheng (lsheng@buaa.edu.cn)
#
# Implement my_imfilter() and gen_hybrid_image()
#
# Based on previous and current work
# by James Hays for CSCI 1430 @ Brown and
# CS 4495/6476 @ Georgia Tech
import numpy as np
from numpy import pi, exp, sqrt
from skimage import io, img_as_ubyte, img_as_float32
from skimage.transform import rescale


def my_imfilter(image, kernel):
    """
    Your function should meet the requirements laid out on the project webpage.
    Apply a filter (using kernel) to an image. Return the filtered image. To
    achieve acceptable runtimes, you MUST use numpy multiplication and summation
    when applying the kernel.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """

    ##################
    # Your code here #
    ##################

    rotate_kernel = np.flip(kernel)

    kernel_row, kernel_col = rotate_kernel.shape

    if (kernel_row % 2 == 0) or (kernel_col % 2 == 0):
        raise Exception('Kernel is not of odd dimensions')

    color_channel = []
    num_channel = 1
    if len(image.shape) == 2:
        # grayscale image
        color_channel.append(image)
    elif len(image.shape) == 3:
        # RGB image
        num_channel = image.shape[2]
        for i in range(image.shape[2]):
            color_channel.append(image[:, :, i])
    else:
        return

    padding_row = kernel_row // 2
    padding_col = kernel_col // 2

    for i in range(num_channel):
        channel = color_channel[i]
        result = np.zeros(channel.shape, dtype=np.float32)
        channel_padded = np.pad(
            channel, [(padding_row, padding_row), (padding_col, padding_col)], mode='constant')
        for col in range(channel.shape[1]):
            for row in range(channel.shape[0]):
                result[row, col] = (
                        rotate_kernel * channel_padded[row: row + kernel_row, col: col + kernel_col]).sum()
        color_channel[i] = result

    if num_channel == 1:
        filtered_image = color_channel[0]
    else:
        filtered_image = np.stack(color_channel, axis=2)
    return filtered_image


"""
EXTRA CREDIT placeholder function
"""


def my_imfilter_fft(image, kernel):
    """
    Your function should meet the requirements laid out in the extra credit section on
    the project webpage. Apply a filter (using kernel) to an image. Return the filtered image.
    Inputs
    - image: numpy nd-array of dim (m,n) or (m, n, c)
    - kernel: numpy nd-array of dim (k, l)
    Returns
    - filtered_image: numpy nd-array of dim of equal 2D size (m,n) or 3D size (m, n, c)
    Errors if:
    - filter/kernel has any even dimension -> raise an Exception with a suitable error message.
    """
    filtered_image = np.zeros(image.shape)

    ##################
    # Your code here #
    print('my_imfilter_fft function in student.py is not implemented')
    ##################

    return filtered_image


def gen_hybrid_image(image1, image2, cutoff_frequency):
    """
     Inputs:
     - image1 -> The image from which to take the low frequencies.
     - image2 -> The image from which to take the high frequencies.
     - cutoff_frequency -> The standard deviation, in pixels, of the Gaussian
                           blur that will remove high frequencies.

     Task:
     - Use my_imfilter to create 'low_frequencies' and 'high_frequencies'.
     - Combine them to create 'hybrid_image'.
    """

    assert image1.shape[0] == image2.shape[0]
    assert image1.shape[1] == image2.shape[1]
    assert image1.shape[2] == image2.shape[2]

    # Steps:
    # (1) Remove the high frequencies from image1 by blurring it. The amount of
    #     blur that works best will vary with different image pairs
    # generate a 1x(2k+1) gaussian kernel with mean=0 and sigma = s, see https://stackoverflow.com/questions/17190649/how-to-obtain-a-gaussian-filter-in-python
    s, k = cutoff_frequency, cutoff_frequency * 2
    probs = np.asarray([exp(-z * z / (2 * s * s)) / sqrt(2 * pi * s * s) for z in range(-k, k + 1)], dtype=np.float32)
    kernel = np.outer(probs, probs)

    # Your code here:
    low_frequencies = my_imfilter(image1, kernel)

    # (2) Remove the low frequencies from image2. The easiest way to do this is to
    #     subtract a blurred version of image2 from the original version of image2.
    #     This will give you an image centered at zero with negative values.
    # Your code here #
    high_frequencies = np.subtract(image2, my_imfilter(image2, kernel))

    # (3) Combine the high frequencies and low frequencies
    # Your code here #
    hybrid_image = np.add(high_frequencies, low_frequencies)

    # (4) At this point, you need to be aware that values larger than 1.0
    # or less than 0.0 may cause issues in the functions in Python for saving
    # images to disk. These are called in proj1_part2 after the call to 
    # gen_hybrid_image().
    # One option is to clip (also called clamp) all values below 0.0 to 0.0, 
    # and all values larger than 1.0 to 1.0.
    high_frequencies = np.clip(high_frequencies, 0.0, 1.0)
    hybrid_image = np.clip(hybrid_image, 0.0, 1.0)

    return low_frequencies, high_frequencies, hybrid_image
