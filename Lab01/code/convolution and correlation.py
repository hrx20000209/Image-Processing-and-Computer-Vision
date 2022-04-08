import scipy.ndimage
import numpy as np
import cv2

kernel = np.asarray([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=np.float32)
image = cv2.imread('../data/test.jpg', 0)
image_convolution = scipy.ndimage.convolve(image, kernel)
image_correlation = scipy.ndimage.correlate(image, kernel)
cv2.imwrite('correlation.jpg', image_correlation)
cv2.imwrite('convolution.jpg', image_convolution)

