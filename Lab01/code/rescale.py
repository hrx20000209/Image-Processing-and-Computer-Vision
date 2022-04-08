import numpy as np
import scipy.ndimage
from time import time
from skimage import color, io, transform
import matplotlib.pyplot as plt

image = color.rgb2gray(io.imread('../data/test.jpg'))
n = 20
kernels = []
conv_time = []
corr_time = []
size = [i * 0.25 for i in range(1, n)]

# 生成不同大小的卷积核
for i in range(3, 16, 2):
    kernel = np.asarray([[1] * i] * i, dtype=np.float32)
    kernels.append(kernel)

for kernel in kernels:
    for name in ['conv', 'corr']:
        time_tmp = []
        for i in range(1, n):
            rescale_image = transform.rescale(image, size[i - 1], anti_aliasing=False)
            start_time = time()
            if name == 'conv':
                image_convolution = scipy.ndimage.convolve(rescale_image, kernel)
            else:
                image_correlation = scipy.ndimage.correlate(rescale_image, kernel)
            end_time = time()
            time_tmp.append(end_time - start_time)
        if name == 'conv':
            conv_time.append(time_tmp)
        else:
            corr_time.append(time_tmp)


plt.title("Convolution")
plt.xlabel("size")
plt.ylabel("time")
for i in range(len(conv_time)):
    num = 2 * (i + 1) + 1
    plt.plot(size, conv_time[i], marker='*', label="{} * {}".format(num, num))

plt.legend(loc="upper left")
plt.savefig('result_convolution.jpg')
plt.show()

plt.title("Correlation")
plt.ylabel("time")
plt.xlabel("size")
for i in range(len(corr_time)):
    num = 2 * (i + 1) + 1
    plt.plot(size, corr_time[i], marker='*', label="{} * {}".format(num, num))

plt.legend(loc="upper left")
plt.savefig('result_correlation.jpg')
plt.show()
