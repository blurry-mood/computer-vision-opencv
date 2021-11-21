import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/pedestrian.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

"""
Average filter
"""
N = 19
kernel = np.ones((N,N),np.float32)
kernel = kernel / kernel.sum()
dst = cv.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.show()

"""
Blurring
"""

blur = cv.blur(img,(N,N))
gaussian_blur = cv.GaussianBlur(img,(N, N),75)
median_blur = cv.medianBlur(img, N)
bilateral_blur = cv.bilateralFilter(img,N,75,75)

plt.subplot(231),plt.imshow(img),plt.title('Original')
plt.subplot(232),plt.imshow(bilateral_blur),plt.title('Bilateral Blurred: keeps edges')
plt.subplot(233),plt.imshow(gaussian_blur),plt.title('Gaussian Blurred')
plt.subplot(234),plt.imshow(blur),plt.title('Mean Blurred')
plt.subplot(235),plt.imshow(median_blur),plt.title('Median Blurred')
plt.show()