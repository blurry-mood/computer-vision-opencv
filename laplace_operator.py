import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

kernel_size = 3
ddepth = cv.CV_16S

img = cv.imread('images/pedestrian.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


"""
Always blur to remove noise
"""

blurred = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

"""
Compute Laplacian
"""
laplacian = cv.Laplacian(gray, ddepth, ksize=kernel_size)
abs_lap = cv.convertScaleAbs(laplacian)

"""
Plot results
"""

plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.subplot(222),plt.imshow(blurred),plt.title('Blurred')
plt.subplot(223),plt.imshow(abs_lap),plt.title('Image Laplacian')
plt.show()