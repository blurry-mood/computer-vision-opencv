import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/j.png', 0)

"""
Always blur to remove noise
"""
img = cv.GaussianBlur(img, (3, 3), 0)
kernel = cv.getStructuringElement(cv.MORPH_RECT,(5,5))

"""
Erosion
"""
erosion = cv.erode(img, kernel, iterations = 1)

"""
Dilation
"""
dilation = cv.dilate(img,kernel,iterations = 1)

"""
Opening
"""
opening = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

"""
Closing
"""
closing = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel)

"""
Gradient
"""
gradient = cv.morphologyEx(img, cv.MORPH_GRADIENT, kernel)

"""
Top Hat
"""
tophat = cv.morphologyEx(img, cv.MORPH_TOPHAT, kernel)

"""
Black Hat
"""
blackhat = cv.morphologyEx(img, cv.MORPH_BLACKHAT, kernel)


"""
Plot results
"""
plt.subplot(241),plt.imshow(img),plt.title('Original')
plt.subplot(242),plt.imshow(erosion),plt.title('Erosion')
plt.subplot(243),plt.imshow(dilation),plt.title('Dilation')
plt.subplot(244),plt.imshow(gradient),plt.title('Gradient')

plt.subplot(245),plt.imshow(tophat),plt.title('Top hat')
plt.subplot(246),plt.imshow(blackhat),plt.title('Black hat')
plt.subplot(247),plt.imshow(opening),plt.title('Opening')
plt.subplot(248),plt.imshow(closing),plt.title('Closing')

plt.show()