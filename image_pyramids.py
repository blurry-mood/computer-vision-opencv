import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/pedestrian.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


lower_reso = cv.pyrDown(img)
higher_reso = cv.pyrUp(lower_reso)

plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.subplot(222),plt.imshow(lower_reso),plt.title('Downsampled')
plt.subplot(223),plt.imshow(higher_reso),plt.title('Upsampled')
plt.subplot(224),plt.imshow(cv.subtract(img, higher_reso[1:])),plt.title('Laplacian')
plt.show()

