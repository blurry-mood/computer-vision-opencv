import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

scale = 1
delta = 0
ddepth = cv.CV_16S

img = cv.imread('images/pedestrian.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)


"""
Always blur to remove noise
"""

blurred = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

"""
Compute gradients using Sobel kernel
"""
grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)
sobel_grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

"""
Compute gradients using Scharr kernel
"""
grad_x = cv.Scharr(gray, ddepth, 1, 0, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
grad_y = cv.Sobel(gray, ddepth, 0, 1, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
abs_grad_x = cv.convertScaleAbs(grad_x)
abs_grad_y = cv.convertScaleAbs(grad_y)
scharr_grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

"""
Plot results
"""

plt.subplot(221),plt.imshow(img),plt.title('Original')
plt.subplot(222),plt.imshow(blurred),plt.title('Blurred')
plt.subplot(223),plt.imshow(sobel_grad),plt.title('Gradient Magnitude using Sobel')
plt.subplot(224),plt.imshow(scharr_grad),plt.title('Gradient Magnitude using Scharr')
plt.show()