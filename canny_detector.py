import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

ratio = 3
kernel_size = 3
window_name = 'Edge Map'
title_trackbar = 'Min Threshold:'
max_lowThreshold = 100

img = cv.imread('images/pedestrian.jpg')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

"""
Always blur to remove noise
"""

blurred = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

"""
Compute edges using Canny detector
"""
def canny(val):
    low_threshold = val
    dst = cv.Canny(gray, low_threshold, low_threshold*ratio, kernel_size)
    mask = dst != 0
    canny_img = blurred * (mask[:,:,None].astype(blurred.dtype))
    cv.imshow(window_name, canny_img)

"""
Plot results
"""

cv.namedWindow(window_name)
cv.createTrackbar(title_trackbar, window_name , 0, max_lowThreshold, canny)
canny(0)
cv.waitKey()