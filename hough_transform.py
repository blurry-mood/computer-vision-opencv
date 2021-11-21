import matplotlib
matplotlib.use('Qt5Agg')

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('images/chess board.png')
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img2 = img.copy()
img3 = img.copy()

"""
Always blur to remove noise
"""
blurred = cv.GaussianBlur(img, (3, 3), 0)
gray = cv.cvtColor(blurred, cv.COLOR_BGR2GRAY)

"""
Detect lines
"""
edges = cv.Canny(gray,50,150,apertureSize = 3)
lines = cv.HoughLines(edges,1,np.pi/180,500)
for line in lines:
    rho,theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 800*(-b))
    y1 = int(y0 + 800*(a))
    x2 = int(x0 - 800*(-b))
    y2 = int(y0 - 800*(a))
    cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)

"""
Detect circles
""" 
circles = cv.HoughCircles(gray,cv.HOUGH_GRADIENT,1,20,
                            param1=50,param2=30,minRadius=10,maxRadius=20)
circles = np.uint16(np.around(circles))
for i in circles[0,:]:
    # draw the outer circle
    cv.circle(img2,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
    cv.circle(img2,(i[0],i[1]),2,(0,0,255),3)

"""
Plot results
"""
plt.subplot(221),plt.imshow(img3),plt.title('Original image')
plt.subplot(222),plt.imshow(edges),plt.title('Canny output')
plt.subplot(223),plt.imshow(img),plt.title('Hough lines')
plt.subplot(224),plt.imshow(img2),plt.title('Hough circles')

plt.show()