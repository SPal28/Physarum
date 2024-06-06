
import numpy as np 
import cv2 as cv
img = cv.imread('/Users/subhangipal/Documents/Physarum/physarum frames data/frame0.jpg')
output = img.copy()

#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)

#using circles method from pyimagesearch
circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
for (x, y, r) in circles:
    cv2.circle(output, (x, y), 2, (0, 255, 0), 4)


cv.imshow('output', output)
cv.waitKey(0)
cv.destroyAllWindows()

