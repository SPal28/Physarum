import numpy as np 
import cv2 as cv

img = cv.imread('/Users/subhangipal/Documents/Physarum/physarum frames data/frame0.jpg')
output = img.copy()

#convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.medianBlur(gray, 5)

#using circles method from pyimagesearch

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)
detected_circles = np.uint16(np.around(circles))
for i in detected_circles[0, :]:
#for value in detected_circles:
    x, y, r = i
    cv.circle(output, (x, y), r, (0, 255, 0), 4)
    cv.circle(output, (x, y), 2, (0, 255, 0), 4)

cv.imshow('output', output)
cv.waitKey(0)
cv.destroyAllWindows()