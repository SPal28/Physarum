import cv2 as cv
import numpy as np

# Function to find the small white circle using Hough Circle Transform
def find_circle_and_calculate_angle(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)
    
    # Detect circles in the image
    circles = cv.HoughCircles(
        gray, 
        cv.HOUGH_GRADIENT, 
        dp=1.9, 
        minDist=400, 
        param1=50, 
        param2=30, 
        minRadius=5, 
        maxRadius=20
    )

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            # Assuming the largest circle found is the target
            center = (x, y)
            img_center = (img.shape[1] // 2, img.shape[0] // 2)
            angle = np.arctan2(center[1] - img_center[1], center[0] - img_center[0]) * 180 / np.pi
            return angle

    return None  # No circle found

# Function to rotate the image to align the single white circle at the 180° line
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# Load the uploaded image
image_path = '/Users/subhangipal/Documents/Physarum/MA074_2/2/MA074.2_001.jpg_part_2.png'  # Update this path as needed
image = cv.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Find the circle and calculate the rotation angle
angle = find_circle_and_calculate_angle(image)
if angle is not None:
    # Rotate the image by 180 degrees minus the calculated angle
    rotated_image = rotate_image(image, 180 - angle)
else:
    rotated_image = image  # No rotation needed if no circle is found

# Save the rotated image
#rotated_image_path = '/mnt/data/rotated_image.png'  # Update this path as needed
#cv.imwrite(rotated_image_path, rotated_image)
#print(f"Rotated image saved to {rotated_image_path}")

# Optionally, display the original and rotated images
cv.imshow('Original Image', image)
cv.imshow('Rotated Image', rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
