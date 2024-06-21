import cv2 as cv
import numpy as np

# Function to find small white circles and calculate the angle for rotation
def calculate_rotation_angle(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, binary = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    centers = []
    for contour in contours:
        if 5 < cv.contourArea(contour) < 100:  # Filtering small contours
            (x, y), radius = cv.minEnclosingCircle(contour)
            if radius < 10:  # Filtering based on size
                centers.append((x, y))

    if len(centers) >= 2:
        centers = sorted(centers, key=lambda c: c[0])  # Sort by x coordinate
        (x1, y1), (x2, y2) = centers[:2]  # Get the first two centers
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Calculate angle
        return angle
    else:
        return 0  # No rotation needed if less than 2 centers are found

# Function to rotate the image to align small white circles horizontally
def rotate_image(img, angle):
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv.warpAffine(img, M, (w, h), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return rotated

# Load the uploaded image
image_path = '/Users/subhangipal/Documents/Physarum/MA074_2/2/MA074.2_001.jpg_part_2.png'
image = cv.imread(image_path)

# Check if the image is loaded correctly
if image is None:
    raise ValueError("Image not loaded. Check the file path.")

# Calculate rotation angle
angle = calculate_rotation_angle(image)
print(f"Rotation angle: {angle}")

# Rotate the image
rotated_image = rotate_image(image, angle)

# Save the rotated image
# rotated_image_path = '/mnt/data/rotated_image.png'
# cv.imwrite(rotated_image_path, rotated_image)
# print(f"Rotated image saved to {rotated_image_path}")

# Optionally, display the original and rotated images
cv.imshow('Original Image', image)
cv.imshow('Rotated Image', rotated_image)
cv.waitKey(0)
cv.destroyAllWindows()
