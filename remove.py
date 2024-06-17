import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = '/Users/subhangipal/Documents/Physarum/indentified circles/indentified circle 6.png'
image = cv2.imread(image_path)

# Convert the image to HSV color space
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range for green color in HSV
lower_green = np.array([40, 100, 100])
upper_green = np.array([80, 255, 255])

# Create a mask for the green circle
mask = cv2.inRange(hsv_image, lower_green, upper_green)

# Find contours of the green circle
contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # Assume the largest contour is the green circle
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Create a mask with the same dimensions as the image
    circle_mask = np.zeros_like(image, dtype=np.uint8)
    
    # Draw the green circle contour on the mask
    cv2.drawContours(circle_mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Create an alpha channel based on the circle mask
    alpha_channel = np.zeros_like(mask, dtype=np.uint8)
    alpha_channel[circle_mask[:, :, 0] == 255] = 255
    
    # Create a 4-channel image (BGR + Alpha)
    b, g, r = cv2.split(image)
    result = cv2.merge([b, g, r, alpha_channel])
    
    # Save the result
    result_path = '/mnt/data/removed_background.png'
    cv2.imwrite(result_path, result)
    
    # Display the result
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title('Background Removed')
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGRA2RGBA))
    plt.show()
    
    print(f"Background removed image saved to {result_path}")
else:
    print("No green circle found.")
