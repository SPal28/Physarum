import cv2
import numpy as np

# Load the image
image_path = '/Users/subhangipal/Documents/Physarum/indentified circles/circles.png'
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
    circle_mask = np.zeros_like(image)
    
    # Draw the green circle contour on the mask
    cv2.drawContours(circle_mask, [largest_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
    
    # Convert the mask to grayscale
    circle_mask_gray = cv2.cvtColor(circle_mask, cv2.COLOR_BGR2GRAY)
    
    # Create a masked image where the green circle is kept, and everything else is black
    result = cv2.bitwise_and(image, image, mask=circle_mask_gray)
    
    # Optionally, you can make the background transparent
    # Create an alpha channel based on the circle mask
    alpha_channel = np.where(circle_mask_gray == 255, 255, 0).astype(np.uint8)
    
    # Add the alpha channel to the result image
    b, g, r = cv2.split(result)
    result_with_alpha = cv2.merge([b, g, r, alpha_channel])
    
    # Save the result
    result_path = '/Users/subhangipal/Documents/Physarum/removed background/removed backgrund.png'
    cv2.imwrite(result_path, result_with_alpha)
    
    print(f"Background removed image saved to {result_path}")
else:
    print("No green circle found.")
