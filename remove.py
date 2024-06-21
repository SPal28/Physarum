import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Function to detect circles in an image
def detect_and_draw_circles(img, save_path):
    output = img.copy()
    # Convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve contrast
    gray = cv.equalizeHist(gray)
    # Apply a larger median blur to reduce noise
    gray = cv.medianBlur(gray, 11)
    # Use HoughCircles to detect circles
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1.2, minDist=400,
                              param1=100, param2=30, minRadius=200, maxRadius=400)
    # Check if circles were found
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        num_circles = len(detected_circles[0])
        print(f"Number of detected circles: {num_circles}")
        for i in detected_circles[0, :]:
            x, y, r = i
            cv.circle(output, (x, y), r, (0, 255, 0), 4)
            cv.circle(output, (x, y), 2, (0, 255, 0), 4)
        # Save the output image
        cv.imwrite(save_path, output)
        print(f"Saved with circles: {save_path}")
    else:
        print("No circles were detected.")

# Function to remove background outside the detected circle
def remove_background(img, save_path):
    hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # Define the range for green color in HSV
    lower_green = np.array([40, 100, 100])
    upper_green = np.array([80, 255, 255])
    # Create a mask for the green circle
    mask = cv.inRange(hsv_image, lower_green, upper_green)
    # Find contours of the green circle
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    if contours:
        # Assume the largest contour is the green circle
        largest_contour = max(contours, key=cv.contourArea)
        # Create a mask with the same dimensions as the image
        circle_mask = np.zeros_like(img, dtype=np.uint8)
        # Draw the green circle contour on the mask
        cv.drawContours(circle_mask, [largest_contour], -1, (255, 255, 255), thickness=cv.FILLED)
        # Create an alpha channel based on the circle mask
        alpha_channel = np.zeros_like(mask, dtype=np.uint8)
        alpha_channel[circle_mask[:, :, 0] == 255] = 255
        # Create a 4-channel image (BGR + Alpha)
        b, g, r = cv.split(img)
        result = cv.merge([b, g, r, alpha_channel])
        # Save the result
        cv.imwrite(save_path, result)
        print(f"Background removed image saved to {save_path}")
    else:
        print("No green circle found.")

# Load the original image
img_path = '/Users/subhangipal/Documents/Physarum/6 petri dishes frames/MA074.2_008 (Large).jpg'
img = cv.imread(img_path)

# Check if the image is loaded correctly
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Get the dimensions of the image
height, width, _ = img.shape

# Calculate the dimensions of each part
part_height = height // 3
part_width = width // 2

# Define the base directory to save the parts
base_dir = '/Users/subhangipal/Documents/Physarum/MA074_2'

# Ensure the base directory exists
os.makedirs(base_dir, exist_ok=True)

# Split the image into 6 parts based on the new dimensions and save to specific folders
for i in range(3):  # 3 rows
    for j in range(2):  # 2 columns
        y_start = i * part_height
        y_end = (i + 1) * part_height
        x_start = j * part_width
        x_end = (j + 1) * part_width

        # Ensure the end coordinates are within the image boundaries
        y_end = min(y_end, height)
        x_end = min(x_end, width)

        part = img[y_start:y_end, x_start:x_end]

        # Check if the part has valid dimensions
        if part.shape[0] == 0 or part.shape[1] == 0:
            print(f"Skipping empty part at ({i}, {j}) with coordinates: ({y_start}, {y_end}, {x_start}, {x_end})")
            continue

        folder_name = str(i*2+j+1)
        folder_path = os.path.join(base_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        filename = os.path.join(folder_path, f'split_image_{i*2+j+1}.png')
        cv.imwrite(filename, part)
        print(f"Saved: {filename} with shape {part.shape}")

        # Detect and draw circles in the split image
        detect_and_draw_circles(part, filename)

        # Remove background outside the detected circle
        remove_background(part, os.path.join(folder_path, f'removed_background_{i*2+j+1}.png'))

# Optionally display the parts
for i in range(3):
    for j in range(2):
        folder_name = str(i*2+j+1)
        filename = os.path.join(base_dir, folder_name, f'removed_background_{i*2+j+1}.png')
        part = cv.imread(filename)
        if part is None:
            print(f"Error loading part {i*2+j+1} from {filename}")
            continue
        cv.imshow(f'Part {i*2+j+1}', part)

cv.waitKey(0)
cv.destroyAllWindows()
