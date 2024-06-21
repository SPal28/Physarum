import os
import numpy as np
import cv2 as cv

# Function to detect circles in an image
def detect_and_draw_circles(img):
    #gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #gray = cv.equalizeHist(gray)
    #gray = cv.medianBlur(gray, 11)
    circles = cv.HoughCircles(img[:,:,2], cv.HOUGH_GRADIENT, dp=1.2, minDist=400,
                              param1=100, param2=30, minRadius=200, maxRadius=400)
    if circles is not None:
        detected_circles = np.uint16(np.around(circles))
        num_circles = len(detected_circles[0])
        print(f"Number of detected circles: {num_circles}")
        return detected_circles[0]
    else:
        print("No circles were detected.")
        return None

# Function to remove background outside the detected circle
def remove_background(img, circles):
    if circles is not None:
        x, y, r = circles[0]
        mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv.circle(mask, (x, y), r, 255, thickness=cv.FILLED)
        masked_img = cv.bitwise_and(img, img, mask=mask)
        b, g, r = cv.split(masked_img)
        alpha_channel = np.zeros_like(b)
        alpha_channel[mask == 255] = 255
        result = cv.merge([b, g, r, alpha_channel])
        return result
    else:
        print("No circles found for background removal.")
        return img

# Define the base directory containing the images to process
base_input_dir = '/Users/subhangipal/Documents/Physarum/cloned frames'
base_output_dir = '/Users/subhangipal/Documents/Physarum/MA074_2'

# Ensure the output base directory exists
os.makedirs(base_output_dir, exist_ok=True)

# Traverse through the folder and process each image
for filename in os.listdir(base_input_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
        img_path = os.path.join(base_input_dir, filename)
        img = cv.imread(img_path)

        # Check if the image is loaded correctly
        if img is None:
            print(f"Error loading image {filename}")
            continue

        # Get the dimensions of the image
        height, width, _ = img.shape

        # Calculate the dimensions of each part
        part_height = height // 3
        part_width = width // 2

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
                folder_path = os.path.join(base_output_dir, folder_name)
                os.makedirs(folder_path, exist_ok=True)

                # Detect and draw circles in the split image
                circles = detect_and_draw_circles(part)

                # Remove background outside the detected circle
                result = remove_background(part, circles)
                result_filename = os.path.join(folder_path, f'{filename}_part_{i*2+j+1}.png')
                cv.imwrite(result_filename, result)
                print(f"Background removed image saved to {result_filename}")

print("Processing complete.")
