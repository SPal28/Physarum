import cv2 as cv

# Load the image
img = cv.imread('/Users/subhangipal/Documents/Physarum/6 petri dishes frames/MA074.2_008 (Large).jpg')  # Update the path to your local image

# Check if the image is loaded correctly
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Get the dimensions of the image
height, width, _ = img.shape

# Calculate the dimensions of each part
part_height = height // 3
part_width = width // 2

# Define font and other drawing parameters
font = cv.FONT_HERSHEY_SIMPLEX
font_scale = 3
font_color = (0, 0, 255)  # Red color in BGR
thickness = 3
line_type = cv.LINE_AA

# Define the desired numbering pattern
numbering_pattern = [
    (0, 0, 1), (1, 0, 2), (2, 0, 3),  # Left column
    (0, 1, 4), (1, 1, 5), (2, 1, 6)   # Right column
]

# Loop through the numbering pattern and add the number labels
for row, col, number in numbering_pattern:
    y_center = row * part_height + part_height // 2
    x_center = col * part_width + part_width // 2

    # Put the number at the center of each part
    cv.putText(img, str(number), (x_center - 10, y_center + 10), font, font_scale, font_color, thickness, line_type)

# Save the labeled image
cv.imwrite('/path/to/save/labeled_image.png', img)  # Update the path to save the labeled image

# Display the labeled image
cv.imshow('Labeled Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
