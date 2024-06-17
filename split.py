import cv2 as cv

# Load the image
img = cv.imread('/Users/subhangipal/Documents/Physarum/numbered image/6 labeled petri dishes.png')  # Update the path to your local image

# Check if the image is loaded correctly
if img is None:
    raise ValueError("Image not loaded. Check the file path.")

# Get the dimensions of the image
height, width, _ = img.shape

# Calculate the dimensions of each part
part_height = height // 3
part_width = width // 2

# Split the image into 6 parts based on the new dimensions
parts_filenames = []
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

        filename = f'/Users/subhangipal/Documents/Physarum/split images/split images_{i*2+j+1}.png'  # Update the path to save the parts
        cv.imwrite(filename, part)
        parts_filenames.append(filename)
        print(f"Saved: {filename} with shape {part.shape}")

# Optionally display the parts
for idx, filename in enumerate(parts_filenames):
    part = cv.imread(filename)
    if part is None:
        print(f"Error loading part {idx+1} from {filename}")
        continue
    cv.imshow(f'Part {idx+1}', part)

cv.waitKey(0)
cv.destroyAllWindows()
