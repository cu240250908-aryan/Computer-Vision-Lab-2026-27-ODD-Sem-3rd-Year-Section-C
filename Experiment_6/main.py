import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Load the image
image = cv2.imread("image6.jpg")

if image is None:
    print("Image not found!")
    exit()

# Step 2: Convert to grayscale and reduce noise
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# ---------------------------------------------------
# Step 3: Global Three holding
# ---------------------------------------------------

start = time.time()

_, global_threshold = cv2.threshold(
    blur, 127, 255, cv2.THRESH_BINARY
)

global_time = time.time() - start

# ---------------------------------------------------
# Step 4: Otsu's Thresholding
# ---------------------------------------------------

start = time.time()

otsu_threshold_value, otsu = cv2.threshold(
    blur, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

otsu_time = time.time() - start

print("Otsu threshold value:", otsu_threshold_value)

# ---------------------------------------------------
# Step 5: Adaptive Thresholding
# ---------------------------------------------------

start = time.time()

adaptive = cv2.adaptiveThreshold(
    blur,
    255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    11,
    2
)

adaptive_time = time.time() - start

# ---------------------------------------------------
# Step 6: Watershed Segmentation
# ---------------------------------------------------

start = time.time()

# Use Otsu result as binary image
binary = otsu.copy()

# Noise removal
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=2
)

# Find sure background
sure_background = cv2.dilate(
    opening,
    kernel,
    iterations=3
)

# Distance transform
distance = cv2.distanceTransform(
    opening,
    cv2.DIST_L2,
    5
)

# Find sure foreground
_, sure_foreground = cv2.threshold(
    distance,
    0.5 * distance.max(),
    255,
    0
)

sure_foreground = np.uint8(sure_foreground)

# Unknown region
unknown = cv2.subtract(
    sure_background,
    sure_foreground
)

# Create markers
_, markers = cv2.connectedComponents(
    sure_foreground
)

markers = markers + 1
markers[unknown == 255] = 0

# Apply watershed
watershed_image = image.copy()
markers = cv2.watershed(
    watershed_image,
    markers
)

watershed_image[markers == -1] = [0, 0, 255]

watershed_time = time.time() - start

# ---------------------------------------------------
# Step 7: K-Means Color Segmentation
# ---------------------------------------------------

start = time.time()

# Convert image into pixel values
data = image.reshape((-1, 3))
data = np.float32(data)

# Number of clusters
k = 3

criteria = (
    cv2.TERM_CRITERIA_EPS +
    cv2.TERM_CRITERIA_MAX_ITER,
    100,
    0.2
)

_, labels, centers = cv2.kmeans(
    data,
    k,
    None,
    criteria,
    10,
    cv2.KMEANS_RANDOM_CENTERS
)

centers = np.uint8(centers)

segmented_data = centers[labels.flatten()]
kmeans_image = segmented_data.reshape(image.shape)

kmeans_time = time.time() - start

# ---------------------------------------------------
# Step 8: Compare Results
# ---------------------------------------------------

print("\nExecution Time:")
print("Global Thresholding:", round(global_time, 5), "seconds")
print("Otsu Thresholding:", round(otsu_time, 5), "seconds")
print("Adaptive Thresholding:", round(adaptive_time, 5), "seconds")
print("Watershed:", round(watershed_time, 5), "seconds")
print("K-Means:", round(kmeans_time, 5), "seconds")

# ---------------------------------------------------
# Save results
# ---------------------------------------------------

cv2.imwrite("gray.jpg", gray)
cv2.imwrite("global_threshold.jpg", global_threshold)
cv2.imwrite("otsu_threshold.jpg", otsu)
cv2.imwrite("adaptive_threshold.jpg", adaptive)
cv2.imwrite("watershed.jpg", watershed_image)
cv2.imwrite("kmeans.jpg", kmeans_image)

# ---------------------------------------------------
# Step 9: Display all segmented images
# ---------------------------------------------------

plt.figure(figsize=(12, 10))

plt.subplot(3, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.imshow(global_threshold, cmap="gray")
plt.title("Global Thresholding")
plt.axis("off")

plt.subplot(3, 2, 3)
plt.imshow(otsu, cmap="gray")
plt.title("Otsu Thresholding")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.imshow(adaptive, cmap="gray")
plt.title("Adaptive Thresholding")
plt.axis("off")

plt.subplot(3, 2, 5)
plt.imshow(cv2.cvtColor(watershed_image, cv2.COLOR_BGR2RGB))
plt.title("Watershed Segmentation")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.imshow(cv2.cvtColor(kmeans_image, cv2.COLOR_BGR2RGB))
plt.title("K-Means Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()

# ---------------------------------------------------
# Step 10: Observations
# ---------------------------------------------------

print("\nObservations:")
print("1. Global three holding separates the image using a fixed threshold.")
print("2. Otsu's method automatically selects a suitable threshold value.")
print("3. Adaptive three holding works better when image illumination is uneven.")
print("4. Watershed segmentation is useful for separating touching objects.")
print("5. K-Means groups pixels according to their color similarity.")
print("6. Different segmentation techniques are suitable for different images.")
print("7. Image segmentation is useful in medical imaging, satellite images,")
print("   object detection, autonomous systems and image analysis.")