import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

# ==========================================
# STEP 1: Load Image
# ==========================================

image = cv2.imread("image3.jpg")

if image is None:
    print("Error: images.jpg not found!")
    print("Please keep images.jpg in the same folder as main.py")
    exit()

# Convert BGR to RGB for Matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ==========================================
# STEP 2: Display Original Image
# ==========================================

plt.figure(figsize=(6, 4))
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# ==========================================
# STEP 3: Gaussian Filtering
# ==========================================

start = time.time()

gaussian_blur = cv2.GaussianBlur(image, (5, 5), 0)

gaussian_time = time.time() - start

gaussian_rgb = cv2.cvtColor(gaussian_blur, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(gaussian_rgb)
plt.title("Gaussian Blur")
plt.axis("off")
plt.show()


# ==========================================
# STEP 4: Median Filtering
# ==========================================

start = time.time()

median_filter = cv2.medianBlur(image, 5)

median_time = time.time() - start

median_rgb = cv2.cvtColor(median_filter, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(median_rgb)
plt.title("Median Filter")
plt.axis("off")
plt.show()


# ==========================================
# STEP 5: Average / Mean Filtering
# ==========================================

start = time.time()

average_filter = cv2.blur(image, (5, 5))

average_time = time.time() - start

average_rgb = cv2.cvtColor(average_filter, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(6, 4))
plt.imshow(average_rgb)
plt.title("Average / Mean Filter")
plt.axis("off")
plt.show()


# ==========================================
# STEP 6: Laplacian Filtering
# ==========================================

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

start = time.time()

laplacian_result = cv2.Laplacian(gray, cv2.CV_64F)

laplacian_time = time.time() - start

laplacian = cv2.convertScaleAbs(laplacian_result)

plt.figure(figsize=(6, 4))
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian Filter")
plt.axis("off")
plt.show()


# ==========================================
# STEP 7: Sobel Edge Detection
# ==========================================

# Sobel X - Vertical Edges
start = time.time()

sobel_x_result = cv2.Sobel(
    gray,
    cv2.CV_64F,
    1,
    0,
    ksize=3
)

sobel_x_time = time.time() - start

sobel_x = cv2.convertScaleAbs(sobel_x_result)


# Sobel Y - Horizontal Edges
start = time.time()

sobel_y_result = cv2.Sobel(
    gray,
    cv2.CV_64F,
    0,
    1,
    ksize=3
)

sobel_y_time = time.time() - start

sobel_y = cv2.convertScaleAbs(sobel_y_result)


# Display Sobel X
plt.figure(figsize=(6, 4))
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X - Vertical Edges")
plt.axis("off")
plt.show()


# Display Sobel Y
plt.figure(figsize=(6, 4))
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y - Horizontal Edges")
plt.axis("off")
plt.show()


# ==========================================
# STEP 8: Time Comparison
# ==========================================

print("\n========================================")
print("COMPARISON OF FILTERING TECHNIQUES")
print("========================================")

print(f"Gaussian Filter Time : {gaussian_time:.6f} seconds")
print(f"Median Filter Time   : {median_time:.6f} seconds")
print(f"Average Filter Time  : {average_time:.6f} seconds")
print(f"Laplacian Time       : {laplacian_time:.6f} seconds")
print(f"Sobel X Time         : {sobel_x_time:.6f} seconds")
print(f"Sobel Y Time         : {sobel_y_time:.6f} seconds")


# ==========================================
# STEP 9: Combined Comparison
# ==========================================

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(image_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(gaussian_rgb)
plt.title("Gaussian")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(median_rgb)
plt.title("Median")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(average_rgb)
plt.title("Average")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(laplacian, cmap="gray")
plt.title("Laplacian")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(sobel_x, cmap="gray")
plt.title("Sobel X")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(sobel_y, cmap="gray")
plt.title("Sobel Y")
plt.axis("off")

plt.tight_layout()
plt.show()


# ==========================================
# STEP 10: Save Output Images
# ==========================================

cv2.imwrite("original.jpg", image)
cv2.imwrite("gaussian_blur.jpg", gaussian_blur)
cv2.imwrite("median_filter.jpg", median_filter)
cv2.imwrite("average_filter.jpg", average_filter)
cv2.imwrite("laplacian.jpg", laplacian)
cv2.imwrite("sobel_x.jpg", sobel_x)
cv2.imwrite("sobel_y.jpg", sobel_y)

print("\n========================================")
print("ALL OUTPUT IMAGES SAVED SUCCESSFULLY!")
print("========================================")

print("original.jpg")
print("gaussian_blur.jpg")
print("median_filter.jpg")
print("average_filter.jpg")
print("laplacian.jpg")
print("sobel_x.jpg")
print("sobel_y.jpg")


# ==========================================
# OBSERVATIONS
# ==========================================

print("\n========================================")
print("OBSERVATIONS")
print("========================================")

print("1. Gaussian Filter reduces noise and smooths the image.")
print("2. Median Filter is effective for salt-and-pepper noise.")
print("3. Average Filter performs general image smoothing.")
print("4. Laplacian Filter highlights edges and fine details.")
print("5. Sobel X detects vertical edges.")
print("6. Sobel Y detects horizontal edges.")
print("7. Low-pass filters are mainly used for smoothing.")
print("8. High-pass filters are mainly used for edge enhancement.")