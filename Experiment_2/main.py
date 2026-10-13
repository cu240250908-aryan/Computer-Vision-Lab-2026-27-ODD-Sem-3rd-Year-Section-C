import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
img = cv2.imread("images/images(1).jpg")


# Convert image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Display original image
plt.figure(figsize=(10, 6))
plt.imshow(gray, cmap="gray")
plt.title("Original Image")
plt.axis("off")
plt.show()

# Original histogram
plt.figure()
plt.hist(gray.ravel(), 256, [0, 256])
plt.title("Histogram of Original Image")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()

# Contrast stretching
min_val = np.min(gray)
max_val = np.max(gray)

stretched = ((gray - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# Display contrast stretched image
plt.figure(figsize=(10, 6))
plt.imshow(stretched, cmap="gray")
plt.title("Contrast Stretched Image")
plt.axis("off")
plt.show()

# Histogram Equalization
equalized = cv2.equalizeHist(gray)

# Display histogram equalized image
plt.figure(figsize=(10, 6))
plt.imshow(equalized, cmap="gray")
plt.title("Histogram Equalized Image")
plt.axis("off")
plt.show()

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
clahe_img = clahe.apply(gray)

# Display CLAHE image
plt.figure(figsize=(10, 6))
plt.imshow(clahe_img, cmap="gray")
plt.title("CLAHE Enhanced Image")
plt.axis("off")
plt.show()

# Compare histograms
plt.figure(figsize=(10, 6))

plt.hist(gray.ravel(), 256, [0, 256], alpha=0.5, label="Original")
plt.hist(equalized.ravel(), 256, [0, 256], alpha=0.5, label="Equalized")
plt.hist(clahe_img.ravel(), 256, [0, 256], alpha=0.5, label="CLAHE")

plt.title("Comparison of Histograms")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# Save processed images
cv2.imwrite("contrast_stretched.jpg", stretched)
cv2.imwrite("histogram_equalized.jpg", equalized)
cv2.imwrite("clahe.jpg", clahe_img)

print("Experiment 2 completed successfully.")