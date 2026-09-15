import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Step 1: Importing required libraries
print("Libraries imported successfully")


# Step 2: Reading and displaying image

img = cv2.imread("img1.jpg")

if img is None:
    print("Image not found")
    print("Please keep image.png in the same folder as this program")
    exit()

# Display using OpenCV
cv2.imshow("Original Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# OpenCV reads image in BGR format
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Display using Matplotlib
plt.imshow(img_rgb)
plt.title("Original Image")
plt.axis("off")
plt.show()


# Step 3: Image properties

height, width, channels = img.shape

print("\nImage Properties")
print("----------------")
print("Width:", width)
print("Height:", height)
print("Dimensions:", img.shape)
print("Number of Channels:", channels)
print("Data Type:", img.dtype)
print("Total Pixels:", width * height)


# Step 4: Saving image in different formats

cv2.imwrite("output.jpg", img)
cv2.imwrite("output.png", img)

print("\nImage saved successfully")
print("JPEG and PNG files have been created")


# Comparing file sizes

jpg_size = os.path.getsize("output.jpg")
png_size = os.path.getsize("output.png")

print("\nFile Size Comparison")
print("--------------------")
print("JPEG Size:", round(jpg_size / 1024, 2), "KB")
print("PNG Size:", round(png_size / 1024, 2), "KB")


# Step 5: Converting image into different color spaces

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)


# Display color spaces

plt.figure(figsize=(10, 7))

plt.subplot(2, 2, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
plt.title("HSV")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB))
plt.title("LAB")
plt.axis("off")

plt.tight_layout()
plt.show()


# Saving converted images

cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("hsv.jpg", hsv)
cv2.imwrite("lab.jpg", lab)


# Step 6: Geometric transformations

# Resizing
resized = cv2.resize(img, (500, 400))


# Rotation
h, w = img.shape[:2]

center = (w // 2, h // 2)

matrix = cv2.getRotationMatrix2D(center, 90, 1)

rotated = cv2.warpAffine(img, matrix, (w, h))


# Horizontal flip
horizontal = cv2.flip(img, 1)


# Vertical flip
vertical = cv2.flip(img, 0)


# Display transformations

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated 90 Degree")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(horizontal, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(vertical, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")

plt.tight_layout()
plt.show()


# Step 7: Negative / Complement of image

negative = 255 - img

plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative Image")
plt.axis("off")
plt.show()


# Step 8: Cropping ROI

# ROI coordinates
x = 50
y = 50
width_roi = 300
height_roi = 250

roi = img[y:y + height_roi, x:x + width_roi]

if roi.size == 0:
    print("\nROI could not be created")
else:
    print("\nROI Properties")
    print("--------------")
    print("ROI Width:", roi.shape[1])
    print("ROI Height:", roi.shape[0])
    print("ROI Dimensions:", roi.shape)
    print("ROI Data Type:", roi.dtype)

    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.title("Region of Interest")
    plt.axis("off")
    plt.show()


# Step 9: Original and processed images comparison

plt.figure(figsize=(12, 9))

plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("Original")
plt.axis("off")

plt.subplot(3, 3, 2)
plt.imshow(gray, cmap="gray")
plt.title("Grayscale")
plt.axis("off")

plt.subplot(3, 3, 3)
plt.imshow(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
plt.title("Resized")
plt.axis("off")

plt.subplot(3, 3, 4)
plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
plt.title("Rotated")
plt.axis("off")

plt.subplot(3, 3, 5)
plt.imshow(cv2.cvtColor(horizontal, cv2.COLOR_BGR2RGB))
plt.title("Horizontal Flip")
plt.axis("off")

plt.subplot(3, 3, 6)
plt.imshow(cv2.cvtColor(vertical, cv2.COLOR_BGR2RGB))
plt.title("Vertical Flip")
plt.axis("off")

plt.subplot(3, 3, 7)
plt.imshow(cv2.cvtColor(negative, cv2.COLOR_BGR2RGB))
plt.title("Negative")
plt.axis("off")

plt.subplot(3, 3, 8)
plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
plt.title("ROI")
plt.axis("off")

plt.tight_layout()
plt.show()


# Saving processed images

cv2.imwrite("resized.jpg", resized)
cv2.imwrite("rotated.jpg", rotated)
cv2.imwrite("horizontal_flip.jpg", horizontal)
cv2.imwrite("vertical_flip.jpg", vertical)
cv2.imwrite("negative.jpg", negative)

if roi.size != 0:
    cv2.imwrite("roi.jpg", roi)


# Step 10: Observations

print("\n====================================")
print("OBSERVATIONS")
print("====================================")

print("1. The original image is a color image having three channels.")
print("2. Grayscale conversion changes the image into a single channel.")
print("3. HSV and LAB provide different ways to represent color information.")
print("4. Resizing changes the width and height of the image.")
print("5. Rotation changes the orientation of the image.")
print("6. Horizontal and vertical flipping create mirror images.")
print("7. Negative operation inverts the pixel values of the image.")
print("8. ROI helps to select and study a particular part of an image.")
print("9. These preprocessing techniques are useful before image analysis.")
print("10. Image preprocessing is commonly used in computer vision applications.")

print("\nAll operations completed successfully.")